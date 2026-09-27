import multiprocessing
from pathlib import Path
import random

multiprocessing.set_start_method("spawn", force=True)

from model.dmm import DMMConfig, DMM
from model.dmm_08m import DMM08MConfig, DMM08M
from loguru import logger
from training.pretrain.aggregated_data_loader import AggregatedMapfArrowDataset
import math
import numpy as np
import os
import time
from contextlib import nullcontext
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.tensorboard import SummaryWriter

# -----------------------------------------------------------------------------
# I/O
train_dir = "train_dir"
exp_dir_name = "trial"
resume_run_dir = ""      # path to run dir to resume from, e.g. "train_dir/trial"

eval_interval = 500      # 0 = disable eval entirely
ckpt_freq = 5000         # save ckpt_{iter}.pt every N steps; 0 = disable
latest_ckpt_freq = 100   # save ckpt_latest.pt every N training steps (overwrites)
log_interval = 1
eval_iters = 40
init_from = "scratch"    # "scratch" | "pretrained" | "resume"
weight_path = ""  # Set explicitly only when loading pretrained weights.
freeze_encoder = False
model_class = "dmm"

# DMM architecture.
n_encoder_layer = 2
n_decoder_layer = 2
n_head = 2
n_embd = 16
latent_embd = 8
latent_tok_n = 8
action_msg_feats = 16

# DMM08M architecture. Only the selected model's fields enter model_args.
dmm08m_width = 96
dmm08m_heads = 4
dmm08m_conv_blocks = 3
dmm08m_mixer_blocks = 2
dmm08m_spatial_size = 5
dmm08m_spatial_mode = "pooled"
dmm08m_local_spatial_size = 7
dmm08m_neighbor_tokenization = "flat"
dmm08m_communication_query_blocks = 2
dmm08m_communication_hidden_multiplier = 2
n_comm_rounds = 2
dropout = 0.0
bias = False

# Observation layout
block_size = 256
field_of_view_size = 11 * 11
agent_info_size = 10
max_num_neighbors = 13
empty_connection_code = -1

# Dynamics
dt = 0.25
tau = 1.0

# Dirichlet z_0 teacher forcing
dirichlet_tf_on = True
dirichlet_tf_beta = 1.0
dirichlet_tf_beta_final = 0.0
dirichlet_tf_anneal_steps = 50_000

# Round z-update teacher forcing
round_tf_on = True
round_tf_beta = 1.0
round_tf_beta_final = 0.0
round_tf_anneal_steps = 50_000

# Optimiser
gradient_accumulation_steps = 16
batch_size = 8
learning_rate = 6e-4
max_iters = 30000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 30000
min_lr = 6e-5

# DDP
backend = "nccl"
device = "cuda"

if "cuda" in device and not torch.cuda.is_available():
    device = "cpu"
    logger.warning(f"Cuda not available, switching to {device}")

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = False
compile_mode = "default"
seed = 1337
seed_data_loader = False
# Declared before the override below so a config file can repoint the dataset.
train_data_files = [
    "training/pretrain/data/train/mazes",
    "training/pretrain/data/train/random",
    "training/pretrain/data/train/house",
]
batch_sizes = []  # empty splits batch_size across those three families
# -----------------------------------------------------------------------------

config_keys = [
    k for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec((Path(__file__).with_name("configurator.py")).read_text())
config = {k: globals()[k] for k in config_keys}

model_types = {
    "dmm": (DMM, DMMConfig),
    "dmm08m": (DMM08M, DMM08MConfig),
}
if model_class not in model_types:
    raise ValueError(f"Unknown model_class: {model_class!r}")
model_type, model_config = model_types[model_class]

# ── DDP init ─────────────────────────────────────────────────────────────────
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

worker_seed = seed + seed_offset
if seed_data_loader:
    # Seed all RNGs before creating the Arrow iterator in this mode.
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

# ── Dataset (created after DDP so device is correct per worker) ───────────────
if not batch_sizes:
    mazes_size = batch_size * 6 // 10
    house_size = batch_size * 2 // 10
    batch_sizes = [mazes_size, batch_size - mazes_size - house_size, house_size]

train_data = AggregatedMapfArrowDataset(train_data_files, device=device, batch_sizes=batch_sizes)
train_data_iter = iter(train_data)


def calculate_epochs(max_iters, dataset_size, batch_size, gradient_accumulation_steps=1):
    effective_batch_size = batch_size * gradient_accumulation_steps
    steps_per_epoch = dataset_size // effective_batch_size
    return max_iters / steps_per_epoch


def human_readable_size(size):
    for unit in ["pairs", "K pairs", "M pairs", "B pairs"]:
        if size < 1000:
            return f"{size:.2f} {unit}"
        size /= 1000
    return f"{size:.2f} B pairs"


if master_process:
    logger.info(f"Train set size: {human_readable_size(train_data.get_full_dataset_size())}")
    num_epochs = calculate_epochs(
        max_iters, train_data.get_full_dataset_size(), batch_size, gradient_accumulation_steps
    )
    logger.info(f"Number of training epochs: {num_epochs:.2f}")

# ── Experiment directory ──────────────────────────────────────────────────────
if init_from == "resume":
    if not resume_run_dir:
        raise ValueError("Set resume_run_dir to the run directory you want to resume from.")
    exp_dir = Path(resume_run_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"resume_run_dir not found: {exp_dir}")
else:
    exp_dir = Path(train_dir) / exp_dir_name

if master_process:
    if exp_dir.exists() and init_from in ("scratch", "pretrained"):
        i = 1
        while (Path(train_dir) / f"{exp_dir_name}_{str(i).zfill(2)}").exists():
            i += 1
        exp_dir = Path(train_dir) / f"{exp_dir_name}_{str(i).zfill(2)}"
        logger.warning(f"Existing exp dir found. Starting fresh in: {exp_dir}")

    os.makedirs(exp_dir, exist_ok=True)
    logger.add(exp_dir / "log.txt", rotation="100 MB", enqueue=True)
    tb_logdir = exp_dir / "tb"
    tb_logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_logdir))

if not seed_data_loader:
    # Otherwise seed PyTorch after creating the Arrow iterator.
    torch.manual_seed(worker_seed)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

empty_token_code = model_config.empty_token_code

model_args = dict(
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
    empty_connection_code=empty_connection_code,
    n_comm_rounds=n_comm_rounds,
    field_of_view_size=field_of_view_size,
    agent_info_size=agent_info_size,
    max_num_neighbors=max_num_neighbors,
    empty_token_code=empty_token_code,
    dt=dt,
    tau=tau,
    dirichlet_tf_on=dirichlet_tf_on,
    dirichlet_tf_beta=dirichlet_tf_beta,
    dirichlet_tf_beta_final=dirichlet_tf_beta_final,
    dirichlet_tf_anneal_steps=dirichlet_tf_anneal_steps,
    round_tf_on=round_tf_on,
    round_tf_beta=round_tf_beta,
    round_tf_beta_final=round_tf_beta_final,
    round_tf_anneal_steps=round_tf_anneal_steps,
)
if model_class == "dmm":
    model_args.update(
        n_encoder_layer=n_encoder_layer,
        n_decoder_layer=n_decoder_layer,
        n_head=n_head,
        n_embd=n_embd,
        latent_embd=latent_embd,
        latent_tok_n=latent_tok_n,
        action_msg_feats=action_msg_feats,
    )
else:
    model_args.update(
        width=dmm08m_width,
        heads=dmm08m_heads,
        conv_blocks=dmm08m_conv_blocks,
        mixer_blocks=dmm08m_mixer_blocks,
        spatial_size=dmm08m_spatial_size,
        spatial_mode=dmm08m_spatial_mode,
        local_spatial_size=dmm08m_local_spatial_size,
        neighbor_tokenization=dmm08m_neighbor_tokenization,
        communication_query_blocks=dmm08m_communication_query_blocks,
        communication_hidden_multiplier=dmm08m_communication_hidden_multiplier,
    )
iter_num = 0
best_val_loss = 1e9
meta_vocab_size = model_config.vocab_size

if init_from == "scratch":
    logger.info("Initializing a new model from scratch")
    model_args["vocab_size"] = meta_vocab_size
    model = model_type(model_config(**model_args))

elif init_from == "pretrained":
    logger.info("Initializing from pretrained weights")
    model_args["vocab_size"] = meta_vocab_size
    model = model_type(model_config(**model_args))
    ckpt = torch.load(Path(weight_path), map_location="cpu", weights_only=False)
    state_dict = {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in ckpt["model"].items()
    }
    model.load_state_dict(state_dict, strict=False)

elif init_from == "resume":
    ckpt_path = exp_dir / "ckpt_latest.pt"
    logger.info(f"Resuming training from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint_class = checkpoint.get("config", {}).get("model_class", "dmm")
    if checkpoint_class != model_class:
        raise ValueError(
            f"Checkpoint model_class={checkpoint_class!r} does not match "
            f"selected model_class={model_class!r}"
        )
    checkpoint_model_args = checkpoint["model_args"]
    for k in model_args:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    model = model_type(model_config(**model_args))
    state_dict = checkpoint["model"]
    state_dict = {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]

logger.info("number of parameters: %.2fM" % (model.get_num_params() / 1e6,))

model.to(device)

if freeze_encoder:
    for param in model.representation_encoder.parameters():
        param.requires_grad = False

scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16"))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None

if compile and "cuda" in device:
    logger.info(f"compiling the model with mode={compile_mode}...")
    try:
        model = torch.compile(model, mode=compile_mode)
    except AttributeError:
        logger.warning("torch.compile unavailable; use the pinned PyTorch 2.13.0+cu126 build")

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model


def save_latest_ckpt():
    if not master_process:
        return
    ckpt = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config,
    }
    path = exp_dir / "ckpt_latest.pt"
    torch.save(ckpt, path)
    logger.info(f"saved latest checkpoint at iter {iter_num} → {path}")


def get_batch(data):
    observations, actions, agent_chat_ids = next(data)
    return observations.to(torch.int), agent_chat_ids.to(torch.int), actions.to(torch.long)


@torch.no_grad()
def estimate_loss():
    out = {}
    raw_model.eval()
    for split in ["train"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, Z = get_batch(train_data_iter)
            with ctx:
                loss, _ = raw_model(X, Y, Z, dirichlet_tf_beta=0.0, round_tf_beta=0.0)
            losses[k] = loss.item()
        out[split] = losses.mean()
    raw_model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


X, Y, Z = get_batch(train_data_iter)
t0 = time.monotonic()

try:
    while True:
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if dirichlet_tf_on:
            prog_d = min(1.0, iter_num / max(dirichlet_tf_anneal_steps, 1))
            current_dirichlet_tf = dirichlet_tf_beta + (dirichlet_tf_beta_final - dirichlet_tf_beta) * prog_d
        else:
            current_dirichlet_tf = 0.0

        if round_tf_on:
            prog_r = min(1.0, iter_num / max(round_tf_anneal_steps, 1))
            current_round_tf = round_tf_beta + (round_tf_beta_final - round_tf_beta) * prog_r
        else:
            current_round_tf = 0.0

        if eval_interval > 0 and iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss()
            logger.info(f"step {iter_num}: train loss {losses['train']:.4f}")
            writer.add_scalar("Loss/train", losses["train"], iter_num)
            writer.add_scalar("Learning Rate", lr, iter_num)
            writer.add_scalar("TF/dirichlet", current_dirichlet_tf, iter_num)
            writer.add_scalar("TF/round", current_round_tf, iter_num)

        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (
                    micro_step == gradient_accumulation_steps - 1
                )
            with ctx:
                loss, per_round_losses = model(X, Y, Z,
                    dirichlet_tf_beta=current_dirichlet_tf,
                    round_tf_beta=current_round_tf,
                )
                loss = loss / gradient_accumulation_steps
            X, Y, Z = get_batch(train_data_iter)
            scaler.scale(loss).backward()

        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if ckpt_freq > 0 and iter_num > 0 and iter_num % ckpt_freq == 0 and master_process:
            named_ckpt = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "config": config,
            }
            ckpt_path = exp_dir / f"ckpt_{iter_num}.pt"
            logger.info(f"saving checkpoint to {ckpt_path}")
            torch.save(named_ckpt, ckpt_path)

        if latest_ckpt_freq > 0 and iter_num % latest_ckpt_freq == 0:
            save_latest_ckpt()

        t1 = time.monotonic()
        dt_step = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            logger.info(
                f"iter {iter_num}: loss {lossf:.4f}, time {dt_step * 1000:.2f}ms, "
                f"dir_tf {current_dirichlet_tf:.3f} rnd_tf {current_round_tf:.3f}"
            )
            writer.add_scalar("Loss/step", lossf, iter_num)
            for n, ls in enumerate(per_round_losses):
                writer.add_scalar(f"Loss_r_{n}", ls.item(), iter_num)

        iter_num += 1

        if iter_num > max_iters:
            break

except BaseException as e:
    logger.exception(f"Training interrupted: {e}")
    save_latest_ckpt()
    raise
finally:
    if master_process:
        writer.close()
    if ddp:
        destroy_process_group()
