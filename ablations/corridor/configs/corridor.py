# DMM on the corridor scenario: the production 3M architecture and teacher-forcing
# floors, with the optimiser cut down to a 14-row dataset. The sweeps override the
# teacher-forcing and round settings from the command line; see the corridor README.
model_class = "dmm"
init_from = "scratch"

train_data_files = ["ablations/corridor/data"]
batch_sizes = [14]
train_dir = "ablations/corridor/runs"
exp_dir_name = "corridor"

block_size = 256
field_of_view_size = 121
agent_info_size = 10
max_num_neighbors = 13

n_encoder_layer = 3
n_decoder_layer = 3
n_head = 3
n_embd = 64 * 3
latent_embd = 32 * 3
latent_tok_n = 32
action_msg_feats = 32 * 3
n_comm_rounds = 4

dt = 0.25
tau = 1.0

dirichlet_tf_on = True
dirichlet_tf_beta = 1.0
dirichlet_tf_beta_final = 0.8
dirichlet_tf_anneal_steps = 1000

round_tf_on = True
round_tf_beta = 1.0
round_tf_beta_final = 0.8
round_tf_anneal_steps = 1000

batch_size = 14
gradient_accumulation_steps = 1
learning_rate = 3e-4
max_iters = 5000
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 3e-5
grad_clip = 1.0

compile = False
eval_interval = 0
ckpt_freq = 0
latest_ckpt_freq = 500
log_interval = 50
