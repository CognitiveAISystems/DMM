# DMM-3M pretraining configuration.
model_class = "dmm"
seed_data_loader = False
max_iters = 1_000_000

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

compile = True

dt = 0.25
tau = 1.0

dirichlet_tf_on = True
dirichlet_tf_beta = 1.0
dirichlet_tf_beta_final = 0.8
dirichlet_tf_anneal_steps = 100_000

round_tf_on = True
round_tf_beta = 1.0
round_tf_beta_final = 0.8
round_tf_anneal_steps = 100_000

batch_size = 128
gradient_accumulation_steps = 4
init_from = "scratch"
freeze_encoder = False

latest_ckpt_freq = 100
eval_interval = 0
ckpt_freq = 25_000
