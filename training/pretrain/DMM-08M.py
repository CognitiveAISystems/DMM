model_class = "dmm08m"
seed_data_loader = True
max_iters = 1_000_000

block_size = 256
field_of_view_size = 121
agent_info_size = 10
max_num_neighbors = 13

dmm08m_width = 96
dmm08m_heads = 4
dmm08m_conv_blocks = 3
dmm08m_mixer_blocks = 2
dmm08m_spatial_size = 5
dmm08m_communication_query_blocks = 2
dmm08m_communication_hidden_multiplier = 2
n_comm_rounds = 4

compile = False
compile_mode = "default"
dtype = "bfloat16"

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

# Global effective batch = 4 GPUs * 100 scenarios/GPU * 2 micro-steps.
# Batch 100 preserves the exact 6:2:2 family composition on every rank.
batch_size = 100
gradient_accumulation_steps = 8
learning_rate = 6e-4
warmup_iters = 2_000
lr_decay_iters = 1_000_000
min_lr = 6e-5

init_from = "scratch"
freeze_encoder = False
exp_dir_name = "dmm_08m_38tok_2block_scratch_ddp4_v3"
latest_ckpt_freq = 100
eval_interval = 0
eval_iters = 40
ckpt_freq = 5_000
log_interval = 1
