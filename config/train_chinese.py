# train a Chinese Four Classics character-level model
# optimized for RTX 3060 (12GB VRAM)

out_dir = 'out-chinese'
eval_interval = 1000
eval_iters = 200
log_interval = 10
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # scratch or resume

# wandb logging
wandb_log = False
wandb_project = 'chinese-classics'
wandb_run_name = 'chinese-gpt'

# dataset
dataset = 'chinese'
gradient_accumulation_steps = 1  # RTX 3060 can handle larger batch_size
batch_size = 64  # Optimized for RTX 3060 (12GB), can increase to 48 if needed
block_size = 256  # Context length, 512 is good balance for Chinese text

# model architecture - medium size for RTX 3060
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2  # Slight dropout for regularization
bias = False  # No bias for efficiency

# optimizer
learning_rate = 3e-4  # Good starting point for medium models
max_iters = 5000  # Train for 50k iterations
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate schedule
decay_lr = True
warmup_iters = 2000  # Warmup for 2k steps
lr_decay_iters = 50000  # Decay over full training
min_lr = 3e-5  # learning_rate / 10

# DDP settings
backend = 'nccl'

# system settings
# device and dtype will be set automatically by train.py based on CUDA availability
compile = False  # Disabled due to Triton compatibility issues on Windows

