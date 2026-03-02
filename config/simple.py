"""Simple configuration without FlexAttention for testing on CPU/MPS."""

# Model configuration (small GPT for fast testing)
model_config = {
    "model_type": "gpt",
    "vocab_size": 50304,  # GPT-2 vocab size rounded to 64
    "num_layers": 4,
    "model_dim": 256,
    "num_heads": 4,
    "ffn_dim": 1024,
    "activation": "swiglu",
}

# Attention configuration - disable FlexAttention for CPU/MPS compatibility
attention_config = {
    "block_size": 64,  # Smaller block size for testing
    "max_window_size": 512,
    "use_flex_attention": False,  # Disable FlexAttention
}

# Embedding configuration
embed_config = {
    "weight_tied": True,
    "enable_embed_split": False,
}

# Training configuration
training_config = {
    "num_iterations": 1000,
    "val_loss_every": 100,
    "val_tokens": 10000,
    "save_checkpoint": False,
    "grad_clip_norm": 1.0,
}

# Data configuration
data_config = {
    "train_seq_len": 256,
    "val_seq_len": 256,
    "train_micro_batch_tokens": 256,
    "val_tokens": 10000,
    "train_files": "data/*.bin",
    "val_files": "data/*.bin",
}

# Optimizer configuration
optimizer_config = {
    "adam": {
        "lr": 0.003,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.1,
    },
    "matrix_optimizer": "muon",
    "muon": {
        "lr": 0.03,
        "momentum": 0.95,
    },
    "lr_multipliers": {
        "c_proj": 0.5,
        "w1": 0.5,
        "w2": 0.5,
    },
    "wd_multipliers": {
        "c_proj": 0.1,
    },
}

# Batch schedule
batch_schedule_config = {
    "schedule_type": "stepped",
    "batch_sizes": [1, 2],
    "transitions": [0.5],
}

# Window schedule
window_schedule_config = {
    "schedule": [3, 5],
    "transitions": [0.5],
}

# Warmup config
warmup_config = {
    "warmup_steps": 2,
    "warmup_seq_len": 128,
}

# LR scheduler
lr_scheduler_config = {
    "scheduler_type": "linear",
    "warmup_steps": 10,
    "use_linear_warmup": True,
    "cooldown_steps": 0,
    "final_lr_ratio": 0.1,
}

# Logging
logging_config = {
    "use_wandb": False,
    "wandb_run_name": None,
}

# Compilation
compilation_config = {
    "compile_model": False,  # Disable compilation for CPU/MPS
}

# Lambda config (for residual connections)
lambda_config = {
    "lambda_init": 0.8,
    "lambda_lr": 0.001,
}

# Gating config
gating_config = {
    "gating_init": 0.0,
}

# Skip config
skip_config = {
    "skip_init": 0.0,
}

# RoPE config
rope_config = {
    "use_rope": True,
    "rope_theta": 10000.0,
}
