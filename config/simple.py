"""Simple configuration without FlexAttention for testing on CPU/MPS."""

# Model configuration (small GPT for fast testing)
NUM_LAYERS = 4

model_config = {
    "model_type": "gpt",
    "vocab_size": 50304,  # GPT-2 vocab size rounded to 64
    "num_layers": NUM_LAYERS,
    "model_dim": 256,
    "num_heads": 4,
    "head_dim": 64,  # model_dim // num_heads
    "ffn_dim": 1024,
    "activation": "swiglu",
    "mlp_init_std_scale": 0.5,
    "lm_head_init_std": 0.005,
    "embed_padding_multiple": 128,
    "eos_token_id": 50256,
    "logits_softcap_scale": 23.0,
    "logits_softcap_shift": 5.0,
    "logits_softcap_divisor": 7.5,
    # Value embedding parameters (indices into ve_computed; must be in range(num_value_embeds))
    # For 4 layers: head 2 + mid 0 + tail 2 = 4; use 2 value embeds so indices 0,1 only
    "value_embed_head_indices": [0, 1],
    "value_embed_mid_layer_count": 4,  # len(head)+len(tail) = 4
    "value_embed_tail_indices": [0, 1],
    "value_embed_gate_scale": 2.0,
    "skip_gate_scale": 2.0,
    "residual_first_layer_index": 0,
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

# Gating config
gating_config = {
    "use_attn_gate": False,
    "use_value_embed_gate": False,
    "use_smear_gate": False,
    "use_skip_gate": False,
    "gate_input_dim": 12,
}

# Skip config (model expects skip_in_layers, skip_out_layers, backout_layer)
skip_config = {
    "skip_in_layers": [1],
    "skip_out_layers": [NUM_LAYERS - 2],
    "backout_layer": NUM_LAYERS - 1,
}

# RoPE config (create_positional_embedding uses type, base_freq, initial_attn_scale)
rope_config = {
    "type": "yarn",
    "base_freq": 1024,
    "initial_attn_scale": 0.1,
}

# Lambda config (model expects resid_lambdas_init, sa_lambdas_init, etc.)
lambda_config = {
    "resid_lambdas_init": 1.1,
    "x0_lambdas_init": 0.0,
    "sa_lambdas_init": [0.5, 1.0],
    "sa_lambdas_init_no_ve": [0.5, 1.0],
    "smear_lambda_init": 0.0,
    "backout_lambda_init": 0.5,
    "skip_lambda_init": -1.5,
}

# Residual connection config (build_residual_connection_fns uses .get with defaults)
residual_connection_config = {
    "mode": "standard",
    "num_streams": 1,
    "num_fracs": 1,
    "tanh": True,
    "disable": None,
    "sinkhorn_iters": 10,
    "sinkhorn_tau": 0.05,
    "mhc_h_res_proj": "sinkhorn",
    "ns_steps": 5,
    "ns_eps": 1e-7,
    "ns_coeffs": (3.0, -3.2, 1.2),
    "mhc_residual_identity_mix": False,
    "mhc_residual_alpha": 0.01,
}

# Low rank config (only read when enabled=True; include for completeness)
low_rank_config = {
    "enabled": False,
    "rank_ratio": 0.25,
    "rank": None,
    "min_rank": 1,
    "max_rank": None,
    "apply_attention": True,
    "apply_mlp": True,
}

# Attention pattern config (value_embed_layers: which ve index per layer; length = num_layers)
attention_pattern_config = {
    "block_mask_pattern": "S" * NUM_LAYERS,
    "value_embed_layers": [0, 1, 0, 1],  # 4 layers, 2 value embeds
    "num_value_embeds": 2,
    "skip_attention_layers": [],
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
        "embed": 1.0,
        "value_embed": 1.0,
        "head": 1.0,
        "scalars": 1.0,
        "x0_lambdas": 1.0,
        "smear_gate": 0.01,
        "skip_gate": 0.05,
    },
    "wd_multipliers": {
        "c_proj": 0.1,
        "embed": 1.0,
        "value_embed": 1.0,
        "head": 1.0,
        "scalars": 0.0,
        "x0_lambdas": 0.0,
        "smear_gate": 0.0,
        "skip_gate": 0.0,
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
