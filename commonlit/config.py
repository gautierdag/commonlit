hyperparameter_defaults = {
    "max_steps": 2500,
    "use_warmup": False,
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "dropout": 0,
    "warmup_steps": 0.05,  # percentage of steps to warmup for
    "dense_dim": None,
    "custom_linear_init": True,
    "bert_model": "roberta-base",
    "freeze_layers": 0,
    "seed": 1000,
    "batch_size": 4,
    "accumulate_grads": 4,
    "stochastic_weight_avg": False,
    "val_check_interval": 50,  # evaluate every 10 steps
    "scheduler_rate": 200,
    "model_max_length": 248,
    "validation_batch_size": 16,
    "finetune_freeze_layers": 0,
    "finetune_learning_rate": 1e-5,
    "finetune_val_check_interval": 16,  # evaluate every 10 steps
    "finetune_use_warmup": True,  # evaluate every 10 steps
    "use_multitask": False,
    "decay_lr": True,
    "use_tanh_constraint": True,
    "commonlit_loss_weight": 1,
    "finetune_commonlit_loss_weight": 1,
    "sqrt_mse_loss": False
}

# hyperparameter_defaults["val_check_interval"] = (
#     hyperparameter_defaults["val_check_interval"]
#     * hyperparameter_defaults["accumulate_grads"]
#)  # actual steps (acummulated gradients)
hyperparameter_defaults["scheduler_rate"] = hyperparameter_defaults[
    "val_check_interval"
]
