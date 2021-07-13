hyperparameter_defaults = {
    "dense_dim": None,
    "custom_linear_init": True,
    "bert_model": "roberta-base",
    "dropout": 0,
    "use_warmup": True,  # whether to use lr cos warmup
    "weight_decay": 0.1,
    "warmup_steps": 0.05,  # percentage of steps to warmup for
    "decay_lr": True, # whether to decay lr over depth
    "use_tanh_constraint": True,
    "commonlit_loss_weight": 1,
    "sqrt_mse_loss": False,
    "num_epochs": 3,
    "seed": 1000,
    "batch_size": 4,
    "accumulate_grads": 4,
    "stochastic_weight_avg": False,
    "scheduler_rate": 200,
    "model_max_length": 248,
    "validation_batch_size": 16,
    "freeze_layers": 0,
    "learning_rate": 1e-4,
    "val_check_interval": 16,  # evaluate every 10 steps
    "mlm_probability": 0.15,
    "pretrain": True,
    "pretrain_max_epochs": 5,
    "pretrain_weight_decay": 0.01,
    "pretrain_learning_rate": 5e-5,
    "pretrain_num_epochs": 5,
    "multitask": False,
    "multitask_freeze_layers": 0,
    "multitask_use_warmup": False,
    "multitask_learning_rate": 1e-5,
    "multitask_commonlit_loss_weight": 1,
    "multitask_val_check_interval": 50,  # evaluate every 50 steps
    "multitask_num_epochs": 2
}

hyperparameter_defaults["scheduler_rate"] = hyperparameter_defaults[
    "val_check_interval"
]
