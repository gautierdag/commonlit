hyperparameter_defaults = {
    "max_steps": 2500,
    "use_warmup": False,
    "learning_rate": 1e-5,
    "weight_decay": 0.1,
    "dropout": 0.25,
    "warmup_steps": 0.06,  # percentage of steps to warmup for
    "dense_dim": None,
    "custom_linear_init": True,
    "bert_model": "roberta-base",
    "freeze_layers": 0,
    "seed": 420,
    "batch_size": 4,
    "accumulate_grads": 4,
    "stochastic_weight_avg": True,
    "val_check_interval": 25,  # evaluate every 10 steps
    "scheduler_rate": 200,
    "model_max_length": 320,
    "validation_batch_size": 4,
    "finetune_freeze_layers": 0,
    "finetune_learning_rate": 1e-6,
    "finetune_val_check_interval": 20,  # evaluate every 10 steps
    "finetune_use_warmup": True,  # evaluate every 10 steps
    "use_multitask": True,
    "decay_lr": True,
    "use_tanh_constraint": True
}

hyperparameter_defaults["val_check_interval"] = (
    hyperparameter_defaults["val_check_interval"]
    * hyperparameter_defaults["accumulate_grads"]
)  # actual steps (acummulated gradients)
hyperparameter_defaults["scheduler_rate"] = hyperparameter_defaults["val_check_interval"]
