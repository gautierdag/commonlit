hyperparameter_defaults = {
    "dense_dim": None,
    "custom_linear_init": True,
    "bert_model": "roberta-large",
    "dropout": 0,
    "use_warmup": True,  # whether to use lr cos warmup
    "weight_decay": 0.01,
    "warmup_steps": 0.1,  # percentage of steps to warmup for
    "decay_lr": True,  # whether to decay lr over depth
    "gradient_clip_val": 1.0,
    "use_tanh_constraint": True,
    "commonlit_loss_weight": 1,
    "sqrt_mse_loss": False,
    "max_epochs": 5,
    "seed": 1000,
    "batch_size": 8,
    "accumulate_grads": 2,
    "stochastic_weight_avg": False,
    "scheduler_rate": 200,
    "model_max_length": 248,
    "validation_batch_size": 32,
    "freeze_layers": 0,
    "learning_rate": 1e-4,
    "val_check_interval": 16,  # evaluate every 10 steps
    "mlm_probability": 0.15,
    "pretrain": True,
    "pretrain_external_files": False,
    "pretrain_max_epochs": 2,
    "pretrain_weight_decay": 0.01,
    "pretrain_learning_rate": 5e-5,
    "multitask": False,
    "multitask_freeze_layers": 0,
    "multitask_use_warmup": True,
    "multitask_batch_size": 4,
    "multitask_accumulate_grads": 4,
    "multitask_learning_rate": 1e-4,
    "multitask_commonlit_loss_weight": 1,
    "multitask_val_check_interval": 0.01,  # evaluate every 10% of an epoch
    "multitask_max_epochs": 1,
    "multitask_chunk_task_batches": 2,  # how to chunk the tasks together
    "multitask_limit_train_batches": 0.2,  # how to chunk the tasks together

}

hyperparameter_defaults["scheduler_rate"] = hyperparameter_defaults[
    "val_check_interval"
]
