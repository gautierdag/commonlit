hyperparameter_defaults = {
    "dense_dim": None,
    "custom_linear_init": True,
    "bert_model": "roberta-large",
    "dropout": 0.1,
    "use_warmup": True,  # whether to use lr cos warmup
    "weight_decay": 0.0,
    "warmup_steps": 0.1,  # percentage of steps to warmup for
    "decay_lr": True,  # whether to decay lr over depth
    "gradient_clip_val": 0.0,
    "output_constraint": "tanh",
    "commonlit_loss_weight": 1,
    "sqrt_mse_loss": False,
    "pooling": "attention",
    "max_epochs": 5,
    "seed": 420,
    "batch_size": 4,
    "accumulate_grads": 8,
    "stochastic_weight_avg": False,
    "scheduler_rate": 200,
    "model_max_length": 300,
    "validation_batch_size": 16,
    "freeze_layers": 0,
    "learning_rate": 1e-5,
    "val_check_interval": 16,  # evaluate every 10 steps
    "resume_optim_from_checkpoint": False,
    "mlm_probability": 0.15,
    "deepspeed": False,  # "deepspeed_stage_3_offload",
    "pretrain": False,
    "pretrain_external_files": True,
    "pretrain_max_epochs": 1,
    "pretrain_weight_decay": 0.01,
    "pretrain_learning_rate": 5e-5,
    "pretrain_batch_size": 2,
    "pretrain_accumulate_grads": 16,
    "pretrain_model_max_length": 300,
    "pretrain_val_check_interval": 0.1,
    "multitask": True,
    "multitask_freeze_layers": 0,
    "multitask_use_warmup": True,
    "multitask_batch_size": 4,
    "multitask_accumulate_grads": 8,
    "multitask_learning_rate": 1e-4,
    "multitask_commonlit_loss_weight": 1,
    "multitask_val_check_interval": 0.02,  # evaluate every 10% of an epoch
    "multitask_max_epochs": 5,
    "multitask_chunk_task_batches": 2,  # how to chunk the tasks together
    "multitask_limit_train_batches": 1.0,  # how to chunk the tasks together
    "multitask_model_max_length": 300,
    "multitask_weight_decay": 0.01,
    # "multitask_validation_batch_size": 16,
}

hyperparameter_defaults["scheduler_rate"] = hyperparameter_defaults[
    "val_check_interval"
]
