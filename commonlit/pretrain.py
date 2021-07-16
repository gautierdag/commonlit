import wandb
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from copy import copy
import gc

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from mlm_model import BertMLMModel


def get_pretrain_params(params):
    """
    Sets all keys that start with pretrain_ as the actual values
    eg: pretrain_learning_rate -> learning_rate
    """
    pretrain_params = copy(params)
    for k in params.keys():
        if "pretrain_" in k:
            parameter_name = k.replace("pretrain_", "")
            pretrain_params[parameter_name] = pretrain_params[k]
    return pretrain_params


def train_pretrain(
    all_params, mlm_train_dataset, mlm_val_dataset, mlm_collate_fn, wandb_group
):
    params = get_pretrain_params(all_params)

    pretrain_train_loader = DataLoader(
        mlm_train_dataset,
        shuffle=True,
        collate_fn=mlm_collate_fn,
        num_workers=6,
        batch_size=params["batch_size"],
    )
    pretrain_val_loader = DataLoader(
        mlm_val_dataset,
        collate_fn=mlm_collate_fn,
        num_workers=6,
        batch_size=params["batch_size"],
    )
    params["max_steps"] = int(
        (len(pretrain_train_loader) * params["max_epochs"]) / params["accumulate_grads"]
    )

    wandb_logger = WandbLogger(
        project="commonlit",
        entity="commonlitreadabilityprize",
        group=wandb_group,
        id=f"{wandb_group}_pretrain",
        config=params,
        job_type="pretrain",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        monitor="mlm_val_loss",
        filename=f"{wandb_group}_pretrain",
        save_weights_only=True,
    )

    model = BertMLMModel(**params)

    print(f"MLM Pre-Training:")
    # Initialize a trainer
    pretrain_trainer = pl.Trainer(
        gpus=1,
        accumulate_grad_batches=params["accumulate_grads"],
        max_epochs=params["max_epochs"],
        progress_bar_refresh_rate=1,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="step"),
        ],
        val_check_interval=params["val_check_interval"],
        stochastic_weight_avg=params["stochastic_weight_avg"],
        log_every_n_steps=params["accumulate_grads"],
    )

    # Train the model ⚡
    pretrain_trainer.fit(
        model,
        train_dataloader=pretrain_train_loader,
        val_dataloaders=[pretrain_val_loader],
    )

    # clean up pretrain logger
    wandb_logger.log_metrics(
        {"mlm_best_val_loss": checkpoint_callback.best_model_score.item()}
    )
    # save model py code
    wandb.save("model.py")
    wandb.finish()

    return (
        checkpoint_callback.best_model_path,
        checkpoint_callback.best_model_score.item(),
    )
