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

from model import BertClassifierModel


def train_finetune_from_checkpoint(
    params,
    checkpoint,
    train_dataset,
    val_dataset,
    collate_fn,
    wandb_group,
    fold=0,
    num_epochs=3,
):
    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=params["batch_size"],
        num_workers=6,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=params["validation_batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=6,
        pin_memory=torch.cuda.is_available(),
    )

    params["max_steps"] = int((len(train_loader) * num_epochs)/ params["accumulate_grads"])
    wandb_logger = WandbLogger(
        project="commonlit",
        entity="commonlitreadabilityprize",
        group=wandb_group,
        id=f"fold_{fold}_{wandb_group}",
        config=params,
        job_type="finetune",
    )

    checkpoint_filename = f"{wandb_group}_fold_{fold}" + "-{val_loss:.2f}"
    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        monitor="val_loss",
        filename=checkpoint_filename,
        mode="min",
        save_weights_only=True,
    )

    # Init our model
    model = BertClassifierModel(**params)

    if checkpoint:
        model = model.load_from_checkpoint(checkpoint, **params)

    print(f"Multi Task Training:")
    # Initialize a trainer
    trainer = pl.Trainer(
        gpus=1,
        accumulate_grad_batches=params["accumulate_grads"],
        max_epochs=num_epochs,
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
    trainer.fit(
        model,
        train_dataloader=train_loader,
        val_dataloaders=[val_loader],
    )
    wandb_logger.log_metrics(
        {"best_val_loss": checkpoint_callback.best_model_score.item()}
    )
    # save model py code
    wandb.save("model.py")
    wandb.finish()

    del trainer, train_loader, val_loader, model, wandb_logger
    gc.collect()

    return (
        checkpoint_callback.best_model_path,
        checkpoint_callback.best_model_score.item(),
    )
