import wandb
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from model import BertClassifierModel


def train_finetune_from_checkpoint(
    params, checkpoint, train_dataset, val_dataset, collate_fn, wandb_group, fold=0
):
    params["learning_rate"] = params["finetune_learning_rate"]
    params["freeze_layers"] = params["finetune_freeze_layers"]
    params["val_check_interval"] = params["finetune_val_check_interval"]
    params["use_warmup"] = params["finetune_use_warmup"]
    
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
    )

    # Init our model
    model = BertClassifierModel(**params)
    model.load_from_checkpoint(checkpoint)

    print(f"Multi Task Training:")
    # Initialize a trainer
    trainer = pl.Trainer(
        gpus=1,
        accumulate_grad_batches=params["accumulate_grads"],
        max_epochs=20,
        progress_bar_refresh_rate=1,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="step"),
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ],
        val_check_interval=params["val_check_interval"],
        stochastic_weight_avg=params["stochastic_weight_avg"],
        log_every_n_steps=params["accumulate_grads"],
        # resume_from_checkpoint=checkpoint
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

    return (checkpoint_callback.best_model_path, checkpoint_callback.best_model_score.item())

