import wandb
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
import pytorch_lightning as pl

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from sampler import BatchSchedulerSampler
from model import BertClassifierModel


def train_multitask(
    params, train_datasets, val_dataset, collate_fn, wandb_group, fold=0
):
    concat_dataset = ConcatDataset(train_datasets)
    multitask_train_loader = DataLoader(
        dataset=concat_dataset,
        sampler=BatchSchedulerSampler(
            dataset=concat_dataset, batch_size=params["batch_size"], chunk_task_batches=2
        ),
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
        id=f"fold_{fold}_{wandb_group}_multitask",
        config=params,
        job_type="multitask",
    )
    
    checkpoint_filename = f"{wandb_group}_fold_{fold}_multi" + "-{val_loss:.2f}"
    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        monitor="val_loss",
        filename=checkpoint_filename,
        mode="min",
        save_weights_only=True,
    )

    # Init our model
    model = BertClassifierModel(**params)

    print(f"Multi Task Training:")
    # Initialize a trainer
    multitask_trainer = pl.Trainer(
        gpus=1,
        accumulate_grad_batches=params["accumulate_grads"],
        max_epochs=20,
        progress_bar_refresh_rate=1,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="step"),
            EarlyStopping(monitor="val_loss", patience=20, mode="min"),
        ],
        val_check_interval=params["val_check_interval"],
        stochastic_weight_avg=params["stochastic_weight_avg"],
        log_every_n_steps=params["accumulate_grads"],
    )
    # Train the model ⚡
    multitask_trainer.fit(
        model,
        train_dataloader=multitask_train_loader,
        val_dataloaders=[val_loader],
    )

    # clean up multitask logger
    wandb_logger.log_metrics(
        {"multitask_best_val_loss": checkpoint_callback.best_model_score.item()}
    )
    # save model py code
    wandb.save("model.py")
    wandb.finish()

    return (checkpoint_callback.best_model_path, checkpoint_callback.best_model_score.item())
