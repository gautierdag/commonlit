from copy import copy
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
from mlm_model import BertMLMModel



def get_multitask_params(params):
    """
    Sets all keys that start with multitask_ as the actual values
    eg: multitask_learning_rate -> learning_rate
    """
    multitask_params = copy(params)
    for k in params.keys():
        if "multitask_" in k:
            parameter_name = k.replace("multitask_", "")
            multitask_params[parameter_name] = multitask_params[k]
    return multitask_params


def train_multitask(
    all_params,
    train_datasets,
    val_dataset,
    collate_fn,
    wandb_group,
    pretrained_checkpoint_path=False,
    checkpoint_type="pretrain",
    fold=0,
):
    params = get_multitask_params(all_params)

    concat_dataset = ConcatDataset(train_datasets)
    multitask_train_loader = DataLoader(
        dataset=concat_dataset,
        sampler=BatchSchedulerSampler(
            dataset=concat_dataset,
            batch_size=params["batch_size"],
            chunk_task_batches=2,
        ),
        collate_fn=collate_fn,
        batch_size=params["batch_size"],
        num_workers=6,
        pin_memory=torch.cuda.is_available(),
    )
    multitask_val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=params["validation_batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=6,
        pin_memory=torch.cuda.is_available(),
    )
    params["max_steps"] = int(
        (len(multitask_train_loader) * params["num_epochs"])
        / params["accumulate_grads"]
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

    if pretrained_checkpoint_path:
        print(f"loading weights from {pretrained_checkpoint_path}")
        if checkpoint_type == "pretrain":
            pretrained_model = BertMLMModel().load_from_checkpoint(pretrained_checkpoint_path)
            model.text_model.load_state_dict(pretrained_model.text_model.roberta.state_dict(), strict=False)
        else:
            assert ValueError("loading from multitask for multitask is not implemented")

    print(f"Multi Task Training:")
    # Initialize a trainer
    multitask_trainer = pl.Trainer(
        gpus=1,
        accumulate_grad_batches=params["accumulate_grads"],
        max_epochs=params["num_epochs"],
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
        val_dataloaders=[multitask_val_loader],
    )

    # clean up multitask logger
    wandb_logger.log_metrics(
        {"multitask_best_val_loss": checkpoint_callback.best_model_score.item()}
    )
    # save model py code
    wandb.save("model.py")
    wandb.finish()

    return (
        checkpoint_callback.best_model_path,
        checkpoint_callback.best_model_score.item(),
    )
