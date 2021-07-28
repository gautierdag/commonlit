import wandb
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import gc

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from model import BertClassifierModel
from mlm_model import BertMLMModel
from dataset import collate_creator


def train_finetune_from_checkpoint(
    params,
    train_dataset,
    val_dataset,
    wandb_group,
    pretrained_checkpoint_path=False,
    checkpoint_type="multi",
    fold=0,
):

    tokenizer = AutoTokenizer.from_pretrained(
        f"../input/huggingfacemodels/{params['bert_model']}/tokenizer",
        model_max_length=params["model_max_length"],
    )
    collate_fn = collate_creator(tokenizer, use_textstat=params["use_textstat"])

    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=params["batch_size"],
        num_workers=6,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=params["validation_batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=6,
    )

    params["max_steps"] = int(
        (len(train_loader) * params["max_epochs"]) / params["accumulate_grads"]
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
        save_weights_only=True,
    )

    # Init our model
    print(params["bert_model"])
    model = BertClassifierModel(**params)
    if pretrained_checkpoint_path:
        print(f"Loading weights from {pretrained_checkpoint_path}")
        if checkpoint_type == "pretrain":
            pretrained_model = BertMLMModel(
                bert_model=params["bert_model"]
            ).load_from_checkpoint(
                pretrained_checkpoint_path, bert_model=params["bert_model"]
            )
            if "roberta" in params["bert_model"]:
                model.text_model.load_state_dict(
                    pretrained_model.text_model.roberta.state_dict(), strict=False
                )
            elif "deberta" in params["bert_model"]:
                model.text_model.load_state_dict(
                    pretrained_model.text_model.deberta.state_dict(), strict=False
                )
            elif "albert" in params["bert_model"]:
                model.text_model.load_state_dict(
                    pretrained_model.text_model.albert.state_dict(), strict=False
                )
            else:
                raise ValueError(f"Unknown bert model {params['bert_model']}")

        elif checkpoint_type == "multi":
            if params[
                "resume_optim_from_checkpoint"
            ]:  # Whether to resume optimiser state
                params["optim_checkpoint_path"] = pretrained_checkpoint_path
            model = model.load_from_checkpoint(pretrained_checkpoint_path, **params)
        else:
            assert ValueError(f"Loading from {checkpoint_type} is not implemented")

    print(f"Final Finetuning Training:")

    # Initialize a trainer
    trainer = pl.Trainer(
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
        gradient_clip_val=params["gradient_clip_val"],
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
    torch.cuda.empty_cache()

    return (
        checkpoint_callback.best_model_path,
        checkpoint_callback.best_model_score.item(),
    )
