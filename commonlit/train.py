from dataset import get_full_train_dataset
import logging
import wandb
import os
from sklearn.model_selection import KFold
from pytorch_lightning.utilities.seed import seed_everything
import torch
from torch.utils.data import Subset

from settings import env_settings
from config import hyperparameter_defaults
from dataset import (
    get_full_train_dataset,
    get_multitask_datasets,
)
from pretrain import train_pretrain
from multitask import train_multitask
from finetune import train_finetune_from_checkpoint

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

os.environ["WANDB_API_KEY"] = env_settings.wandb_api_key
os.environ["TOKENIZERS_PARALLELISM"] = "false"

wandb.login()

seed_everything(hyperparameter_defaults["seed"])

skf = KFold(n_splits=5, shuffle=True, random_state=hyperparameter_defaults["seed"])

dataset = get_full_train_dataset()
multitask_datasets = get_multitask_datasets()

# Log Group
group = f"{hyperparameter_defaults['bert_model']}_textstats_" + wandb.util.generate_id()
group = group.replace("/", "-")


# Pretraining (done outside fold using all data)
checkpoint_type = "pretrain"
pretrained_checkpoint_path = False
if hyperparameter_defaults["pretrain"]:
    print("Training with MLM objective")
    best_pretrain_model, _ = train_pretrain(
        hyperparameter_defaults,
        group,
    )
    pretrained_checkpoint_path = best_pretrain_model


pretrained_checkpoint_path = "models/microsoft-deberta-large_1hee8w4o_pretrain.ckpt"

for fold, (train_ids, val_ids) in enumerate(skf.split(dataset)):
    torch.cuda.empty_cache()
    print(f"Starting fold {fold} for {group}")
    if fold in [0, 1, 2]:
        continue

    train_dataset = Subset(dataset, train_ids)
    val_dataset = Subset(dataset, val_ids)

    checkpoint_path = pretrained_checkpoint_path

    best_multitask_model = False
    if hyperparameter_defaults["multitask"]:
        train_datasets = multitask_datasets + [train_dataset]
        print("Training with Multi Task objective")
        best_multitask_model, best_multitask_score = train_multitask(
            hyperparameter_defaults,
            train_datasets,
            val_dataset,
            group,
            fold=fold,
            pretrained_checkpoint_path=checkpoint_path,
            checkpoint_type=checkpoint_type,
        )
        print(f"Best multitask score for fold {fold}: {best_multitask_score}")
        checkpoint_type = "multi"
        checkpoint_path = best_multitask_model

    # checkpoint_type = "multi"
    # checkpoint_path = pretrained_multi_checkpoints[fold]

    print("Finetuning")
    _, best_finetune_score = train_finetune_from_checkpoint(
        hyperparameter_defaults,
        train_dataset,
        val_dataset,
        group,
        fold=fold,
        pretrained_checkpoint_path=checkpoint_path,
        checkpoint_type=checkpoint_type,
    )
    print(f"Best finetuning score for fold {fold}: {best_finetune_score}")

    if best_finetune_score > 0.5:
        print("Not under 0.5 - cancelling run for other folds")
        break
