from dataset import collate_creator, get_full_train_dataset, get_ybins
import logging
import wandb
import os
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import Subset

from settings import env_settings
from config import hyperparameter_defaults
from dataset import (
    get_full_train_dataset,
    get_ybins,
    get_multitask_datasets,
    collate_creator,
)
from multitask import train_multitask
from finetune import train_finetune_from_checkpoint

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

os.environ["WANDB_API_KEY"] = env_settings.wandb_api_key
wandb.login()

seed_everything(hyperparameter_defaults["seed"])


skf = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=hyperparameter_defaults["seed"]
)

dataset = get_full_train_dataset()
y_bins = get_ybins()
multitask_datasets = get_multitask_datasets()

tokenizer = AutoTokenizer.from_pretrained(
    f"../input/huggingfacemodels/{hyperparameter_defaults['bert_model']}/tokenizer",
    model_max_length=hyperparameter_defaults["model_max_length"],
)
collate_fn = collate_creator(tokenizer)

group = (
    f"{hyperparameter_defaults['bert_model']}_" + wandb.util.generate_id()
)
for fold, (train_ids, val_ids) in enumerate(skf.split(dataset, y=y_bins)):
    print(f"Starting fold {fold} for {group}")
    train_dataset = Subset(dataset, train_ids)
    val_dataset = Subset(dataset, val_ids)

    best_multitask_model = False
    if hyperparameter_defaults["use_multitask"]:
        train_datasets = multitask_datasets + [train_dataset]
        print("Training with Multi Task objective")
        best_multitask_model, best_multitask_score = train_multitask(
            hyperparameter_defaults,
            train_datasets,
            val_dataset,
            collate_fn,
            group,
            fold=fold,
        )
        print(f"Best multitask score for fold {fold}: {best_multitask_score}")

    print("Finetuning")
    _, best_finetune_score = train_finetune_from_checkpoint(hyperparameter_defaults,
        best_multitask_model,
        train_dataset,
        val_dataset,
        collate_fn,
        group,
        fold=fold,
    )
    print(f"Best finetuning score for fold {fold}: {best_finetune_score}")