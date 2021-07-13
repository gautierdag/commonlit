import random
import pandas as pd
import torch
import os
from datasets import Dataset, DatasetDict
from datasets import load_dataset

from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
from typing import List


class PretrainComparaisonDataset(torch.utils.data.Dataset):
    """
    Dataset class for aligned text (simple vs hard - same content)
    """

    def __init__(self, df, dataset_name="wiki"):
        self.df = df
        self.dataset_name = dataset_name
        assert all(["text" in col for col in df.columns])
        self.column_idxs = list(range(len(df.columns)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        i, j = random.sample(self.column_idxs, 2)
        target = int((i > j))
        # if y=1 then it assumed the first input should be ranked higher (have a larger column index) than the second input,
        # and vice-versa for y=−1
        sample = {
            "text_input1": self.df.iloc[idx, i],
            "text_input2": self.df.iloc[idx, j],
            "dataset_name": self.dataset_name,
        }
        if target == 0:
            target = -1
        sample["target"] = target
        return sample


class CommonDataset(torch.utils.data.Dataset):
    def __init__(self, df, dataset_name="commonlit"):
        self.df = df
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = {
            "text_input": self.df.iloc[idx].excerpt,
            "target": self.df.iloc[idx].target,
            "dataset_name": self.dataset_name,
        }
        return sample


def collate_creator(tokenizer):
    def collate_fn(batch):
        items = {}
        for key in batch[0].keys():
            if "text_input" in key:
                items[key] = tokenizer(
                    [batch_item[key] for batch_item in batch],
                    padding=True,
                    return_tensors="pt",
                    truncation=True,
                )
        items["target"] = torch.tensor(
            [batch_item["target"] for batch_item in batch]
        ).float()
        items["dataset_name"] = batch[0]["dataset_name"]
        return items

    return collate_fn

def get_train_df():
    train = pd.read_csv("../input/commonlitreadabilityprize/train.csv")
    return train


def get_full_train_dataset() -> torch.utils.data.Dataset:
    train = get_train_df()
    dataset = CommonDataset(train)
    return dataset


def get_ybins():
    train = pd.read_csv("../input/commonlitreadabilityprize/train.csv")
    num_bins = int(np.floor(1 + np.log2(len(train))))
    est = KBinsDiscretizer(n_bins=num_bins, encode="ordinal", strategy="quantile")
    y_labels = train.target.values.reshape(-1, 1)
    est.fit(y_labels)
    y_bins = est.transform(y_labels)
    return y_bins


def get_ck_12_df() -> pd.DataFrame:
    ck_12_df = pd.read_csv("../input/ck12excerpts/ck12.csv")
    ck_12_df["text"] = ck_12_df["excerpt"]
    ck_12_df["target"] = (ck_12_df.level - 3) / 9
    ck_12_df = ck_12_df[["text", "target"]]
    ck_12_df["dataset"] = "ck12"
    ck_12_df.rename(columns={"text": "excerpt"}, inplace=True)
    ck_12_df = ck_12_df[
        (ck_12_df.excerpt.str.len() > 500) & (ck_12_df.excerpt.str.len() < 1500)
    ].reset_index(drop=True)
    ck_12_df.dropna(inplace=True)
    return ck_12_df


def get_ck_12_dataset() -> torch.utils.data.Dataset:
    ck_12_df = get_ck_12_df()
    return CommonDataset(ck_12_df, dataset_name="ck_12")


def get_weebit_df() -> pd.DataFrame:
    weebit_df = pd.read_csv(
        "../input/readability-external/weebit_reextracted.tsv", sep="\t"
    )
    weebit_df["target"] = (weebit_df.readability - 2) / 4
    weebit_df = weebit_df[["text", "target"]]
    weebit_df["dataset"] = "weebit"
    weebit_df.rename(columns={"text": "excerpt"}, inplace=True)
    weebit_df = weebit_df[
        (weebit_df.excerpt.str.len() > 500) & (weebit_df.excerpt.str.len() < 1500)
    ].reset_index(drop=True)
    weebit_df.dropna(inplace=True)
    return weebit_df


def get_weebit_dataset() -> torch.utils.data.Dataset:
    weebit_df = get_weebit_df()
    return CommonDataset(weebit_df, dataset_name="weebit")


def get_wiki_df() -> pd.DataFrame:
    wiki_df = pd.read_csv("../input/simple-wiki/simple_wiki.csv")
    small_texts_ids = wiki_df[
        (wiki_df.text.str.len() < 500) | (wiki_df.text.str.len() > 1500)
    ].id.unique()
    wiki_df = wiki_df[~wiki_df.id.isin(small_texts_ids)]
    simple_df = wiki_df[wiki_df.target == 0][["text", "id"]].rename(
        columns={"text": "simple_text"}
    )
    normal_df = wiki_df[wiki_df.target == 1][["text", "id"]].rename(
        columns={"text": "hard_text"}
    )
    wiki_df = simple_df.merge(normal_df, on="id", how="inner")[
        ["simple_text", "hard_text"]
    ].reset_index(drop=True)
    wiki_df.dropna(inplace=True)
    return wiki_df


def get_wiki_dataset() -> torch.utils.data.Dataset:
    wiki_df = get_wiki_df()
    return PretrainComparaisonDataset(wiki_df, dataset_name="wiki")


def get_onestop_df() -> pd.DataFrame:
    onestop = []
    i = 0
    p = "../input/onestopenglishcorpus"
    for csv_file in os.listdir(p):
        filename = f"{p}/{csv_file}"
        if "all_data" in filename:
            continue
        try:
            tmp_df = pd.read_csv(filename, encoding="cp1252")
        except Exception as e:
            print(e)
            print(filename)
            continue
        elementary = "".join(
            tmp_df[~tmp_df[tmp_df.columns[0]].isna()][tmp_df.columns[0]].values
        )
        intermediate = "".join(
            tmp_df[~tmp_df[tmp_df.columns[1]].isna()][tmp_df.columns[1]].values
        )
        advanced = "".join(
            tmp_df[~tmp_df[tmp_df.columns[2]].isna()][tmp_df.columns[2]].values
        )
        onestop.append(
            {"text_easy": elementary, "text_med": intermediate, "text_hard": advanced}
        )
        i += 1
    return pd.DataFrame(onestop)


def get_onestop_dataset() -> torch.utils.data.Dataset:
    onestop_df = get_onestop_df()
    return PretrainComparaisonDataset(onestop_df, dataset_name="onestop")


def get_race_df() -> pd.DataFrame:
    return pd.read_csv("../input/racehighschooldataset/race_highschool_dataset.csv")


def get_race_dataset() -> torch.utils.data.Dataset:
    race_df = get_race_df()
    return CommonDataset(race_df, dataset_name="race")


def get_multitask_datasets() -> List[torch.utils.data.Dataset]:
    wiki_dataset = get_wiki_dataset()
    onestop_dataset = get_onestop_dataset()
    race_dataset = get_race_dataset()
    ck_12_dataset = get_ck_12_dataset()
    weebit_dataset = get_weebit_dataset()
    return [wiki_dataset, onestop_dataset, race_dataset, ck_12_dataset, weebit_dataset]


def get_mlm_dataset(train_idxs, val_idxs, tokenizer, use_external_files=False):
    max_seq_length = tokenizer.__dict__["model_max_length"]
    train = get_train_df()
    train_df = train.loc[train_idxs][["excerpt"]].copy().reset_index(drop=True)
    val_df = train.loc[val_idxs][["excerpt"]].copy().reset_index(drop=True).rename(columns={"excerpt": "text"})

    # add text from other sources
    if use_external_files:
        wiki_df = get_wiki_df()
        onestop_df = get_onestop_df()
        race_df = get_race_df()
        ck_12_df = get_ck_12_df()
        weebit_df = get_weebit_df()

        train_text_df = (
            pd.concat(
                [
                    ck_12_df.excerpt,
                    race_df.excerpt,
                    weebit_df.excerpt,
                    wiki_df.simple_text,
                    wiki_df.hard_text,
                    onestop_df.text_easy,
                    onestop_df.text_med,
                    onestop_df.text_hard,
                    train_df.excerpt
                ]
            )
            .rename("text")
            .reset_index(drop=True).to_frame()
        )

    # combined_df
    else:
        train_text_df =  train_df.rename(columns={"excerpt": "text"})

    # combined_df
    train_dataset = Dataset.from_pandas(train_text_df)
    val_dataset = Dataset.from_pandas(val_df)

    raw_datasets = DatasetDict({"train": train_dataset, "val": val_dataset})

    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=["text"],
        load_from_cache_file=False,
        desc="Running tokenizer on every text in dataset",
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + max_seq_length]
                for i in range(0, total_length, max_seq_length)
            ]
            for k, t in concatenated_examples.items()
        }
        return result

    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        desc=f"Grouping texts in chunks of {max_seq_length}",
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["val"]
    return (train_dataset, eval_dataset)
