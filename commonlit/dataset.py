import random
import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
from typing import List


class PretrainComparaisonDataset(Dataset):
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


class CommonDataset(Dataset):
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


def get_full_train_dataset() -> Dataset:
    train = pd.read_csv("../input/commonlitreadabilityprize/train.csv")
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


def get_ck_12_dataset() -> Dataset:
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
    return CommonDataset(ck_12_df, dataset_name="ck_12")


def get_weebit_dataset() -> Dataset:
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
    return CommonDataset(weebit_df, dataset_name="weebit")


def get_wiki_dataset() -> Dataset:
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
    return PretrainComparaisonDataset(wiki_df, dataset_name="wiki")


def get_onestop_dataset() -> Dataset:
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
    onestop_df = pd.DataFrame(onestop)
    return PretrainComparaisonDataset(onestop_df, dataset_name="onestop")


def get_race_dataset() -> Dataset:
    race_df = pd.read_csv("../input/racehighschooldataset/race_highschool_dataset.csv")
    return CommonDataset(race_df, dataset_name="race")


def get_multitask_datasets() -> List[Dataset]:
    wiki_dataset = get_wiki_dataset()
    onestop_dataset = get_onestop_dataset()
    race_dataset = get_race_dataset()
    ck_12_dataset = get_ck_12_dataset()
    weebit_dataset = get_weebit_dataset()
    return [wiki_dataset, onestop_dataset, race_dataset, ck_12_dataset, weebit_dataset]
