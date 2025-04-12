from pathlib import Path
import re

import torch
import pandas as pd
from transformers import PreTrainedTokenizer

from data.base_dataset import BaseDataset
from data.utils import download_lra


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def clean_listop_text(text: str) -> str:
    text = text.replace("(", "").replace(")", "")
    text = normalize_spaces(text)
    return text


class ListOps(BaseDataset):
    name: str = "listops"
    website: str = ""

    def __init__(
        self,
        data: str,
        tokenizer: PreTrainedTokenizer = None,
        max_length: int = 512,
        shuffle: bool = True,
        device: str = "cpu",
    ):
        super().__init__(
            data=data,
            tokenizer=tokenizer,
            max_length=max_length,
            shuffle=shuffle,
            device=device,
        )

    def __len__(self):
        return len(self.data["text"])

    def __getitem__(self, index: int):
        index = self.shuffled_order[index]

        # Clean text
        clean_text = clean_listop_text(self.data["text"][index])

        # Tokenize
        token_dict = self.tokenizer(
            clean_text,
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        return (
            token_dict["input_ids"].squeeze(0).to(self.device),
            torch.tensor(self.data["label"][index], device=self.device, dtype=torch.long),
        )

    @classmethod
    def download_dataset(cls, path: Path = None) -> None:
        if path is None:
            path = Path("./datastorage")

        download_lra(path)

    @classmethod
    def load_raw_splits(cls, path: str, **kwargs):
        if path is None:
            path = Path("./datastorage/lra_release/listops-1000")

        train = pd.read_csv(
            f"{path}/basic_train.tsv",
            sep="\t",
        )
        val = pd.read_csv(
            f"{path}/basic_val.tsv",
            sep="\t",
        )
        test = pd.read_csv(
            f"{path}/basic_test.tsv",
            sep="\t",
        )
        return {
            "train": {"text": train["Source"], "label": train["Target"]},
            "val": {"text": val["Source"], "label": val["Target"]},
            "test": {"text": test["Source"], "label": test["Target"]},
        }
