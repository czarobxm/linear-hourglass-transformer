from pathlib import Path

import pandas as pd
from transformers import PreTrainedTokenizer

from data.base_dataset import BaseDataset


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
        # Tokenize
        token_dict = self.tokenizer(
            self.data["text"][index],
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        return (
            token_dict["input_ids"].squeeze(0).to(self.device),
            self.data["label"][index].to(self.device),
        )

    @classmethod
    def download_dataset(cls, path: Path) -> None:
        raise ValueError(
            "Download the dataset from the LRA (Long Range Arena) github page (https://github.com/google-research/long-range-arena) and put the unzipped folder in the datastorage folder"
        )

    @classmethod
    def load_raw_splits(cls, path: str, **kwargs):
        if path is None:
            path = Path("./datastorage/lra_release 3/listops-1000")

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
            "val": {"text": val["Source"], "label": test["Target"]},
            "test": {"text": test["Source"], "label": test["Target"]},
        }
