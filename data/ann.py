import csv
import os
import subprocess
import random
from typing import Dict
from pathlib import Path

import torch
from transformers import PreTrainedTokenizer

from data.base_dataset import BaseDataset
from data.utils import download_lra


def csv_row_generator(file_path):
    with open(file_path, newline="", encoding="utf-8") as file:
        for row in file:
            yield row


class ANN(BaseDataset):
    name: str = ""
    website: str = ""

    def __init__(
        self,
        data: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 4096,
        shuffle: bool = True,
        separator_token: str = " [SEP] ",
        tokens_per_text: int = 2045,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            data=data,
            tokenizer=tokenizer,
            max_length=max_length,
            shuffle=shuffle,
            device=device,
        )
        self.separator_token = separator_token
        self.tokens_per_text = tokens_per_text
        self.create_data_iterator()

    def create_data_iterator(self) -> None:
        if self.shuffle:
            dir_name, base_name = os.path.split(self.data["path"])
            name, ext = os.path.splitext(base_name)
            shuffled_filename = os.path.join(dir_name, f"{name}_shuffled{ext}")
            self.data["iterator"] = csv_row_generator(shuffled_filename)
        else:
            self.data["iterator"] = csv_row_generator(self.data["path"])

    def __len__(self) -> int:
        return self.data["length"]

    def __getitem__(self, index: int) -> Dict[str, str]:
        try:
            line = next(self.data["iterator"]).strip()
        except StopIteration as exc:
            # Reinitialize the iterator if we reach the end
            self.create_data_iterator()
            raise StopIteration from exc
        label, _, _, text_1, text_2 = line.split("\t")
        label = label.strip("\"' ")
        text = (
            text_1[: self.tokens_per_text]
            + self.separator_token
            + text_2[: self.tokens_per_text]
        )

        # Tokenize
        token_dict = self.tokenizer(
            text,
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        return (
            token_dict["input_ids"].squeeze(0).to(self.device),
            torch.tensor(float(label), dtype=torch.long, device=self.device),
        )

    @classmethod
    def download_dataset(cls, path: Path = None) -> None:
        if path is None:
            path = Path("./datastorage")

        download_lra(path)

    @classmethod
    def load_raw_splits(cls, path: Path = None, **kwargs) -> Dict[str, str]:
        if path is None:
            path = Path("./datastorage/lra_release 3/lra_release/tsv_data")
        if not path.exists():
            cls.download_dataset(path)

        train_path = f"{path}/new_aan_pairs.train.tsv"
        val_path = f"{path}/new_aan_pairs.eval.tsv"
        test_path = f"{path}/new_aan_pairs.test.tsv"

        train_length = subprocess.run(
            ["wc", "-l", train_path], stdout=subprocess.PIPE, text=True, check=True
        )
        train_length = int(train_length.stdout.strip().split()[0])

        val_length = subprocess.run(
            ["wc", "-l", val_path], stdout=subprocess.PIPE, text=True, check=True
        )
        val_length = int(val_length.stdout.strip().split()[0])

        test_length = subprocess.run(
            ["wc", "-l", test_path], stdout=subprocess.PIPE, text=True, check=True
        )
        test_length = int(test_length.stdout.strip().split()[0])

        return {
            "train": {
                "path": train_path,
                "length": train_length,
            },
            "val": {
                "path": val_path,
                "length": val_length,
            },
            "test": {
                "path": test_path,
                "length": test_length,
            },
        }
