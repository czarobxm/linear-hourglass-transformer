import linecache
import subprocess
from typing import Dict
from pathlib import Path

import torch
from transformers import PreTrainedTokenizer

from data.base_dataset import BaseDataset
from data.utils import download_lra


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

    def __len__(self) -> int:
        result = subprocess.run(
            ["wc", "-l", self.data["file"]], stdout=subprocess.PIPE, text=True, check=True
        )
        return int(result.stdout.strip().split()[0])

    def __getitem__(self, index: int) -> Dict[str, str]:
        line = linecache.getline(self.data["file"], index + 1).strip()
        label, _, _, text_1, text_2 = line.split("\t")
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

        return {
            "train": {
                "file": train_path,
            },
            "val": {
                "file": val_path,
            },
            "test": {
                "file": test_path,
            },
        }
