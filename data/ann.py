from typing import Dict
from pathlib import Path

import pandas as pd
from transformers import PreTrainedTokenizer

from data.base_dataset import BaseDataset


class ANN(BaseDataset):
    name: str = ""
    website: str = ""

    def __init__(
        self,
        data: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        shuffle: bool = True,
        separator_token: str = " [SEP] ",
        tokens_per_text: int = 512,
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
        return len(self.data["text_1"])

    def __getitem__(self, index: int) -> Dict[str, str]:
        text = (
            self.data["text_1"][index][: self.tokens_per_text]
            + self.separator_token
            + self.data["text_2"][index][: self.tokens_per_text]
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
            self.data["label"][index].to(self.device),
        )

    @classmethod
    def download_dataset(cls, path: Path) -> None:
        raise ValueError(
            "Download the dataset from the LRA (Long Range Arena) github page (https://github.com/google-research/long-range-arena) and put the unzipped folder in the datastorage folder"
        )

    @classmethod
    def load_raw_splits(cls, path: Path = None, **kwargs) -> Dict[str, str]:
        if path is None:
            path = Path("./datastorage/lra_release 3/lra_release/tsv_data/tsv_data")
        if path.exists():
            cls.download_dataset(path)

        train = pd.read_csv(
            f"{path}/new_aan_pairs.train.tsv",
            sep="\t",
            header=None,
        )
        val = pd.read_csv(
            f"{path}/new_aan_pairs.eval.tsv",
            sep="\t",
            header=None,
        )
        test = pd.read_csv(
            f"{path}/new_aan_pairs.test.tsv",
            sep="\t",
            header=None,
        )

        return {
            "train": {
                "text_1": train[3].tolist(),
                "text_2": train[4].tolist(),
                "label": train[0].tolist(),
            },
            "val": {
                "text_1": val[3].tolist(),
                "text_2": val[4].tolist(),
                "label": val[0].tolist(),
            },
            "test": {
                "text_1": test[3].tolist(),
                "text_2": test[4].tolist(),
                "label": test[0].tolist(),
            },
        }
