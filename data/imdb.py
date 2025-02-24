from pathlib import Path

import torch
from transformers import PreTrainedTokenizer
from datasets import load_dataset

from data.base_dataset import BaseDataset


class IMDB(BaseDataset):
    name: str = "enwik9"
    website: str = "https://mattmahoney.net/dc/textdata.html"

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
            torch.Tensor([self.data["label"][index]], device=self.device),
        )

    @classmethod
    def download_dataset(cls, path: Path = None):
        if path is None:
            path = Path("./datastorage/imdb")
        load_dataset(
            "stanfordnlp/imdb", cache_dir=path, split="train", resume_download=None
        )
        load_dataset(
            "stanfordnlp/imdb", cache_dir=path, split="test", resume_download=None
        )

    @classmethod
    def load_raw_splits(cls, path: Path, **kwargs):
        if path is None:
            path = Path("./datastorage/imdb")

        train = load_dataset(
            "stanfordnlp/imdb", cache_dir=path, split="train", resume_download=None
        )
        test = load_dataset(
            "stanfordnlp/imdb", cache_dir=path, split="test", resume_download=None
        )
        return {
            "train": {"text": train["text"], "label": train["label"]},
            "val": {"text": test["text"][:12500], "label": test["label"][:12500]},
            "test": {"text": test["text"][12500:], "label": test["label"][12500:]},
        }
