from pathlib import Path

import torch

# from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer


class MethodNotSupportedError(Exception):
    pass


class BaseDataset(torch.utils.data.Dataset):
    name: str = "base_dataset"
    website: str = ""

    def __init__(
        self,
        data: str,
        tokenizer: AutoTokenizer = None,
        max_length: int = 512,
        shuffle: bool = True,
        device: str = "cpu",
    ):
        self.data = data

        self.tokenizer = tokenizer
        self.attention_masks = None
        self.token_type_ids = None
        self.max_length = max_length

        self.create_shuffle_order(shuffle)

        self.device = device

    def create_shuffle_order(self, shuffle: bool = True):
        if shuffle:
            self.shuffled_order = torch.randperm(
                len(self.data) // self.max_length + 1
            ).tolist()
        else:
            self.shuffled_order = torch.arange(len(self)).tolist()

    def __len__(self):
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )

    def __getitem__(self, index):
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )

    @classmethod
    def load_raw_splits(cls, path: str, **kwargs):
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )

    @classmethod
    def download_dataset(cls, path: str):
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )

    @classmethod
    def create_split_datasets(
        cls,
        split: str = "all",
        tokenizer: AutoTokenizer = None,
        path: str = None,
        max_length: int = 512,
        shuffle: bool = True,
        device: str = "cpu",
        **kwargs,
    ):
        path = Path(path)
        splits = cls.load_raw_splits(path=path, **kwargs)
        all_kwargs = kwargs
        all_kwargs.update(
            dict(
                tokenizer=tokenizer,
                max_length=max_length,
                shuffle=shuffle,
                device=device,
            )
        )
        if split == "all":
            return (
                cls(data=splits["train"], **all_kwargs),
                cls(data=splits["val"], **all_kwargs),
                cls(data=splits["test"], **all_kwargs),
            )
        else:
            raise ValueError(f"Invalid value for split argument: {split}")


class BaseArtificialDataset(BaseDataset):
    name = "base_artificial_dataset"

    @classmethod
    def download_dataset(cls, path: str):
        raise MethodNotSupportedError("Artificial datasets do not need to be downloaded")

    @classmethod
    def create_artificial_datasets(cls, path: str, **kwargs):
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )

    @classmethod
    def create_split_datasets(
        cls,
        split: str = "all",
        tokenizer: AutoTokenizer = None,
        path: str = None,
        max_length: int = 512,
        shuffle: bool = True,
        device: str = "cpu",
        create_artificial_datasets: bool = False,
        **kwargs,
    ):
        if create_artificial_datasets:
            cls.create_artificial_datasets(path, **kwargs)

        return super().create_split_datasets(
            split=split,
            tokenizer=tokenizer,
            path=path,
            max_length=max_length,
            shuffle=shuffle,
            device=device,
            **kwargs,
        )
