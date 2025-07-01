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
        super().__init__()
        self.data = data

        self.tokenizer = tokenizer
        self.attention_masks = None
        self.token_type_ids = None
        self.max_length = max_length

        self.shuffle = shuffle
        self.create_shuffle_order(shuffle)

        self.device = device

    def create_shuffle_order(self, shuffle: bool = True):
        if shuffle:
            self.shuffled_order = torch.randperm(len(self)).tolist()
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
        use_validation: bool = False,
        tokenizer: AutoTokenizer = None,
        path: str = None,
        max_length: int = 512,
        shuffle: bool = True,
        device: str = "cpu",
        **kwargs,
    ):
        path = Path(path)
        splits = cls.load_raw_splits(path=path, use_validation=use_validation, **kwargs)
        all_kwargs = kwargs
        all_kwargs.update(
            dict(
                tokenizer=tokenizer,
                max_length=max_length,
                shuffle=shuffle,
                device=device,
            )
        )
        if isinstance(splits["test"], list):
            return (
                cls(data=splits["train"], **all_kwargs),
                cls(data=splits["val"], **all_kwargs),
                [cls(data=test_split, **all_kwargs) for test_split in splits["test"]],
            )
        return (
            cls(data=splits["train"], **all_kwargs),
            cls(data=splits["val"], **all_kwargs),
            cls(data=splits["test"], **all_kwargs),
        )


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
        use_validation: bool = False,
        tokenizer: AutoTokenizer = None,
        path: str = None,
        max_length: int = 512,
        shuffle: bool = True,
        device: str = "cpu",
        create_artificial_datasets: bool = False,
        **kwargs,
    ):
        path = Path(path)
        # if create_artificial_datasets:
        #     cls.create_artificial_datasets(path, **kwargs)

        return super().create_split_datasets(
            use_validation=use_validation,
            tokenizer=tokenizer,
            path=path,
            max_length=max_length,
            shuffle=shuffle,
            device=device,
            **kwargs,
        )
