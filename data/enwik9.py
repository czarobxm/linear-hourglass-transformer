from pathlib import Path
from typing import Dict, TextIO, Union
import urllib
import zipfile


from transformers import PreTrainedTokenizer

from data.base_dataset import BaseDataset


DATA_TYPE = Dict[str, Union[TextIO, int, int]]


class Enwik9(BaseDataset):
    name: str = "enwik9"
    website: str = "https://mattmahoney.net/dc/textdata.html"

    def __init__(
        self,
        data: DATA_TYPE = None,
        tokenizer: PreTrainedTokenizer = None,
        max_length: int = 512,
        shuffle: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            data=data,
            tokenizer=tokenizer,
            max_length=max_length,
            shuffle=shuffle,
            device=device,
        )

    def __len__(self):
        return int((self.data["end"] - self.data["start"]) // self.max_length) + 1

    def __getitem__(self, index):
        if index > len(self):
            raise IndexError("Index out of range")
        # Get shuffled index
        index = self.shuffled_order[index]
        start_idx = self.data["start"] + index * self.max_length

        self.data["file"].seek(start_idx)
        text = self.data["file"].read(self.max_length).decode("utf-8", errors="ignore")

        # Tokenize
        token_dict = self.tokenizer(
            text,
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return token_dict["input_ids"].squeeze(0).to(self.device)

    @classmethod
    def download_dataset(cls, path: Path = None):
        if path is None:
            path = Path("./datastorage/enwik9")

        path.mkdir(exist_ok=True, parents=True)

        # Download
        if not (path / "enwik9.zip").exists():
            urllib.request.urlretrieve(
                "http://mattmahoney.net/dc/enwik9.zip", path / "enwik9.zip"
            )

        # Unzip
        with zipfile.ZipFile(path / "enwik9.zip", "r") as zip_ref:
            zip_ref.extractall(path)

        # Remove zip file
        zip_path = path / "enwik9.zip"
        zip_path.unlink()

    @classmethod
    def load_raw_splits(  # pylint: disable=arguments-differ
        cls, path: Path, use_validation: bool, **kwargs
    ) -> Dict[str, DATA_TYPE]:
        if path is None:
            path = Path("./datastorage/enwik9")
        if not (path / "enwik9").exists():
            cls.download_dataset(path)

        file = open(path / "enwik9", "rb")

        train = {"file": file, "start": 0, "end": 900_000_000}
        if use_validation:
            val = {"file": file, "start": 900_000_000, "end": 950_000_000}
            test = {"file": file, "start": 950_000_000, "end": 1_000_000_000}
        else:
            val = {"file": file, "start": 0, "end": 0}
            test = {"file": file, "start": 900_000_000, "end": 1_000_000_000}

        return {
            "train": train,
            "val": val,
            "test": test,
        }
