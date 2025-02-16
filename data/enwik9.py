from pathlib import Path
from typing import Dict
import urllib
import zipfile


from transformers import PreTrainedTokenizer

from data.base_dataset import BaseDataset


class Enwik9(BaseDataset):
    name: str = "enwik9"
    website: str = "https://mattmahoney.net/dc/textdata.html"

    def __init__(
        self,
        data: str = "./datastorage/enwik9",
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
        return int(len(self.data) // self.max_length) + 1

    def __getitem__(self, index):
        if index > len(self):
            raise IndexError("Index out of range")
        # Get shuffled index
        start_idx = index * self.max_length
        end_idx = (index + 1) * self.max_length - 1

        # Tokenize
        token_dict = self.tokenizer(
            self.data[start_idx:end_idx],
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

        path.mkdir(path, exist_ok=True)

        # Download
        if not path.exists(path / "enwik9"):
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
    def load_raw_splits(cls, path: Path, **kwargs) -> Dict[str, str]:
        if path is None:
            path = Path("./datastorage/enwik9")
        if not (path / "enwik9").exists():
            cls.download_dataset(path)
        with open(path / "enwik9", "r", encoding="utf-8") as file:
            text = file.read()
        train = text[:90_000_000]
        val = text[90_000_000:95_000_000]
        test = text[95_000_000:]
        return {"train": train, "val": val, "test": test}
