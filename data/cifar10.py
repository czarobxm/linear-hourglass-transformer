import os

import torchvision
from transformers import PreTrainedTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt

from data.base_dataset import BaseDataset


class Cifar10(BaseDataset):
    name: str = "cifar10"
    website: str = ""

    def __init__(
        self,
        data: str,
        tokenizer: PreTrainedTokenizer,
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
        return len(self.data["image"])

    def __getitem__(self, index):
        return (
            self._prepare_image(self.data["image"][index]).to(self.device),
            self.data["label"][index].to(self.device),
        )

    def show_img(self, index) -> None:
        """Show random image from dataset"""
        img = torchvision.transforms.functional.pil_to_tensor(self.data["image"][index])
        plt.imshow(img.permute(1, 2, 0).int())

    @staticmethod
    def _prepare_image(img):
        img = torchvision.transforms.functional.pil_to_tensor(img) / 255.0
        img = img.transpose(1, 2).flatten(2)
        return img

    @classmethod
    def load_raw_splits(cls, path: str, **kwargs):
        if path is None:
            path = os.path.abspath("./datastorage/cifar10")
        os.makedirs(path, exist_ok=True)

        train = load_dataset(
            "uoft-cs/cifar10", cache_dir=path, split="train", resume_download=None
        )
        test = load_dataset(
            "uoft-cs/cifar10", cache_dir=path, split="test", resume_download=None
        )
        return {
            "train": {"image": train["img"], "label": train["label"]},
            "val": {"image": test["img"][:5000], "label": test["label"][:5000]},
            "test": {"image": test["img"][5000:], "label": test["label"][5000:]},
        }
