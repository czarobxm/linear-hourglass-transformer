from pathlib import Path

import torch
import pandas as pd
from transformers import PreTrainedTokenizer
import torchvision
import torchvision
from PIL import Image
import matplotlib.pyplot as plt


from data.base_dataset import BaseDataset
from data.utils import download_lra


class Pathfinder(BaseDataset):
    name: str = "pathfinder"
    website: str = ""

    def __init__(
        self,
        data: str,
        tokenizer: PreTrainedTokenizer,
        shuffle: bool = True,
        max_length: int = 512,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            data=data,
            tokenizer=tokenizer,
            max_length=max_length,
            shuffle=shuffle,
            device=device,
        )

    def __len__(self) -> int:
        return len(self.data["label"])

    def _load_img(self, index):
        return Image.open(
            self.data["path_base"] / self.data["path_suffix"][index]
        ).convert("L")

    def show_img(self, index) -> None:
        """Show random image from dataset"""
        img = torchvision.transforms.functional.pil_to_tensor(self._load_img(index))
        plt.imshow(img.permute(1, 2, 0).int())

    def __getitem__(self, index):
        img = torchvision.transforms.functional.pil_to_tensor(self._load_img(index))
        return img.to(torch.long).to(self.device).flatten(), torch.tensor(
            [self.data["label"][index]], device=self.device, dtype=torch.long
        )

    @classmethod
    def download_dataset(cls, path: Path = None) -> None:
        if path is None:
            path = Path("./datastorage")

        download_lra(path)

    @classmethod
    def load_raw_splits(cls, path: Path, **kwargs):
        if path is None:
            path = Path(
                "./datastorage/lra_release 3/lra_release/pathfinder32/curv_baseline"
            )

        paths_to_images = []
        labels = []

        metadata_path = path / "metadata"

        for file in metadata_path.iterdir():
            df = pd.read_csv(path / "metadata" / file.name, sep=" ", header=None)
            path_to_images_single_file = df[0] + "/" + df[1]
            paths_to_images.extend(path_to_images_single_file.tolist())
            labels.extend(df[2].tolist())

        return {
            "train": {
                "path_base": path,
                "path_suffix": paths_to_images[:160_000],
                "label": labels[:160_000],
            },
            "val": {
                "abs_path": path,
                "path_suffix": paths_to_images[160_000:180_000],
                "label": labels[160_000:180_000],
            },
            "test": {
                "abs_path": path,
                "path_suffix": paths_to_images[180_000:],
                "label": labels[180_000:],
            },
        }
