import os

import pandas as pd
from transformers import PreTrainedTokenizer
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


from data.base_dataset import BaseDataset


class Pathfinder(BaseDataset):
    name: str = "pathfinder"
    website: str = ""

    def __init__(
        self,
        data: str,
        path: str,
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
            path=path,
            device=device,
        )

    def __len__(self) -> int:
        return len(self.data["text"])

    def _load_img(self, index):
        return Image.open(
            self.data["path_base"] + "/" + self.data["path_suffix"][index]
        ).convert("RGB")

    def show_img(self, index) -> None:
        """Show random image from dataset"""
        img = torchvision.transforms.functional.pil_to_tensor(self.data["image"][index])
        plt.imshow(img.permute(1, 2, 0).int())

    def __getitem__(self, index):
        image = self._load_img(index)
        transform = transforms.ToTensor()
        image_tensor = transform(image)
        return image_tensor, self.data["label"][index]

    @classmethod
    def load_raw_splits(cls, path: str):
        if path is None:
            path = os.path.abspath(
                "./datastorage/lra_release 3/lra_release/pathfinder32/curv_baseline"
            )

        paths_to_images = []
        labels = []

        for file in os.listdir(path + "/metadata"):
            df = pd.read_csv(path + "/metadata/" + file, sep=" ", header=None)
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
