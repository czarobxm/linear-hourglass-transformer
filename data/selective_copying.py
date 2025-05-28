"""Based on https://github.com/MinhZou/selective-copying-mamba/blob/main/data_generator.py"""

from pathlib import Path

import torch
from transformers import PreTrainedTokenizer

from data.base_dataset import BaseArtificialDataset


def process_kwargs(kwargs):
    print(kwargs)
    length = None
    is_list = None
    for key, value in kwargs.items():
        if is_list is None:
            is_list = isinstance(value, list)

        if length is None:
            print(key, value, type(value), isinstance(value, list) == is_list)
            length = len(value)

        if len(value) != length:
            raise ValueError(
                f"All lists must have the same length. "
                f"Expected {length}, but got {len(value)} for key: {key}"
            )

        if isinstance(value, list) ^ is_list:  # XOR gate
            raise ValueError(
                f"All lists must be of the same type. "
                f"Expected list, but got {type(value)} for key: {key}"
            )

    if not is_list:
        return [kwargs]

    length = len(next(iter(kwargs.values())))  # Get the length of the lists
    result = []

    for i in range(length):
        entry = {key: value[i] for key, value in kwargs.items()}
        result.append(entry)
    return result


def generate_selective_copying_data(
    context_len: int,
    query_len: int,
    vocab_size: int,
    num_samples: int,
):
    num_tokens_to_memorize = int(context_len * 0.6)

    random_integers = torch.randint(1, vocab_size, (num_samples, num_tokens_to_memorize))
    zero_matrix = torch.zeros((num_samples, context_len)).long()
    positions = torch.rand(num_samples, context_len).argsort(dim=1)[
        :, :num_tokens_to_memorize
    ]  # Get first `elems_to_copy` indices
    row_indices = (
        torch.arange(num_samples).unsqueeze(1).expand(-1, num_tokens_to_memorize)
    )  # Row indices

    positions = positions.sort(dim=1)[0]
    zero_matrix[row_indices, positions] = random_integers.long()  # Assign random integers

    inputs = torch.cat(
        [
            zero_matrix,
            torch.zeros((num_samples, num_tokens_to_memorize)).long() + vocab_size + 1,
        ],
        dim=1,
    )
    labels = torch.cat(
        [
            torch.ones((num_samples, context_len)).long() * -100,
            random_integers[:, :query_len],
        ],
        dim=1,
    )
    return inputs, labels


def create_path(
    path: Path,
    inputs_or_labels,
    context_len: int = 5,
    vocab_size: int = 5,
    query_len: int = 15,
    num_samples: int = 10,
    **_kwargs,
):
    full_path = (
        f"{path}/"
        f"{inputs_or_labels}-"
        f"vocab_size_{vocab_size}-"
        f"context_len_{context_len}-"
        f"query_len_{query_len}-"
        f"num_samples_{num_samples}.pt"
    )
    if not path.exists():
        path.mkdir(exist_ok=True, parents=True)
    return full_path


class SelectiveCopying(BaseArtificialDataset):
    name: str = "selective_copying"

    def __init__(
        self,
        data: str,
        tokenizer: PreTrainedTokenizer = None,
        max_length: int = 512,
        shuffle: bool = True,
        device: str = "cpu",
        vocab_size: int = 5,
        context_len: int = 5,
        query_len: int = 15,
        num_samples: int = 10,
    ):
        super().__init__(
            data=data,
            tokenizer=tokenizer,
            max_length=max_length,
            shuffle=shuffle,
            device=device,
        )
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.query_len = query_len
        self.num_samples = num_samples

    def __len__(self):
        return self.data["inputs"].shape[0]

    def __getitem__(self, index):
        return (
            self.data["inputs"][index].to(self.device),
            self.data["labels"][index].to(self.device),
        )

    @classmethod
    def create_artificial_datasets(cls, path: str, **kwargs):
        if path is None:
            path = Path("./datastorage/sequence_modelling")
        kwargs = process_kwargs(kwargs)
        for kwg in kwargs:
            inputs, labels = generate_selective_copying_data(**kwg)
            torch.save(inputs, create_path(path, "inputs", **kwg))
            torch.save(labels, create_path(path, "labels", **kwg))

    @classmethod
    def load_raw_splits(cls, path: str, **kwargs):
        if path is None:
            path = Path("./datastorage/sequence_modelling")

        kwargs.pop("use_validation")
        kwargs = process_kwargs(kwargs)

        print(kwargs)

        if isinstance(kwargs, list):
            inputs = []
            labels = []
            for kwg in kwargs:
                inputs.append(
                    torch.load(
                        create_path(path=path, inputs_or_labels="inputs", **kwg),
                    )
                )
                labels.append(
                    torch.load(
                        create_path(path=path, inputs_or_labels="labels", **kwg),
                    )
                )

        if len(inputs) == 1:
            length = len(inputs[0])
            return {
                "train": {
                    "inputs": inputs[: int(length * 0.8)],
                    "labels": labels[: int(length * 0.8)],
                },
                "val": {
                    "inputs": inputs[int(length * 0.8) : int(length * 0.9)],
                    "labels": labels[int(length * 0.8) : int(length * 0.9)],
                },
                "test": {
                    "inputs": inputs[int(length * 0.9) :],
                    "labels": labels[int(length * 0.9) :],
                },
            }
        else:
            length = len(inputs[0])
            return {
                "train": {
                    "inputs": inputs[0][: int(length * 0.8)],
                    "labels": labels[0][: int(length * 0.8)],
                },
                "val": {
                    "inputs": inputs[0][int(length * 0.8) :],
                    "labels": labels[0][int(length * 0.8) :],
                },
                "test": [
                    {
                        "inputs": inputs[i],
                        "labels": labels[i],
                    }
                    for i in range(1, len(inputs))
                ],
            }
