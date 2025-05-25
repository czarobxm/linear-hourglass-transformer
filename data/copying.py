"""Based on https://github.com/MinhZou/selective-copying-mamba/blob/main/data_generator.py"""

from pathlib import Path

import torch
from transformers import PreTrainedTokenizer

from data.base_dataset import BaseArtificialDataset


def process_kwargs(kwargs):
    if isinstance(kwargs["sequence_length"], str) and isinstance(
        kwargs["num_samples"], str
    ):
        train_samples, test_samples = kwargs["num_samples"].split(",")
        train, test_start, test_stop = kwargs["sequence_length"].split(",")
        train, test_start, test_stop = int(train), int(test_start), int(test_stop)

        train_kwargs = kwargs.copy()
        train_kwargs["sequence_length"] = int(train)
        train_kwargs["num_samples"] = int(train_samples)

        test_kwargs = kwargs.copy()
        test_kwargs = []
        for value in range(test_start, test_stop + 1, 2):
            kwg = kwargs.copy()
            kwg["sequence_length"] = int(value)
            kwg["num_samples"] = int(test_samples)
            test_kwargs.append(kwg)
        return train_kwargs, test_kwargs
    return kwargs, kwargs


def generate_copying_data(
    sequence_length: int,
    vocab_size: int,
    num_samples: int,
):
    """
    Generates random sequences for the copying task.
    The inputs are of the form:  [x_1,  x_2,  ..., x_n,  0,   x_1, x_2, ..., x_n-1, x_n, 1, ..., 1],
    The outputs are of the form: [-100, -100, ..., -100, x_1, x_2, x_3, ..., x_n  , 1,   1, ..., 1],
    where n is the sequence_length and 0 is the separator token and 1 is the padding token.
    The intput sequence length is between 60% and 100% of the sequence_length parameter and
    the length distribution follows the uniform distribution.
    """
    min_sequence_length = int(sequence_length * 0.6)
    inputs_length = 2 * sequence_length + 2

    all_inputs = []
    all_labels = []

    for seq_len in range(min_sequence_length, sequence_length + 1):
        # print(seq_len)
        num_samples_per_length = max(
            1, int(num_samples / (sequence_length - min_sequence_length + 1))
        )

        random_sequence = torch.randint(
            low=2, high=vocab_size, size=(num_samples_per_length, seq_len)
        )
        mask_matrix = torch.zeros((num_samples_per_length, seq_len)).long() - 100
        separator = torch.zeros((num_samples_per_length, 1)).long()

        inputs_not_padded = torch.cat(
            [random_sequence, separator, random_sequence], dim=1
        )
        labels_not_padded = torch.cat([mask_matrix, separator, random_sequence], dim=1)

        inputs = torch.ones((num_samples_per_length, inputs_length)).long()
        labels = torch.ones((num_samples_per_length, inputs_length)).long() * -100

        inputs[:, : 2 * seq_len + 1] = inputs_not_padded
        labels[:, : 2 * seq_len + 1] = labels_not_padded

        all_inputs.append(inputs)
        all_labels.append(labels)

    return torch.cat(all_inputs, dim=0), torch.cat(all_labels, dim=0)


def create_path(
    path: Path,
    inputs_or_labels,
    sequence_length: int = 20,
    vocab_size: int = 5,
    num_samples: int = 10,
    **_kwargs,
):
    full_path = (
        f"{path}/"
        f"{inputs_or_labels}-"
        f"vocab_size_{vocab_size}-"
        f"sequence_length_{sequence_length}-"
        f"num_samples_{num_samples}.pt"
    )
    if not path.exists():
        path.mkdir(exist_ok=True, parents=True)
    return full_path


class Copying(BaseArtificialDataset):
    name: str = "copying"

    def __init__(
        self,
        data: str,
        tokenizer: PreTrainedTokenizer = None,
        max_length: int = 512,
        shuffle: bool = True,
        device: str = "cpu",
        vocab_size: int = 5,
        sequence_length: int = 20,
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
        self.sequence_length = sequence_length
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

        inputs, labels = generate_copying_data(**kwargs[0])
        torch.save(inputs, create_path(path, "inputs", **kwargs[0]))
        torch.save(labels, create_path(path, "labels", **kwargs[0]))

        if isinstance(kwargs[1], list):
            for kwg in kwargs[1]:
                inputs, labels = generate_copying_data(**kwg)
                torch.save(inputs, create_path(path, "inputs", **kwg))
                torch.save(labels, create_path(path, "labels", **kwg))
        else:
            inputs, labels = generate_copying_data(**kwargs[1])
            torch.save(inputs, create_path(path, "inputs", **kwargs[1]))
            torch.save(labels, create_path(path, "labels", **kwargs[1]))

    @classmethod
    def load_raw_splits(cls, path: str, **kwargs):
        if path is None:
            path = Path("./datastorage/sequence_modelling")

        kwargs.pop("use_validation")
        kwargs = process_kwargs(kwargs)

        train_inputs = torch.load(
            create_path(path=path, inputs_or_labels="inputs", **kwargs[0]),
        )
        train_labels = torch.load(
            create_path(path=path, inputs_or_labels="labels", **kwargs[0]),
        )

        if isinstance(kwargs[1], list):
            test_inputs = []
            test_labels = []
            for kwg in kwargs[1]:
                test_inputs.append(
                    torch.load(
                        create_path(path=path, inputs_or_labels="inputs", **kwg),
                    )
                )
                test_labels.append(
                    torch.load(
                        create_path(path=path, inputs_or_labels="labels", **kwg),
                    )
                )
        else:
            test_inputs = torch.load(
                create_path(path=path, inputs_or_labels="inputs", **kwargs[2]),
            )
            test_labels = torch.load(
                create_path(path=path, inputs_or_labels="labels", **kwargs[2]),
            )

        train_length = len(train_inputs)
        return {
            "train": {
                "inputs": train_inputs[: int(train_length * 0.8)],
                "labels": train_labels[: int(train_length * 0.8)],
            },
            "val": {
                "inputs": train_inputs[int(train_length * 0.8) :],
                "labels": train_labels[int(train_length * 0.8) :],
            },
            "test": [
                {
                    "inputs": test_inputs[i],
                    "labels": test_labels[i],
                }
                for i in range(1, len(test_inputs))
            ],
        }
