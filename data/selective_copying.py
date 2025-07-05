"""Based on https://github.com/MinhZou/selective-copying-mamba/blob/main/data_generator.py"""

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

from data.base_dataset import BaseArtificialDataset


def pad_and_concat_tensors(tensor_list, value):
    """
    Pad tensors to the maximum width and concatenate them along the batch dimension.

    Args:
        tensor_list: List of 1D or 2D tensors where:
                    - 1D tensors: shape (seq_len,) - will be treated as single batch
                    - 2D tensors: shape (batch_size, seq_len) - multiple batches
        debug: Print shapes for debugging

    Returns:
        torch.Tensor: Concatenated tensor with all tensors padded to max width
    """
    # Convert 1D tensors to 2D by adding batch dimension
    processed_tensors = []
    for tensor in tensor_list:
        if tensor.dim() == 1:
            # Shape (seq_len,) -> (1, seq_len)
            processed_tensors.append(tensor.unsqueeze(0))
        elif tensor.dim() == 2:
            # Already (batch_size, seq_len)
            processed_tensors.append(tensor)
        else:
            raise ValueError(
                f"Expected 1D or 2D tensors, got {tensor.dim()}D tensor with shape {tensor.shape}"
            )

    # Find the maximum sequence length (dimension 1)
    max_seq_len = max(tensor.size(1) for tensor in processed_tensors)

    # Pad each tensor to max_seq_len
    padded_tensors = []
    for _, tensor in enumerate(processed_tensors):
        current_seq_len = tensor.size(1)
        if current_seq_len < max_seq_len:
            # Pad the sequence dimension (last dimension)
            # (left_pad, right_pad) for the sequence dimension
            padding = (0, max_seq_len - current_seq_len)
            padded_tensor = F.pad(tensor, padding, "constant", value)
        else:
            padded_tensor = tensor
        padded_tensors.append(padded_tensor)

    # Concatenate along the batch dimension (dimension 0)
    result = torch.cat(padded_tensors, dim=0)

    return result


def process_kwargs(kwargs):
    if isinstance(kwargs["query_len"], str) and isinstance(kwargs["num_samples"], str):
        train_samples, test_samples = kwargs["num_samples"].split(",")
        train, test_start, test_stop = kwargs["query_len"].split(",")
        train, test_start, test_stop = int(train), int(test_start), int(test_stop)

        train_kwargs = kwargs.copy()
        train_kwargs["num_samples"] = int(train_samples)
        train_kwargs["query_len"] = int(train)
        train_kwargs["context_len"] = 2 * int(train)

        test_kwargs = []
        for value in range(test_start, test_stop + 1, 2):
            kwg = kwargs.copy()
            kwg["query_len"] = int(value)
            kwg["num_samples"] = int(test_samples)
            kwg["context_len"] = 2 * int(value)
            test_kwargs.append(kwg)
        return train_kwargs, test_kwargs
    return kwargs, kwargs


def generate_selective_copying_data(
    context_len: int,
    query_len: int,
    vocab_size: int,
    num_samples: int,
):
    num_tokens_to_memorize = int(context_len * 0.5)

    random_integers = torch.randint(2, vocab_size, (num_samples, num_tokens_to_memorize))
    zero_matrix = torch.zeros((num_samples, context_len)).long()
    ones = torch.ones((num_samples, 2)).long()
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
            ones,
            random_integers,
        ],
        dim=1,
    )
    labels = torch.cat(
        [
            torch.ones((num_samples, context_len + 2)).long() * -100,
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

        inputs, labels = generate_selective_copying_data(**kwargs[0])
        torch.save(inputs, create_path(path, "inputs", **kwargs[0]))
        torch.save(labels, create_path(path, "labels", **kwargs[0]))

        if isinstance(kwargs[1], list):
            all_inputs, all_labels = [], []
            for kwg in kwargs[1]:
                inputs, labels = generate_selective_copying_data(**kwg)
                all_inputs.append(inputs)
                all_labels.append(labels)

            all_inputs = pad_and_concat_tensors(all_inputs, value=1)
            all_labels = pad_and_concat_tensors(all_labels, value=-100)

            torch.save(all_inputs, create_path(path, "inputs_test", **kwargs[0]))
            torch.save(all_labels, create_path(path, "labels_test", **kwargs[0]))
        else:
            inputs, labels = generate_selective_copying_data(**kwargs[1])
            torch.save(inputs, create_path(path, "inputs_test", **kwargs[0]))
            torch.save(labels, create_path(path, "labels_test", **kwargs[0]))

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

        test_inputs = torch.load(
            create_path(path=path, inputs_or_labels="inputs_test", **kwargs[0]),
        )
        test_labels = torch.load(
            create_path(path=path, inputs_or_labels="labels_test", **kwargs[0]),
        )

        train_length = len(train_inputs)
        return {
            "train": {
                "inputs": train_inputs[: int(train_length)],
                "labels": train_labels[: int(train_length)],
            },
            "val": {"inputs": train_inputs, "labels": train_labels},
            "test": {
                "inputs": test_inputs,
                "labels": test_labels,
            },
        }
