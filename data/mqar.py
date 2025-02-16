from pathlib import Path

import torch
import numpy as np

from data.base_dataset import BaseArtificialDataset


def generate_mqar_data(
    vocab_size: int,
    num_examples: int = 32,
    input_seq_len: int = 512,
    seed: int = 42,
    power_a: float = 0.01,
    num_kv_pairs: int = 8,
    random_non_queries: bool = True,
):
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    assert num_kv_pairs * 4 <= input_seq_len

    np.random.seed(seed)

    # two tokens for key and value
    context_size = num_kv_pairs * 2

    # create keys so that each key is present exactly once in each example
    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(
        np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs
    )

    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(
        np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs
    )

    # create sequences
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    # compute power law
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()

    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(
        np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs
    )

    # queries and answers
    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
    examples = np.concatenate([kvs, queries], axis=1)

    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)

    inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])

    # replace all the 0 with random values
    if random_non_queries:
        inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]

    return inputs, labels


def create_path(
    path: Path,
    inputs_or_labels: str,
    vocab_size: str,
    num_examples: str,
    input_seq_len: str,
    seed: str,
    power_a: str,
    num_kv_pairs: str,
    random_non_queries: str,
):
    full_path = (
        f"{path}/"
        f"{inputs_or_labels}-"
        f"vocab_size_{vocab_size}-"
        f"num_examples_{num_examples}-"
        f"input_seq_len_{input_seq_len}-"
        f"seed_{seed}-"
        f"power_a_{power_a}-"
        f"num_kv_pairs_{num_kv_pairs}-"
        f"random_non_queries_{random_non_queries}.pt"
    )
    if not path.exists():
        path.mkdir(path, exist_ok=True)
    return full_path


class MQAR(BaseArtificialDataset):
    name: str = "mqar"

    def __init__(
        self,
        data: str,
        tokenizer: str,
        vocab_size: int,
        num_examples: int,
        input_seq_len: int,
        max_length: int = 512,
        shuffle: bool = True,
        power_a: float = 0.01,
        num_kv_pairs: int = 8,
        random_non_queries: bool = True,
        seed: int = 42,
        device: str = "cpu",
    ):
        super().__init__(
            data=data,
            tokenizer=tokenizer,
            max_length=max_length,
            shuffle=shuffle,
            device=device,
        )
        self.vocab_size = vocab_size
        self.num_examples = num_examples
        self.input_seq_len = input_seq_len
        self.power_a = power_a
        self.num_kv_pairs = num_kv_pairs
        self.random_non_queries = random_non_queries
        self.seed = seed

    def __len__(self):
        return self.data["inputs"].shape[0]

    def __getitem__(self, index):
        return (
            self.data["inputs"][index].to(self.device),
            self.data["labels"][index].to(self.device),
        )

    @classmethod
    def load_raw_splits(
        cls,
        path: Path = None,
        vocab_size: int = 128,
        num_examples: int = 32,
        input_seq_len: int = 64,
        seed: int = 42,
        power_a: float = 0.01,
        num_kv_pairs: int = 8,
        random_non_queries: bool = True,
        **kwargs,
    ):
        if path is None:
            path = Path("./datastorage/mqar")

        if isinstance(path, str):
            path = Path(path)

        params = {
            "vocab_size": vocab_size,
            "num_examples": num_examples,
            "input_seq_len": input_seq_len,
            "seed": seed,
            "power_a": power_a,
            "num_kv_pairs": num_kv_pairs,
            "random_non_queries": random_non_queries,
        }
        inputs = torch.load(
            create_path(path=path, inputs_or_labels="inputs", **params),
        )
        labels = torch.load(
            create_path(path=path, inputs_or_labels="labels", **params),
        )
        length = inputs.shape[0]
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

    @classmethod
    def create_artificial_datasets(
        cls,
        path: Path,
        **kwargs,
    ):
        if path is None:
            path = Path("./datastorage/mqar")
        inputs, labels = generate_mqar_data(**kwargs)
        torch.save(inputs, create_path(path=path, inputs_or_labels="inputs", **kwargs))
        torch.save(labels, create_path(path=path, inputs_or_labels="labels", **kwargs))
