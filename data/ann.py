import csv
import os
import random
import subprocess
import sys
import tempfile
from typing import Dict
from pathlib import Path

import torch
from transformers import PreTrainedTokenizer

from data.base_dataset import BaseDataset
from data.utils import download_lra


def csv_row_generator(file_path):
    with open(file_path, newline="", encoding="utf-8") as file:
        for row in file:
            yield row


def shuffle_large_csv(file_path, temp_dir=None, sort_command="sort"):
    """
    Shuffles a large CSV file using an intermediate file and external sort command.

    Args:
        file_path (str): Path to the input CSV file.
        temp_dir (str, optional): Directory to use for the intermediate file.
                                  Defaults to the directory of the input file.
        sort_command (str): Path or name of the external sort utility (e.g., 'sort' or 'gsort').
    """
    # Field size limit handling (same as before)
    try:
        csv.field_size_limit(2 * 1024 * 1024)
    except OverflowError:
        print("Warning: Could not set CSV field size limit.")
        csv.field_size_limit(131072)

    dir_name, base_name = os.path.split(file_path)
    name, ext = os.path.splitext(base_name)
    shuffled_filename = os.path.join(dir_name, f"{name}_shuffled{ext}")
    intermediate_filename = os.path.join(
        temp_dir or dir_name, f"{name}_intermediate_for_sort{ext}"
    )

    print(f"Step 1: Creating intermediate file with random keys: {intermediate_filename}")
    header = None
    try:
        # Step 1: Write each row (except header) with a random key to an intermediate file
        with open(file_path, newline="", encoding="utf-8") as infile, open(
            intermediate_filename, "w", newline="", encoding="utf-8"
        ) as intermediate_file:
            reader = csv.reader(infile)
            writer = csv.writer(
                intermediate_file, quoting=csv.QUOTE_MINIMAL
            )  # Adjust quoting if needed

            try:
                header = next(reader)  # Read the header
            except StopIteration:
                print("Warning: Input CSV is empty.")
                # Clean up intermediate file if created
                if os.path.exists(intermediate_filename):
                    os.remove(intermediate_filename)
                # Create empty output file?
                if header:
                    with open(
                        shuffled_filename, "w", newline="", encoding="utf-8"
                    ) as outfile:
                        writer_out = csv.writer(outfile)
                        writer_out.writerow(header)
                return  # Exit

            for i, row in enumerate(reader):
                # Generate a random key, using a fixed-width format can sometimes help external sort
                # Ensure the key doesn't contain the delimiter (,)
                rand_key = f"{random.random():.17f}"  # String format
                # Prepend key. Ensure the key itself doesn't mess up CSV parsing later
                # Using a simple comma delimiter for the key might be fine if keys don't contain commas
                # and data fields are properly quoted by the csv writer.
                # Using QUOTE_MINIMAL to avoid unnecessary quotes.
                writer.writerow([rand_key] + row)

        print(f"Step 2: Sorting intermediate file using external '{sort_command}'...")
        # Step 2: Use external sort command
        # Ensure the intermediate file is closed before sorting
        # sort options:
        # -t, : Use comma as the field delimiter
        # -k1,1n : Sort numerically (-n) based on the first field (-k1,1)
        # -o : Specify the output file
        # It's safer to write the sorted output to a temporary file first,
        # then add the header and move it to the final location.
        sorted_temp_filename = os.path.join(
            temp_dir or dir_name, f"{name}_sorted_temp{ext}"
        )

        sort_cmd_list = [
            sort_command,
            "-t,",  # Delimiter is comma
            "-k1,1n",  # Sort numerically on the first field
            "-o",
            sorted_temp_filename,  # Output file
            intermediate_filename,  # Input file
        ]

        try:
            # Add LC_ALL=C for potentially faster sorting and consistent behavior
            env = os.environ.copy()
            env["LC_ALL"] = "C"
            process = subprocess.run(
                sort_cmd_list, check=True, capture_output=True, text=True, env=env
            )
            print("External sort completed.")
        except FileNotFoundError:
            print(
                f"Error: Sort command '{sort_command}' not found. Please install it or provide the correct path."
            )
            raise
        except subprocess.CalledProcessError as e:
            print(f"Error during external sort:")
            print(f"Command: {' '.join(e.cmd)}")
            print(f"Return Code: {e.returncode}")
            print(f"Stderr: {e.stderr}")
            print(f"Stdout: {e.stdout}")
            raise  # Re-raise the exception

        print(f"Step 3: Writing final shuffled file: {shuffled_filename}")
        # Step 3: Write header and the sorted content (without keys) to final file
        with open(
            sorted_temp_filename, newline="", encoding="utf-8"
        ) as sorted_infile, open(
            shuffled_filename, "w", newline="", encoding="utf-8"
        ) as outfile:
            writer = csv.writer(outfile)
            reader = csv.reader(sorted_infile)

            if header:
                writer.writerow(header)  # Write header

            for row in reader:
                if row:  # Make sure row is not empty
                    writer.writerow(
                        row[1:]
                    )  # Write row skipping the first element (the sort key)

        # Clean up the temporary sorted file
        if os.path.exists(sorted_temp_filename):
            print(f"Cleaning up temporary sorted file: {sorted_temp_filename}")
            os.remove(sorted_temp_filename)

        print(f"Shuffled file created: {shuffled_filename}")

    finally:
        # Step 4: Clean up the intermediate file
        if os.path.exists(intermediate_filename):
            print(f"Cleaning up intermediate file: {intermediate_filename}")
            os.remove(intermediate_filename)


class ANN(BaseDataset):
    name: str = ""
    website: str = ""

    def __init__(
        self,
        data: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 4096,
        shuffle: bool = True,
        separator_token: str = " [SEP] ",
        tokens_per_text: int = 2045,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            data=data,
            tokenizer=tokenizer,
            max_length=max_length,
            shuffle=shuffle,
            device=device,
        )
        self.separator_token = separator_token
        self.tokens_per_text = tokens_per_text

        if shuffle:
            shuffle_large_csv(self.data["path"])
            dir_name, base_name = os.path.split(self.data["path"])
            name, ext = os.path.splitext(base_name)
            shuffled_filename = os.path.join(dir_name, f"{name}_shuffled{ext}")
            self.data["iterator"] = csv_row_generator(shuffled_filename)
        else:
            self.data["iterator"] = csv_row_generator(self.data["path"])

    def __len__(self) -> int:
        return self.data["length"]

    def __getitem__(self, index: int) -> Dict[str, str]:
        line = next(self.data["iterator"]).strip()
        label, _, _, text_1, text_2 = line.split("\t")
        label = label.strip("\"' ")
        text = (
            text_1[: self.tokens_per_text]
            + self.separator_token
            + text_2[: self.tokens_per_text]
        )

        # Tokenize
        token_dict = self.tokenizer(
            text,
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        return (
            token_dict["input_ids"].squeeze(0).to(self.device),
            torch.tensor(float(label), dtype=torch.long, device=self.device),
        )

    @classmethod
    def download_dataset(cls, path: Path = None) -> None:
        if path is None:
            path = Path("./datastorage")

        download_lra(path)

    @classmethod
    def load_raw_splits(cls, path: Path = None, **kwargs) -> Dict[str, str]:
        if path is None:
            path = Path("./datastorage/lra_release 3/lra_release/tsv_data")
        if not path.exists():
            cls.download_dataset(path)

        train_path = f"{path}/new_aan_pairs.train.tsv"
        val_path = f"{path}/new_aan_pairs.eval.tsv"
        test_path = f"{path}/new_aan_pairs.test.tsv"

        train_length = subprocess.run(
            ["wc", "-l", train_path], stdout=subprocess.PIPE, text=True, check=True
        )
        train_length = int(train_length.stdout.strip().split()[0])

        val_length = subprocess.run(
            ["wc", "-l", val_path], stdout=subprocess.PIPE, text=True, check=True
        )
        val_length = int(val_length.stdout.strip().split()[0])

        test_length = subprocess.run(
            ["wc", "-l", test_path], stdout=subprocess.PIPE, text=True, check=True
        )
        test_length = int(test_length.stdout.strip().split()[0])

        return {
            "train": {
                "path": train_path,
                "length": train_length,
            },
            "val": {
                "path": val_path,
                "length": val_length,
            },
            "test": {
                "path": test_path,
                "length": test_length,
            },
        }
