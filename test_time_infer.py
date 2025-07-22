import csv

import hydra
import torch

from conf.definitions import ExperimentCfg
from training_setup import (
    initialize_model,
    setup_tokenizer,
    setup_random_seed,
)


@hydra.main(
    version_base=None, config_path="conf/experiments", config_name="enwik9_vanilla_8x4096"
)
def main(cfg: ExperimentCfg) -> None:
    device = cfg.device
    setup_random_seed(cfg.seed)
    tokenizer = setup_tokenizer(cfg.tokenizer)
    model = initialize_model(cfg.model, tokenizer.vocab_size, device=device)

    warmup_steps = 256
    test_steps = 16384

    for _ in range(warmup_steps):
        x = torch.randint(1, tokenizer.vocab_size, (1, 4096)).to(device)
        model(x)

    for _ in range(1, test_steps + 1):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        x = torch.randint(1, tokenizer.vocab_size, (1, 1)).to(device)
        inference_times = []
        for _ in range(test_steps):
            start.record()
            model(x, inference=True)
            end.record()
            torch.cuda.synchronize()
            inference_times.append(start.elapsed_time(end))
            print(inference_times[-1])

    with open("inference_times.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(inference_times)


if __name__ == "__main__":
    main()
