import hydra
import torch

from conf.definitions import ExperimentCfg
from training_setup import (
    initialize_model,
    setup_tokenizer,
    setup_random_seed,
    setup_training,
)


@hydra.main(
    version_base=None, config_path="conf/experiments", config_name="enwik9_vanilla_8x4096"
)
def main(cfg: ExperimentCfg) -> None:
    print(cfg)
    device = cfg.device
    setup_random_seed(cfg.seed)
    tokenizer = setup_tokenizer(cfg.tokenizer)
    model = initialize_model(cfg.model, tokenizer.vocab_size, device=device)

    training_setup = setup_training(cfg.training, model, 1000)
    optimizer = training_setup["optimizer"]
    criterion = training_setup["loss_fn"]

    warmup_steps = 256
    test_steps = 128
    seq_lens = [128, 512, 1024, 2048, 4096, 8096, 16384, 32768, 65536]

    for _ in range(warmup_steps):
        x = torch.randint(1, tokenizer.vocab_size, (1, 4096)).to(device)
        model(x)

    for seq_len in seq_lens:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        x = torch.randint(1, tokenizer.vocab_size, (1, seq_len)).to(device)
        inference_times = []
        for _ in range(test_steps):
            print(model(x).shape, x.shape)
            loss = criterion(
                model(x).view(seq_len, tokenizer.vocab_size), x.view(seq_len)
            )
            start.record()
            loss.backward()
            end.record()
            torch.cuda.synchronize()
            optimizer.step()
            optimizer.zero_grad()
            inference_times.append(start.elapsed_time(end))

        print(
            f"Mean time per backward step for seq_len {seq_len}: {sum(inference_times) / len(inference_times)} ms"
        )


if __name__ == "__main__":
    main()
