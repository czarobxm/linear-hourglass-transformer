import csv
import os
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
    # Save original working directory before Hydra changes it
    original_cwd = hydra.utils.get_original_cwd()

    device = cfg.device
    setup_random_seed(cfg.seed)
    tokenizer = setup_tokenizer(cfg.tokenizer)
    model = initialize_model(cfg.model, tokenizer.vocab_size, device=device)

    warmup_steps = 256
    test_steps = 163840

    # Warmup phase
    with torch.no_grad():
        for i in range(warmup_steps):
            x = torch.randint(1, tokenizer.vocab_size, (1, 1)).to(device)
            model(x, inference=True)

    # Get absolute path for the CSV file in the original working directory
    output_file = os.path.join(
        original_cwd,
        f"{cfg.model.mha.method_params.method}_{cfg.model.structure}_inference_times.csv",
    )
    print(f"Saving results to: {output_file}")

    # Open CSV file for writing and write header
    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "inference_time_ms"])  # Header row

        with torch.no_grad():
            # Testing phase - save after each measurement
            for step in range(1, test_steps + 1):
                try:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    x = torch.randint(1, tokenizer.vocab_size, (1, 2)).to(device)

                    start.record()
                    model(x, inference=True)
                    end.record()
                    torch.cuda.synchronize()

                    inference_time = start.elapsed_time(end)
                    print(f"Step {step}: {inference_time:.4f} ms")

                    # Write immediately to file to preserve data
                    writer.writerow([step, inference_time])
                    file.flush()  # Force write to disk

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"OOM error at step {step}: {e}")
                        print(f"Results saved up to step {step-1}")
                        break
                    else:
                        raise e
                except Exception as e:
                    print(f"Unexpected error at step {step}: {e}")
                    print(f"Results saved up to step {step-1}")
                    break


if __name__ == "__main__":
    main()
