import logging
import os

import hydra
from hydra.core.config_store import ConfigStore
from transformers import AutoTokenizer

from conf.definitions import ExperimentCfg
from training_setup import (
    initialize_model,
    setup_tokenizer,
)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Registering the Config class with the name 'config'.
cs = ConfigStore.instance()
cs.store(name="experiment", node=ExperimentCfg)


@hydra.main(
    version_base=None, config_path="conf/experiments", config_name="enwik9_vanilla_8x4096"
)
def main(cfg: ExperimentCfg) -> None:
    print(cfg.model)
    device = cfg.device

    logging.info("Creating tokenizer...")
    tokenizer: AutoTokenizer = setup_tokenizer(cfg.tokenizer)
    logging.info("Tokenizer created.")

    model = initialize_model(cfg.model, tokenizer.vocab_size, device=device)
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_param}")


if __name__ == "__main__":
    main()  # pylint: disable=E1120
