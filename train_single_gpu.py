import logging

import hydra
from hydra.core.config_store import ConfigStore
from transformers import AutoTokenizer

from conf.definitions import Experiment
from training_func import train
from training_setup import (
    setup_logging,
    create_dataloaders,
    setup_neptune,
    initialize_model,
    setup_training,
    setup_tokenizer,
)


# Registering the Config class with the name 'config'.
cs = ConfigStore.instance()
cs.store(name="experiment", node=Experiment)


@hydra.main(
    version_base=None, config_path="conf/experiments", config_name="enwik9_vanilla_8x4096"
)
def main(cfg: Experiment) -> None:
    device = cfg.device

    logger = setup_logging()
    logging.info("Starting training script.")

    logging.info("Creating tokenizer...")
    tokenizer: AutoTokenizer = setup_tokenizer(cfg.tokenizer)
    logging.info("Tokenizer created.")

    logging.info("Creating DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        cfg.dataset, cfg.training, tokenizer
    )
    logging.info("Data loaders created.")

    run = setup_neptune(cfg.neptune)
    logging.info("Neptune run initialized.")

    model = initialize_model(cfg.model, tokenizer.vocab_size, device=device)
    print(tokenizer.vocab_size)
    logging.info("Model %s initialized: ", cfg.model)

    training_setup = setup_training(cfg.training, model, len(train_loader))
    logging.info("Training setup completed.")

    logging.info("Starting training...")
    train(
        cfg=cfg,
        model=model,
        optimizer=training_setup["optimizer"],
        scheduler=training_setup["scheduler"],
        loss_fn=training_setup["loss_fn"],
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        run=run,
        logger=logger,
    )
    logging.info("Training finished.")

    logging.info("Average loss on test set: %s", run["metrics/test_avg_loss"])
    logging.info("Accuracy on test set: %s", "metrics/test_acc")
    logging.info("Evaluation finished.")

    run.stop()


if __name__ == "__main__":
    main()  # pylint: disable=E1120
