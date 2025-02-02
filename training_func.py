import logging
from typing import Tuple
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import neptune
from omegaconf import OmegaConf
from tqdm import tqdm

from models.base import BaseModel
from conf.definitions import Experiment


def log_batch_neptune(
    stage: str,
    run: neptune.Run,
    loss: float,
    correct: int,
    total: int,
    n_iter: int,
    lr: float,
) -> None:
    metrics = {
        f"{stage}_loss": loss,
        f"{stage}_batch_accuracy": correct / total,
        "n_iter": n_iter,
        "lr": lr,
    }

    for key, value in metrics.items():
        run[f"metrics/{key}"].append(value)


def prepare_inputs_and_targets(
    data: torch.Tensor, task: str, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    if task == "classification":
        return data[0].to(device), data[1].unsqueeze(-1).to(torch.float32).to(device)
    elif task == "sequence_modelling":
        inputs = data.detach().clone().to(device)
        targets = data.contiguous().view(-1).to(device)
        return inputs, targets
    else:
        raise ValueError(f"Unsupported task: {task}")


def train_one_batch(
    data: torch.Tensor,
    model: BaseModel,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    update_weights: bool,
    task: str,
) -> Tuple[float, int, int, float, float]:
    # Prepare inputs and targets
    inputs, targets = prepare_inputs_and_targets(data, task, model.device)

    # Forward pass
    outputs = model(inputs).view(targets.shape[0], -1)

    # Compute loss
    loss = loss_fn(outputs, targets)

    # Backward pass
    loss.backward()
    if update_weights:
        optimizer.step()
        optimizer.zero_grad()

    # Compute metrics
    correct = (outputs.argmax(-1) == targets).sum().item()
    total = outputs.shape[0]

    return loss.item(), correct, total


def train_one_epoch(
    train_loader: DataLoader,
    model: BaseModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn: nn.Module,
    gradient_accumulation_steps: int,
    run: neptune.Run,
    task: str,
    n_iter: int = 0,
):
    running_loss = 0
    correct = 0
    total = 0
    accumulated_steps = 0

    for inputs in tqdm(train_loader, "Training:"):
        # Handle gradient accumulation
        if accumulated_steps == gradient_accumulation_steps - 1:
            update_weights = True
            accumulated_steps = 0
        else:
            update_weights = False
            accumulated_steps += 1

        # Train one batch
        loss, cor, tot = train_one_batch(
            inputs, model, optimizer, loss_fn, update_weights, task
        )

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()

        # Log batch metrics
        n_iter += 1
        running_loss += loss
        correct += cor
        total += tot
        lr = optimizer.param_groups[0]["lr"]
        log_batch_neptune(
            stage="train",
            run=run,
            loss=loss,
            correct=cor,
            total=tot,
            n_iter=n_iter,
            lr=lr,
        )
    # Log epoch metrics
    run["metrics/train_epoch_avg_loss"].append(running_loss / len(train_loader))
    run["metrics/train_epoch_accuracy"].append(correct / total)
    return n_iter


def evaluate_one_batch(
    data: torch.Tensor, model: BaseModel, loss_fn: nn.Module, task: str
) -> Tuple[float, int, int]:
    # Prepare inputs and targets
    inputs, targets = prepare_inputs_and_targets(data, task, model.device)
    # Forward pass
    outputs = model(inputs).view(targets.shape[0], -1)
    # Compute loss
    loss = loss_fn(outputs, targets)
    # Compute metrics
    correct = (outputs.argmax(-1) == targets).sum().item()
    return loss.item(), correct, outputs.shape[0]


def evaluate_one_epoch(
    val_loader: DataLoader,
    model: BaseModel,
    loss_fn: nn.Module,
    run: neptune.Run,
    task: str,
    stage: str = "val",
) -> float:
    running_vloss = 0
    correct = total = 0

    with torch.no_grad():
        for vdata in val_loader:
            vloss, batch_correct, batch_total = evaluate_one_batch(
                vdata, model, loss_fn, task
            )
            running_vloss += vloss
            correct += batch_correct
            total += batch_total

    avg_loss = running_vloss / len(val_loader)
    accuracy = correct / total
    run[f"metrics/{stage}_avg_loss"].append(avg_loss)
    run[f"metrics/{stage}_acc"].append(accuracy)
    return running_vloss


def train(
    cfg: Experiment,
    model: BaseModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    run: neptune.Run,
    logger: logging.Logger,
) -> None:
    run["config"] = OmegaConf.to_container(cfg)  # Save the config to Neptune

    n_iter = 0
    for epoch in range(cfg.training.epochs):
        logger.info(f"Epoch {epoch+1}/{cfg.training.epochs}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        n_iter = train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            loss_fn,
            cfg.training.gradient_accumulation_steps,
            run,
            cfg.task,
            n_iter,
        )

        if cfg.training.use_validation:
            evaluate_one_epoch(val_loader, model, loss_fn, run, cfg.task, "val")

        save_model(model, run, epoch)

    evaluate_one_epoch(test_loader, model, loss_fn, run, cfg.task, "test")


def save_model(model: BaseModel, run: neptune.Run, epoch: int) -> None:
    path = Path(f"models_checkpoints/{run['sys/id'].fetch()}")
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / f"epoch-{epoch}.pth")
