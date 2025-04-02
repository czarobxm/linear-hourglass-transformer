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
from conf.definitions import ExperimentCfg


def log_batch_neptune(
    stage: str,
    run: neptune.Run,
    loss: float,
    correct: int,
    total: int,
    n_iter: int,
    lr: float,
    losses_history: list = None,
    rolling_window_sizes: list = None,
    running_avgs: list = None,
    smoothing_factors: list = None,
) -> None:
    metrics = {
        f"{stage}_loss": loss,
        f"{stage}_batch_accuracy": correct / total,
        "n_iter": n_iter,
        "lr": lr,
    }

    # Log running average losses with different smoothing factors
    if running_avgs is not None and smoothing_factors is not None:
        for i, factor in enumerate(smoothing_factors):
            factor_key = str(factor).replace(".", "_")
            metrics[f"{stage}_running_avg_loss_{factor_key}"] = running_avgs[i]

    # Log rolling mean losses if history and window sizes are provided
    if losses_history is not None and rolling_window_sizes is not None:
        for window_size in rolling_window_sizes:
            if len(losses_history) >= window_size:
                rolling_mean = sum(losses_history[-window_size:]) / window_size
                metrics[f"{stage}_loss_rolling_{window_size}"] = rolling_mean

    for key, value in metrics.items():
        run[f"metrics/{key}"].append(value)


def prepare_inputs_and_targets(
    data: torch.Tensor, task: str, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    if task == "classification":
        return data[0].to(device), data[1].to(device)
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
    outputs = model(inputs)
    outputs = outputs.view(targets.shape[0], -1)

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
    rolling_window_sizes: list = None,
    smoothing_factors: list = None,  # List of smoothing factors
    n_iter: int = 0,
    prev_running_avgs: list = None,  # Previous running averages from last epoch
    prev_losses_history: list = None,  # Previous losses history from last epoch
    max_history_size: int = None,  # Maximum size to keep in losses history
):
    running_loss = 0
    correct = 0
    total = 0
    accumulated_steps = 0

    # Initialize losses history with previous values or create new one
    if prev_losses_history is not None:
        losses_history = prev_losses_history.copy()
    else:
        losses_history = []

    # Trim history if needed
    if max_history_size is not None and len(losses_history) > max_history_size:
        losses_history = losses_history[-max_history_size:]

    # Initialize running averages with previous values or create new ones
    running_avgs = None
    if smoothing_factors:
        if prev_running_avgs is None or len(prev_running_avgs) != len(smoothing_factors):
            # Initialize with None values if not provided or length mismatch
            running_avgs = [None] * len(smoothing_factors)
        else:
            running_avgs = prev_running_avgs.copy()

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

        # Store loss in history
        losses_history.append(loss)

        # Trim history if needed
        if max_history_size is not None and len(losses_history) > max_history_size:
            losses_history = losses_history[-max_history_size:]

        # Update running averages with different smoothing factors
        if smoothing_factors and running_avgs:
            for i, factor in enumerate(smoothing_factors):
                if running_avgs[i] is None:
                    running_avgs[i] = loss
                else:
                    running_avgs[i] = factor * loss + (1 - factor) * running_avgs[i]

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
            losses_history=losses_history,
            rolling_window_sizes=rolling_window_sizes,
            running_avgs=running_avgs,
            smoothing_factors=smoothing_factors,
        )

    # Log epoch metrics
    run["metrics/train_epoch_avg_loss"].append(running_loss / len(train_loader))
    run["metrics/train_epoch_accuracy"].append(correct / total)

    return n_iter, running_avgs, losses_history


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
    cfg: ExperimentCfg,
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

    # Initialize tracking variables
    n_iter = 0
    running_avgs = None
    losses_history = None
    best_val_loss = float("inf")

    # Determine max history size if rolling window sizes are specified
    max_history_size = (
        max(cfg.neptune.rolling_window_sizes)
        if hasattr(cfg.neptune, "rolling_window_sizes")
        and cfg.neptune.rolling_window_sizes
        else None
    )

    for epoch in range(cfg.training.epochs):
        logger.info(f"Epoch {epoch+1}/{cfg.training.epochs}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        print("train loader length: ", len(train_loader))

        # Updated call to train_one_epoch with new parameters
        n_iter, running_avgs, losses_history = train_one_epoch(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            run=run,
            task=cfg.task,
            rolling_window_sizes=(
                cfg.neptune.rolling_window_sizes
                if hasattr(cfg.neptune, "rolling_window_sizes")
                else None
            ),
            smoothing_factors=(
                cfg.neptune.smoothing_factors
                if hasattr(cfg.neptune, "smoothing_factors")
                else None
            ),
            n_iter=n_iter,
            prev_running_avgs=running_avgs,
            prev_losses_history=losses_history,
            max_history_size=max_history_size,
        )

        if cfg.training.use_validation:
            val_loss = evaluate_one_epoch(
                val_loader, model, loss_fn, run, cfg.task, "val"
            )
            save_model(model, run, epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model, run, epoch, best=True)

        save_model(model, run, epoch)
    if cfg.training.use_validation:
        # load model with best validation loss
        model.load_state_dict(
            torch.load(f"models_checkpoints/{run['sys/id'].fetch()}/best.pth")
        )

    evaluate_one_epoch(test_loader, model, loss_fn, run, cfg.task, "test")

    delete_model_checkpoints(run)


def save_model(
    model: BaseModel, run: neptune.Run, epoch: int, best: bool = False
) -> None:
    path = Path(f"models_checkpoints/{run['sys/id'].fetch()}")
    path.mkdir(parents=True, exist_ok=True)
    if best:
        torch.save(model.state_dict(), path / "best.pth")
    else:
        torch.save(model.state_dict(), path / f"epoch-{epoch}.pth")


def delete_model_checkpoints(run: neptune.Run) -> None:
    path = Path(f"models_checkpoints/{run['sys/id'].fetch()}")
    if path.exists():
        for file in path.iterdir():
            file.unlink()
        path.rmdir()
