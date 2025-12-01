"""
Training orchestration for edge_GNN.

Inputs:
    - model: EdgeGNN instance.
    - loss_fn: CompositeLoss.
    - dataloaders: PyG DataLoaders for train/val/test splits.
Outputs:
    - Checkpoints (.ckpt), normalization state, and metrics pickle for visualization.
Purpose:
    Provides a clean training loop with AMP, early stopping, gradient clipping, and logging.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader


class Trainer:
    """Wraps training, evaluation, and checkpointing."""

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        log_dir: str,
        amp: bool = True,
        grad_clip: float = 5.0,
        scheduler=None,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.amp = amp
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp)

    def _step_batch(self, batch, train: bool = True) -> Tuple[float, Dict[str, float]]:
        batch = batch.to(self.device)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=self.amp):
                pred_dict, stacked = self.model(batch)
                loss, metrics = self.loss_fn(pred_dict, batch.y, batch)
            if train:
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler:
                    self.scheduler.step()
            return loss.item(), metrics

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_metrics: Dict[str, float] = {}
        for batch in loader:
            loss, metrics = self._step_batch(batch, train=True)
            # Running mean of metrics.
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
        n = max(1, len(loader))
        for k in epoch_metrics:
            epoch_metrics[k] /= n

        # TensorBoard logging.
        for k, v in epoch_metrics.items():
            self.writer.add_scalar(f"train/{k}", v, epoch)
        return epoch_metrics

    def eval_epoch(self, loader: DataLoader, epoch: int, split: str = "val") -> Dict[str, float]:
        self.model.eval()
        epoch_metrics: Dict[str, float] = {}
        with torch.no_grad():
            for batch in loader:
                loss, metrics = self._step_batch(batch, train=False)
                for k, v in metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
        n = max(1, len(loader))
        for k in epoch_metrics:
            epoch_metrics[k] /= n
        for k, v in epoch_metrics.items():
            self.writer.add_scalar(f"{split}/{k}", v, epoch)
        return epoch_metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        checkpoint_path: str,
        save_every: int = 10,
        early_stop_patience: int = 30,
        metrics_path: str = "artifacts/metrics.pkl",
    ):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        best_val = float("inf")
        best_epoch = 0
        history: Dict[str, Dict[str, list]] = {"train": {}, "val": {}}

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.eval_epoch(val_loader, epoch, split="val")

            for k, v in train_metrics.items():
                history["train"].setdefault(k, []).append(v)
            for k, v in val_metrics.items():
                history["val"].setdefault(k, []).append(v)

            # Checkpointing.
            if val_metrics["loss/total"] < best_val:
                best_val = val_metrics["loss/total"]
                best_epoch = epoch
                self._save_checkpoint(checkpoint_path, epoch)

            if epoch % save_every == 0:
                self._save_checkpoint(checkpoint_path.replace(".ckpt", f"_epoch{epoch}.ckpt"), epoch)

            # Early stopping.
            if epoch - best_epoch > early_stop_patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {early_stop_patience} epochs).")
                break

        with open(metrics_path, "wb") as f:
            pickle.dump(history, f)
        return history

    def _save_checkpoint(self, path: str, epoch: int):
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epoch": epoch,
            },
            path,
        )
        print(f"[Checkpoint] Saved to {path}")

    @torch.no_grad()
    def predict(self, loader: DataLoader):
        """Run inference over a loader; returns list of stacked predictions and ground truth."""

        self.model.eval()
        preds, targets = [], []
        for batch in loader:
            batch = batch.to(self.device)
            pred_dict, stacked = self.model(batch)
            preds.append(stacked.cpu())
            targets.append(batch.y.cpu())
        return preds, targets
