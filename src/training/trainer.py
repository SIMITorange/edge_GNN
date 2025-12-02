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
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, LinearLR, SequentialLR
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
        use_warmup: bool = False,
        warmup_epochs: int = 10,
        total_epochs: int = 200,
        scheduler_type: str = "cosine",
        min_lr: float = 1e-5,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.amp = amp
        self.grad_clip = grad_clip
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp) if device.type == "cuda" else torch.amp.GradScaler("cpu", enabled=False)
        
        # Build learning rate scheduler if requested
        if scheduler is None and use_warmup:
            if warmup_epochs > 0:
                warmup_scheduler = LinearLR(
                    optimizer, 
                    start_factor=0.1, 
                    total_iters=warmup_epochs
                )
                if scheduler_type == "cosine":
                    main_scheduler = CosineAnnealingLR(
                        optimizer,
                        T_max=total_epochs - warmup_epochs,
                        eta_min=min_lr,
                    )
                elif scheduler_type == "exponential":
                    main_scheduler = ExponentialLR(optimizer, gamma=0.95)
                else:
                    main_scheduler = LinearLR(optimizer, start_factor=1.0, total_iters=1)
                
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[warmup_epochs],
                )
            else:
                if scheduler_type == "cosine":
                    scheduler = CosineAnnealingLR(
                        optimizer,
                        T_max=total_epochs,
                        eta_min=min_lr,
                    )
                elif scheduler_type == "exponential":
                    scheduler = ExponentialLR(optimizer, gamma=0.95)
        
        self.scheduler = scheduler

    def _step_batch(self, batch, train: bool = True) -> Tuple[float, Dict[str, float]]:
        batch = batch.to(self.device)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp):
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
            return loss.item(), metrics

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_metrics: Dict[str, float] = {}
        batch_count = 0
        for batch in loader:
            loss, metrics = self._step_batch(batch, train=True)
            # Running mean of metrics.
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
            batch_count += 1
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

        print(f"{'Epoch':<6} {'Train Loss':<14} {'Val Loss':<14} {'Est. Time Remaining':<20}")
        print("-" * 60)

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.eval_epoch(val_loader, epoch, split="val")

            for k, v in train_metrics.items():
                history["train"].setdefault(k, []).append(v)
            for k, v in val_metrics.items():
                history["val"].setdefault(k, []).append(v)

            # Learning rate schedule step at epoch level (if using sequential/cosine scheduler)
            if self.scheduler is not None:
                self.scheduler.step()

            # Checkpointing.
            if val_metrics["loss/total"] < best_val:
                best_val = val_metrics["loss/total"]
                best_epoch = epoch
                self._save_checkpoint(checkpoint_path, epoch)

            if epoch % save_every == 0:
                self._save_checkpoint(checkpoint_path.replace(".ckpt", f"_epoch{epoch}.ckpt"), epoch)

            # Print detailed metrics every 10 epochs
            if epoch % 10 == 0 or epoch == 1:
                train_loss = train_metrics.get("loss/total", 0.0)
                val_loss = val_metrics.get("loss/total", 0.0)
                
                # Detailed per-target breakdown
                targets = list(set(k.split('/')[0] for k in train_metrics.keys() if '/' in k))
                targets = [t for t in targets if t != 'consistency' and t != 'loss']
                
                print(f"{epoch:<6} {train_loss:<14.6f} {val_loss:<14.6f} Best: epoch {best_epoch} (val={best_val:.6f})")
                
                # Print per-target losses
                for target in targets:
                    train_l1 = train_metrics.get(f"{target}/l1", 0.0)
                    val_l1 = val_metrics.get(f"{target}/l1", 0.0)
                    train_rel = train_metrics.get(f"{target}/rel", 0.0)
                    val_rel = val_metrics.get(f"{target}/rel", 0.0)
                    print(f"  {target:<30} Train L1: {train_l1:.6f}  Val L1: {val_l1:.6f}  "
                          f"Train RelL1: {train_rel:.6f}  Val RelL1: {val_rel:.6f}")
                
                # Print consistency metric if available
                if "consistency/grad" in train_metrics:
                    consistency = train_metrics.get("consistency/grad", 0.0)
                    print(f"  {'Consistency/Grad':<30} {consistency:.6f}")
                print()

            # Early stopping.
            if epoch - best_epoch > early_stop_patience:
                print(f"\n⏹️  Early stopping at epoch {epoch} (no val improvement for {early_stop_patience} epochs).")
                break

        print("\n" + "="*80)
        print(f"✓ Training complete!")
        print(f"  Best validation loss: {best_val:.6f} (epoch {best_epoch})")
        print(f"  Best checkpoint: {checkpoint_path}")
        print(f"  Metrics: {metrics_path}")
        print("="*80 + "\n")
        
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
