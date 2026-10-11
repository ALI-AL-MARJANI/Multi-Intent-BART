"""Training loop: AdamW, linear warmup/decay, early stopping on dev overall accuracy."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from gemis.models.gemis import GEMISModel
from gemis.training.evaluation import predict_frames
from gemis.training.metrics import compute_metrics

logger = logging.getLogger(__name__)


class GEMISTrainer:
    """Train with teacher forcing, evaluate with greedy decoding after each epoch.

    ``best.pt`` holds the epoch with the highest dev overall accuracy;
    ``last.pt`` is written after every epoch and can be resumed from.
    """

    def __init__(
        self,
        model: GEMISModel,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        output_dir: str | Path,
        device: torch.device,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        max_epochs: int = 30,
        warmup_ratio: float = 0.01,
        gradient_clip: float = 1.0,
        gradient_accumulation_steps: int = 1,
        early_stopping_patience: int = 0,
        mixed_precision: bool = False,
        max_new_tokens: int = 96,
        run_config: dict[str, Any] | None = None,
        show_progress: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.max_epochs = max_epochs
        self.gradient_clip = gradient_clip
        self.accumulation = max(1, gradient_accumulation_steps)
        self.patience = early_stopping_patience
        self.max_new_tokens = max_new_tokens
        self.run_config = run_config or {}
        self.show_progress = show_progress

        self.use_amp = mixed_precision and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        steps_per_epoch = -(-len(train_loader) // self.accumulation)
        total_steps = steps_per_epoch * max_epochs
        self.optimizer = AdamW(model.parameters(), lr=float(learning_rate), weight_decay=weight_decay)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )

        self.epoch = 0
        self.best_metric = -1.0
        self.best_epoch = 0
        self.history: list[dict[str, Any]] = []

    # ── public API ───────────────────────────────────────────────────────────

    def train(self) -> list[dict[str, Any]]:
        """Run until ``max_epochs`` or early stopping. Returns the per-epoch log."""
        while self.epoch < self.max_epochs:
            self.epoch += 1
            started = time.time()
            train_loss = self._train_epoch()
            golds, preds, _ = predict_frames(self.model, self.dev_loader, self.device, self.max_new_tokens)
            metrics = compute_metrics(golds, preds)
            record = {
                "epoch": self.epoch,
                "train_loss": train_loss,
                **{f"dev_{k}": v for k, v in metrics.items()},
                "seconds": round(time.time() - started, 1),
            }
            self.history.append(record)
            logger.info(
                "epoch %d/%d  loss %.4f  dev slot_f1 %.2f  intent_acc %.2f  overall_acc %.2f  (%.0fs)",
                self.epoch,
                self.max_epochs,
                train_loss,
                metrics["slot_f1"],
                metrics["intent_accuracy"],
                metrics["overall_accuracy"],
                record["seconds"],
            )

            if metrics["overall_accuracy"] > self.best_metric:
                self.best_metric = metrics["overall_accuracy"]
                self.best_epoch = self.epoch
                self._save("best.pt", with_optimizer=False)
            self._save("last.pt", with_optimizer=True)

            if self.patience and self.epoch - self.best_epoch >= self.patience:
                logger.info("early stopping: no dev improvement for %d epochs", self.patience)
                break
        return self.history

    def resume(self, path: str | Path) -> None:
        """Restore model, optimizer, scheduler and counters from ``last.pt``."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.epoch = state["epoch"]
        self.best_metric = state["best_metric"]
        self.best_epoch = state["best_epoch"]
        self.history = state["history"]
        logger.info("resumed from %s at epoch %d", path, self.epoch)

    # ── internals ────────────────────────────────────────────────────────────

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss, n_batches = 0.0, 0
        self.optimizer.zero_grad()
        progress = tqdm(
            self.train_loader,
            desc=f"epoch {self.epoch}/{self.max_epochs}",
            leave=False,
            disable=not self.show_progress,
        )
        for step, batch in enumerate(progress, start=1):
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                loss = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    decoder_input_ids=batch["decoder_input_ids"].to(self.device),
                    labels=batch["labels"].to(self.device),
                )["loss"]
            self.scaler.scale(loss / self.accumulation).backward()
            total_loss += loss.item()
            n_batches += 1

            if step % self.accumulation == 0 or step == len(self.train_loader):
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                progress.set_postfix(loss=f"{total_loss / n_batches:.4f}")
        return total_loss / max(n_batches, 1)

    def _save(self, filename: str, with_optimizer: bool) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": self.model.export_config(),
                "optimizer_state_dict": self.optimizer.state_dict() if with_optimizer else None,
                "scheduler_state_dict": self.scheduler.state_dict() if with_optimizer else None,
                "epoch": self.epoch,
                "best_metric": self.best_metric,
                "best_epoch": self.best_epoch,
                "history": self.history,
                "run_config": self.run_config,
            },
            self.output_dir / filename,
        )
