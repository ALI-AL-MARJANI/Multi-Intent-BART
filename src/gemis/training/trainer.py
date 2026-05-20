"""Training loop for GEMIS.

Uses a native PyTorch loop with AdamW and a linear warmup schedule.
Teacher forcing is applied during training; greedy decoding at evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from gemis.models.gemis import GEMISModel
from gemis.training.metrics import Frame, compute_metrics, parse_target_sequence  # noqa: F401

logger = logging.getLogger(__name__)


class GEMISTrainer:
    """Wraps GEMISModel with a complete train / eval loop.

    Args:
        model:          instantiated GEMISModel.
        train_loader:   DataLoader for the training set.
        dev_loader:     DataLoader for the development set.
        learning_rate:  AdamW learning rate.
        num_epochs:     total training epochs.
        warmup_steps:   linear warmup steps for the scheduler.
        output_dir:     where to save checkpoints.
        device:         torch device string (e.g. "cuda" or "cpu").
        gradient_clip:  max grad norm for clipping.
    """

    def __init__(
        self,
        model: GEMISModel,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        learning_rate: float = 2e-5,
        num_epochs: int = 30,
        warmup_steps: int = 200,
        output_dir: str = "checkpoints",
        device: str = "cuda",
        gradient_clip: float = 1.0,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.num_epochs = num_epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gradient_clip = gradient_clip
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)

        self.model.to(self.device)

        # scheduler counts optimizer steps (after accumulation), not raw batches
        total_optimizer_steps = (len(train_loader) // self.gradient_accumulation_steps) * num_epochs
        self.optimizer = AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-2)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
        )

        self.best_overall_acc: float = -1.0

    # ── public API ─────────────────────────────────────────────────────────────

    def train(self) -> None:
        for epoch in range(1, self.num_epochs + 1):
            train_loss = self._train_epoch(epoch)
            metrics = self._evaluate()

            logger.info(
                "Epoch %d/%d — loss: %.4f | slot_f1: %.2f | intent_acc: %.2f | overall_acc: %.2f",
                epoch, self.num_epochs, train_loss,
                metrics["slot_f1"], metrics["intent_accuracy"], metrics["overall_accuracy"],
            )

            if metrics["overall_accuracy"] > self.best_overall_acc:
                self.best_overall_acc = metrics["overall_accuracy"]
                self._save_checkpoint("best.pt")
                logger.info("  ↑ New best overall accuracy: %.2f", self.best_overall_acc)

        self._save_checkpoint("last.pt")

    # ── internal ───────────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        accum = self.gradient_accumulation_steps

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}", leave=False)
        for step, batch in enumerate(pbar):
            batch.pop("words", None)  # list[list[str]], not a tensor
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
                labels=batch["labels"],
                pointer_targets=batch["pointer_targets"],
            )
            # scale loss so gradients are equivalent to a full-batch mean
            loss: torch.Tensor = outputs["loss"] / accum
            loss.backward()

            total_loss += outputs["loss"].item()
            n_batches += 1

            if (step + 1) % accum == 0 or (step + 1) == len(self.train_loader):
                if self.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

        return total_loss / n_batches if n_batches else 0.0

    def _evaluate(self) -> dict[str, float]:
        self.model.eval()
        golds: list[Frame] = []
        preds: list[Frame] = []

        tokenizer = self.model.tokenizer

        with torch.no_grad():
            for batch in self.dev_loader:
                words_batch = batch.pop("words", None)  # list[list[str]], not a tensor
                batch = {k: v.to(self.device) for k, v in batch.items()}

                generated = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    input_words_batch=words_batch,
                )

                for i, gen_ids in enumerate(generated):
                    gold_ids = [t for t in batch["labels"][i].tolist() if t != -100]
                    golds.append(parse_target_sequence(gold_ids, tokenizer))
                    preds.append(parse_target_sequence(gen_ids, tokenizer))

        return compute_metrics(golds, preds)

    def _save_checkpoint(self, filename: str) -> None:
        path = self.output_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        logger.info("Checkpoint saved → %s", path)
