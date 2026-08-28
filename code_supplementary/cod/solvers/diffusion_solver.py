import json
import math
import os
from datetime import datetime

import numpy as np
import torch

from cod.data.base import BaseDataModule
from cod.utils.sched import get_lr
from cod.utils.training import compute_effective_lr
from .base import BaseSolver


class LatentNormalizer(torch.nn.Module):
    def __init__(self, normalizer_path):
        super().__init__()
        values = np.load(normalizer_path)
        mean = torch.from_numpy(values["mean"]).float()
        std = torch.from_numpy(values["std"]).float().clamp_min_(1e-6)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def normalize(self, latents):
        return (latents - self.mean) / self.std

    def denormalize(self, latents):
        return latents * self.std + self.mean


class LatentDiffusionSolver(BaseSolver):
    def __init__(
        self,
        dm: BaseDataModule,
        model,
        normalizer_path: str,
        lr: float = 1e-4,
        scale_lr_by_batch_size: bool = True,
        min_lr: float = 1e-6,
        weight_decay: float = 0.05,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
        warmup_epochs: int = 40,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sigma_data: float = 1.0,
        validation_seed: int = 2027,
        metrics_jsonl_path: str = None,
        metrics_log_interval: int = 20,
    ):
        super().__init__(track_max_score=False)
        self.dm = dm
        self.model = model
        self.normalizer = LatentNormalizer(normalizer_path)
        self.lr = lr
        self.scale_lr_by_batch_size = bool(scale_lr_by_batch_size)
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.adam_betas = (float(adam_beta1), float(adam_beta2))
        self.adam_eps = float(adam_eps)
        self.warmup_epochs = warmup_epochs
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_data = sigma_data
        self.validation_seed = validation_seed
        self.metrics_jsonl_path = metrics_jsonl_path
        self.metrics_log_interval = max(1, int(metrics_log_interval))
        self._val_losses = []

    def configure_optimizers(self):
        lr = self.lr
        if self.scale_lr_by_batch_size:
            lr = compute_effective_lr(
                self.lr, self.dm.batch_size, self.get_trainer_if_exists()
            )
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=lr,
            betas=self.adam_betas,
            eps=self.adam_eps,
            weight_decay=self.weight_decay,
        )

        def schedule(epoch):
            if epoch < self.warmup_epochs:
                return max(1e-8, (epoch + 1) / max(1, self.warmup_epochs))
            progress = (epoch - self.warmup_epochs) / max(
                1, self.trainer.max_epochs - self.warmup_epochs
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return self.min_lr / lr + (1.0 - self.min_lr / lr) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def _edm_loss(self, latents, labels, generator=None):
        expected = (self.model.num_latents, self.model.channels)
        if latents.ndim != 3 or tuple(latents.shape[1:]) != expected:
            raise ValueError(
                f"EDM training expects full [B,{expected[0]},{expected[1]}] "
                f"causal-prefix arrays, got {tuple(latents.shape)}"
            )
        shape = (latents.size(0), 1, 1)
        normal = torch.randn(shape, device=latents.device, generator=generator)
        sigma = (normal * self.p_std + self.p_mean).exp()
        noise = torch.randn(
            latents.shape, device=latents.device, dtype=latents.dtype, generator=generator
        ) * sigma
        weight = (sigma.square() + self.sigma_data ** 2) / (
            sigma * self.sigma_data
        ).square()
        denoised = self.model(latents + noise, sigma, labels)
        return (weight * (denoised - latents).square()).mean()

    def training_step(self, batch, batch_idx):
        latents = self.normalizer.normalize(batch["latents"])
        loss = self._edm_loss(latents, batch["category_ids"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        self._write_metric(loss, batch_idx)
        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.log(
            "train/lr", get_lr(self.trainer.optimizers),
            on_step=True, on_epoch=False, rank_zero_only=True,
        )

    def on_validation_epoch_start(self):
        self._val_losses = []

    def validation_step(self, batch, batch_idx):
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.validation_seed + batch_idx)
        latents = self.normalizer.normalize(batch["latents"])
        loss = self._edm_loss(latents, batch["category_ids"], generator=generator)
        self._val_losses.append(loss.detach())
        self.log("val/loss", loss, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        if self._val_losses:
            loss = torch.stack(self._val_losses).mean()
            self.track_score(loss)

    def _write_metric(self, loss, batch_idx):
        if not self.metrics_jsonl_path or self.global_rank != 0:
            return
        if self.global_step % self.metrics_log_interval != 0:
            return
        directory = os.path.dirname(self.metrics_jsonl_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        record = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": int(self.current_epoch),
            "batch_idx": int(batch_idx),
            "global_step": int(self.global_step),
            "loss": float(loss.detach().cpu()),
        }
        with open(self.metrics_jsonl_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
