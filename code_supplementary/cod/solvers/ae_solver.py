import functools
import json
import os
from datetime import datetime

import torch
from torch.nn import functional as F

from pointops.functions import pointops
from cod.data.base import BaseDataModule
from cod.losses.recon import OccupancyReconstructionLoss, UncertaintyLoss
from cod.metrics.occupancy import Accuracy, IoU
from cod.models.vae.base import BaseAutoencoder
from cod.utils.recon import chunked_reconstruct
from cod.utils.sched import CosineAnnealingLR, get_lr
from cod.utils.training import compute_effective_lr
from cod.utils.vis import points_to_img
from .base import BaseSolver


class AutoencoderSolver(BaseSolver):
    """Stage-1 ZipTok3D tokenizer solver."""

    def __init__(
        self,
        dm: BaseDataModule,
        model: BaseAutoencoder,
        lr: float = 1e-4,
        scale_lr_by_batch_size: bool = True,
        weight_decay: float = 0.01,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
        use_cosine_annealing: bool = False,
        warmup_epochs: int = 5,
        logit_threshold: float = 0,
        eval_chunk_size: int = 100000,
        val_vis_interval: int = 10,
        coeff_uncertainty: float = 0,
        coeff_init: float = 1,
        coeff_recon: float = 1,
        elt_distill: bool = False,
        elt_min_loop: int = 1,
        elt_max_loop: int = 1,
        elt_distill_weight: float = 0.25,
        elt_distill_loss: str = "bce",
        elt_distill_weight_step: float = 0.0,
        elt_distill_weight_max: float = 0.25,
        elt_schedule_interval: int = 5,
        elt_plane_warmup_epochs: int = 10,
        elt_plane_distill_weight: float = 0.0,
        elt_plane_distill_weight_step: float = 0.0,
        elt_plane_distill_weight_max: float = 0.0,
        elt_loop_warmup_start_min_loop: int = 1,
        max_logit_abs: float = 30.0,
        metrics_jsonl_path: str = None,
        metrics_log_interval: int = 50,
    ):
        super().__init__(track_max_score=True)
        self.lr = lr
        self.scale_lr_by_batch_size = bool(scale_lr_by_batch_size)
        self.weight_decay = float(weight_decay)
        self.adam_betas = (float(adam_beta1), float(adam_beta2))
        self.adam_eps = float(adam_eps)
        self.use_cosine_annealing = use_cosine_annealing
        self.warmup_epochs = warmup_epochs
        self.scheduler = None
        self.eval_chunk_size = eval_chunk_size
        self.val_vis_interval = val_vis_interval
        self.coeff_uncertainty = coeff_uncertainty
        self.coeff_init = coeff_init
        self.coeff_recon = coeff_recon

        self.elt_distill = elt_distill
        self.elt_min_loop = int(elt_min_loop)
        self.elt_max_loop = int(elt_max_loop)
        self.elt_distill_weight = elt_distill_weight
        self.elt_distill_loss = elt_distill_loss
        self.elt_distill_weight_step = elt_distill_weight_step
        self.elt_distill_weight_max = elt_distill_weight_max
        self.elt_schedule_interval = max(1, int(elt_schedule_interval))
        self.elt_plane_warmup_epochs = int(elt_plane_warmup_epochs)
        self.elt_plane_distill_weight = elt_plane_distill_weight
        self.elt_plane_distill_weight_step = elt_plane_distill_weight_step
        self.elt_plane_distill_weight_max = elt_plane_distill_weight_max
        self.elt_loop_warmup_start_min_loop = int(elt_loop_warmup_start_min_loop)
        self.max_logit_abs = max_logit_abs
        self.metrics_jsonl_path = metrics_jsonl_path
        self.metrics_log_interval = max(1, int(metrics_log_interval))

        self.model = model
        self.dm = dm
        self.logit_threshold = logit_threshold
        self.criterion = OccupancyReconstructionLoss(vol_coeff=1.0, near_coeff=0.1)
        self.uncertainty_criterion = UncertaintyLoss()
        self.accuracy_metric = Accuracy()
        self.iou_metric = IoU()

        self.make_points_img = functools.partial(
            points_to_img, zdir="y", view_angle=(30, -45), point_size=3
        )
        self.create_image_log_buffer("recon_gt", log_once=True)
        self.create_image_log_buffer("recon")

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
        if self.use_cosine_annealing:
            self.scheduler = CosineAnnealingLR(
                optimizer,
                warmup_epochs=self.warmup_epochs,
                total_epochs=self.trainer.max_epochs,
                lr=lr,
                min_lr=1e-6,
            )
        return [optimizer], []

    def _scheduled_values(self):
        stage = self.current_epoch // self.elt_schedule_interval
        distill_weight = min(
            self.elt_distill_weight + stage * self.elt_distill_weight_step,
            self.elt_distill_weight_max,
        )
        current_min_loop = max(
            self.elt_min_loop,
            self.elt_loop_warmup_start_min_loop - stage,
        )
        if self.current_epoch < self.elt_plane_warmup_epochs:
            plane_weight = 0.0
        else:
            plane_stage = (
                self.current_epoch - self.elt_plane_warmup_epochs
            ) // self.elt_schedule_interval
            plane_weight = min(
                self.elt_plane_distill_weight
                + plane_stage * self.elt_plane_distill_weight_step,
                self.elt_plane_distill_weight_max,
            )
        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        alpha = max(0.0, 1.0 - float(self.global_step) / total_steps)
        return current_min_loop, distill_weight, plane_weight, alpha

    def _sample_student_loop(self, min_loop):
        high = min(self.elt_max_loop, self.model.decoder.num_loops)
        if high <= 1:
            return None
        low = min(max(1, min_loop), high - 1)
        return int(torch.randint(low, high, (1,), device=self.device).item())

    def training_step(self, batch, batch_idx):
        min_loop, distill_weight, plane_weight, alpha = self._scheduled_values()
        student_loop = self._sample_student_loop(min_loop) if self.elt_distill else None

        pc = batch["surface"]
        query_points = batch["query_points"]
        labels = batch["labels"]
        outputs = self.model(
            pc,
            query_points,
            num_decode_loops=self.elt_max_loop,
            student_loop=student_loop,
        )
        num_vol_points = batch["num_vol_points"][0].item()

        recon_loss, vol_loss, near_loss = self.criterion(
            outputs["logits"], labels, num_vol_points=num_vol_points
        )
        init_loss = self.criterion(
            outputs["init_out"], labels, num_vol_points=num_vol_points
        )[0]
        uncertainty_loss = self.uncertainty_criterion(
            outputs["uncertainty"], outputs["init_out"], labels
        )
        loss = (
            self.coeff_recon * recon_loss
            + self.coeff_init * init_loss
            + self.coeff_uncertainty * uncertainty_loss
        )

        student_gt_loss = None
        distillation_loss = None
        elt_loss = None
        plane_loss = None
        if student_loop is not None:
            student_logits = outputs["student_logits"].clamp(
                -self.max_logit_abs, self.max_logit_abs
            )
            teacher_logits = outputs["logits"].detach().clamp(
                -self.max_logit_abs, self.max_logit_abs
            )
            student_gt_loss = self.criterion(
                student_logits, labels, num_vol_points=num_vol_points
            )[0]
            if self.elt_distill_loss == "bce":
                # Match L_rec's volume/near-surface weighting while using the
                # detached final prediction as a soft target.
                distillation_loss = self.criterion(
                    student_logits,
                    torch.sigmoid(teacher_logits),
                    num_vol_points=num_vol_points,
                )[0]
            elif self.elt_distill_loss == "mse":
                distillation_loss = F.mse_loss(student_logits, teacher_logits)
            else:
                raise ValueError(f"unknown ELT loss: {self.elt_distill_loss}")
            elt_loss = alpha * student_gt_loss + (1.0 - alpha) * distillation_loss
            loss = loss + distill_weight * elt_loss

            if plane_weight > 0:
                plane_loss = F.mse_loss(
                    outputs["student_planes"], outputs["planes"].detach()
                )
                loss = loss + plane_weight * plane_loss

        nonfinite = float(not torch.isfinite(loss).item())
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        metrics = {
            "loss": loss,
            "recon_loss": recon_loss,
            "vol_loss": vol_loss,
            "near_loss": near_loss,
            "init_loss": init_loss,
            "uncertainty_loss": uncertainty_loss,
            "elt_student_loop": student_loop,
            "elt_min_loop": min_loop,
            "elt_lambda": alpha,
            "elt_distill_weight": distill_weight,
            "elt_plane_distill_weight": plane_weight,
            "elt_student_gt_loss": student_gt_loss,
            "elt_distill_loss": distillation_loss,
            "elt_loss": elt_loss,
            "elt_plane_distill_loss": plane_loss,
            "nonfinite_loss": nonfinite,
        }
        for name, value in metrics.items():
            if value is not None:
                self.log(
                    f"train/{name}", value if torch.is_tensor(value) else float(value),
                    rank_zero_only=True,
                )
        self._write_metrics(metrics, batch_idx)
        return {"loss": loss}

    def _write_metrics(self, metrics, batch_idx):
        if not self.metrics_jsonl_path or self.global_rank != 0:
            return
        if self.global_step % self.metrics_log_interval != 0:
            return
        record = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": int(self.current_epoch),
            "batch_idx": int(batch_idx),
            "global_step": int(self.global_step),
            "coeff_init": self.coeff_init,
            "coeff_recon": self.coeff_recon,
        }
        for name, value in metrics.items():
            if value is None:
                record[name] = None
            elif torch.is_tensor(value):
                record[name] = float(value.detach().cpu())
            else:
                record[name] = float(value)
        directory = os.path.dirname(self.metrics_jsonl_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.metrics_jsonl_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.scheduler is not None:
            epoch = (
                self.trainer.global_step
                * self.trainer.accumulate_grad_batches
                / len(self.trainer.train_dataloader)
            )
            self.scheduler.step(epoch)
        self.log(
            "train/lr", get_lr(self.trainer.optimizers),
            on_step=True, on_epoch=False, rank_zero_only=True,
        )

    def validation_step(self, batch, batch_idx):
        pc = batch["surface"]
        query_points = batch["query_points"]
        labels = batch["labels"]
        chunk_size = self.eval_chunk_size if self.eval_chunk_size > 0 else query_points.size(1)
        if self.is_debug:
            chunk_size = 10000
            query_points = query_points[:, :chunk_size * 2]
            labels = labels[:, :chunk_size * 2]

        logits, preds = chunked_reconstruct(
            self.model,
            pc,
            query_points,
            chunk_size,
            self.logit_threshold,
            num_decode_loops=self.elt_max_loop,
        )
        batch_size = pc.size(0)
        for b in range(batch_size):
            idx = batch_idx * batch_size + b
            if self.val_vis_interval > 0 and idx % self.val_vis_interval == 0:
                pred_points = query_points[b, preds[b].bool()]
                if pred_points.size(0) > 2048:
                    pred_points = pointops.fps(pred_points.unsqueeze(0).float(), 2048)[0]
                self.add_log_buffer_items("recon", self.make_points_img(pred_points))
                self.add_log_buffer_items("recon_gt", self.make_points_img(pc[b]))
        self.accuracy_metric.update(preds, labels)
        self.iou_metric.update(preds, labels)

    def on_validation_epoch_end(self):
        self.log_buffered()
        accuracy = self.accuracy_metric.compute()
        iou = self.iou_metric.compute()
        self.log("val/accuracy", accuracy, on_epoch=True, rank_zero_only=True, sync_dist=True)
        self.log("val/iou", iou, on_epoch=True, rank_zero_only=True, sync_dist=True)
        self.track_score(iou)
        print(f"accuracy: {accuracy:.1f}%, iou: {iou:.1f}%")
