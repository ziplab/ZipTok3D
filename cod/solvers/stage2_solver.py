import functools
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from pointops.functions import pointops
from cod.data.base import BaseDataModule
from cod.losses.recon import OccupancyReconstructionLoss
from cod.metrics.occupancy import Accuracy, IoU
from cod.models.vae.base import BaseAutoencoder
from cod.utils.recon import chunked_reconstruct
from cod.utils.sched import get_lr
from cod.utils.training import compute_effective_lr, load_model_weights_from_checkpoint
from cod.utils.vis import points_to_img
from .base import BaseSolver


class VAEStage2Solver(BaseSolver):
    """Causal prefix-VAE used by ZipTok3D's stage-2 generator."""

    def __init__(
        self,
        dm: BaseDataModule,
        model: BaseAutoencoder,
        autoencoder_checkpoint_path: str,
        lr: float = 1e-4,
        scale_lr_by_batch_size: bool = True,
        weight_decay: float = 0.01,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
        logit_threshold: float = 0,
        eval_chunk_size: int = 100000,
        val_vis_interval: int = 10,
        coeff_kl: float = 1e-3,
        coeff_feat: float = 1.0,
        coeff_recon: float = 1.0,
        prefix_budgets: Sequence[int] = (1, 2, 4, 8, 16),
        prefix_loss_weights: Sequence[float] = (1, 1, 1, 1, 1),
        num_decode_loops: int = 6,
    ):
        super().__init__(track_max_score=True)
        if len(prefix_budgets) != len(prefix_loss_weights):
            raise ValueError("prefix_budgets and prefix_loss_weights must have equal length")
        self.lr = lr
        self.scale_lr_by_batch_size = bool(scale_lr_by_batch_size)
        self.weight_decay = float(weight_decay)
        self.adam_betas = (float(adam_beta1), float(adam_beta2))
        self.adam_eps = float(adam_eps)
        self.eval_chunk_size = eval_chunk_size
        self.val_vis_interval = val_vis_interval
        self.coeff_kl = coeff_kl
        self.coeff_feat = coeff_feat
        self.coeff_recon = coeff_recon
        self.prefix_budgets = [int(x) for x in prefix_budgets]
        self.prefix_loss_weights = [float(x) for x in prefix_loss_weights]
        self.num_decode_loops = int(num_decode_loops)
        self.model = model
        self.dm = dm
        self.logit_threshold = logit_threshold

        self.targets_norm = nn.LayerNorm(model.embed_dim, elementwise_affine=False)
        self.criterion = OccupancyReconstructionLoss(vol_coeff=1.0, near_coeff=0.1)
        self.accuracy_metric = Accuracy()
        self.iou_metric = IoU()
        self.make_points_img = functools.partial(
            points_to_img, zdir="y", view_angle=(30, -45), point_size=3
        )
        self.create_image_log_buffer("recon_gt", log_once=True)
        self.create_image_log_buffer("recon")

        state_dict = load_model_weights_from_checkpoint(
            autoencoder_checkpoint_path, prefix="model"
        )
        self.model.load_autoencoder_weights(state_dict)

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
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[60, 70, 80, 90], gamma=0.5
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        pc = batch["surface"]
        query_points = batch["query_points"]
        labels = batch["labels"]
        num_vol_points = batch["num_vol_points"][0].item()

        with torch.no_grad():
            encoded = self.model.encode_embed(pc)
        latent, posterior = self.model.encode_latents(
            encoded, active_num_latents=max(self.prefix_budgets)
        )
        reconstructed, _ = self.model.decode_latents(latent)
        targets = self.targets_norm(encoded[:, :reconstructed.size(1)]).detach()

        total_weight = sum(self.prefix_loss_weights)
        feature_loss = reconstructed.new_zeros(())
        recon_loss = reconstructed.new_zeros(())
        for budget, weight in zip(self.prefix_budgets, self.prefix_loss_weights):
            if budget > reconstructed.size(1):
                continue
            prefix = reconstructed[:, :budget]
            prefix_mask = torch.zeros(
                prefix.shape[:2], dtype=torch.bool, device=prefix.device
            )
            feat_k = F.mse_loss(prefix, targets[:, :budget])
            planes = self.model.decode_embed(
                prefix,
                mask=prefix_mask,
                num_decode_loops=self.num_decode_loops,
            )[0]
            logits = self.model.decode_queries(planes, query_points)
            recon_k = self.criterion(
                logits, labels, num_vol_points=num_vol_points
            )[0]
            feature_loss = feature_loss + weight * feat_k / total_weight
            recon_loss = recon_loss + weight * recon_k / total_weight
            self.log(f"train/feat_k{budget}", feat_k, rank_zero_only=True)
            self.log(f"train/recon_k{budget}", recon_k, rank_zero_only=True)

        kl_loss = posterior.kl().mean()
        loss = (
            self.coeff_feat * feature_loss
            + self.coeff_recon * recon_loss
            + self.coeff_kl * kl_loss
        )
        self.log("train/feat_loss", feature_loss, rank_zero_only=True)
        self.log("train/recon_loss", recon_loss, rank_zero_only=True)
        self.log("train/kl_loss", kl_loss, rank_zero_only=True)
        self.log("train/loss", loss, rank_zero_only=True)
        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
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
            active_num_latents=self.model.active_num_latents,
            num_decode_loops=self.num_decode_loops,
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
