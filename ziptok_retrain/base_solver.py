"""COD-VAE losses plus optional progressive intra-loop occupancy self-distillation.

Adds configurable loss/optimizer coefficients and fails on non-finite loss;
the original reconstruction, initialization and uncertainty objectives remain.
Optional distillation anchors keep alpha continuous across a resumed batch-size change.
"""

import functools
from typing import Optional, Tuple

import torch

from pointops.functions import pointops
from cod.data.base import BaseDataModule
from cod.models.vae.base import BaseAutoencoder
from cod.solvers.base import BaseSolver
from cod.utils.vis import points_to_img
from cod.metrics.occupancy import Accuracy, IoU
from cod.losses.recon import OccupancyReconstructionLoss, UncertaintyLoss
from cod.losses.distillation import progressive_alpha, intermediate_losses
from cod.utils.sched import CosineAnnealingLR, get_lr
from cod.utils.training import compute_effective_lr


class AutoencoderSolver(BaseSolver):

    def __init__(self,
                 dm: BaseDataModule,
                 model: BaseAutoencoder,
                 lr: float = 1e-4,
                 use_cosine_annealing: bool = False,
                 warmup_epochs: int = 5,
                 logit_threshold: float = 0,

                 eval_chunk_size: int = 100000,
                 val_vis_interval: int = 10,
                 coeff_uncertainty: float = 0,
                 coeff_init: float = 1,
                 coeff_recon: float = 1,
                 scale_lr_by_batch_size: bool = False,
                 weight_decay: float = 0.01,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eval_num_latents: Optional[int] = None,
                 min_lr: float = 1e-6,
                 coeff_intermediate: float = 0.0,
                 distill_schedule_steps: Optional[int] = None,
                 distill_anchor_step: int = 0,
                 distill_anchor_alpha: float = 1.0,
                 ):
        super().__init__(track_max_score=True)

        if lr <= 0 or min_lr < 0 or weight_decay < 0:
            raise ValueError('lr must be positive; min_lr and weight_decay must be nonnegative')
        if warmup_epochs < 0:
            raise ValueError('warmup_epochs must be nonnegative')
        if len(betas) != 2 or any(not 0 <= beta < 1 for beta in betas):
            raise ValueError('betas must contain two values in [0, 1)')
        if any(value < 0 for value in (coeff_recon, coeff_init, coeff_uncertainty, coeff_intermediate)):
            raise ValueError('loss coefficients must be nonnegative')
        if distill_schedule_steps is not None and (isinstance(distill_schedule_steps, bool)
                or not isinstance(distill_schedule_steps, int) or distill_schedule_steps <= 0):
            raise ValueError('distill_schedule_steps must be a positive integer or null')
        if coeff_intermediate > 0 and getattr(model.decoder, 'num_loops', 1) < 2:
            raise ValueError('intermediate supervision requires decoder num_loops >= 2')
        progressive_alpha(distill_anchor_step, 1, anchor_step=distill_anchor_step,
                          anchor_alpha=distill_anchor_alpha)  # Validate without needing a Trainer.
        self.coeff_intermediate = coeff_intermediate
        self.distill_schedule_steps = distill_schedule_steps
        self.distill_anchor_step = distill_anchor_step
        self.distill_anchor_alpha = float(distill_anchor_alpha)
        if eval_num_latents is not None:
            if isinstance(eval_num_latents, bool) or not isinstance(eval_num_latents, int):
                raise ValueError('eval_num_latents must be an integer or null')
            if not 1 <= eval_num_latents <= model.num_latents:
                raise ValueError('eval_num_latents must be in [1, model.num_latents]')

        self.lr = lr
        self.use_cosine_annealing = use_cosine_annealing
        self.warmup_epochs = warmup_epochs
        self.scheduler = None
        self.eval_chunk_size = eval_chunk_size
        self.val_vis_interval = val_vis_interval
        self.coeff_uncertainty = coeff_uncertainty
        self.coeff_init = coeff_init
        self.coeff_recon = coeff_recon
        self.scale_lr_by_batch_size = scale_lr_by_batch_size
        self.weight_decay = weight_decay
        self.betas = tuple(betas)
        self.eval_num_latents = eval_num_latents
        self.min_lr = min_lr

        self.model = model
        self.dm = dm
        self.logit_threshold = logit_threshold

        self.criterion = OccupancyReconstructionLoss(vol_coeff=1.0, near_coeff=0.1)
        self.uncertainty_criterion = UncertaintyLoss()
        self.accuracy_metric = Accuracy()
        self.iou_metric = IoU()

        self.make_points_img = functools.partial(points_to_img, zdir='y', view_angle=(30, -45), point_size=3)
        self.create_image_log_buffer('recon_gt', log_once=True)
        self.create_image_log_buffer('recon')
        self.create_image_log_buffer('uncertainty')

    def configure_optimizers(self):
        lr = self.lr
        if self.scale_lr_by_batch_size:
            lr = compute_effective_lr(lr, self.dm.batch_size, self.get_trainer_if_exists())
        params = [x for x in self.parameters() if x.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=self.weight_decay, betas=self.betas)
        if self.use_cosine_annealing:
            if self.warmup_epochs >= self.trainer.max_epochs:
                raise ValueError('warmup_epochs must be less than trainer.max_epochs')
            if self.min_lr > lr:
                raise ValueError('min_lr must not exceed the effective learning rate')
            self.scheduler = CosineAnnealingLR(optimizer, warmup_epochs=self.warmup_epochs,
                                               total_epochs=self.trainer.max_epochs, lr=lr, min_lr=self.min_lr)

        return [optimizer], []

    def training_step(self, batch, batch_idx):
        pc = batch['surface']
        query_points = batch['query_points']
        intermediate_step = None
        if self.coeff_intermediate > 0:
            anchor_step = getattr(self, 'distill_anchor_step', 0)
            anchor_alpha = getattr(self, 'distill_anchor_alpha', 1.0)
            remaining_steps = (self.distill_schedule_steps if self.distill_schedule_steps is not None
                               else self.trainer.estimated_stepping_batches - anchor_step)
            alpha = progressive_alpha(self.global_step, remaining_steps,
                                      anchor_step=anchor_step, anchor_alpha=anchor_alpha)
            intermediate_step = int(torch.randint(1, self.model.decoder.num_loops, ()).item())
        outputs = (self.model(pc, query_points, intermediate_step=intermediate_step)
                   if intermediate_step is not None else self.model(pc, query_points))
        ## loss
        logits = outputs['logits']
        labels = batch['labels']
        num_vol_points = batch['num_vol_points'][0].item()

        recon_loss, vol_loss, near_loss = self.criterion(logits, labels, num_vol_points=num_vol_points)
        self.log('train/vol_loss', vol_loss.item(), rank_zero_only=True)
        self.log('train/near_loss', near_loss.item(), rank_zero_only=True)
        self.log('train/recon_loss', recon_loss.item(), rank_zero_only=True)

        init_out = outputs['init_out']
        init_loss = self.criterion(init_out, labels, num_vol_points)[0]
        self.log('train/init_loss', init_loss.item(), rank_zero_only=True)

        uncertainty_loss = self.uncertainty_criterion(outputs['uncertainty'], init_out, labels)
        self.log('train/uncertainty_loss', uncertainty_loss.item(), rank_zero_only=True)

        loss = recon_loss * self.coeff_recon + init_loss * self.coeff_init + uncertainty_loss * self.coeff_uncertainty
        if intermediate_step is not None:
            intermediate, intermediate_gt, distill = intermediate_losses(
                outputs['intermediate_logits'], logits, labels, num_vol_points, alpha)
            loss = loss + self.coeff_intermediate * intermediate
            for name, value in {'distill_alpha': alpha, 'intermediate_depth': intermediate_step,
                                'intermediate_gt_loss': intermediate_gt.detach(),
                                'distill_loss': distill.detach(),
                                'intermediate_loss': intermediate.detach()}.items():
                self.log('train/' + name, value, on_step=True, on_epoch=False, rank_zero_only=True)
        if not torch.isfinite(loss).all():
            raise FloatingPointError('Non-finite stage-1 loss; inspect inputs and individual loss terms')
        self.log('train/loss', loss.item())
        if 'prefix_lengths' in outputs:
            self.log('train/prefix_length', outputs['prefix_lengths'].float().mean(), rank_zero_only=True)

        return {
            'loss': loss,
        }

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.scheduler is not None:
            epoch = self.trainer.current_epoch + (batch_idx + 1) / len(self.trainer.train_dataloader)
            self.scheduler.step(epoch)

        lr = get_lr(self.trainer.optimizers)
        self.log('train/lr', lr, on_step=True, on_epoch=False, rank_zero_only=True)

    def validation_step(self, batch, batch_idx):
        pc = batch['surface']
        query_points = batch['query_points']
        labels = batch['labels']
        eval_chunk_size = self.eval_chunk_size if self.eval_chunk_size > 0 else query_points.size(1)
        if self.is_debug:
            ## for sanity check
            eval_chunk_size = 10000
            query_points = query_points[:, :eval_chunk_size * 2]
            labels = labels[:, :eval_chunk_size * 2]

        z = self.model.encode(pc)[0]
        planes = self.model.decode(z, active_num_latents=self.eval_num_latents)[0]
        logits = torch.cat([
            self.model.decode_queries(planes, query_points[:, i:i + eval_chunk_size])
            for i in range(0, query_points.size(1), eval_chunk_size)
        ], dim=1)
        preds = (logits > self.logit_threshold).float()

        ## visualization (set val_vis_interval = -1 to disable)
        batch_size = pc.size(0)
        for b in range(pc.size(0)):
            idx = batch_idx * batch_size + b
            if (self.val_vis_interval > 0) and (idx % self.val_vis_interval == 0):
                pred_points = query_points[b, preds[b].bool()]
                if pred_points.size(0) > 2048:
                    pred_points = pointops.fps(pred_points.unsqueeze(0).float(), 2048)[0]

                pred_img = self.make_points_img(pred_points)
                gt_img = self.make_points_img(pc[b])
                self.add_log_buffer_items('recon', pred_img)
                self.add_log_buffer_items('recon_gt', gt_img)

        self.accuracy_metric.update(preds, labels)
        self.iou_metric.update(preds, labels)

    def on_validation_epoch_end(self):
        self.log_buffered()
        accuracy = self.accuracy_metric.compute()
        iou = self.iou_metric.compute()

        self.log('val/accuracy', accuracy, on_epoch=True, rank_zero_only=True, sync_dist=True)
        self.log('val/iou', iou, on_epoch=True, rank_zero_only=True, sync_dist=True)
        self.track_score(iou)
        print(f'accuracy: {accuracy:.1f}%, iou: {iou:.1f}%')
