"""MultiBudgetSolver + AllLoopSolver that supervises every intermediate loop in every step.

AllLoopSolver differences from MultiBudgetSolver:
  * training_step runs ELT on ALL intermediate loops (1 .. num_loops-1) every step,
    rather than one randomly-sampled loop;
  * main forward uses model.forward() so uncertainty is handled correctly by the
    autoencoder's internal implementation;
  * ELT decodes use a second model.encode() call with z.detach(), so each
    intermediate loop is trained toward the final-loop teacher without adding an
    extra encoder gradient path (encoder is trained only by the main loss);
  * val/iou_k1 and val/iou_k128 are both logged (val_budgets must include both).
"""
from typing import Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .base_solver import AutoencoderSolver
from cod.losses.distillation import progressive_alpha, intermediate_losses
from cod.metrics.occupancy import IoU
from .prefix import normalize_weights, scheduled_probs


class MultiBudgetSolver(AutoencoderSolver):

    def __init__(self,
                 dm, model,
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
                 prefix_budget_weights: Optional[Sequence[float]] = None,
                 prefix_budget_weights_final: Optional[Sequence[float]] = None,
                 prefix_weight_anneal_epochs: Optional[Sequence[float]] = None,
                 val_budgets: Sequence[int] = (1, 2, 4, 128),
                 score_budgets: Sequence[int] = (1, 2, 4),
                 log_budget_losses: bool = True,
                 ):
        super().__init__(
            dm=dm, model=model, lr=lr, use_cosine_annealing=use_cosine_annealing,
            warmup_epochs=warmup_epochs, logit_threshold=logit_threshold,
            eval_chunk_size=eval_chunk_size, val_vis_interval=val_vis_interval,
            coeff_uncertainty=coeff_uncertainty, coeff_init=coeff_init, coeff_recon=coeff_recon,
            scale_lr_by_batch_size=scale_lr_by_batch_size, weight_decay=weight_decay,
            betas=betas, eval_num_latents=eval_num_latents, min_lr=min_lr,
            coeff_intermediate=coeff_intermediate, distill_schedule_steps=distill_schedule_steps,
            distill_anchor_step=distill_anchor_step, distill_anchor_alpha=distill_anchor_alpha)
        budgets = list(self.model.prefix_budgets)
        if not hasattr(self.model, 'set_budget_probs'):
            raise TypeError('MultiBudgetSolver needs ziptok_retrain.model.WeightedPrefixAutoencoder')
        uniform = [1.0] * len(budgets)
        self._w_start = normalize_weights(prefix_budget_weights or uniform, len(budgets))
        self._w_final = (normalize_weights(prefix_budget_weights_final, len(budgets))
                         if prefix_budget_weights_final is not None else None)
        if (self._w_final is None) != (prefix_weight_anneal_epochs is None):
            raise ValueError('prefix_budget_weights_final and prefix_weight_anneal_epochs go together')
        self._anneal = None if prefix_weight_anneal_epochs is None else tuple(prefix_weight_anneal_epochs)
        scheduled_probs(0, self._w_start, self._w_final, self._anneal)

        self.val_budgets = sorted({int(k) for k in val_budgets})
        self.score_budgets = sorted({int(k) for k in score_budgets})
        for k in self.val_budgets:
            if not 1 <= k <= self.model.num_latents:
                raise ValueError(f'val budget {k} outside [1, {self.model.num_latents}]')
        if not set(self.score_budgets) <= set(self.val_budgets):
            raise ValueError('score_budgets must be a subset of val_budgets')
        self.full_budget = self.model.num_latents
        if self.full_budget not in self.val_budgets:
            raise ValueError('val_budgets must include the full budget for val/iou continuity')
        self.budget_iou = nn.ModuleDict({str(k): IoU() for k in self.val_budgets})
        self.log_budget_losses = bool(log_budget_losses)
        self.model.set_budget_probs(self._w_start)

    def current_budget_probs(self, epoch: float) -> torch.Tensor:
        return scheduled_probs(epoch, self._w_start, self._w_final, self._anneal)

    def on_train_start(self):
        if self.scheduler is not None:
            self.scheduler.step(float(self.trainer.current_epoch))

    def on_train_batch_start(self, batch, batch_idx):
        num_batches = max(1, int(self.trainer.num_training_batches))
        epoch = self.trainer.current_epoch + batch_idx / num_batches
        probs = self.current_budget_probs(epoch)
        self.model.set_budget_probs(probs)
        self.log('train/p_k1', float(probs[0]), on_step=True, on_epoch=False, rank_zero_only=True)

    def training_step(self, batch, batch_idx):
        pc = batch['surface']
        query_points = batch['query_points']
        intermediate_step = None
        alpha = None
        if self.coeff_intermediate > 0:
            remaining_steps = (self.distill_schedule_steps if self.distill_schedule_steps is not None
                               else self.trainer.estimated_stepping_batches - self.distill_anchor_step)
            alpha = progressive_alpha(self.global_step, remaining_steps,
                                      anchor_step=self.distill_anchor_step,
                                      anchor_alpha=self.distill_anchor_alpha)
            intermediate_step = int(torch.randint(1, self.model.decoder.num_loops, ()).item())
        outputs = (self.model(pc, query_points, intermediate_step=intermediate_step)
                   if intermediate_step is not None else self.model(pc, query_points))
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
        loss = (recon_loss * self.coeff_recon + init_loss * self.coeff_init
                + uncertainty_loss * self.coeff_uncertainty)
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
            raise FloatingPointError('Non-finite loss')
        self.log('train/loss', loss.item())
        lengths = outputs['prefix_lengths']
        self.log('train/prefix_length', lengths.float().mean(), rank_zero_only=True)
        if self.log_budget_losses:
            self._log_budget_losses(logits.detach(), labels, num_vol_points, lengths)
        return {'loss': loss}

    @torch.no_grad()
    def _log_budget_losses(self, logits, labels, num_vol_points, lengths):
        logits = logits.float()
        vol  = F.binary_cross_entropy_with_logits(
            logits[:, :num_vol_points], labels[:, :num_vol_points].float(), reduction='none').mean(1)
        near = F.binary_cross_entropy_with_logits(
            logits[:, num_vol_points:], labels[:, num_vol_points:].float(), reduction='none').mean(1)
        per_sample = self.criterion.vol_coeff * vol + self.criterion.near_coeff * near
        for budget in torch.unique(lengths).tolist():
            mask = lengths == budget
            self.log(f'train_k/recon_k{int(budget)}', per_sample[mask].mean(),
                     on_step=False, on_epoch=True, rank_zero_only=True)

    def validation_step(self, batch, batch_idx):
        pc           = batch['surface']
        query_points = batch['query_points']
        labels       = batch['labels']
        eval_chunk_size = self.eval_chunk_size if self.eval_chunk_size > 0 else query_points.size(1)
        if self.is_debug:
            eval_chunk_size = 10000
            query_points = query_points[:, :eval_chunk_size * 2]
            labels       = labels[:, :eval_chunk_size * 2]
        z = self.model.encode(pc)[0]
        for budget in self.val_budgets:
            planes = self.model.decode(z, active_num_latents=budget)[0]
            logits = torch.cat([
                self.model.decode_queries(planes, query_points[:, i:i + eval_chunk_size])
                for i in range(0, query_points.size(1), eval_chunk_size)
            ], dim=1)
            preds = (logits > self.logit_threshold).float()
            self.budget_iou[str(budget)].update(preds, labels)
            if budget == self.full_budget:
                self.accuracy_metric.update(preds, labels)
            del planes, logits, preds

    def on_validation_epoch_end(self):
        ious = {k: self.budget_iou[str(k)].compute() for k in self.val_budgets}
        for k, value in ious.items():
            self.log(f'val/iou_k{k}', value, on_epoch=True, sync_dist=True)
        self.log('val/iou', ious[self.full_budget], on_epoch=True, rank_zero_only=True, sync_dist=True)
        self.log('val/accuracy', self.accuracy_metric.compute(), on_epoch=True,
                 rank_zero_only=True, sync_dist=True)
        score = torch.stack([ious[k] for k in self.score_budgets]).mean()
        self.track_score(score)
        summary = ', '.join(f'K{k}={float(v):.3f}' for k, v in ious.items())
        print(f'val iou: {summary}; score(mean K{self.score_budgets})={float(score):.3f}')


class AllLoopSolver(MultiBudgetSolver):
    """ELT supervision on ALL intermediate loops every training step.

    Main forward uses model.forward() so uncertainty is handled correctly.
    ELT decodes use a detached z from a second model.encode() call, so the
    intermediate-loop decoder is trained by ELT while the encoder is trained
    only by the main loss (which already provides K-budget signal).
    """

    def training_step(self, batch, batch_idx):
        pc           = batch['surface']
        query_points = batch['query_points']
        labels       = batch['labels']
        num_vol_points = batch['num_vol_points'][0].item()

        # --- Main forward (encoder + all 6 loops; correct uncertainty) --------
        main_out  = self.model(pc, query_points)
        logits    = main_out['logits']
        init_out  = main_out['init_out']
        lengths   = main_out['prefix_lengths']

        recon_loss, vol_loss, near_loss = self.criterion(logits, labels, num_vol_points=num_vol_points)
        init_loss        = self.criterion(init_out, labels, num_vol_points)[0]
        uncertainty_loss = self.uncertainty_criterion(main_out['uncertainty'], init_out, labels)

        loss = (recon_loss * self.coeff_recon
                + init_loss  * self.coeff_init
                + uncertainty_loss * self.coeff_uncertainty)

        for name, val in [('recon_loss', recon_loss), ('vol_loss', vol_loss),
                          ('near_loss', near_loss), ('init_loss', init_loss),
                          ('uncertainty_loss', uncertainty_loss)]:
            self.log(f'train/{name}', val.item(), rank_zero_only=True)
        self.log('train/loss', loss.item())
        self.log('train/prefix_length', lengths.float().mean(), rank_zero_only=True)
        if self.log_budget_losses:
            self._log_budget_losses(logits.detach(), labels, num_vol_points, lengths)

        # --- All-loop ELT ------------------------------------------------
        if self.coeff_intermediate > 0:
            remaining_steps = (self.distill_schedule_steps if self.distill_schedule_steps is not None
                               else self.trainer.estimated_stepping_batches - self.distill_anchor_step)
            alpha = progressive_alpha(self.global_step, remaining_steps,
                                      anchor_step=self.distill_anchor_step,
                                      anchor_alpha=self.distill_anchor_alpha)
            self.log('train/distill_alpha', alpha, on_step=True, on_epoch=False, rank_zero_only=True)

            # Second encode with detach: ELT backward only flows through intermediate decoder,
            # not the encoder. Main loss already provides K-budget signal to the encoder.
            # Detached immediately so no backward graph is retained for the encoder.
            z_elt  = self.model.encode(pc)[0].detach()
            n_mid  = self.model.decoder.num_loops - 1   # steps 1 .. num_loops-1
            elt_sum = torch.tensor(0.0, device=logits.device)
            for step_j in range(1, self.model.decoder.num_loops):
                dec_j      = self.model.decode(z_elt, active_num_latents=lengths, intermediate_step=step_j)
                mid_logits = self.model.decode_queries(dec_j[4], query_points)
                step_loss, _, _ = intermediate_losses(mid_logits, logits, labels, num_vol_points, alpha)
                elt_sum = elt_sum + step_loss
                del dec_j, mid_logits
            elt_avg = elt_sum / n_mid
            loss = loss + self.coeff_intermediate * elt_avg
            self.log('train/intermediate_loss', elt_avg.detach(),
                     on_step=True, on_epoch=False, rank_zero_only=True)

        if not torch.isfinite(loss).all():
            raise FloatingPointError('Non-finite loss')
        return {'loss': loss}
