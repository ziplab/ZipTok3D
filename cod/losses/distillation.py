"""Progressive intermediate occupancy supervision with a detached final teacher.

Only intermediate predictions transition from labels to self-distillation;
the final reconstruction and COD-VAE auxiliary objectives keep their labels.
Optional step/alpha anchors preserve continuity when a resumed run changes batch size.
"""
import math
from numbers import Integral, Real
import torch
from .recon import OccupancyReconstructionLoss


def progressive_alpha(step, total_steps, *, anchor_step=0, anchor_alpha=1.0):
    """Decay from an optional anchor over total_steps remaining Trainer steps."""
    if isinstance(anchor_step, bool) or not isinstance(anchor_step, Integral) or anchor_step < 0:
        raise ValueError('distill_anchor_step must be a nonnegative integer')
    if (isinstance(anchor_alpha, bool) or not isinstance(anchor_alpha, Real)
            or not math.isfinite(anchor_alpha) or not 0 <= anchor_alpha <= 1):
        raise ValueError('distill_anchor_alpha must be finite and in [0, 1]')
    if not math.isfinite(float(total_steps)) or total_steps <= 0:
        raise ValueError('total_steps must be finite and positive')
    if not math.isfinite(float(step)) or step < 0:
        raise ValueError('step must be finite and nonnegative')
    if step < anchor_step:
        raise ValueError('global_step is before distill_anchor_step; check the resume checkpoint and config')
    return float(anchor_alpha) * max(0.0, 1.0 - float(step - anchor_step) / float(total_steps))


def intermediate_losses(student_logits, final_logits, labels, num_vol_points, alpha):
    """Return blended, ground-truth and distillation losses (before beta weighting)."""
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0, 1]')
    if student_logits.shape != final_logits.shape or labels.shape != student_logits.shape:
        raise ValueError('student, teacher and label shapes must match')
    if not 0 < num_vol_points < labels.shape[1]:
        raise ValueError('both volume and near-surface queries are required')
    criterion = OccupancyReconstructionLoss(vol_coeff=1.0, near_coeff=0.1)
    gt = criterion(student_logits, labels, num_vol_points)[0]
    teacher = torch.sigmoid(final_logits.detach().float().clamp(-30, 30))
    dist = criterion(student_logits.float().clamp(-30, 30), teacher, num_vol_points)[0]
    return alpha * gt + (1.0 - alpha) * dist, gt, dist
