import os
from typing import List, Union

import torch
import numpy as np
import cv2
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch import LightningModule, Callback
from torchmetrics.metric import Metric

import engine


class BaseSolver(LightningModule):

    def __init__(self, track_max_score: bool = True):
        super().__init__()

        self.checkpoint_epoch = -1
        self.output_dir = engine.to_experiment_dir('outputs')

        self._track_max_score = track_max_score
        self._best_score = None
        self._additional_callbacks: List[Callback] = [_DefaultTaskCallback()]
        self._metrics = None
        self._media_log_buffers = {}
        self._log_once_dict = {}
        self._logged_dict = {}
        self._debug = False

        os.makedirs(self.output_dir, exist_ok=True)

    def get_trainer_if_exists(self):
        if hasattr(self, 'trainer'):
            return self.trainer
        return None

    @property
    def is_debug(self):
        return self._debug

    def enable_debug(self):
        self._debug = True

    def configure_callbacks(self):
        return self._additional_callbacks

    def add_callback(self, callback: Callback):
        self._additional_callbacks.append(callback)

    @property
    def best_score(self):
        return self._best_score

    def track_score(self, score: any):
        if self._best_score is None:
            should_update_best = True
        else:
            current_is_higher = score > self._best_score
            should_update_best = current_is_higher == self._track_max_score
        if should_update_best:
            self._best_score = score

        self.log('score', score, on_epoch=True, sync_dist=True)
        self.log('best', self._best_score, on_epoch=True, sync_dist=True)

    def restore_checkpoint(self, checkpoint_path: str):
        print(f'checkpoint loaded: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['state_dict']
        self.load_state_dict(state_dict, strict=True)
        self.checkpoint_epoch = checkpoint.get('epoch', -1)

    def reset_metrics(self):
        if self._metrics is None:
            self._metrics = [value for value in self.modules() if isinstance(value, Metric)]

        for metric in self._metrics:
            metric.reset()

    def create_image_log_buffer(self, name, log_once: bool = False):
        self._media_log_buffers[name] = []
        self._log_once_dict[name] = log_once

    @rank_zero_only
    def add_log_buffer_items(self, name: str, items: Union[any, list, tuple]):
        if name not in self._media_log_buffers:
            raise Exception('image buffer not created: ' + name)
        # check image is iterable with if
        if not isinstance(items, (list, tuple)):
            items = [items]
        images = []
        for item in items:
            if isinstance(item, torch.Tensor):
                item = item.detach().cpu().numpy()
            images.append(item)
        self._media_log_buffers[name].extend(images)

    @rank_zero_only
    def log_buffered(self):
        for name in self._media_log_buffers.keys():
            if self._should_skip_log(name):
                continue
            images = self._media_log_buffers[name]
            if len(images) > 0:
                self.log_image(name, images)

    def log_image(self, name, images, max_length=None, **kwargs):
        if self._should_skip_log(name):
            return

        wandb_logger = None
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                wandb_logger = logger
                break

        if wandb_logger is None:
            return

        if isinstance(images, np.ndarray) or isinstance(images, torch.Tensor):
            images = [images]

        parsed_images = []
        for img in images:
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            is_float = (img.dtype == np.float32) or (img.dtype == np.float64)
            if is_float and (img.max() <= 1.0):
                img = (img * 255).astype(np.uint8)
            if max_length is not None:
                length = max(img.shape[:2])
                ratio = max_length / length
                if ratio < 1:
                    new_size = (int(img.shape[1] * ratio), int(img.shape[0] * ratio))
                    img = cv2.resize(img, new_size)
            parsed_images.append(img)

        wandb_logger.log_image(name, parsed_images, **kwargs)
        if not self.trainer.sanity_checking:
            self._logged_dict[name] = True

    def reset_image_log_buffers(self):
        for name in self._media_log_buffers.keys():
            self._media_log_buffers[name] = []

    def _should_skip_log(self, name):
        if name not in self._log_once_dict:
            return False
        log_once = self._log_once_dict[name]
        if not log_once:
            return False
        return (name in self._logged_dict) and (self._logged_dict[name])


class _DefaultTaskCallback(Callback):
    def on_sanity_check_end(self, trainer, solver):
        if not isinstance(solver, BaseSolver):
            return
        solver._best_score = None

    def on_validation_epoch_start(self, trainer, solver) -> None:
        if not isinstance(solver, BaseSolver):
            return
        solver.reset_metrics()
        solver.reset_image_log_buffers()

    def on_after_backward(self, trainer, solver) -> None:
        if not solver.is_debug:
            return

        for name, p in solver.named_parameters():
            if (p.grad is None) and p.requires_grad:
                raise Exception(f'unused parameters detected: {name}')
