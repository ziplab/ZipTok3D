import os
from os import path
import time
import re
from typing import Union

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from .config import instantiate
from .experiment import to_experiment_dir, get_experiment_name

__all__ = ['prepare_trainer', 'get_checkpoint_dir', 'find_best_checkpoint_path', 'parse_gpus_str']


def prepare_trainer(cfg, gpus: str, debug: bool = False):
    gpus = parse_gpus_str(gpus)
    callbacks = []

    ## checkpoint callback handling
    checkpoint_cfg = cfg.get('checkpoint', None)
    if checkpoint_cfg is not None:
        checkpoint_filename = checkpoint_cfg.get('filename', 'best-{epoch}-{score:.4f}')
        save_weights_only = checkpoint_cfg.get('save_weights_only', True)
        checkpoint_dir = get_checkpoint_dir(cfg, remove_dir_cfg=True)
        checkpoint_cfg.pop('save_dir', None)
        checkpoint_callback = instantiate(checkpoint_cfg, ModelCheckpoint,
                                          dirpath=checkpoint_dir,
                                          filename=checkpoint_filename,
                                          save_weights_only=save_weights_only,
                                          save_last=True,
                                          monitor=checkpoint_cfg.get('monitor', 'score')
                                          )
        checkpoint_callback.CHECKPOINT_NAME_LAST = 'last'
        callbacks.append(checkpoint_callback)

    ## logging callback handling
    logger = None
    wandb_cfg = cfg.get('wandb', None)
    if wandb_cfg is not None and not debug:
        logger = _create_wandb_callback(cfg, wandb_cfg)

    trainer_cfg = cfg.get('trainer', OmegaConf.create({}))
    strategy = trainer_cfg.pop('strategy', None)
    if (gpus == 1) or (isinstance(gpus, list) and (len(gpus) == 1)):
        strategy = None

    if debug:
        gpus = 1
        callbacks = []
        strategy = None
        logger = None

    ## handling additional callbacks
    additional_callbacks = trainer_cfg.pop('callbacks', [])
    for callback_cfg in additional_callbacks:
        callback = instantiate(callback_cfg)
        callbacks.append(callback)

    if strategy is not None:
        trainer_cfg.strategy = strategy
    return instantiate(trainer_cfg, pl.Trainer,
                       allow_unknown_params=True,
                       logger=logger,
                       callbacks=callbacks,
                       fast_dev_run=debug,
                       devices=gpus,
                       )


def get_checkpoint_dir(cfg, remove_dir_cfg: bool = False):
    checkpoint_cfg = cfg.get('checkpoint', None)
    checkpoint_dir = 'checkpoints'
    if checkpoint_cfg is not None:
        checkpoint_dir = checkpoint_cfg.get('save_dir', 'checkpoints')
        if remove_dir_cfg:
            del checkpoint_cfg['save_dir']

    return to_experiment_dir(checkpoint_dir)


def find_best_checkpoint_path(checkpoint_dir):
    files = [x for x in os.listdir(checkpoint_dir) if x.endswith('.ckpt') and x.startswith('best')]
    pattern = re.compile(r"best-epoch=([0-9]+)-score=([0-9.]+).ckpt")
    file_score_tuples = []
    for file in files:
        m = pattern.match(file)
        if m is None:
            continue

        file_score_tuples.append((file, m.group(2)))

    if len(file_score_tuples) == 0:
        return None
    file_score_tuples.sort(key=lambda x: -1 * float(x[-1]))
    best_file = file_score_tuples[0][0]
    return path.join(checkpoint_dir, best_file)


def parse_gpus_str(gpus_str: str) -> Union[int, list]:
    gpus_str = gpus_str.strip()
    if re.match(r'^(\d+, ?)+\d$', gpus_str):
        return [int(x.strip()) for x in gpus_str.split(',')]

    m = re.match(r'^\[((?:\d+, ?)*\d)]$', gpus_str)
    if m is not None:
        return [int(x.strip()) for x in m.group(1).split(',')]

    return int(gpus_str)


def _create_wandb_callback(cfg, wandb_cfg):
    secret_path = wandb_cfg.get('secret_path', None)
    if secret_path is None:
        return None

    secret_cfg = OmegaConf.load(secret_path)
    os.environ['WANDB_API_KEY'] = secret_cfg.api_key
    name = get_experiment_name()
    run_id = f'{int(time.time())}_{name}'
    hyper_params = OmegaConf.to_container(cfg, resolve=True)
    return WandbLogger(save_dir=to_experiment_dir(),
                       name=name,
                       id=run_id,
                       project=wandb_cfg.get('project', 'myproject'),
                       entity=secret_cfg.entity,
                       config=hyper_params,
                       )
