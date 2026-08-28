import torch
from lightning.pytorch import Trainer


def compute_effective_lr(lr: float, batch_size: int, trainer: Trainer, base_batch_size: int = 256):
    num_effective_batches = 1
    if trainer is not None:
        num_effective_batches = trainer.accumulate_grad_batches * trainer.world_size
    effective_batch_size = batch_size * num_effective_batches
    return lr * effective_batch_size / base_batch_size


def load_model_weights_from_checkpoint(checkpoint_path: str, prefix: str = 'model'):
    state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
    target_dict = {}
    target_key = f'{prefix}.'
    for k, v in state_dict.items():
        if k.startswith(target_key):
            target_dict[k.replace(target_key, '')] = v

    return target_dict
