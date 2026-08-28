"""Cache deterministic stage-2 posterior means for EDM training."""

import argparse
import json
import os
from os import path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import engine
from cod.utils.training import load_model_weights_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="stage-2 prefix VAE training config")
    parser.add_argument("checkpoint", help="trained stage-2 prefix VAE checkpoint")
    parser.add_argument("output", help="latent cache directory")
    parser.add_argument(
        "--tokens",
        type=int,
        default=16,
        help="maximum causal prefix length to cache for EDM training (paper setting: 16)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123456)
    return parser.parse_args()


@torch.no_grad()
def cache_split(model, dataset, split, output, args):
    if hasattr(dataset, "repeat"):
        dataset.repeat = 1
    if hasattr(dataset, "transform"):
        dataset.transform = None
    if hasattr(dataset, "use_queries"):
        dataset.use_queries = False
    if hasattr(dataset, "use_full_surface"):
        dataset.use_full_surface = False
    split_offset = {"train": 0, "val": 1, "test": 2}[split]
    generator = torch.Generator().manual_seed(args.seed + split_offset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        generator=generator,
    )
    latent_batches = []
    label_batches = []
    for batch in tqdm(loader, desc=f"cache {split}"):
        points = batch["surface"].to(args.device, non_blocking=True)
        encoded = model.encode_embed(points)
        latent, _ = model.encode_latents(
            encoded,
            active_num_latents=args.tokens,
            sample_posterior=False,
        )
        latent_batches.append(latent.cpu().numpy())
        labels = batch["category_ids"]
        if labels.ndim > 1:
            labels = labels[:, 0]
        label_batches.append(labels.cpu().numpy().astype(np.int64))

    latents = np.concatenate(latent_batches, axis=0).astype(np.float32)
    labels = np.concatenate(label_batches, axis=0).astype(np.int64)
    split_dir = path.join(output, split)
    os.makedirs(split_dir, exist_ok=True)
    np.save(path.join(split_dir, "latents.npy"), latents)
    np.save(path.join(split_dir, "labels.npy"), labels)
    print(f"{split}: cached {len(latents)} arrays with shape {latents.shape[1:]} -> {split_dir}")
    return latents


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = engine.load_config(args.config)
    model = engine.instantiate(cfg.model)
    state = load_model_weights_from_checkpoint(args.checkpoint, prefix="model")
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval()
    if not 1 <= args.tokens <= model.active_num_latents:
        raise ValueError(
            f"--tokens must be in [1, {model.active_num_latents}], got {args.tokens}"
        )
    dm = engine.instantiate(cfg.data)

    os.makedirs(args.output, exist_ok=True)
    train_latents = None
    for split in ["train", "val", "test"]:
        latents = cache_split(model, dm.get_dataset(split), split, args.output, args)
        if split == "train":
            train_latents = latents
    mean = train_latents.mean(axis=(0, 1), keepdims=True)
    std = train_latents.std(axis=(0, 1), keepdims=True).clip(1e-6)
    np.savez(path.join(args.output, "normalizer.npz"), mean=mean, std=std)
    metadata = {
        "num_latents": args.tokens,
        "channels": int(train_latents.shape[-1]),
        "posterior": "deterministic_mean",
        "input_augmentation": "disabled",
        "dataset_repeat": 1,
        "seed": args.seed,
        "normalizer": "training_set_channel_statistics",
        "splits": ["train", "val", "test"],
    }
    with open(path.join(args.output, "metadata.json"), "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
