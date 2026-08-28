"""Generate ShapeNet meshes from a physical prefix of the causal stage-2 EDM."""

import argparse
import os
from os import path

import torch

import engine
from cod.solvers.diffusion_solver import LatentNormalizer
from cod.utils.recon import create_grid_queries, occupancy_to_mesh
from cod.utils.training import load_model_weights_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("generation_config")
    parser.add_argument("generation_checkpoint")
    parser.add_argument("vae_config")
    parser.add_argument("vae_checkpoint")
    parser.add_argument("output")
    parser.add_argument("--category", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=18)
    parser.add_argument(
        "--tokens",
        type=int,
        default=2,
        help="physical causal-prefix length to sample (paper setting: 2)",
    )
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument(
        "--loops",
        type=int,
        default=5,
        help="number of shared stage-1 refinement passes (paper setting: 5)",
    )
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=250000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def load_model(config_path, checkpoint_path, device):
    cfg = engine.load_config(config_path)
    model = engine.instantiate(cfg.model)
    state = load_model_weights_from_checkpoint(checkpoint_path, prefix="model")
    model.load_state_dict(state, strict=True)
    return cfg, model.to(device).eval()


@torch.no_grad()
def main():
    args = parse_args()
    if args.num_samples < 1 or args.batch_size < 1:
        raise ValueError("--num-samples and --batch-size must be positive")
    gen_cfg, diffusion = load_model(
        args.generation_config, args.generation_checkpoint, args.device
    )
    _, vae = load_model(args.vae_config, args.vae_checkpoint, args.device)
    normalizer = LatentNormalizer(gen_cfg.solver.normalizer_path).to(args.device)
    max_loops = vae.autoencoder.decoder.num_loops
    if not 1 <= args.loops <= max_loops:
        raise ValueError(f"--loops must be in [1, {max_loops}], got {args.loops}")
    if not 1 <= args.tokens <= diffusion.num_latents:
        raise ValueError(
            f"--tokens must be in [1, {diffusion.num_latents}], got {args.tokens}"
        )
    if args.tokens > vae.active_num_latents:
        raise ValueError(
            "sampled EDM prefix exceeds the prefix VAE's supported length: "
            f"{args.tokens} > {vae.active_num_latents}"
        )

    queries = create_grid_queries(args.resolution).to(args.device)
    os.makedirs(args.output, exist_ok=True)
    generator = torch.Generator(device=args.device).manual_seed(args.seed + args.category)
    for batch_start in range(0, args.num_samples, args.batch_size):
        current_batch = min(args.batch_size, args.num_samples - batch_start)
        labels = torch.full(
            (current_batch,), args.category, dtype=torch.long, device=args.device
        )
        normalized = diffusion.sample(
            labels,
            num_steps=args.num_steps,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            rho=args.rho,
            generator=generator,
            num_latents=args.tokens,
        )
        latents = normalizer.denormalize(normalized)
        if tuple(latents.shape[1:]) != (args.tokens, diffusion.channels):
            raise RuntimeError(
                f"expected generated arrays [B,{args.tokens},"
                f"{diffusion.channels}], got {tuple(latents.shape)}"
            )
        features, prefix_mask = vae.decode_latents(latents)
        planes = vae.decode_embed(
            features,
            mask=prefix_mask,
            num_decode_loops=args.loops,
        )[0]

        for batch_index in range(current_batch):
            sample_index = batch_start + batch_index
            logits = []
            for start in range(0, queries.size(1), args.chunk_size):
                chunk = queries[:, start:start + args.chunk_size]
                logits.append(vae.decode_queries(
                    planes[batch_index:batch_index + 1], chunk
                ))
            volume = torch.cat(logits, dim=1)[0]
            try:
                mesh = occupancy_to_mesh(volume, args.resolution)
            except (RuntimeError, ValueError) as exc:
                print(f"sample {sample_index}: empty or invalid isosurface ({exc})")
                continue
            mesh.export(path.join(
                args.output,
                f"category{args.category:02d}_{sample_index:04d}.obj",
            ))
        print(f"generated {batch_start + current_batch}/{args.num_samples}")


if __name__ == "__main__":
    main()
