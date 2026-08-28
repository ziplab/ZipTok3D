"""Benchmark trained Stage-2 generation under the paper's efficiency protocol."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import engine
from cod.solvers.diffusion_solver import LatentNormalizer
from cod.utils.recon import create_grid_queries
from cod.utils.training import load_model_weights_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generation_config")
    parser.add_argument("generation_checkpoint")
    parser.add_argument("vae_config")
    parser.add_argument("vae_checkpoint")
    parser.add_argument("--category", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--loops", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=18)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--query-chunk", type=int, default=131072)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", default="generation_efficiency.json")
    return parser.parse_args()


def load_model(config_path, checkpoint_path, device):
    cfg = engine.load_config(config_path)
    model = engine.instantiate(cfg.model)
    state = load_model_weights_from_checkpoint(checkpoint_path, prefix="model")
    model.load_state_dict(state, strict=True)
    return cfg, model.to(device).float().eval()


def measure(operation, batch_size, warmup, repeats, device):
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeats):
        start.record()
        operation()
        end.record()
        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end))
    mean = float(np.mean(times))
    return {
        "mean_batch_ms": mean,
        "std_batch_ms": float(np.std(times)),
        "throughput_samples_per_second": batch_size * 1000.0 / mean,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
    }


def query_field(vae, planes, grid, chunk_size):
    outputs = []
    for start in range(0, grid.size(1), chunk_size):
        queries = grid[:, start:start + chunk_size].expand(planes.size(0), -1, -1)
        outputs.append(vae.decode_queries(planes, queries))
    return torch.cat(outputs, dim=1)


@torch.inference_mode()
def main():
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("the benchmark requires CUDA")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    generation_cfg, diffusion = load_model(
        args.generation_config, args.generation_checkpoint, args.device
    )
    _, vae = load_model(args.vae_config, args.vae_checkpoint, args.device)
    normalizer = LatentNormalizer(generation_cfg.solver.normalizer_path).to(args.device)
    if not 1 <= args.tokens <= min(diffusion.num_latents, vae.active_num_latents):
        raise ValueError(
            "--tokens exceeds the causal EDM or prefix VAE maximum length"
        )
    if not 1 <= args.loops <= vae.autoencoder.decoder.num_loops:
        raise ValueError("invalid stage-1 refinement depth")

    labels = torch.full(
        (args.batch_size,), args.category, dtype=torch.long, device=args.device
    )
    grid = create_grid_queries(128).to(args.device)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    def sample():
        return diffusion.sample(
            labels,
            num_steps=args.num_steps,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            rho=args.rho,
            generator=generator,
            num_latents=args.tokens,
        )

    sampling_measurement = measure(
        sample, args.batch_size, args.warmup, args.repeats, args.device
    )

    def full_operation():
        latent = normalizer.denormalize(sample())
        features, mask = vae.decode_latents(latent)
        planes = vae.decode_embed(
            features, mask=mask, num_decode_loops=args.loops
        )[0]
        return query_field(vae, planes, grid, args.query_chunk)

    full_measurement = measure(
        full_operation, args.batch_size, args.warmup, args.repeats, args.device
    )
    payload = {
        "protocol": {
            "weights": "trained checkpoints",
            "latent_shape": [args.tokens, diffusion.channels],
            "batch_size": args.batch_size,
            "precision": "FP32",
            "tf32": False,
            "flash_sdp": False,
            "steps": args.num_steps,
            "sampler": "Heun",
            "rho": args.rho,
            "sigma_min": args.sigma_min,
            "sigma_max": args.sigma_max,
            "stage1_refinement_passes": args.loops,
            "grid_resolution": 128,
            "query_chunk": args.query_chunk,
            "warmup_batches": args.warmup,
            "measured_batches": args.repeats,
            "excluded": ["model loading", "marching cubes", "file output"],
        },
        "sampling": sampling_measurement,
        "full_generation": full_measurement,
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
