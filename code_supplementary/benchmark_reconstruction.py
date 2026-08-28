"""Benchmark a trained tokenizer under the paper's reconstruction protocol."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import engine
from cod.utils.recon import create_grid_queries
from cod.utils.training import load_model_weights_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--loops", type=int, default=5)
    parser.add_argument("--split", choices=("val", "test"), default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--query-chunk", type=int, default=131072)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--output", default="reconstruction_efficiency.json")
    return parser.parse_args()


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
        "throughput_shapes_per_second": batch_size * 1000.0 / mean,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
    }


def decode_prefix(model, prefix, loops):
    return model.decoder.decode(prefix, mask=None, num_loops=loops)[0]


def query_field(model, planes, grid, chunk_size):
    batch_size = planes.size(0)
    outputs = []
    for start in range(0, grid.size(1), chunk_size):
        queries = grid[:, start:start + chunk_size].expand(batch_size, -1, -1)
        outputs.append(model.decode_queries(planes, queries))
    return torch.cat(outputs, dim=1)


@torch.inference_mode()
def main():
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("the benchmark requires CUDA")
    if min(args.batch_size, args.query_chunk, args.warmup, args.repeats) < 1:
        raise ValueError("batch size, query chunk, warmup, and repeats must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    cfg = engine.load_config(args.config)
    model = engine.instantiate(cfg.model)
    state = load_model_weights_from_checkpoint(args.checkpoint, prefix="model")
    model.load_state_dict(state, strict=True)
    model.to(args.device).float().eval()
    if not 1 <= args.tokens <= model.num_latents:
        raise ValueError(f"--tokens must be in [1, {model.num_latents}]")
    if not 1 <= args.loops <= model.decoder.num_loops:
        raise ValueError(f"--loops must be in [1, {model.decoder.num_loops}]")

    dm = engine.instantiate(cfg.data)
    split = args.split or dm.evaluation_split
    dataset = dm.get_dataset(split)
    dataset.use_queries = False
    dataset.use_full_surface = False
    if len(dataset) < args.batch_size:
        raise RuntimeError("selected split is smaller than the benchmark batch")
    points = torch.stack([dataset[index]["surface"] for index in range(args.batch_size)])
    points = points.to(args.device)
    grid = create_grid_queries(128).to(args.device)

    full_latents = model.encode(points)[0]
    prefix = full_latents[:, :args.tokens].contiguous()
    decoder_measurement = measure(
        lambda: decode_prefix(model, prefix, args.loops),
        args.batch_size, args.warmup, args.repeats, args.device,
    )
    decoder_measurement["latency_ms_per_shape"] = (
        decoder_measurement["mean_batch_ms"] / args.batch_size
    )

    def full_operation():
        encoded = model.encode(points)[0][:, :args.tokens].contiguous()
        planes = decode_prefix(model, encoded, args.loops)
        return query_field(model, planes, grid, args.query_chunk)

    full_measurement = measure(
        full_operation, args.batch_size, args.warmup, args.repeats, args.device
    )
    decoder_parameters = sum(
        parameter.numel()
        for module in (model.decoder, model.head)
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    payload = {
        "protocol": {
            "weights": "trained checkpoint",
            "split": split,
            "tokens": args.tokens,
            "refinement_passes": args.loops,
            "batch_size": args.batch_size,
            "precision": "FP32",
            "tf32": False,
            "flash_sdp": False,
            "grid_resolution": 128,
            "query_chunk": args.query_chunk,
            "warmup_batches": args.warmup,
            "measured_batches": args.repeats,
            "excluded": ["data loading", "CPU-GPU transfer", "marching cubes", "file output"],
        },
        "decoder_parameters": decoder_parameters,
        "decoder_parameters_m": decoder_parameters / 1e6,
        "decoder": decoder_measurement,
        "full_reconstruction": full_measurement,
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
