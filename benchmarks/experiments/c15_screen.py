"""Stress-first component experiments against unchanged production models.

Example: python benchmarks/experiments/c15_screen.py --kind norm --out results/c15-norm.json
Models use separate production graph caches; monkeypatches exist only while
the candidate executes or captures. Never use this module as a submission.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from benchmarks.experiments.c15_components import gemm_gelu, norm, welford_norm
from transformer_opt import dispatch, kernels
from transformer_opt.submission import (
    BaselineTransformer, TransformerConfig, UserOptimizedTransformer,
    compare_outputs, copy_model_weights, generate_random_case,
)
from tools.capture_environment import capture_environment

ORIGINAL_NORM, ORIGINAL_FUSED, ORIGINAL_GELU = F.layer_norm, kernels.fused_residual_layer_norm, F.gelu
ORIGINAL_REFERENCE = dispatch._reference_attention


@contextmanager
def candidate_context(kind, amount):
    if kind in ("norm", "welford"):
        def plain(x, normalized_shape, weight=None, bias=None, eps=1e-5):
            if weight is None or tuple(normalized_shape) != (x.shape[-1],):
                return ORIGINAL_NORM(x, normalized_shape, weight, bias, eps)
            if kind == "welford":
                if x.shape[-1] not in (32, 128):
                    return ORIGINAL_NORM(x, normalized_shape, weight, bias, eps)
                return welford_norm(x, weight, bias, eps, rows_per_program=amount)
            return norm(x, weight, bias, eps, rows_per_program=amount)
        def fused(x, update, weight, bias, eps, valid_token_mask=None, **kwargs):
            return norm(x, weight, bias, eps, update=update, mask=valid_token_mask,
                        rows_per_program=amount, **kwargs)
        F.layer_norm = plain
        if kind == "norm": kernels.fused_residual_layer_norm = fused
    elif kind == "gelu":
        F.gelu = lambda value, approximate="none": value
    elif kind == "chunk":
        def chunked(q, k, v, mask, causal, scale):
            out = torch.empty_like(q, memory_format=torch.contiguous_format)
            for start in range(0, q.shape[0], amount):
                end = start + amount
                result = ORIGINAL_REFERENCE(q[start:end], k[start:end], v[start:end],
                                            None if mask is None else mask[start:end], causal, scale)
                out[start:end].copy_(result)
            return out
        dispatch._reference_attention = chunked
    try:
        yield
    finally:
        F.layer_norm, kernels.fused_residual_layer_norm, F.gelu = ORIGINAL_NORM, ORIGINAL_FUSED, ORIGINAL_GELU
        dispatch._reference_attention = ORIGINAL_REFERENCE


def samples(functions, repeats, rounds):
    raw, round_medians = {name: [] for name in functions}, []
    for fn in functions.values():
        for _ in range(3): fn()
    torch.cuda.synchronize()
    for round_index in range(rounds):
        names = list(functions)
        if round_index % 2: names.reverse()
        medians = {}
        for name in names:
            events = []
            for _ in range(repeats):
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record(); functions[name](); end.record()
                events.append((start, end))
            torch.cuda.synchronize()
            values = [a.elapsed_time(b) for a, b in events]
            raw[name].extend(values); medians[name] = statistics.median(values)
        round_medians.append(medians)
    medians = {name: statistics.median(values) for name, values in raw.items()}
    return {"raw_ms": raw, "round_medians_ms": round_medians, "median_ms": medians,
            "control_over_candidate": medians["control"] / medians["candidate"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("norm", "welford", "gelu", "chunk"), required=True)
    parser.add_argument("--amount", type=int, default=8)
    parser.add_argument("--tile", type=int, nargs=5, default=(32, 64, 32, 4, 2))
    parser.add_argument("--rows", type=int, nargs="+", default=(2, 3, 7, 8, 12))
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): parser.error("output already exists")
    if args.amount <= 0 or args.repeats <= 0 or args.rounds < 2: parser.error("invalid sampling or amount")
    torch.set_float32_matmul_precision("high")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("c15_*.py")}
    payload = {"status": "RUNNING", "environment": capture_environment(), "experiment_sources": hashes,
               "parameters": vars(args) | {"out": str(args.out)}, "results": []}
    cases = json.loads((ROOT / "benchmarks/final_evaluator_shapes.json").read_text())["explicit_cases"]
    if set(args.rows) - {c["source_row"] for c in cases if c["execution"] == "run"}:
        parser.error("unknown or non-executable row")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        if case["source_row"] not in args.rows: continue
        config = TransformerConfig(**case["config"])
        result = {"case_id": case["id"], "status": "RUNNING", "trials": []}
        payload["results"].append(result)
        for weight_seed, seeds in ((8080, ((2026, .25, 0.), (4096, 1., .25), (1701, 4., .25))),
                                    (1234, tuple((s, 1., 0.) for s in range(1234, 1239)))):
            torch.manual_seed(weight_seed)
            baseline = BaselineTransformer(config).cuda().eval()
            control = UserOptimizedTransformer(config).cuda().eval()
            candidate = UserOptimizedTransformer(config).cuda().eval()
            copy_model_weights(baseline, control, strict=True); copy_model_weights(baseline, candidate, strict=True)
            if args.kind == "gelu":
                for layer in candidate.layers:
                    linear = layer.ffn_in
                    linear.forward = lambda x, linear=linear: gemm_gelu(x, linear.weight, linear.bias, args.tile)
            def forward():
                with candidate_context(args.kind, args.amount): return candidate(x, mask)
            with torch.inference_mode():
                for seed, scale, padding in seeds:
                    x, mask = generate_random_case(config, torch.device("cuda"), torch.float32, seed, padding, scale)
                    reference, before, after = baseline(x, mask), control(x, mask), forward()
                    a, b = (compare_outputs(reference, value, atol=.001, rtol=.01) for value in (before, after))
                    result["trials"].append({"weight_seed": weight_seed, "input_seed": seed, "scale": scale,
                        "padding": padding, "compared_elements": b.total_elements,
                        "control_failed": a.failed_elements, "candidate_failed": b.failed_elements,
                        "candidate_max_abs": b.max_abs_error, "bitwise_equal_to_control": torch.equal(before, after)})
                    del reference, before, after
                    print(case["id"], result["trials"][-1], flush=True)
                if weight_seed == 1234:
                    # Retain diagnostics for rejected candidates, but no performance
                    # claim or promotion is allowed when any stress trial failed.
                    result["status"] = "REJECTED_ACCURACY" if any(t["candidate_failed"] for t in result["trials"]) else "PASS"
                    if result["status"] == "PASS":
                        result["timing"] = samples({"control": lambda: control(x, mask), "candidate": forward}, args.repeats, args.rounds)
                        print("TIMING", case["id"], result["timing"]["median_ms"], result["timing"]["control_over_candidate"], flush=True)
            del baseline, control, candidate, x, mask
            gc.collect(); torch.cuda.empty_cache()
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
    payload["status"] = "COMPLETE"
    args.out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__": main()
