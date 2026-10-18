"""A/B rounded, normalized-probability attention against the prior eager route.

The control disables only ``prefer_rounded_attention`` in both dispatch sites,
restoring the previous kernel and reference-layer policy. No baseline or model
weights change. Separate model instances isolate each variant's optional graph.
All elements and five input seeds are checked before retaining timings.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from tools.capture_environment import capture_environment
from transformer_opt.kernels import attention
from transformer_opt.submission import (
    BaselineTransformer, TransformerConfig, UserOptimizedTransformer,
    compare_outputs, copy_model_weights, generate_random_case,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--large-repeats", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--execution-mode", choices=("auto", "eager"), default="eager")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output already exists; preserve recorded evidence")
    if min(args.repeats, args.large_repeats) <= 0 or args.rounds < 2:
        parser.error("repeats must be positive; rounds must be at least two")
    matrix = json.loads((ROOT / "benchmarks/final_evaluator_shapes.json").read_text())
    cases = [case for case in matrix["explicit_cases"] if case["execution"] == "run"]
    unknown = set(args.cases or []) - {case["id"] for case in cases}
    if unknown:
        parser.error(f"unknown cases: {sorted(unknown)}")
    cases = [case for case in cases if case["id"] in args.cases] if args.cases else [
        case for case in cases if case["source_row"] in (2, 3, 7, 12)
    ]
    if not attention.triton_available():
        parser.error("CUDA and Triton are required")
    torch.set_float32_matmul_precision("high")
    environment = capture_environment()
    source = Path(attention.__file__).read_bytes().replace(b"\r\n", b"\n")
    predicate = attention.prefer_rounded_attention
    disabled = lambda *args, **kwargs: False
    results = []
    try:
        for case in cases:
            config = TransformerConfig(**case["config"])
            torch.manual_seed(1234)
            baseline = BaselineTransformer(config).cuda().eval()
            models = {enabled: UserOptimizedTransformer(
                config, execution_mode=args.execution_mode,
            ).cuda().eval() for enabled in (False, True)}
            for model in models.values():
                copy_model_weights(baseline, model, strict=True)

            def forward(enabled):
                attention.prefer_rounded_attention = predicate if enabled else disabled
                return models[enabled](x, mask)

            with torch.inference_mode():
                trials = []
                for seed in range(1234, 1239):
                    x, mask = generate_random_case(config, torch.device("cuda"),
                                                   torch.float32, seed, 0., 1.)
                    expected = baseline(x, mask)
                    actual = forward(True)
                    check = compare_outputs(expected, actual, rtol=.01, atol=.001)
                    trials.append({"seed": seed, "failed_elements": check.failed_elements,
                                   "compared_elements": check.total_elements,
                                   "max_abs_error": check.max_abs_error})
                    if not check.passed:
                        raise RuntimeError(f"{case['id']}: failed={check.failed_elements}")
                    del expected, actual
                memory = {}
                for name, enabled in (("control", False), ("candidate", True)):
                    forward(enabled)
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    output = forward(enabled)
                    torch.cuda.synchronize()
                    memory[name] = {"allocated_before_bytes": before,
                                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                    "incremental_peak_bytes": torch.cuda.max_memory_allocated() - before}
                    del output
                repeats = args.large_repeats if case["source_row"] == 6 else args.repeats
                raw = {"control": [], "candidate": []}
                per_round = []
                counts = {}
                for enabled in (False, True):
                    for _ in range(3):
                        forward(enabled)
                torch.cuda.synchronize()
                for round_index in range(args.rounds):
                    order = [("control", False), ("candidate", True)]
                    if round_index % 2:
                        order.reverse()
                    round_medians = {}
                    for name, enabled in order:
                        models[enabled].reset_attention_backend_counts()
                        events = []
                        for _ in range(repeats):
                            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                            start.record()
                            forward(enabled)
                            end.record()
                            events.append((start, end))
                        torch.cuda.synchronize()
                        values = [start.elapsed_time(end) for start, end in events]
                        raw[name].extend(values)
                        round_medians[name] = statistics.median(values)
                        counts[name] = dict(models[enabled].attention_backend_counts)
                    per_round.append(round_medians["control"] / round_medians["candidate"])
                medians = {name: statistics.median(values) for name, values in raw.items()}
                result = {"case_id": case["id"], "config": case["config"], "status": "PASS",
                          "accuracy_trials": trials, "failed_elements": 0,
                          "compared_elements": sum(t["compared_elements"] for t in trials),
                          "raw_ms": raw, "median_ms": medians,
                          "control_over_candidate": medians["control"] / medians["candidate"],
                          "per_round_control_over_candidate": per_round,
                          "memory": memory, "backend_counts_last_round": counts}
                results.append(result)
                print(f"{case['id']}: {result['control_over_candidate']:.3f}x; "
                      f"rounds={per_round}", flush=True)
            del baseline, model, models, x, mask
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        attention.prefer_rounded_attention = predicate
    payload = {"schema_version": 1, "status": "PASS", "environment": environment,
               "attention_source_sha256": hashlib.sha256(source).hexdigest(),
               "control_transformation": "prefer_rounded_attention returns False at both dispatch sites",
               "parameters": {"execution_mode": args.execution_mode, "precision": "high",
                              "accuracy_seeds": list(range(1234, 1239)), "warmup": 3,
                              "repeats": args.repeats, "large_repeats": args.large_repeats,
                              "rounds": args.rounds, "selected_cases": args.cases},
               "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
