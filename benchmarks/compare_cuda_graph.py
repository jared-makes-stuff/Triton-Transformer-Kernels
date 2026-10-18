"""Compare automatic CUDA-graph inference with identical eager model math.

The explicit organizer baseline is unchanged. Capture/setup is excluded from
steady-state timing; input copies, output clones, and invalidation checks are
included. Raw CUDA-event samples alternate order in each round.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from benchmarks.compare_causal_frontier import alternating_samples
from benchmarks.torch_transformer_benchmark import generate_random_case
from tools.capture_environment import capture_environment
from transformer_opt.submission import (
    BaselineTransformer, TransformerConfig, UserOptimizedTransformer,
    compare_outputs, copy_model_weights,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output already exists; preserve recorded evidence")
    if args.repeats <= 0 or args.rounds < 2:
        parser.error("repeats must be positive; rounds must be at least two")
    matrix = json.loads((ROOT / "benchmarks/final_evaluator_shapes.json").read_text())
    cases = [case for case in matrix["explicit_cases"] if case["execution"] == "run"]
    unknown = set(args.cases or []) - {case["id"] for case in cases}
    if unknown:
        parser.error(f"unknown cases: {sorted(unknown)}")
    # Other rows intentionally retain eager execution and need no capture A/B.
    cases = [case for case in cases if case["id"] in args.cases] if args.cases else [
        case for case in cases if case["source_row"] not in (5, 6, 8, 11, 13)
    ]
    torch.set_float32_matmul_precision("high")
    environment = capture_environment()
    results = []
    for case in cases:
        config = TransformerConfig(**case["config"])
        torch.manual_seed(1234)
        baseline = BaselineTransformer(config).cuda().eval()
        control = UserOptimizedTransformer(config, execution_mode="eager").cuda().eval()
        candidate = UserOptimizedTransformer(config).cuda().eval()
        copy_model_weights(baseline, control, strict=True)
        copy_model_weights(baseline, candidate, strict=True)
        with torch.inference_mode():
            compared = failed = 0
            identical = True
            start = time.perf_counter()
            for seed in range(1234, 1239):
                x, mask = generate_random_case(config, torch.device("cuda"),
                                               torch.float32, seed, 0.0, 1.0)
                allocated_before_candidate = torch.cuda.memory_allocated()
                actual = candidate(x, mask)
                if seed == 1234:
                    # Packed weights, vendor workspace, capture storage, and
                    # the returned output; this is not isolated graph memory.
                    first_call_retained_bytes = (
                        torch.cuda.memory_allocated() - allocated_before_candidate
                    )
                check = compare_outputs(baseline(x, mask), actual, rtol=.01, atol=.001)
                failed += check.failed_elements
                compared += check.total_elements
                identical &= torch.equal(actual, control(x, mask))
            correctness_and_setup_seconds = time.perf_counter() - start
            if failed or not identical:
                raise RuntimeError(f"{case['id']}: failed={failed}; bitwise eager={identical}")
            candidate.reset_attention_backend_counts()
            control.reset_attention_backend_counts()
            timing = alternating_samples(
                {"control": lambda: control(x, mask),
                 "candidate": lambda: candidate(x, mask)},
                repeats=args.repeats, rounds=args.rounds,
            )
            result = {
                "case_id": case["id"], "config": case["config"], "status": "PASS",
                "accuracy_trials": 5, "compared_elements": compared,
                "failed_elements": failed, "bitwise_equal_to_eager": identical,
                "correctness_and_setup_seconds": correctness_and_setup_seconds,
                "candidate_first_call_retained_bytes_including_output": first_call_retained_bytes,
                **timing,
                "candidate_backend_counts": dict(candidate.attention_backend_counts),
                "control_backend_counts": dict(control.attention_backend_counts),
                "cuda_graph_replays": candidate.cuda_graph_replays,
            }
            results.append(result)
            print(f"{case['id']}: {timing['control_over_candidate']:.3f}x eager/graph; "
                  f"{timing['median_ms']}", flush=True)
        del baseline, candidate, control, x, mask, actual
        gc.collect()
        torch.cuda.empty_cache()
    payload = {
        "schema_version": 1, "status": "PASS", "environment": environment,
        "parameters": {"repeats": args.repeats, "rounds": args.rounds, "warmup": 10,
                       "selected_cases": args.cases, "accuracy_trials": 5,
                       "setup_excluded": True, "input_copy_and_output_clone_included": True},
        "results": results,
        "geometric_mean_eager_over_graph": statistics.geometric_mean(
            result["control_over_candidate"] for result in results
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
