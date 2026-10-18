"""A/B a causal loop bound without modifying the selected implementation.

This retains a bounded hypothesis as a reproducible experiment.
The candidate is generated in a temporary module from the current kernel;
production dispatch and the organizer benchmark are never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from transformer_opt import dispatch
from transformer_opt.kernels import attention
from transformer_opt.submission import (
    BaselineTransformer,
    TransformerConfig,
    UserOptimizedTransformer,
    compare_outputs,
    copy_model_weights,
    generate_random_case,
)
from tools.capture_environment import capture_environment


OLD_LOOP = "        for start_n in range(0, N_CTX, BLOCK_N):"
NEW_LOOP = """        end_n = N_CTX
        if CAUSAL:
            end_n = tl.minimum(N_CTX, (query_block + 1) * BLOCK_M)
        for start_n in range(0, end_n, BLOCK_N):"""


def candidate_source(source: str) -> str:
    if source.count(OLD_LOOP) != 1:
        raise ValueError("expected exactly one unpruned attention loop")
    return source.replace(OLD_LOOP, NEW_LOOP)


def alternating_samples(functions, *, repeats: int, rounds: int):
    samples = {name: [] for name in functions}
    for fn in functions.values():
        for _ in range(10):
            fn()
    torch.cuda.synchronize()
    for round_index in range(rounds):
        names = list(functions)
        if round_index % 2:
            names.reverse()
        for name in names:
            events = []
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                functions[name]()
                end.record()
                events.append((start, end))
            torch.cuda.synchronize()
            samples[name].extend(start.elapsed_time(end) for start, end in events)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {
        "raw_ms": samples,
        "median_ms": medians,
        "control_over_candidate": medians["control"] / medians["candidate"],
    }


def run_experiment(control, candidate, repeats: int, rounds: int, selected_cases=None):
    results = []
    # Final rows plus a longer direct-kernel regime and a held-out model.
    shapes = [(64, 128, 4, 32), (64, 128, 1, 128),
              (64, 128, 16, 8), (2, 512, 8, 64)]
    for shape in shapes:
        torch.manual_seed(1234)
        q = torch.randn(shape, device="cuda")
        k, v = torch.randn_like(q), torch.randn_like(q)
        actual, expected = candidate(q, k, v, causal=True), control(q, k, v, causal=True)
        accuracy = compare_outputs(expected, actual, atol=0.001, rtol=0.01)
        if not accuracy.passed:
            raise RuntimeError(f"candidate/control correctness failure: {shape}")
        functions = {
            "control": lambda: control(q, k, v, causal=True),
            "candidate": lambda: candidate(q, k, v, causal=True),
        }
        result = {"kind": "direct_attention", "shape_bshd": shape,
                  "bitwise_equal_to_control": torch.equal(actual, expected),
                  **alternating_samples(functions, repeats=repeats, rounds=rounds)}
        results.append(result)
        print(f"direct {shape}: {result['control_over_candidate']:.3f}x", flush=True)

    matrix = json.loads((ROOT / "benchmarks/final_evaluator_shapes.json").read_text())
    cases = [case for case in matrix["explicit_cases"] if case["execution"] == "run"]
    cases.append({"id": "heldout-long-causal", "config": {
        "batch_size": 2, "seq_len": 512, "d_model": 512, "num_heads": 8,
        "ffn_dim": 2048, "num_layers": 2, "causal": True}})
    if selected_cases:
        unknown = set(selected_cases) - {case["id"] for case in cases}
        if unknown:
            raise ValueError(f"unknown model cases: {sorted(unknown)}")
        cases = [case for case in cases if case["id"] in selected_cases]
    else:
        cases = [case for case in cases if case.get("source_row") in (1, 5, 9, 11)
                 or case["id"] == "heldout-long-causal"]
    for case in cases:
        config = TransformerConfig(**case["config"])
        torch.manual_seed(1234)
        # Parameters need version counters for packed-QKV cache invalidation.
        with torch.inference_mode(False):
            baseline = BaselineTransformer(config).cuda().eval()
            # Kernel A/B swaps Python callables. Capturing one variant and
            # replaying it for both would invalidate the causal comparison.
            model = UserOptimizedTransformer(config, execution_mode="eager").cuda().eval()
            copy_model_weights(baseline, model, strict=True)

        def forward(fn):
            dispatch.triton_attention = fn
            return model(x, mask)

        comparisons = 0
        for seed in (1234, 1235, 1236, 1237, 1238):
            x, mask = generate_random_case(config, torch.device("cuda"), torch.float32,
                                           seed, 0.0, 1.0)
            accuracy = compare_outputs(baseline(x, mask), forward(candidate),
                                       atol=0.001, rtol=0.01)
            if not accuracy.passed:
                raise RuntimeError(f"candidate/reference correctness failure: {case['id']}")
            comparisons += x.numel()
        model.reset_attention_backend_counts()
        result = {
            "kind": "end_to_end", "case_id": case["id"], "config": case["config"],
            "accuracy_trials": 5, "compared_elements": comparisons, "failed_elements": 0,
            **alternating_samples({"control": lambda: forward(control),
                                   "candidate": lambda: forward(candidate)},
                                  repeats=repeats, rounds=rounds),
            "attention_backend_counts": dict(model.attention_backend_counts),
        }
        results.append(result)
        print(f"model {case['id']}: {result['control_over_candidate']:.3f}x", flush=True)
        del baseline, model, x, mask
        torch.cuda.empty_cache()
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--case", action="append", dest="cases",
                        help="exact final case ID; repeat to select multiple model cases")
    args = parser.parse_args(argv)
    if args.repeats <= 0 or args.rounds < 2:
        parser.error("repeats must be positive and rounds must be at least two")
    if args.out.exists():
        parser.error("output already exists; choose a new path to preserve evidence")
    if not attention.triton_available():
        parser.error("this experiment requires CUDA and Triton")
    source = Path(attention.__file__).read_text(encoding="utf-8")
    # Reconstruct the exact pre-change control after the candidate is promoted.
    if source.count(NEW_LOOP) == 1 and OLD_LOOP not in source:
        source = source.replace(NEW_LOOP, OLD_LOOP)
    candidate = candidate_source(source)
    control = dispatch.triton_attention
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        # Match the organizer's default; causal Triton dots also force IEEE.
        torch.backends.cuda.matmul.allow_tf32 = False
        with tempfile.TemporaryDirectory(prefix="causal-frontier-") as tmp:
            modules = {}
            for name, content in (("frontier_control", source),
                                  ("frontier_candidate", candidate)):
                path = Path(tmp) / f"{name}.py"
                path.write_text(content, encoding="utf-8")
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                modules[name] = module
            with torch.inference_mode():
                results = run_experiment(modules["frontier_control"].triton_attention,
                                         modules["frontier_candidate"].triton_attention,
                                         args.repeats, args.rounds, args.cases)
            payload = {
                "schema_version": 1, "status": "PASS", "decision": "unreviewed",
                "environment": capture_environment(),
                "control_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "candidate_source_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
                "transformation": {"old": OLD_LOOP, "new": NEW_LOOP},
                "parameters": {"warmup": 10, "repeats": args.repeats,
                               "rounds": args.rounds, "allow_tf32": False,
                               "selected_cases": args.cases},
                "results": results,
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
    finally:
        dispatch.triton_attention = control
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        sys.modules.pop("frontier_control", None)
        sys.modules.pop("frontier_candidate", None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
