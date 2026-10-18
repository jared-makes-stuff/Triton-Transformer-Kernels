"""A/B the two guarded TF32x3 attention routes against identical IEEE dots."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from benchmarks.compare_causal_frontier import run_experiment
from tools.capture_environment import capture_environment
from transformer_opt import dispatch
from transformer_opt.kernels import attention


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output already exists; preserve recorded evidence")
    if args.repeats <= 0 or args.rounds < 2:
        parser.error("repeats must be positive; rounds must be at least two")
    candidate = Path(attention.__file__).read_text(encoding="utf-8")
    # Campaign 14's short-sequence kernel takes precedence over the historical
    # TF32x3 route. Disable that newer launch in BOTH generated modules so this
    # experiment still compares the advertised IEEE and three-product dots.
    rounded = "if prefer_rounded_attention(q, valid_token_mask, causal):"
    if candidate.count(rounded) != 1:
        raise ValueError("expected exactly one rounded-attention launch guard")
    candidate = candidate.replace(rounded, "if False and prefer_rounded_attention(q, valid_token_mask, causal):")
    original = "PRECISE_TENSOR_CORES=(\n"
    if candidate.count(original) != 1:
        raise ValueError("expected exactly one guarded precision argument")
    control = candidate.replace(original, "PRECISE_TENSOR_CORES=(False and\n")
    previous = dispatch.triton_attention
    torch.set_float32_matmul_precision("high")
    try:
        with tempfile.TemporaryDirectory(prefix="attention-precision-") as tmp:
            modules = {}
            for name, content in (("precision_control", control), ("precision_candidate", candidate)):
                path = Path(tmp) / f"{name}.py"
                path.write_text(content, encoding="utf-8")
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                modules[name] = module
            with torch.inference_mode():
                results = run_experiment(
                    modules["precision_control"].triton_attention,
                    modules["precision_candidate"].triton_attention,
                    args.repeats, args.rounds,
                    ["final-09-b64-d128-h1-s128", "final-13-b64-d128-h4-s1024"],
                )
            payload = {
                "schema_version": 1, "status": "PASS", "environment": capture_environment(),
                "control_source_sha256": hashlib.sha256(control.encode()).hexdigest(),
                "candidate_source_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
                "parameters": {"repeats": args.repeats, "rounds": args.rounds,
                               "warmup": 10, "execution_mode": "eager",
                               "rounded_attention_disabled_in_both_variants": True},
                "results": results,
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
    finally:
        dispatch.triton_attention = previous
        for name in ("precision_control", "precision_candidate"):
            sys.modules.pop(name, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
