"""Counterbalanced production-path A/B for the exact row-7 LayerNorm route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from transformer_opt.submission import (
    BaselineTransformer, TransformerConfig, UserOptimizedTransformer,
    compare_outputs, copy_model_weights, generate_random_case,
)
from tools.capture_environment import capture_environment


class PriorNormalization(UserOptimizedTransformer):
    def _use_short_layer_norm(self, norm, x):
        return False


def samples(functions, repeats, rounds):
    raw, round_medians = {name: [] for name in functions}, []
    for fn in functions.values():
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    for round_index in range(rounds):
        names = list(functions)
        if round_index % 2:
            names.reverse()
        medians = {}
        for name in names:
            events = []
            for _ in range(repeats):
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                functions[name]()
                end.record()
                events.append((start, end))
            torch.cuda.synchronize()
            values = [a.elapsed_time(b) for a, b in events]
            raw[name].extend(values)
            medians[name] = statistics.median(values)
        round_medians.append(medians)
    medians = {name: statistics.median(values) for name, values in raw.items()}
    return {"raw_ms": raw, "round_medians_ms": round_medians, "median_ms": medians,
            "control_over_candidate": medians["control"] / medians["candidate"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args(argv)
    if args.out.exists(): parser.error("output already exists")
    if args.repeats <= 0 or args.rounds < 2: parser.error("invalid sampling")
    torch.set_float32_matmul_precision("high")
    config = TransformerConfig(64, 128, 32, 4, 32, 4, True)
    payload = {"status": "RUNNING", "environment": capture_environment(),
               "parameters": {"repeats": args.repeats, "rounds": args.rounds,
                              "execution_mode": "auto", "input_copy_and_output_clone_included": True},
               "trials": []}
    for weight_seed, inputs in ((8080, ((2026, .25, 0.), (4096, 1., .25), (1701, 4., .25))),
                                (1234, tuple((s, 1., 0.) for s in range(1234, 1239)))):
        torch.manual_seed(weight_seed)
        baseline = BaselineTransformer(config).cuda().eval()
        control = PriorNormalization(config).cuda().eval()
        candidate = UserOptimizedTransformer(config).cuda().eval()
        copy_model_weights(baseline, control, strict=True)
        copy_model_weights(baseline, candidate, strict=True)
        with torch.inference_mode():
            for seed, scale, padding in inputs:
                x, mask = generate_random_case(config, torch.device("cuda"), torch.float32, seed, padding, scale)
                reference, before, after = baseline(x, mask), control(x, mask), candidate(x, mask)
                check = compare_outputs(reference, after, atol=.001, rtol=.01)
                identical = torch.equal(before, after)
                payload["trials"].append({"weight_seed": weight_seed, "input_seed": seed,
                    "scale": scale, "padding": padding, "failed_elements": check.failed_elements,
                    "compared_elements": check.total_elements, "bitwise_equal_to_control": identical})
                if not check.passed or not identical: raise RuntimeError("strict comparator or bitwise control gate failed")
            if weight_seed == 1234:
                control.reset_attention_backend_counts(); candidate.reset_attention_backend_counts()
                payload["timing"] = samples({"control": lambda: control(x, mask),
                                             "candidate": lambda: candidate(x, mask)}, args.repeats, args.rounds)
                payload["backend_counts"] = {"control": control.attention_backend_counts,
                                             "candidate": candidate.attention_backend_counts}
                payload["graph_replays"] = {"control": control.cuda_graph_replays,
                                            "candidate": candidate.cuda_graph_replays}
    payload["status"] = "PASS"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "timing": payload["timing"]["median_ms"],
                      "control_over_candidate": payload["timing"]["control_over_candidate"]}))


if __name__ == "__main__": main()
