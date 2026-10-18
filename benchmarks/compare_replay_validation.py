"""Current replay versus C15's repeated validation walks, with identical math.

Both variants retain all state fields, hooks, training/inference guards, copies,
clones and stream ordering. The control restores the previous validation walks
inside the same replay lock; it does not disable validation.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from benchmarks.compare_short_layer_norm import samples
from transformer_opt.submission import (BaselineTransformer, TransformerConfig, UserOptimizedTransformer,
    compare_outputs, copy_model_weights, generate_random_case)
from tools.capture_environment import capture_environment


def prior_signature(model, x, mask):
    parameters = tuple((id(p), p.data_ptr(), p._version, p.device, p.dtype) for p in model.parameters())
    modules = tuple((id(m), getattr(m, 'eps', None), getattr(m, 'normalized_shape', None)) for m in model.modules())
    attention = tuple((l.attention.d_model, l.attention.num_heads, l.attention.head_dim,
                       l.attention.scale) for l in model.layers)
    return (parameters, modules, attention, model.config, model.attention_backend,
            x.device, tuple(x.shape), x.dtype, mask is not None,
            torch.backends.cuda.matmul.allow_tf32, torch.is_inference_mode_enabled())


class PriorReplayValidation(UserOptimizedTransformer):
    def _graph_signature(self, x, valid_token_mask):
        if any(m.training or m._forward_hooks or m._forward_pre_hooks for m in self.modules()):
            return None
        if any(p.is_inference() for p in self.parameters()):
            return None
        return prior_signature(self, x, valid_token_mask)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=80)
    parser.add_argument('--rounds', type=int, default=4)
    args = parser.parse_args(argv)
    if args.out.exists(): parser.error('output already exists')
    if args.repeats <= 0 or args.rounds < 2: parser.error('invalid sampling')
    torch.set_float32_matmul_precision('high')
    payload = {'status': 'RUNNING', 'environment': capture_environment(),
               'control_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'parameters': {'repeats': args.repeats, 'rounds': args.rounds,
                              'validation_enabled_in_both': True, 'execution_mode': 'auto'}, 'results': []}
    cases = json.loads((ROOT/'benchmarks/final_evaluator_shapes.json').read_text())['explicit_cases']
    for case in cases:
        if case['source_row'] not in (1, 2, 3, 4, 7, 9, 10, 12): continue
        c = TransformerConfig(**case['config'])
        row = {'case_id': case['id'], 'trials': []}
        payload['results'].append(row)
        for weights, inputs in ((8080, ((2026, .25, 0.), (4096, 1., .25), (1701, 4., .25))),
                                (1234, tuple((seed, 1., 0.) for seed in range(1234, 1239)))):
            torch.manual_seed(weights)
            baseline = BaselineTransformer(c).cuda().eval()
            control = PriorReplayValidation(c).cuda().eval()
            candidate = UserOptimizedTransformer(c).cuda().eval()
            copy_model_weights(baseline, control, strict=True); copy_model_weights(baseline, candidate, strict=True)
            with torch.inference_mode():
                for seed, scale, padding in inputs:
                    x, mask = generate_random_case(c, torch.device('cuda'), torch.float32, seed, padding, scale)
                    ref, before, after = baseline(x, mask), control(x, mask), candidate(x, mask)
                    old, new = (compare_outputs(ref, y, atol=.001, rtol=.01) for y in (before, after))
                    identical = torch.equal(before, after)
                    assert identical and candidate._graph_signature(x, mask) == prior_signature(candidate, x, mask)
                    # Arithmetic is unchanged: retain known old-route stress
                    # failures, require zero on the normal final-matrix seeds.
                    if weights == 1234 and not new.passed: raise RuntimeError('final seed failed')
                    row['trials'].append({'weight_seed': weights, 'seed': seed, 'scale': scale,
                        'padding': padding, 'control_failed': old.failed_elements,
                        'candidate_failed': new.failed_elements, 'compared': new.total_elements,
                        'bitwise_equal_to_control': identical, 'signature_equal_to_prior': True})
                if weights == 1234:
                    control.reset_attention_backend_counts(); candidate.reset_attention_backend_counts()
                    row['timing'] = samples({'control': lambda: control(x, mask),
                                             'candidate': lambda: candidate(x, mask)}, args.repeats, args.rounds)
                    cpu = {}
                    for name, item in (('control', control), ('candidate', candidate)):
                        start = time.perf_counter_ns()
                        for _ in range(1000): item._graph_signature(x, mask)
                        cpu[name] = (time.perf_counter_ns() - start) / 1e6
                    row['signature_cpu_us_per_call'] = cpu
                    row['backend_counts'] = {'control': control.attention_backend_counts,
                                             'candidate': candidate.attention_backend_counts}
                    row['graph_replays'] = {'control': control.cuda_graph_replays, 'candidate': candidate.cuda_graph_replays}
                    print(case['id'],row['timing']['control_over_candidate'],cpu,flush=True)
        del baseline, control, candidate, x, mask, ref, before, after
        gc.collect(); torch.cuda.empty_cache()
    payload['status'] = 'PASS'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload,indent=2)+'\n')
    return 0

if __name__ == '__main__': raise SystemExit(main())
