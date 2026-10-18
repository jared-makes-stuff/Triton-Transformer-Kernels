"""Stress-first A/B of complete production replay versus isolated C16 candidates."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from benchmarks.compare_short_layer_norm import samples
from benchmarks.experiments.c16_candidates import FastValidation, FusedFFN
from transformer_opt.submission import (BaselineTransformer, TransformerConfig, UserOptimizedTransformer,
    copy_model_weights, compare_outputs, generate_random_case)
from tools.capture_environment import capture_environment


class FastFusedFFN(FusedFFN, FastValidation):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('validation', 'ffn', 'ffn-fast'), required=True)
    parser.add_argument('--tile', type=int, nargs=5, default=(16, 64, 32, 4, 1))
    parser.add_argument('--rows', type=int, nargs='+', default=(2, 3, 7, 12))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): parser.error('output already exists')
    torch.set_float32_matmul_precision('high')
    Candidate = {'validation': FastValidation, 'ffn': FusedFFN, 'ffn-fast': FastFusedFFN}[args.kind]
    Control = FastValidation if args.kind == 'ffn-fast' else UserOptimizedTransformer
    payload = {'status': 'RUNNING', 'environment': capture_environment(),
               'parameters': vars(args) | {'out': str(args.out)},
               'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in Path(__file__).parent.glob('c16_*.py')}, 'results': []}
    cases = json.loads((ROOT/'benchmarks/final_evaluator_shapes.json').read_text())['explicit_cases']
    for case in cases:
        if case['source_row'] not in args.rows: continue
        c = TransformerConfig(**case['config'])
        result = {'case_id': case['id'], 'trials': []}
        payload['results'].append(result)
        for weights, inputs in ((8080, ((2026, .25, 0.), (4096, 1., .25), (1701, 4., .25))),
                                (1234, tuple((s, 1., 0.) for s in range(1234, 1239)))):
            torch.manual_seed(weights)
            baseline = BaselineTransformer(c).cuda().eval()
            control = Control(c).cuda().eval()
            candidate = Candidate(c).cuda().eval()
            candidate.tile = args.tile
            copy_model_weights(baseline, control, strict=True); copy_model_weights(baseline, candidate, strict=True)
            with torch.inference_mode():
                for seed, scale, padding in inputs:
                    x, mask = generate_random_case(c, torch.device('cuda'), torch.float32, seed, padding, scale)
                    ref, before, after = baseline(x, mask), control(x, mask), candidate(x, mask)
                    check, old = (compare_outputs(ref, y, atol=.001, rtol=.01) for y in (after, before))
                    result['trials'].append({'weight_seed': weights, 'seed': seed, 'scale': scale,
                        'padding': padding, 'failed': check.failed_elements, 'control_failed': old.failed_elements,
                        'compared': check.total_elements, 'bitwise_control': torch.equal(before, after)})
                    if args.kind == 'validation':
                        assert candidate._validated_signature(x, mask) == candidate._graph_signature(x, mask)
                    print(case['id'],result['trials'][-1],flush=True)
                    del ref, before, after
                if weights == 1234:
                    result['status'] = 'REJECTED_ACCURACY' if any(t['failed'] for t in result['trials']) else 'PASS'
                    if result['status'] == 'PASS':
                        result['timing'] = samples({'control': lambda: control(x, mask),
                                                    'candidate': lambda: candidate(x, mask)}, 80, 4)
                        print('TIMING',case['id'],result['timing']['median_ms'],result['timing']['control_over_candidate'],flush=True)
            del baseline, control, candidate, x, mask
            gc.collect(); torch.cuda.empty_cache()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload,indent=2)+'\n')
    payload['status'] = 'COMPLETE'
    args.out.write_text(json.dumps(payload,indent=2)+'\n')

if __name__ == '__main__': main()
