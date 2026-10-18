"""Profile replay validation cost; unchecked replay is a diagnostic, not a route."""
import argparse
import cProfile
import hashlib
import io
import json
from pathlib import Path
import pstats
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from benchmarks.compare_short_layer_norm import samples
from transformer_opt.submission import TransformerConfig, UserOptimizedTransformer
from tools.capture_environment import capture_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): parser.error('output already exists')
    torch.set_float32_matmul_precision('high')
    payload = {'status': 'RUNNING', 'environment': capture_environment(),
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'unchecked_replay_is_selectable': False, 'results': []}
    for batch, width in ((1, 128), (64, 32), (64, 128)):
        model = UserOptimizedTransformer(TransformerConfig(batch, 128, width, 4, width, 4, True)).cuda().eval()
        x = torch.randn(batch, 128, width, device='cuda')
        with torch.inference_mode():
            model(x); entry = model._inference_graph
            cpu = {}
            for name, fn in {'guard': lambda: model._can_replay_cuda_graph(x, None),
                             'signature': lambda: model._graph_signature(x, None)}.items():
                values = []
                for _ in range(8):
                    start = time.perf_counter_ns()
                    for _ in range(1000): fn()
                    values.append((time.perf_counter_ns() - start) / 1000 / 1000)
                cpu[name] = {'microseconds_per_call': values, 'median_us': statistics.median(values)}
            profile = cProfile.Profile()
            profile.enable()
            for _ in range(1000): model._graph_signature(x, None)
            profile.disable()
            stream = io.StringIO()
            pstats.Stats(profile, stream=stream).sort_stats('cumtime').print_stats(16)
            timing = samples({'control': lambda: model(x), 'candidate': lambda: entry.replay(x, None)}, 80, 4)
            row = {'shape': list(x.shape), 'cpu_validation': cpu,
                   'signature_cprofile': stream.getvalue(), 'diagnostic_only_timing': timing}
            payload['results'].append(row)
            print(row['shape'],cpu,'complete_over_unchecked',timing['control_over_candidate'],flush=True)
        del model, x, entry
        torch.cuda.empty_cache()
    payload['status'] = 'COMPLETE'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2)+'\n')

if __name__ == '__main__': main()
