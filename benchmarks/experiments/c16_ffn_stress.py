"""Additional independent-weight gate for the only promising fused-FFN shape."""
import hashlib, json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from benchmarks.experiments.c16_screen import FastFusedFFN
from benchmarks.experiments.c16_candidates import FastValidation
from transformer_opt.submission import (BaselineTransformer, TransformerConfig, copy_model_weights,
    compare_outputs, generate_random_case)
from tools.capture_environment import capture_environment

torch.set_float32_matmul_precision('high')
out = Path(sys.argv[1])
if out.exists(): raise FileExistsError(out)
payload = {'status': 'RUNNING', 'environment': capture_environment(),
           'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'trials': []}
config = TransformerConfig(64, 128, 32, 4, 32, 4, True)
for weight_seed in (0, 1701, 4096, 8080):
    torch.manual_seed(weight_seed)
    baseline = BaselineTransformer(config).cuda().eval()
    control = FastValidation(config).cuda().eval()
    candidate = FastFusedFFN(config).cuda().eval(); candidate.tile = (16, 32, 32, 4, 1)
    copy_model_weights(baseline, control, strict=True); copy_model_weights(baseline, candidate, strict=True)
    with torch.inference_mode():
        for scale in (.125, .25, 1., 4.):
            for padding in (0., .25):
                x, mask = generate_random_case(config, torch.device('cuda'), torch.float32, 5050, padding, scale)
                reference, before, after = baseline(x, mask), control(x, mask), candidate(x, mask)
                old, new = (compare_outputs(reference, y, atol=.001, rtol=.01) for y in (before, after))
                trial = {'weight_seed': weight_seed, 'input_seed': 5050, 'scale': scale,
                         'padding': padding, 'control_failed': old.failed_elements,
                         'candidate_failed': new.failed_elements, 'compared': new.total_elements}
                payload['trials'].append(trial); print(trial, flush=True)
payload['status'] = 'PASS' if all(t['candidate_failed'] == 0 for t in payload['trials']) else 'FAIL'
out.write_text(json.dumps(payload, indent=2)+'\n')
