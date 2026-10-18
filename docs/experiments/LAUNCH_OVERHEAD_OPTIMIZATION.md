# Campaign 13 — launch overhead and guarded tensor-core precision

> Historical record: [Campaign 16 replay validation](CAMPAIGN-016.md) is the current selection. Measurements below retain Campaign 13's original source identity.

Date: 2026-10-18 (Asia/Singapore). Decision: **keep bounded replay and two
precision routes; the 10x aggregate target remains unmet**.

Selected schema-2 implementation identity:
`de8b6dd8831cb8637ac49cf8f2acfdcf69219e2cb9c303f60af81d123f34e868`.
Campaign 12 artifacts remain unchanged at `abb639ce...`; they are historical,
not relabeled as current-source evidence.

## Objective and constraints

The requested target was 10x geometric-mean end-to-end speedup across the final
matrix. We retained the frozen explicit PyTorch baseline, dimensions, strict
`abs_error <= 0.001 OR abs_error <= 0.01 * abs(reference)` comparator, five
accuracy seeds, and the single authorized resource skip. An individual row
above 10x does not satisfy the aggregate target.

## Accepted implementation

- **CUDA replay:** exact final rows 1, 2, 3, 4, 7, 9, 10, and 12 capture the
  existing forward. Every call still checks state, copies current inputs/masks,
  replays GPU work, and clones an independent output. Signature changes force
  recapture; device/dtype moves clear both caches. A lock and completion event
  order static-buffer reuse across caller streams. Setup calls do not inflate
  observed backend counters. Other shapes, training, gradients, hooks, autocast,
  compilation, and outer capture retain eager execution. Capture errors surface.
- **Attention precision:** exact float32 BSHD shapes `(64,128,1,128)` and
  `(64,1024,4,32)` on SM120 use Triton 3.7.1's `tf32x3` mode for QK and P@V.
  Ordinary causal TF32 remains disabled. Other shapes/devices retain the prior
  precision and all accuracy-safe backend/hybrid routes remain unchanged.

Implementation: [submission](../../transformer_opt/submission.py),
[attention](../../transformer_opt/kernels/attention.py),
[state/stream regression tests](../../tests/test_cuda_graph_inference.py).
Use `execution_mode="eager"` for the unchanged-math control and
`clear_cuda_graph()` to release the single capture.

## Controlled measurements

The [graph A/B](../results/rtx-5070-ti-2026-10-18-c13-graph-ab.json) checks five
different input seeds against both the baseline and eager model. All eight
captured cases have zero failed elements and bitwise-equal eager/replay outputs.
Each variant retains 320 CUDA-event samples across four alternating rounds.
Input copies, output clones, and signature checks are timed; capture is setup.

| Final row | Eager / replay median ratio |
| --- | ---: |
| 1 | 1.170x |
| 2 | 5.014x |
| 3 | 4.247x |
| 4 | 2.632x |
| 7 | 1.443x |
| 9 | 1.078x |
| 10 | 1.136x |
| 12 | 3.735x |

Every row's four per-round ratios exceed one. The geometric mean over these
eight A/B cases is 2.136x; this is not the 13-row baseline-to-optimized score.
The first eligible candidate call retains 32.945–44.765 MiB in this experiment,
including the returned output, packed weights, vendor workspace, and capture
storage. This is not an isolated graph-memory measurement. Replay therefore
trades additional persistent memory for lower launch overhead.

The [precision A/B](../results/rtx-5070-ti-2026-10-18-c13-precision-ab.json)
uses identical eager models and reconstructs an IEEE control by disabling only
the new precision guard. Four alternating rounds retain 160 samples per variant.
Row 9 improves **1.105x**, and row 13 improves **1.182x**. Both pass five baseline
accuracy seeds. The other three direct-kernel control shapes remain bitwise
equal; the changed direct row-9 shape passes the strict comparator.

## Complete-model results

| Run | Correctness | Final-matrix geometric mean |
| --- | --- | ---: |
| [Primary](../results/rtx-5070-ti-2026-10-18-c13-final.json) | 13/13 executable PASS; zero failed elements | 4.186514x |
| [Confirmation](../results/rtx-5070-ti-2026-10-18-c13-final-confirmation.json) | Same complete matrix and correctness | 3.744398x |

Each matrix compares 938,885,120 output elements over five seeds per row. The
100,000-token case is excluded from pass and timing aggregates. Backend totals
remain 1,260 Triton / 0 SDPA / 196 reference calls, including graph replays.

The 32-token row 12 measures **7.768x / 6.786x**.
Row 11 measures **15.882x / 8.047x**. These individual
results do not establish 10x across the matrix. The wide row 8 measures
**1.053x / 1.034x**, with small gains. Large-batch optimized
latency is **2,653.790 ms / 2,453.685 ms**; its baseline varies too.
That unchanged eager route measures **0.749x / 0.950x**,
including a regression. We did not establish the cause of the timing drift.

Campaign 12's 2.422–2.776x full-matrix pair is useful history, but comparing
separate-process geometric means does not isolate the causal gain from either
new optimization. The two same-process A/B gates provide that evidence.

## Rejected and narrowed experiments

| Experiment | Observation | Decision |
| --- | --- | --- |
| Replay for larger cases | Rows 5, 8, 11, and 13 lacked enough consistent benefit to justify retained storage; row 6 is memory intensive | Keep those rows eager |
| TF32x3 for every causal attention shape | All 13 strict five-seed cases passed, but several small-row timings regressed or varied by round | Admit only consistently improving rows 9 and 13 |
| Custom tensor-core linear layers throughout | Failed two output elements on final row 1 | Reject; no speed claim |
| Custom linears with vendor packed-QKV retained | Failed one row-1 element at seed 1235 | Reject |
| Custom linears after an exact first layer | Failed one row-1 element at seed 1238 | Reject and stop that approach |

Exploratory scripts/results remain under ignored `results/c13*`. An early
prototype also stopped while recording environment metadata after mixing the
legacy TF32 toggle with the newer precision setter. Subsequent canonical runs
use only `torch.set_float32_matmul_precision("high")`; failed prototype output
is not used as a retained benchmark claim. No baseline/comparator relaxation,
output memoization, input-specific answer caching, or new resource skip was used.

## Validation and reproduction

Final review reproduced two cache bugs: inference-mode buffers could be reused
under `no_grad`, and changed attention scale could leave stale captured math.
Both new regression tests failed before the fix. The signature now includes
inference context, attention configuration, and normalization shape as well as
parameter versions and epsilon. The seven focused cache tests pass after the
fix. Pre-review artifacts remain byte-preserved under ignored
`results/c13-pre-review/`; the selected evidence below is remeasured on the final
source rather than relabeling those measurements.

The [held-out run](../results/rtx-5070-ti-2026-10-18-c13-heldout.json) is 7/7 PASS.
The [source-derived run](../results/rtx-5070-ti-2026-10-18-c13-source-derived.json)
is 28/28 executable PASS, with zero failed elements across 459,776,000 comparisons
and the one authorized resource skip. Fresh row-5/6/9/11/13 profiles pass explicit
custom-kernel expectations; row 9 proves 12 attention and 24 residual/LayerNorm
kernel launches across three actual graph replays. The
[validation record](../results/optimization-validation-2026-10-18.json)
retains exact commands, test outcomes, and artifact hashes.
Historical Campaign 12 artifact hashes remain covered by regression tests.

```powershell
& .venv\Scripts\python.exe benchmarks\compare_cuda_graph.py --out results\graph-ab.json
& .venv\Scripts\python.exe benchmarks\compare_attention_precision.py --out results\precision-ab.json
& .venv\Scripts\python.exe benchmarks\run_organizer_validation.py --matrix benchmarks\final_evaluator_shapes.json --out results\final.json
& .venv\Scripts\python.exe -m pytest tests -q
```

Current weak cases are the reference-attention width-1024 model and very large
batch. Further progress requires a new measured approach to their compute or
memory costs and the strict numerical boundary encountered by custom linears.
No forecast establishes that those changes can reach 10x. All work is local;
no commit, branch, push, deployment, or publication was performed.
