# Campaign 14 — explicitly rounded short-sequence attention

> Historical record: [Campaign 16 replay validation](CAMPAIGN-016.md) is the current selection. Campaign 14's measurements and rejected prototypes remain attached to their original identities.

Date: 2026-10-18 (Asia/Singapore). Selected identity (schema 2):
`ec46e346b8ab259de4da6d5db7bb90c7235984b6ef52bfecf5f5a89357a94992`. Decision: **accept only four rounded shapes; reject the broad candidate**.
The requested **10x aggregate target remains unmet**. The accepted complete
pair measures **4.976933x / 4.599836x**.

## Contract and implementation

The frozen explicit PyTorch baseline, dimensions, five seeds, strict
`abs_error <= .001 OR abs_error <= .01 * abs(reference)` comparator, parameter
surface, and sole authorized 100,000-token resource skip remain unchanged.
An individual 10x row does not satisfy the 13-row geometric-mean target.

The installed Triton 3.7.1 `tl.dot` source warns that ordinary TF32 inputs may
be truncated. A local explicit-rounding linear probe matched cuBLAS bit for
bit on three large tested matrix shapes; one small shape still differed by
5.96e-7. This local observation is not a universal cuBLAS claim. It motivated
round-to-nearest Q/K and normalized-probability/V inputs for attention.

[`attention.py`](../../transformer_opt/kernels/attention.py) adds
`_attention_fwd_rounded`. Its full key tile permits softmax normalization before
rounding P for P@V; rounding unnormalized online values changes that boundary.
Q/K and normalized probabilities/V use `cvt.rn.tf32.f32`, followed by ordinary
TF32 tensor-core dots. Scores/probabilities stay on chip. The wrapper retains
packed/strided QKV, strided bool masks, causal isolation and zero all-masked rows.

Only four BSHD shapes are admitted: `(1,128,4,32)`, `(4,128,4,32)`,
`(64,128,4,8)` and `(64,32,4,32)`, with SM120, fp32, TF32 enabled, disabled
gradients and causal mode. Model reference-guard bypass additionally requires
four layers and FFN width equal to model width. Canonical row 7 (both widths
32) uses four rounded layers. Row 6 retains its two-reference/two-custom split.
Other precision/device settings retain the prior routes. Vendor linears,
residual fusion, and the original eight-shape graph guard remain.

## Controlled accepted measurements

[`compare_rounded_attention.py`](../../benchmarks/compare_rounded_attention.py)
disables the predicate at both selection sites for the prior-route control.
Separate identical-weight models isolate graph caches. Each of the four cases
checks all elements for five fresh inputs, then retains 160 samples per variant
across four alternating rounds. Copies, clones, state checks and GPU work stay
timed. The [production A/B](../results/rtx-5070-ti-2026-10-18-c14-rounded-auto-ab.json) uses auto mode;
the [eager A/B](../results/rtx-5070-ti-2026-10-18-c14-rounded-ab.json) records near-parity/regressions
without replay. Acceptance follows the measured actual execution mode.

| Final row | Production control / candidate median | Per-round range |
| --- | ---: | ---: |
| 02 | 1.197x | 0.625–1.931x |
| 03 | 1.201x | 1.201–1.213x |
| 07 | 1.517x | 1.517–1.522x |
| 12 | 1.251x | 1.242–1.252x |

These controlled ratios attribute the four-shape change. Cross-process aggregate
movement also contains baseline and workstation timing variability.

## Complete final matrices

[Primary](../results/rtx-5070-ti-2026-10-18-c14-final.json) and
[confirmation](../results/rtx-5070-ti-2026-10-18-c14-final-confirmation.json) both pass 13/13 executable
rows, with zero failed elements across 938,885,120 comparisons each. The resource
skip is not a pass and is excluded from speedup aggregates. Precision is float32,
padding is zero, and input scale/seed policy follows the frozen final manifest.

| Final row | Primary | Confirmation |
| --- | ---: | ---: |
| 01 | 4.085x | 6.015x |
| 02 | 13.952x | 11.763x |
| 03 | 8.388x | 18.030x |
| 04 | 7.611x | 7.128x |
| 05 | 3.161x | 3.787x |
| 06 | 2.996x | 1.043x |
| 07 | 4.394x | 5.185x |
| 08 | 1.054x | 1.068x |
| 09 | 2.570x | 2.435x |
| 10 | 2.305x | 1.871x |
| 11 | 13.441x | 8.071x |
| 12 | 10.185x | 6.857x |
| 13 | 8.882x | 8.230x |

Timing is steady-state CUDA-event latency on the recorded RTX 5070 Ti, excluding
construction, compilation, weight packing and capture. It is a comparison with
explicit attention, not a universal advantage over SDPA. The wide row remains a
bottleneck; large-batch and several baseline timings vary substantially.

## Rejected evidence and numerical limitation

The broader eleven-shape prototype passed two default five-seed matrices at
5.385785x and 4.613902x (`f91e1797...`). These are retained as
[rejected primary](../results/rtx-5070-ti-2026-10-18-c14-wide-prototype-final.json) and
[rejected confirmation](../results/rtx-5070-ti-2026-10-18-c14-wide-prototype-final-confirmation.json),
not selected current-source results. Its 1.962x large-batch A/B and 61.1% lower
incremental allocation peak are **not accepted implementation claims**.

An additional screen changed weight seed to 8080 and used input cases
`(seed=2026,scale=.25,padding=0)`, `(4096,1,.25)` and `(1701,4,.25)`.
The [retained failure record](../results/rtx-5070-ti-2026-10-18-c14-rejected-stress.json) shows rounded
failures on rows 1, 4, 5, 6, 9, 10 and 11. At quarter scale, row 6 failed 619
elements versus **560 in the pre-existing route**; row 11 failed 2 versus 460.
Improved error is still a failure under the zero-failure comparator.

The failure is not fixed by more accurate IEEE/TF32x3 dots, disabling packed QKV,
using vendor normalization, disabling FMA, or matching exp/reduction/division
more closely. Those bounded probes are retained in the failure record. No root
cause or general accuracy fix is claimed. The accepted four shapes passed the
additional weight/scale/padding cases; larger rejected shapes cannot select the
new kernel. Global old-route TF32/quarter-scale correctness remains an explicit
pre-existing limitation and is deferred beyond this optimization attempt.

Other rejected probes: reference attention chunked by 512 was bitwise equal over
five seeds and lowered allocation peaks, but measured 0.974x; custom rounded
linears had limited/negative model benefits; two-pass long attention passed
five seeds but measured 0.949x; a head-width-256 full tile exceeded shared memory.
No baseline relaxation, output memoization, or additional skip was used.

## Review, validation and reproduction

Review found that the width-32 row has FFN width 32, not 128. The final guard
matches model/FFN width and its regression test checks the neighboring width.
Profiler validation also initially rejected the new real kernel name. It now
accepts only exact `_attention_fwd` and `_attention_fwd_rounded` keys and still
rejects substring decoys; a regression test covers both owned names and decoys.
PowerShell treated profiler stderr status text as a terminating error in the
first wrapper; subsequent runs retain stderr and enforce the process exit code.

A final negative test reproduced a missing-dependency bug: matching CUDA metadata
could select the rounded policy with Triton absent. The policy now requires
Triton explicitly; that regression fails before the guard and passes afterward.
All selected evidence was rerun on the final `ec46e346...` identity. The prior
fully tested `2d09c946...` snapshot is preserved under ignored
`results/c14-pre-availability`, rather than relabeling its measurements.
Raw-hashed `docs/results/*.json` files also disable Git text normalization so
future checkouts retain the exact newline bytes used by the evidence manifests.

The [validation record](../results/optimization-validation-2026-10-18-c14.json)
owns exact observed checks and artifact hashes. Fresh complete final, held-out,
source-derived, replay A/B and actual kernel profiles use the final identity.
Earlier C14 snapshots remain under ignored `results/c14-pre-review`; historical
C12/C13 JSON hashes are still enforced. Tests formerly binding C13 as current
now bind fresh C14, with explicit immutable-C13 replacement coverage.

```powershell
& .venv\Scripts\python.exe benchmarks/compare_rounded_attention.py --execution-mode auto --out results/rounded-auto.json
& .venv\Scripts\python.exe benchmarks/run_organizer_validation.py --matrix benchmarks/final_evaluator_shapes.json --out results/final.json
& .venv\Scripts\python.exe -m pytest tests -q
```

Requirements, code flow, design, result index, README/case study, chart and Colab
pin are reconciled to the accepted four-shape implementation. This linked repo
record supplies in-scope decision/implementation/evidence closure; external
durable memory was not written. No commit, branch, push, deployment or public
publication was performed.
