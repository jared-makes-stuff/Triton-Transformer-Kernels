# Campaign 16: replay validation and whole-FFN fusion

Status: accepted replay validation traversal on 2026-10-18; FFN fusion rejected. Continue toward the 10x geometric-mean target in
[requirements AC-12](../REQUIREMENTS.md), using Campaign 15 source
`eb4217f4a589e66d150cfe3ce36d1eacc4eded4d308984bf5b62e1ed7ed764f4`
as the unchanged control. The baseline, strict zero-failure comparator,
dimensions, timing policy and sole authorized resource skip stay fixed.

## Evidence and hypotheses

[Campaign 15](CAMPAIGN-015.md) accepted exact row-7 Welford normalization.
Its isolated single-GEMM/exact-GELU candidates were near parity, so repeating
that epilogue alone is excluded. Larger attention chunks reproduced existing
stress failures and are also excluded.

[PyTorch's CUDA graph explanation](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
and [NVIDIA's graph guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
explain amortizing host setup/launch work. The current adapter still traverses
modules and parameters repeatedly on every replay to preserve invalidation.
First measure that cost and a diagnostic lower bound before proposing a
semantics-preserving traversal improvement. A replay without validation is
only an upper-bound diagnostic, never a selectable implementation.

The second hypothesis is materially broader than Campaign 15's epilogue:
keep the first linear's full hidden tile on chip, apply the same exact-erf
GELU, round at the existing TF32 boundary, and compute the second linear in
the same Triton program. This could replace three launches and an intermediate
write, but tensor-core accumulation differences and register/shared-memory
pressure may erase the gain or fail accuracy. Use only widths 32/128 initially,
and bound the query tile/configuration screen to four variants.

## Execution and gates

1. Capture live source/environment and hash the accepted evidence. Measure
   replay guard/signature CPU cost and complete versus unchecked replay timing
   on three small cases; retain raw samples and label the unchecked result.
2. Implement at most one evidence-led validation traversal refinement if the
   measured cost is material. Preserve every existing invalidation field,
   module/hook/training guard, lock/event ordering and fresh-output behavior.
3. Independently screen whole-FFN fusion on the four stress-robust final rows
   2, 3, 7 and 12, using separate model graph caches. Weight seed 8080 and the
   quarter-scale/padded cases precede five normal seeds. Reject any nonzero
   failed-element count before performance testing. Retain IEEE/TF32 policy.
4. Promote only repeatable whole-forward wins with negative, mutation, hook,
   profiler and regression coverage. Freeze implementation/tooling first, then
   obtain fresh complete primary/confirmation, held-out and source-derived
   evidence. Otherwise retain Campaign 15 source and current claims.

Bound: two hypotheses, four FFN tiles, at most one refinement per hypothesis.
Run GPU jobs sequentially. The primary agent implements and then reviews from
requirements, architecture, numerical QA, privacy, documentation and adversarial
perspectives; no independent-agent review or delegation is claimed. No commits,
branches or external actions are authorized. Keep historical JSON immutable.

The linked plan, workflow and evidence ledger provide in-repository task-graph
closure. External durable memory remains read-only under the user's explicit
save rule. No project-context document exists.

## Observed selection and counterfactual

Selected source: `5bcf25d733d92d571d4531194e9b92fd0a8d3d16a74c72659b5413ca03836dc5`. Only `transformer_opt/submission.py` changes production
behavior in this pass. It keeps the eight-shape replay policy and all kernels.
State collection validates training/hooks and inference-only parameters while
walking live module/parameter registries once. Identity deduplication preserves
the prior ordering for shared modules, parameters and cycles. It still records
storage pointers (including `.data` replacement without a version increment),
versions, device/dtype, module identity, epsilon/normalized shape, attention
configuration, input/mask and precision/inference policy.

The [controlled A/B](../results/rtx-5070-ti-2026-10-18-c16-replay-validation-ab.json) restores the old
repeated state/eligibility walks inside the same replay lock. Both variants
retain input copies, fresh output clones, all state checks and completion-event
ordering. Each checks five normal inputs plus three weight/scale/padding
stress inputs. All outputs are bitwise equal and every valid signature equals
the former collector on the same model. Known old-route stress failures are
reported on both sides; no arithmetic was changed to conceal them.

| Final row | Whole-model control/candidate | Old state check µs | New state check µs |
| --- | ---: | ---: | ---: |
| 01 | 1.000x | 154.1 | 77.8 |
| 02 | 1.659x | 154.0 | 79.3 |
| 03 | 1.638x | 153.3 | 76.6 |
| 04 | 1.000x | 151.3 | 77.3 |
| 07 | 1.164x | 153.6 | 78.8 |
| 09 | 1.000x | 157.2 | 79.6 |
| 10 | 1.000x | 155.0 | 78.5 |
| 12 | 1.220x | 154.8 | 78.3 |

Each variant retains 320 raw CUDA-event samples in four alternating rounds.
The CPU column is an independent 1000-call state-collection measurement;
it is not added to or substituted for end-to-end timing. Other shapes can be
device-bound and remain near parity despite lower host overhead. Small
near-parity regressions are retained; no uniform speedup is claimed.

## Rejected fusion and new limitations

The [research ledger](../results/rtx-5070-ti-2026-10-18-c16-component-screens.json) embeds five exact
source versions and eight diagnostic/prototype records. The initial replay
guard/signature costs were about 59/95 µs on the smallest case. Unchecked replay
was only an upper-bound diagnostic and is never selected by the implementation.

Whole-FFN fusion used explicit TF32 round-to-nearest and exact-erf GELU, with
four tile variants. Before the validation bottleneck was removed, timing was
near parity. On the faster-validation control, row 7 improved approximately
1.078–1.107x in the bounded screen, while other rows were mixed/regressed.
The only promising shape then failed an additional 32-input screen: weights
0/1701/4096/8080, input seed 5050, scales .125/.25/1/4 and zero/25% padding.
Several cases had zero control failures but 1–4 candidate failures. **All FFN
fusion is rejected**, despite the earlier eight-input passes and timing gains.

That stronger screen also exposed unchanged row-7 control failures with some
other weights at quarter/eighth scale. Campaign 15's eight-input bitwise gate
remains valid for its recorded inputs; it was never a universal guarantee.
The replay optimization preserves these outputs bit for bit and is not a
numerical repair. A broader precision fix remains outside this bounded pass.

## Complete matrices

[Primary](../results/rtx-5070-ti-2026-10-18-c16-final.json): **4.189651x**.
[Confirmation](../results/rtx-5070-ti-2026-10-18-c16-final-confirmation.json):
**4.170246x**. Both pass 13/13 executable
rows, zero failed elements across 938,885,120 comparisons per run. The sole
100000-token resource skip remains a non-pass and is excluded from aggregates.
The requested **10x aggregate target remains unmet**.

| Final row | Primary | Confirmation |
| --- | ---: | ---: |
| 01 | 2.087x | 2.096x |
| 02 | 12.653x | 12.705x |
| 03 | 11.128x | 10.635x |
| 04 | 5.812x | 5.750x |
| 05 | 2.425x | 2.436x |
| 06 | 1.775x | 1.776x |
| 07 | 7.853x | 7.307x |
| 08 | 1.051x | 1.054x |
| 09 | 2.215x | 2.226x |
| 10 | 1.887x | 2.033x |
| 11 | 7.087x | 7.084x |
| 12 | 7.716x | 7.950x |
| 13 | 8.837x | 8.343x |

Cross-process aggregate movement includes baseline/workstation variability.
The controlled A/B is the attribution evidence for this change. Construction,
compilation, cache population and capture are excluded; copies, clones,
validation and GPU work remain inside timed forwards.

## Review and closure

[Validation record](../results/optimization-validation-2026-10-18-c16.json)
contains the observed tests, artifact/source hashes and hygiene results.
Regression coverage exercises ordered deduplication, shared storage, registry
cycles, replacement without version increments, parameter/module replacement,
metadata, hooks, training and inference-only tensors, alongside existing
cross-stream and fresh-output tests. The first GPU replacement fixture created
inference tensors and correctly bypassed replay; it was corrected to use
ordinary replacement tensors for the recapture test. The separate inference
fallback regression remains. An initial prototype assertion also compared
different models' storage identities; it was corrected to compare collectors
on the same model. Neither required a comparator or product relaxation.

Requirements AC-15, design, code flow, current-result index, README/case study,
figure and Colab source pin are reconciled. Current-source assertions use
fresh C16 evidence, with immutable-C15 replacement coverage. The linked repo
ledger supplies task-graph closure; external durable memory is untouched.
No commit, branch, push, release, deployment or public publication was performed.

```powershell
& .venv\Scripts\python.exe benchmarks/compare_replay_validation.py --out results/replay-validation-ab.json
& .venv\Scripts\python.exe benchmarks/run_organizer_validation.py --matrix benchmarks/final_evaluator_shapes.json --out results/final.json
& .venv\Scripts\python.exe -m pytest tests -q
```
