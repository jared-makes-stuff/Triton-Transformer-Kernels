# Campaign 15: component bottlenecks after rounded attention

Status: accepted exact row-7 Welford normalization on 2026-10-18; other candidates rejected.
[Campaign 16](CAMPAIGN-016.md) is the current selection. This pass targeted a
10x final-matrix geometric mean under [AC-12](../REQUIREMENTS.md) and began from
the accepted Campaign 14 source,
`ec46e346b8ab259de4da6d5db7bb90c7235984b6ef52bfecf5f5a89357a94992`
retaining it until the candidate cleared every promotion gate. The baseline,
tolerance, shapes, skip and timing policies were unchanged.

## Inputs and research

- [Campaign 14](ROUNDED_ATTENTION_OPTIMIZATION.md): explicit TF32 rounding
  improved four shapes, but broader attention failed different-weight,
  quarter-scale tests. More accurate dot products did not repair that failure.
- Its row-7 CUPTI profile attributes 478.676 of 840.516 microseconds to
  LayerNorm (56.95%), versus 109.540 to attention (13.03%). These are device
  event totals from three forwards, not a new wall-clock measurement.
- Row 6 retains two reference attention layers, large temporary allocations,
  and variable timing. A previous chunk size of 512 saved memory but regressed
  latency; this pass tests larger chunks rather than repeating that choice.
- [Triton's matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
  describes grouped tile scheduling and fusing activation while accumulators
  remain in registers. This motivates a bounded tile screen with explicit
  TF32 round-to-nearest and the existing exact-erf GELU formula.
- [NVIDIA's cuBLAS epilogue documentation](https://docs.nvidia.com/cuda/cublas/index.html#cublasltepilogue-t)
  specifies a tanh GELU approximation. It is not a drop-in replacement for the
  baseline's `approximate="none"`; that shortcut is excluded.
- [NVIDIA's Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html#occupancy)
  lists 99 KiB maximum shared memory per block on SM120. The prior wide
  attention tile exceeded that limit. Do not transfer datacenter tile sizes
  blindly to this GPU.
- [Triton's persistent tutorial](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)
  demonstrates FP16/FP8 kernels and warns about small shared-memory devices.
  It is useful scheduling research, not evidence that its low-precision
  implementation satisfies this repository's FP32 contract.
- [PyTorch 2.13's CUDA LayerNorm source](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/cuda/layer_norm_kernel.cu)
  provides the compatibility reference for four adjacent values per lane,
  Welford accumulation, descending warp combination, and the affine expression.
  The Triton refinement implements that operation sequence; bitwise checks bind
  the observation to the installed binary and target device rather than making
  a universal framework-version guarantee.

## Bounded plan

1. **Normalization:** process multiple independent rows per program, preserving
   feature-axis reductions, epsilon, optional bias, and residual/mask semantics.
   Test 4/8/16 rows per program locally; compare whole forwards in the existing
   production execution mode, with separate model graph caches.
2. **Linear/activation:** screen a small tile family for TF32-RNE GEMM plus
   exact-erf GELU. Retain vendor linears unless a complete forward wins. Do not
   infer an end-to-end improvement from an isolated GEMM win.
3. **Large-batch reference attention:** test chunks 1024/2048/4096, retaining
   reference operations and current two-reference/two-Triton routing. Require
   zero comparator failures and repeated counterbalanced timing; report memory
   separately. Do not promote an inconclusive timing result.

Stop each hypothesis after its bounded tile/chunk screen and at most one
evidence-directed refinement. All GPU work runs sequentially. The primary
agent performs implementation and subsequent review; no delegated or
independent-agent approval is claimed.

## Gates and closure

Stress comes before timing and full campaigns: weights 8080 with inputs
2026/0.25/unpadded, 4096/1.0/25%-padded, and 1701/4.0/25%-padded; then the
five normal seeds. Record both control and candidate failures, including
pre-existing failures. A new selected route requires zero failures; a known
control failure is not permission to accept a new failing route.

If a candidate survives, integrate only its tested guard, cover negative
dispatch and profiler ownership, freeze source, run two complete final
matrices plus affected regression/held-out/source-derived checks, and update
current documentation. Otherwise retain Campaign 14 source and claims, publish
the rejection ledger, and state that 10x remains unmet. Keep all historical
JSON bytes intact. Preserve version counters by constructing parameters
outside inference mode. Retain experiment source hashes and environment.

The in-repository plan, workflow, results, and linked requirements form the
task evidence graph. External durable memory is not written because the user
requires explicit save authorization. No project-context file exists.

## Observed results and decisions

Selected fingerprint: `eb4217f4a589e66d150cfe3ce36d1eacc4eded4d308984bf5b62e1ed7ed764f4`. [Production A/B](../results/rtx-5070-ti-2026-10-18-c15-norm-ab.json)
measures **1.344754x** whole-forward control/candidate speedup. All four
alternating rounds improve (1.324–1.370x), with 320 CUDA samples per variant.
Eight accuracy inputs pass the strict reference comparator and are bitwise
equal to the prior normalization route. Construction/capture remain excluded;
fresh input copies, output clones, state checks and GPU work remain timed.

The [row-7 profiler](../results/rtx-5070-ti-2026-10-18-c15-row07-profile.json) records 27 exact
`_short_layer_norm_fwd` launches and 12 rounded attention launches over three
forwards. The other rows retain their former normalization paths. A global
forward hook disables replay; local/global hooks retain ordinary LayerNorm
execution. Parameters and state-dict names remain unchanged.

The [component screen ledger](../results/rtx-5070-ti-2026-10-18-c15-component-screens.json) retains:

- Naive multirow normalization: all five screened rows failed quarter-scale
  stress (1, 1, 1, 1093, and 2 failed elements for rows 2, 3, 7, 8, and 12).
  Normal five-seed passes were insufficient. Rejected before timing.
- Welford refinement: rows 2, 3, 7 and 12 were bitwise equal to control on eight
  inputs. Only row 7 improved timing (1.317–1.324x in the exploratory tile
  screen); other rows measured 0.970–0.990x and remain unchanged. Eight rows per
  program was selected; 4/16-row variants were also retained.
- Exact-erf GELU plus RNE GEMM: 32x64 and 64x128 tiles were near parity on
  robust shapes (roughly 0.974–1.024x); no consistent material gain justified
  replacing vendor linears. Other rows retained pre-existing stress failures.
- Reference chunks 1024/2048/4096: every tested output was bitwise equal to the
  control, including five normal seeds. Each retains 560 quarter-scale failures
  with weights 8080. Rejected at the accuracy gate; no new latency or memory
  claim is made for these chunks.

## Complete matrices and interpretation

[Primary](../results/rtx-5070-ti-2026-10-18-c15-final.json): **3.911727x**.
[Confirmation](../results/rtx-5070-ti-2026-10-18-c15-final-confirmation.json):
**3.797882x**. Both pass all 13 executable
rows, zero failed elements across 938,885,120 comparisons each. Row 14 remains
the sole source-authorized resource skip and is not counted as a pass.
The requested **10x aggregate target was not reached**.

| Final row | Primary | Confirmation |
| --- | ---: | ---: |
| 01 | 2.123x | 2.045x |
| 02 | 7.339x | 7.295x |
| 03 | 6.580x | 6.897x |
| 04 | 5.861x | 5.773x |
| 05 | 2.451x | 2.498x |
| 06 | 1.781x | 1.787x |
| 07 | 6.390x | 6.656x |
| 08 | 1.049x | 1.047x |
| 09 | 2.268x | 2.159x |
| 10 | 2.764x | 1.835x |
| 11 | 7.604x | 7.530x |
| 12 | 6.736x | 6.972x |
| 13 | 8.894x | 8.899x |

The current aggregate pair is lower than the historical Campaign 14 pair.
The optimized row-7 latency improves in a same-process counterfactual, while
the baseline and workstation timing window also changed. No causal aggregate
improvement or regression is inferred from that cross-process movement.
Wide GEMM/FFN work, strict rounding sensitivity and the large-batch hybrid
remain the practical bottlenecks. A further doubling of aggregate performance
is not supported by these incremental results.

## Review and validation closure

The [validation record](../results/optimization-validation-2026-10-18-c15.json)
owns final observed checks, artifact/source hashes, and review findings.
Existing current-source assertions were moved to fresh C15 matrices/replay
evidence; C14 attention comparisons remain historical with explicit identity
and immutable-artifact coverage. No comparator or valid accuracy test was
weakened. The stronger stress gate excluded failed candidates before expensive
final campaigns. The simple reduction's failure is preserved, not hidden.

Requirements AC-14, design, code flow, result index, README/case study, chart,
and Colab pin are reconciled to this source. This linked evidence ledger closes
the in-repository graph; external durable memory was not written. No commit,
branch, push, release or external publication was performed.

```powershell
& .venv\Scripts\python.exe benchmarks/compare_short_layer_norm.py --out results/short-norm-ab.json
& .venv\Scripts\python.exe benchmarks/run_organizer_validation.py --matrix benchmarks/final_evaluator_shapes.json --out results/final.json
& .venv\Scripts\python.exe -m pytest tests -q
```
