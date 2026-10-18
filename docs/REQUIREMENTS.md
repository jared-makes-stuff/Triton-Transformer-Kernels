# Transformer GPU Kernel Requirements

Use the [documentation hub](README.md) for the repository reading order. This
file owns the executable contract and its unresolved organizer assumptions.

Status: implementation contract reconciled with the two participant-supplied
organizer downloads received on 2026-08-27 and the 14-row Track 3 test-shape
table published in the organizer document's Section 3.7. PyTorch is the
selected framework. `benchmarks/final_evaluator_shapes.json` preserves the
published row order and records every execution assumption required because
the table omits framework, dtype, padding, tolerance, timing, and backward
requirements. `benchmarks/official_shapes.json` remains a project-owned
held-out matrix rather than the final organizer matrix.

Selected local entry: `transformer_opt/submission.py::UserOptimizedTransformer`.
Campaign 16 collects replay guards and state in one traversal while preserving
all existing invalidation fields. Its schema-2 identity is
`5bcf25d733d92d571d4531194e9b92fd0a8d3d16a74c72659b5413ca03836dc5`.
See [the replay validation record](experiments/CAMPAIGN-016.md).
Campaign 15's eight-row Welford LayerNorm remains historical at
`eb4217f4a589e66d150cfe3ce36d1eacc4eded4d308984bf5b62e1ed7ed764f4`;
see [the component optimization record](experiments/CAMPAIGN-015.md).
Campaign 14's short-sequence attention with explicit round-to-nearest TF32
inputs and normalized probabilities remains historical at
`ec46e346b8ab259de4da6d5db7bb90c7235984b6ef52bfecf5f5a89357a94992`.
See [the rounding optimization record](experiments/ROUNDED_ATTENTION_OPTIMIZATION.md)
for the 10x tuning target, acceptance gates, measurements, and limitations.
Campaign 13's bounded replay and precision evidence remains historical at
`de8b6dd8831cb8637ac49cf8f2acfdcf69219e2cb9c303f60af81d123f34e868`;
see [its record](experiments/LAUNCH_OVERHEAD_OPTIMIZATION.md).
Campaign 12's causal loop-bound evidence remains historical at
`abb639ce7fa68c9af9f21d776ca3d3afaa9e8f22be7e0ae8116ec4f71068c5ad`;
see [its publication record](experiments/PUBLICATION_OPTIMIZATION.md).

Historical Campaign 11 measurements remain attached to
`908a0d708cd8f70f44d5f14fda93d3cafb1cc18345f43914e715594cfa7b7ef9`.
The earlier validation-hardened, unpruned control is
`a186b679885e9e787b3deba0ad710855ae4c2486ae491b53e4e64bfa13e7f9cf`. It is the historical control for Campaign 12.
Historical JSON is not relabeled as current evidence. The organizer downloads,
reference comparator, support envelope, and model state-dict contract remain
unchanged. Campaigns 14 through 16 are bounded by the policies below.

## Source-of-truth order

1. The untouched organizer PyTorch download identified by
   `benchmarks/reference/organizer_downloads.json`.
2. The organizer's final shape table identified by
   `benchmarks/final_evaluator_shapes.json` and its live source metadata.
3. The older result-linked snapshot identified by
   `benchmarks/reference/manifest.json`.
4. Track 3 in `hackathon-docs/hackathon-details.md`, lines 674-780.
5. Reproduced behavior on the target GPU.
6. Design and explanatory documentation in this repository.

If these sources conflict, update this file and the reference manifest before
tuning kernels. Never relax a checked-in correctness rule to match a looser
description.

This file owns the implementation contract. The canonical chronology and
current-versus-historical evidence map is
[`experiments/OPTIMIZATION_HISTORY.md`](experiments/OPTIMIZATION_HISTORY.md);
campaign ledgers and raw JSON remain the authority for individual executions.

## Organizer download reconciliation

- Untouched PyTorch SHA-256:
  `5529c96a80799b51f68092e1444a30b17994554dffdf52da98ba701489a7f36e`.
- Untouched TensorFlow SHA-256:
  `00e99b6e1d19e961039b66eb3d3c055b36cc50f0436da2558f5f1fbe292ef798`.
- `benchmarks/run_organizer_torch.py` injects only the submitted class into the
  untouched PyTorch harness; tests AST-compare protected baseline definitions.
- The TensorFlow download is retained as an alternative-framework and shape
  cross-check. It is not treated as a second implementation requirement.
- `benchmarks/run_organizer_validation.py` translates every feasible published
  dimension signal into the selected PyTorch harness. Its 28 executable cases
  all pass; the TensorFlow source's designated 100000-token quadratic stress
  case is the only authorized resource skip and is never counted as a pass.
- The organizer document now publishes 14 exact rows covering batch size,
  QKV/model width, heads, sequence length, layers, causal mode, and FFN width.
  Thirteen rows are executable on the selected PyTorch path. Row 14's batch,
  QKV-dimension, head-count, and 100000-token sequence axes match the supplied
  TensorFlow stress case that explicitly permits resource-preflighting; it
  remains an authorized skip and is not counted as a pass. Its layer, causal,
  and FFN values remain the distinct values published in the final table.
- The final table does not state dtype or padding. Until the organizer says
  otherwise, final-shape validation uses the selected PyTorch parser defaults
  of float32 and no padding, then explicitly selects the stricter executable
  comparator below.

The files are not interchangeable evaluator specifications. PyTorch defaults
to one configurable float32 case. Its untouched module docstring names the
0.001/0.01 OR rule, but its parser actually defaults to 0.002/0.02; repository
evidence matrices explicitly pass the stricter 0.001/0.01 values. TensorFlow
defaults to a float16 compact dimension sweep and the prose-level 0.002/0.02 OR
rule. The final table supplies dimensions but does not resolve those
framework-level differences. See `hackathon-docs/ORGANIZER_INPUTS.md` for the
exact differences and remaining clarifications.

## Required behavior

The selected PyTorch implementation reuses the pre-LayerNorm Transformer and
contract helpers from the untouched `benchmarks/torch_transformer_benchmark.py`.
The optimized submission adapter is `transformer_opt/submission.py`:

```text
for each block:
  x = x + MHA(LayerNorm(x), valid_token_mask, causal)
  x = x + Linear(GELU(Linear(LayerNorm(x)), approximate="none"))
final output = LayerNorm(x)
```

- Multi-head attention uses separate Q, K, V, and output projections.
- Invalid key positions cannot be attended to.
- Causal attention cannot attend to future positions.
- Invalid output rows are zero after every block and after the final norm.
- The optimized model must accept the same inputs, preserve parameter names,
  and load the reference state dict with `strict=True`.
- Training/backward support is not required by the current inference harness.

## Correctness contract

For every output element, the checked-in benchmark requires:

```text
abs(optimized - reference) <= 0.001
OR
abs(optimized - reference) <= 0.01 * abs(reference)
```

The Track 3 prose states looser bounds (`abs < 0.002`, `relative < 0.02`). This
project intentionally targets the stricter executable benchmark rule. A case
passes only when there are zero failing elements. NaN and infinity mismatches
fail.

Known limitation: Campaign 16's additional seed-5050 stress screen found
unchanged selected row-7 failures for some other weight seeds at quarter/eighth
input scale. The final five-seed matrix and the specifically recorded stress
passes remain valid; they do not establish correctness for all distributions.
The replay traversal change preserves those outputs bit for bit. The
[Campaign 16 record](experiments/CAMPAIGN-016.md) retains these unresolved
failures and rejects the FFN candidate that introduced additional ones.

Correctness validation must cover:

- CPU float32 semantic regressions;
- CUDA float32 for end-to-end auto routing and forced-custom coverage, float16
  for direct-kernel validation, and bfloat16 for the exact fallback;
- causal and non-causal attention;
- no mask, all-valid mask, partial prefix padding, and minimum one-token prefix;
- multiple seeds and input scales;
- sequence lengths on and immediately around tile boundaries;
- every executable row in the final organizer shape table;
- custom-kernel positive dispatch and unsupported-shape fallback dispatch.

## Kernel and dispatch contract

- At least one repository-owned GPU kernel must execute in the submitted path.
- The primary custom kernel is forward scaled dot-product attention implemented
  in Triton with online softmax and fp32 softmax accumulation.
- Causal attention visits K/V tiles only through `min(S, (query_block + 1) * BLOCK_M)`;
  the diagonal tile retains its elementwise causal mask. Non-causal traversal
  remains full-length. This bound must preserve boundary, mask, and strict
  end-to-end correctness.
- The custom path must not materialize a `[B,H,S,S]` score tensor or dense
  causal mask.
- PyTorch SDPA and explicit reference-style attention are the safe fallback and
  comparison backends.
- `auto` dispatch may select custom Triton only inside its tested support
  envelope; forced `triton` mode must fail clearly when unsupported.
- Backend choice must be inspectable in tests/results. Import or compilation
  errors may not be swallowed as successful custom execution.
- Eager CUDA float32 inference may cache a derived packed QKV projection for
  measured `d_model <= 512` shapes and exact `d_model == 1024`. Widths 513-1023,
  widths above 1024, CPU, low precision, compiled execution, gradients, or an
  otherwise unsupported layout must retain separate projections. The derived
  cache must invalidate after parameter/state/device/dtype changes, remain
  non-persistent, and preserve strict state-dict compatibility.
- The measured fixed launch policy uses 32x32 tiles for `head_dim == 128` only
  through sequence 128 and returns to 32x64 at sequence 129 and above. Boundary
  tests must preserve that exact guard; it changes launch geometry, not the
  support envelope, arithmetic, public API, or persisted state.
- Direct Triton attention supports `head_dim == 8` by zero-padding the compile-
  time dot width to 16 lanes, masking padded Q/K/V loads, storing only the real
  eight output lanes, and scaling by the real head dimension. The measured
  64x64 launch is selected for the final row-11 target. Multi-layer `auto`
  dispatch uses Triton in every layer for exact final row 11. Final row 7 uses
  all four rounded layers under the SM120 policy below. Otherwise it keeps
  layer zero on reference math and uses the prior padded kernel for the other
  three. That older kernel failed one strict
  element with four approximate layers. Other width-eight shapes remain on
  explicit reference math until separately measured.
- Exact final row 6 (`B=10000,S=128,d_model=128,heads=4,layers=4,causal=true`)
  retains two reference layers followed by two prior Triton layers. The broader
  rounded prototype passed the normal five-seed matrix but failed the additional
  quarter-scale stress gate, so it is not selected for row 6. For
  eval-mode CUDA float32 inference outside compilation, this exact runtime shape
  also fuses each residual add with the following LayerNorm. Exact final row 5
  (`B=128,S=128,d_model=128,heads=4,layers=4,causal=true`), row 9
  (`B=64,S=128,d_model=128,heads=1,layers=4,causal=true`), and row 11
  (`B=64,S=128,d_model=128,heads=16,layers=4,causal=true`) reuse the same
  fused forward while retaining all four Triton attention layers. Rows 5 and 9
  are admitted only by exact static configuration and runtime-shape guards; their
  neighboring batches, sequence lengths, dimensions, head counts, layer counts,
  causal modes, and feed-forward widths remain unfused. The fused
  kernel must
  preserve optional bias, epsilon, strict state-dict compatibility, padded-row
  zeroing, and each shape's established attention routing. Noncontiguous masks, neighboring
  shapes, training, CPU, other dtypes, and compiled execution retain the eager
  PyTorch path.
- The exact project-held-out `B=2,S=512,d_model=512,heads=8,layers=2,
  causal=true` envelope uses SDPA for both unpadded and prefix-padded inputs.
  Controlled screens, repeated held-out matrices, backend counts, and profiler
  events show it removes the prior long-causal regression. This project-owned
  route does not broaden the organizer-final claim.
- The benchmark-default non-causal float32 custom path follows the benchmark's
  TF32 toggle. Causal custom attention normally uses IEEE fp32 dot products.
  On compute capability 12.0 with TF32 enabled and gradients disabled, only the
  four short-sequence BSHD shapes for final rows 2, 3, 7, and 12 listed in
  `prefer_rounded_attention` use explicit
  `cvt.rn.tf32.f32` rounding for Q/K and normalized probabilities/V before each
  single-product TF32 dot. The full key tile is 32 or 128 tokens; no quadratic
  global scratch is materialized. The model bypasses older reference-layer
  guards only with four layers and FFN width equal to model width, including
  final row 7 (both widths are 32). Rejected larger shapes retain the prior route.
  Packed/strided QKV and bool
  masks retain their strides. Other shapes retain the prior kernel; its exact
  `(64,128,1,128)` and `(64,1024,4,32)` SM120 fp32 shapes use `tf32x3`.
  Truncated single-product causal TF32 remains disabled.
  Automatic end-to-end low-precision runs, unsupported custom head widths, and
  unmeasured causal batches above 128 use the explicit reference-style path
  after testing exposed rare SDPA or fused-attention misses. Exact row 6 above
  is the measured large-batch exception. CPU, other unsupported layouts,
  and training with gradients fall back unless explicitly added and tested.
- `execution_mode="auto"` may capture and replay the unchanged forward for exact
  final rows 1, 2, 3, 4, 7, 9, 10, and 12. This requires eager, contiguous CUDA
  float32, eval mode, disabled gradients/autocast, auto attention, matching
  runtime/model dimensions, and an absent or contiguous bool mask. Hooks,
  compilation, outer graph capture, and unmeasured shapes retain eager execution.
  Rows 5, 6, 8, 11, and 13 retain eager execution: their graph screens did not
  justify the additional persistent storage.
- Replay copies current input/mask values and returns a fresh output tensor.
  It must preserve prior outputs, strict state-dict keys, and actual backend
  counts. Parameter identity/storage/version/device/dtype changes, module
  replacement, LayerNorm shape/epsilon or attention-configuration changes,
  inference/no-grad context changes, and TF32-policy changes invalidate
  the single-entry cache. Device/dtype moves release graph and packed caches.
  Host locking plus a CUDA completion event order reuse across caller streams.
  Capture errors propagate. `execution_mode="eager"` disables replay;
  `clear_cuda_graph()` releases its private storage. Captures are not serialized.
- Replay collects ordered, identity-deduplicated module and parameter state in
  one traversal of the live registries. It retains every former identity,
  storage pointer, version, device, dtype, module epsilon/normalized shape,
  attention configuration, input/mask and execution-policy field. Module
  training/hooks and inference-only parameters reject capture before replay.
  State collection remains inside the replay lock; no validation is skipped
  or amortized across calls. `_can_replay_cuda_graph` retains its complete
  boolean eligibility semantics for callers.
- Exact final row 7 (`B=64,S=128,d_model=32,heads=4,ffn_dim=32,layers=4,
  causal=true`) uses an eight-row Welford LayerNorm kernel on SM120 for
  contiguous CUDA float32 inference. Its four-value per-lane accumulation and
  descending reduction preserve vendor math in the direct and model tests.
  Affine weight, optional bias and epsilon retain their normal semantics.
  Neighboring configurations, other devices/dtypes/layouts, gradients,
  training, autocast, compilation, replaced LayerNorm types and local/global
  forward hooks retain module execution. Global forward hooks also disable
  graph replay so their Python observers run on every call. Attention routing,
  residual operations and mask zeroing are unchanged by this optimization.

## Benchmark integrity

- Correctness runs before performance timing.
- GPU timing uses CUDA events after warm-up and synchronization.
- Baseline and optimized ordering alternates by round.
- Every requested case has an explicit `PASS`, `FAIL`, `OOM`, or `ERROR` result.
- Unexpected runtime errors fail the run. OOM may be reported separately but is
  never converted into a pass.
- A run with zero completed cases fails.
- Result artifacts include the code revision, dirty state, command, raw timing
  samples, framework/runtime versions, GPU name/capability, driver, and case
  definition.
- Performance claims are tied to curated JSON, its implementation fingerprint,
  and its recorded Git dirty state, not console excerpts or an assumed commit.

## Target and dependencies

Primary tuning target:

- NVIDIA GeForce RTX 5070 Ti, compute capability 12.0, 16,303 MiB VRAM;
- native Windows 11, build 26200, NVIDIA driver 616.56;
- Python 3.12.10;
- PyTorch 2.13.0+cu130;
- CUDA runtime 13.0;
- Triton 3.7.1 from `triton-windows==3.7.1.post27`.

Portable CPU tests remain required. Other CUDA devices use the same guarded
dispatcher and may fall back to SDPA until measured.

## Acceptance criteria

- **AC-1:** A direct Triton attention kernel executes on the target GPU and
  matches the reference across the declared support envelope.
- **AC-2:** End-to-end optimized Transformer outputs pass the stricter checked-in
  tolerance for every executed matrix case.
- **AC-3:** Dispatch tests prove both custom selection and safe fallback; no
  silent fallback is reported as custom execution.
- **AC-4:** Sweep/matrix error accounting cannot report success for unexpected
  exceptions, OOM-only runs, or zero completed cases.
- **AC-5:** Target-GPU evidence includes raw timings, environment metadata,
  backend counts, and profiler evidence that the custom kernel ran.
- **AC-6:** README, kernel design, technical report, and demo runbook contain no
  placeholders or unverified performance statements.
- **AC-7:** The downloaded PyTorch/TensorFlow files match frozen hashes, and the
  optimized class passes the untouched organizer PyTorch harness without
  modifying its baseline, comparator, argument parsing, or timing code.
- **AC-8:** Every feasible case derived from the two supplied contracts passes
  in an isolated process, and skip accounting accepts only the exact
  source-authorized 100000-token stress case.
- **AC-9:** The 14 published final-shape rows are preserved in source order;
  all 13 executable rows pass the untouched selected PyTorch comparator in
  isolated processes, and the exact 100000-token resource skip is excluded
  from the pass count.
- **AC-10:** Exact-row residual/LayerNorm fusion remains limited to final rows 5,
  6, 9, and 11 in eval-mode eager CUDA float32 inference; direct, stress, neighbor,
  training, gradient, dtype, device, layout, mask, memory, and profiler gates
  must remain current for the selected fingerprint.
- **AC-11:** CUDA replay preserves changing input/mask values, independent output
  ownership, parameter/state/epsilon invalidation, stream ordering, observability,
  and negative guards. Setup and persistent graph memory must be disclosed;
  input copies, output clones, and signature checks remain inside timed forwards.
- **AC-12:** The continued tuning goal is 10x geometric-mean final-matrix speedup.
  This is a tuning target, not an accepted result or permission to change the
  baseline, tolerance, skip policy, dimensions, or scoring. Report the actual
  complete-matrix pair even when the target is not reached.
- **AC-13:** Rounded attention is accepted only on the four shapes that passed
  both the five-seed final gate and the additional weight/scale/padding stress
  cases. Preserve rejected broad-prototype and pre-existing quarter-scale
  failures as limitations; do not claim that every input distribution passes.
- **AC-14:** Short LayerNorm stays restricted to exact row 7 and must preserve
  bitwise model outputs versus the prior route on the five normal seeds and
  additional weight/scale/padding stress cases. Require direct affine, tail,
  optional-bias and epsilon checks, negative guard/hook coverage, actual
  profiler events, and counterbalanced whole-model timing before acceptance.
- **AC-15:** Replay validation optimization preserves the previous ordered
  signature and bitwise model outputs across all eight existing replay shapes.
  Tests must cover shared/cyclic module registries, shared parameters/storage,
  storage replacement without a version increment, parameter/module replacement,
  dtype and normalization/attention metadata, hooks, training, inference tensors,
  independent outputs and caller streams. Reference failures in broader stress
  must be retained as limitations; unchanged arithmetic is not a numerical fix.

## Deliverables

- Public, structured, commented repository.
- Reproduction instructions and limitations in README.
- Custom GPU kernel implementation and guarded integration.
- Correctness/performance results from the target machine.
- Technical report including environment, optimization rationale, AI tooling,
  measurements, and limitations.
- Byte-preserved organizer downloads, checksum manifest, and exact-harness
  PyTorch default plus full source-derived validation evidence.
- Track 3 requirement-to-evidence compliance matrix.
- Demo-video runbook for an end-to-end public walkthrough.

## Open organizer questions

- Required dtypes and padding modes for the final 14-row shape table.
- Whether additional non-causal rows are required beyond the published causal
  rows.
- Whether timing includes model construction/compilation or steady-state forward
  only.
- Whether gradients/backward are evaluated.
- Exact dependency and source-file modification restrictions.
- Whether a later workshop/evaluator revision supersedes either supplied file.
- Whether the current live download attachments are byte-identical to the two
  files frozen on 2026-08-27; the attachment UI was visible on 2026-08-28 but
  did not expose bytes for a fresh checksum through the read-only browser path.

These unknowns do not block validating the published dimensions under the
selected PyTorch defaults, but they block claiming that every unstated
evaluator policy is known.


## Portfolio publication intent (2026-10-16)

The repository also serves as a technical portfolio for software engineering and
quantitative-development-adjacent roles. The public landing page must expose the
problem, owned kernels, reproducible results, and numerical/performance trade-offs
before the competition campaign history. The case study must explain why memory
traffic, numerical error, experimental controls, and safe routing matter.

- Use `we` for project narrative, as requested by the project owner. Do not infer
  participant names, team size, individual roles, or external competition outcomes.
- Distinguish explicit PyTorch attention, PyTorch SDPA, and custom Triton execution.
- Tie current performance claims to fresh source-fingerprinted artifacts; preserve
  Campaign 11 records as historical evidence.
- Keep quantitative relevance specific to numerical computing and performance
  engineering. Do not imply trading, backtesting, market-data, P&L, or production
  low-latency experience from this Transformer inference project.
- Publication preparation is local. Remote publication, commits, and portfolio
  website changes are outside this task's authorized execution scope.
- The documentation, CPU CI, GPU tests, ignored scratch outputs, and third-party
  attribution must support review without discarding historical experiment evidence.
