# Campaign 12 — final optimization and portfolio preparation

> Historical record: [Campaign 16 replay validation](CAMPAIGN-016.md) is the current selection. Measurements below retain Campaign 12's original source identity.

Date: 2026-10-16 (Asia/Singapore). Decision: **keep causal tile pruning**.

The selected implementation has schema-2 fingerprint
`abb639ce7fa68c9af9f21d776ca3d3afaa9e8f22be7e0ae8116ec4f71068c5ad`.
The unpruned maintenance control is
`a186b679885e9e787b3deba0ad710855ae4c2486ae491b53e4e64bfa13e7f9cf`.
Campaign 11's historical measured fingerprint and JSON remain unchanged.

## Hypothesis and change

Earlier campaigns rejected causal-frontier pruning because it did not improve
their end-to-end target. We re-tested it on the cumulative implementation.
The original kernel visited every K/V tile for every query block, including
tiles whose positions were entirely in the future.

```diff
- for start_n in range(0, N_CTX, BLOCK_N):
+ end_n = N_CTX
+ if CAUSAL:
+     end_n = tl.minimum(N_CTX, (query_block + 1) * BLOCK_M)
+ for start_n in range(0, end_n, BLOCK_N):
```

Skipped tiles have zero probability for every query in that block. The final
visited tile still applies the existing elementwise causal mask. Non-causal
traversal remains full-length. This change does not alter dot precision, dtype
boundaries, launch geometry, backend selection, or model parameters.

## Controlled A/B evidence

[`compare_causal_frontier.py`](../../benchmarks/compare_causal_frontier.py)
generates both kernel modules in a temporary directory. It never rewrites the
production module or organizer files. Control and candidate share inputs,
weights, launch policy, precision settings, warm-up, and device. CUDA-event
sample batches alternate order across four rounds. Raw samples and exact kernel
source hashes are retained.

| Gate | Observation | Evidence |
| --- | --- | --- |
| Direct kernels, first screen | 1.223×, 1.263×, 1.248×, 1.432×; all four outputs bitwise equal to control | [Screen](../results/rtx-5070-ti-2026-10-16-frontier-screen.json) |
| Direct kernels, repetition | 1.223×, 1.263×, 1.249×, 1.417×; all four outputs bitwise equal | [Confirmation](../results/rtx-5070-ti-2026-10-16-frontier-confirmation.json) |
| End-to-end screens | Rows 5, 9, and 11 improve in both runs; row 1 spans 0.984–1.044×; unchanged held-out SDPA spans 0.995–1.006× | Same two artifacts, 200 samples per variant |
| 1,024-token final row 13 | Control 31.466896 ms → candidate 15.945936 ms, **1.973349×** | [Large-case gate](../results/rtx-5070-ti-2026-10-16-frontier-large-cases.json), 80 samples per variant |
| Row-13 round ratios | 2.001×, 1.965×, 1.978×, 1.955× | Recomputed from the four 20-sample batches |
| Large-batch row 6 | Aggregate control/candidate median ratio 1.804×, but round ratios range 0.798–2.133× | Same artifact; **inconclusive for a stable latency gain** |
| Hybrid row 7 | Aggregate ratio 1.015×, with substantial round variability | Same artifact; no material speed claim |

The direct shapes are `[B,S,H,D] = [64,128,4,32]`, `[64,128,1,128]`,
`[64,128,16,8]`, and `[2,512,8,64]`. Direct equality applies to those inputs,
not a universal bitwise-equivalence claim. Model candidates are checked against
the explicit baseline for five seeds at the strict repository OR tolerance.

The first exploratory script invocation constructed parameters inside inference
mode and stopped with `Inference tensors do not track version counter` before
any model timing. We corrected the experiment to construct parameters with
version tracking. Both recorded screen artifacts come from subsequent successful
runs. No kernel fallback or comparator relaxation was used to repair that error.

## Complete-model validation and variability

| Run | Correctness | Geometric-mean speedup vs explicit PyTorch |
| --- | --- | ---: |
| [Unchanged control](../results/rtx-5070-ti-2026-10-16-publication-control-final.json) | 13/13 executable PASS; 0 failed elements | 2.072090× |
| [Campaign 12 primary](../results/rtx-5070-ti-2026-10-16-c12-final.json) | 13/13 executable PASS; 0 failed elements | 2.775997× |
| [Campaign 12 confirmation](../results/rtx-5070-ti-2026-10-16-c12-final-confirmation.json) | 13/13 executable PASS; 0 failed elements | 2.421721× |

Each final matrix checks 938,885,120 output elements across 65 accuracy trials.
Each excludes the one authorized 100,000-token resource skip from pass and timing
aggregates. Backend accounting remains Triton 1,260 / SDPA 0 / reference 196.
The final dimensions are organizer-published; float32, no padding, strict
tolerances, and timing settings are explicitly repository-selected assumptions.

The full matrices are separate processes and show substantial latency drift.
In particular, row 6's optimized median is 1,503.563 ms in the primary and
3,722.332 ms in the confirmation; its baseline also varies substantially. Its
end-to-end baseline/optimized ratios are **1.766× and 0.563×**. The experiment
does not establish the cause of this variability. We do not infer a causal
optimization percentage by comparing whole-matrix geomeans or hide that row.
The README figure shows both runs' range, not a confidence interval.

The acceptance rationale is the repeatable direct-kernel improvements,
consistent row-13 model improvement across all four alternating rounds, strict
correctness, and unchanged compatibility/routing. A stable large-batch latency
improvement remains unproven.

## Additional current-source gates

- [Held-out matrix](../results/rtx-5070-ti-2026-10-16-c12-heldout.json):
  **7/7 PASS**, five accuracy trials per case, 1.482? measured geomean.
- [Source-derived organizer matrix](../results/rtx-5070-ti-2026-10-16-c12-source-derived.json):
  **28/28 executable PASS**, one authorized skip, zero failed elements across
  459,776,000 comparisons. It includes float32, float16, bfloat16, causal, and
  padded cases; sensitive low-precision routes retain exact reference math.
- Fresh profiles for [row 5](../results/rtx-5070-ti-2026-10-16-c12-row05-profile.json),
  [row 9](../results/rtx-5070-ti-2026-10-16-c12-row09-profile.json), and
  [row 11](../results/rtx-5070-ti-2026-10-16-c12-row11-profile.json) each prove
  12 Triton attention launches and 24 fused residual/LayerNorm launches over
  three forwards. All explicit profiler expectations pass.
- The [row-6 profile](../results/rtx-5070-ti-2026-10-16-c12-row06-profile.json)
  proves two Triton plus two reference attention calls and eight fused
  residual/LayerNorm launches in one forward, preserving the accuracy-safe hybrid.
- The [row-13 profile](../results/rtx-5070-ti-2026-10-16-c12-row13-profile.json)
  checks actual custom-attention execution for the strongest stable A/B result.
- `python -m pytest tests -q`: **215 passed**, 14 upstream deprecation warnings.
  With `CUDA_VISIBLE_DEVICES=-1`: **111 passed, 104 expected GPU skips**.
- `python tools/check_publication.py`: **zero findings** for local Markdown link
  targets and common credential literals. Python compilation checks also pass.
  The [validation record](../results/publication-validation-2026-10-16.json)
  retains exact commands, results, artifact hashes, and review limitations.

The GitHub Actions CPU workflow is prepared locally; its Linux job has not run
remotely. The Colab notebook is locally syntax/pin checked; remote execution
requires publication of the matching source. Neither is reported as a live pass.

## Reproduce the isolated experiment

```bash
python benchmarks/compare_causal_frontier.py --out results/frontier.json
python benchmarks/compare_causal_frontier.py --case final-13-b64-d128-h4-s1024 \
  --repeats 20 --rounds 4 --out results/frontier-long.json
```

The script can reconstruct the pre-pruning control from the current source. It
rejects unknown loop forms and refuses to overwrite an output artifact. Its
temporary modules are removed and production dispatch is restored on exit.
Use the [reproduction guide](../guides/REPRODUCING.md) for full matrices/profiling.

## Publication changes

- Reframed the README around the problem, implementation, skills, and auditable
  current results, with a figure generated directly from the two final artifacts.
- Added a [technical case study and portfolio copy](../PORTFOLIO.md), explaining
  SWE and numerical/performance-engineering relevance without trading claims.
- Added CPU CI, contributor guidance, publication hygiene checks, and third-party
  attribution. Protected the frozen organizer files from Git line-ending
  conversion. CPU CI has no performance or GPU-correctness claim.
- Updated the notebook's strict implementation pin and replaced its hardcoded
  competition clone URL with a validated, credential-free GitHub HTTPS input.
- Retained historical experiment JSON and labeled competition-era reports.
  Scratch outputs, figure dependencies, and original-file backups stay ignored.

The checkout began with only `.gitattributes` tracked and all project files
untracked. Local edits preserve the supplied work; no commit, branch, push,
website change, external publication, or persistent-memory update was performed.
The earlier competition name “SpeedROCm” remains in the historical bundle;
the portfolio uses “Triton Transformer Kernels” and claims NVIDIA CUDA evidence.
