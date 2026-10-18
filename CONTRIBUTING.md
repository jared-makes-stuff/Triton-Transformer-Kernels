# Working on the kernels

Use Python 3.12 and the [recorded environment](docs/guides/REPRODUCING.md).
Run commands from the repository root. The Python package intentionally reuses
the frozen organizer reference rather than shipping a second baseline.

## Before changing execution

Read [requirements](docs/REQUIREMENTS.md), [kernel design](docs/KERNEL_DESIGN.md),
and the relevant dispatch and GPU tests. Keep state-dict compatibility, mask
semantics, strict correctness, and observable fallback routing intact.

The two downloaded files in `benchmarks/` are reference inputs. Do not edit their
baseline, comparator, timing, or parser to make an optimization pass. Their raw
checksums are in `benchmarks/reference/organizer_downloads.json`.

## Validate a change

```bash
python -m pytest tests -q
python tools/check_publication.py
python benchmarks/run_organizer_validation.py \
  --matrix benchmarks/final_evaluator_shapes.json --out results/final.json
python benchmarks/run_matrix.py --device cuda --accuracy-trials 5 \
  --out results/heldout.json
```

CPU CI checks contracts, runners, and evidence. GPU tests skip when CUDA/Triton
is absent; a green CPU job alone does not establish kernel correctness or speed.
On CUDA, run the direct boundary/mask tests and end-to-end gates affected by the
change, then profile the selected backend with `benchmarks/profile_cases.py`.

## Make performance evidence reviewable

- Test correctness before timing. Report failed elements, OOM, and errors.
- Warm up compilation and caches; use CUDA events and alternating run order.
- Keep raw samples, seeds, shapes, environment, actual backends, and source hashes.
- Compare against an unchanged control in the same session and repeat the result.
- Keep scratch outputs under ignored `results/`. Curate only useful evidence
  under `docs/results/`, with its decision and limitations documented.
- Preserve historical JSON. A new implementation fingerprint needs new results;
  it does not inherit old measurements.

Update the requirements when intent changes, kernel design when execution
changes, and result index when new evidence is selected. The
[portfolio case study](docs/PORTFOLIO.md) must continue to distinguish the
explicit-attention baseline, PyTorch SDPA, and project-owned Triton kernels.

## Review scope

Describe the bottleneck, change, correctness evidence, latency/memory trade-off,
and remaining limits. Prefer measured improvements over broad shape exceptions
or speculative tuning. Keep benchmark artifacts free of credentials, absolute
home paths, and profiler traces that are unnecessary for reproducing the claim.
