# Reproduce the results

Run commands from the repository root. Performance evidence is from native
Windows 11, an RTX 5070 Ti (compute capability 12.0, 16,303 MiB), driver 616.56,
Python 3.12.10, PyTorch 2.13.0+cu130, CUDA 13.0, and Triton 3.7.1.
Other hardware requires its own measurements.

## Native Windows CUDA

```powershell
py -3.12 -m venv .venv
$python = ".venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
& $python -m pip install -r requirements.txt
& $python -m pytest tests -q
```

`requirements.txt` pins the observed environment. Install the CUDA PyTorch wheel
first so pip retains that build. Native Windows uses `triton-windows`; a normal
CPython installation provides the headers/import library needed for its driver
shim.

## CPU development and CI

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.5.2 pytest==9.1.1
python -m pytest tests -q
```

On Windows, activate `.venv\Scripts\Activate.ps1` instead. CPU tests cover
semantics, dispatch contracts, failure accounting, and immutable artifacts.
GPU skips do not count as GPU evidence.

## CUDA verification

The commands below use `python`; in PowerShell, substitute `& $python` from the
setup above. Output under `results/` is ignored by Git.

```bash
# Untouched organizer harness; strict repository tolerance selected explicitly.
python benchmarks/run_organizer_torch.py --device cuda \
  --atol 0.001 --rtol 0.01 --evidence-out results/organizer.json

# All 14 published dimensions: 13 executable rows and one authorized resource skip.
python benchmarks/run_organizer_validation.py \
  --matrix benchmarks/final_evaluator_shapes.json --out results/final.json

# Broader source-derived contract: 28 executable cases and the authorized skip.
python benchmarks/run_organizer_validation.py --out results/source-derived.json

# Seven project-owned held-out cases; five accuracy trials per case.
python benchmarks/run_matrix.py --device cuda --attention-backend auto \
  --accuracy-trials 5 --out results/heldout.json

# Observe actual GPU execution, rather than inferring it from imports.
python benchmarks/profile_cases.py --case long-attention --dtype float32 \
  --attention-backend auto --expect-backend triton --steps 5 \
  --out results/profile.json --trace results/profile-trace.json

# Isolate replay from identical eager math; copies/clones/checks remain timed.
python benchmarks/compare_cuda_graph.py --out results/graph-ab.json

# Compare rounded attention with the prior routes in the actual execution mode.
python benchmarks/compare_rounded_attention.py --execution-mode auto --out results/rounded-ab.json
# Use --execution-mode eager to isolate the same routes without graph replay.

# Current host optimization: both variants preserve every validation field.
python benchmarks/compare_replay_validation.py --out results/replay-validation-ab.json

# Isolate the row-7 Welford normalization on the current execution path.
python benchmarks/compare_short_layer_norm.py --out results/short-norm-ab.json
```

The organizer parser defaults to `atol=0.002, rtol=0.02`, despite an older
module docstring. The repository matrices explicitly select the stricter
`abs_error <= 0.001 OR abs_error <= 0.01 * abs(reference)` rule. Any failing
element fails correctness. OOM, unexpected errors, and zero executed cases
cannot become a pass. The resource skip is excluded from all speedup aggregates.

## Optional Linux/WSL and Colab paths

The measured results are native Windows results. For WSL, install a C compiler,
matching Python headers, and a CUDA-compatible PyTorch environment; the Linux
PyTorch CUDA distribution supplies its matching Triton dependency. The existing
`scripts/run-wsl.ps1` wrapper uses `~/.venvs/tiktok-techjam-2026`; set
`TIKTOK_TECHJAM_PYTHON` to override that interpreter. `tools/triton-cc` prefers
gcc/clang and can use a user-scoped Zig fallback.

[`notebooks/colab_benchmark.ipynb`](../../notebooks/colab_benchmark.ipynb) is a
clean, fingerprint-pinned full-repository runner. It asks for a credential-free
GitHub HTTPS URL, checks out `main`, and stops if the source fingerprint differs.
Publish the matching implementation before using that remote workflow. Fresh
Colab timings describe the assigned GPU and cannot inherit the RTX 5070 Ti numbers.

## Reading the evidence

Use the [result index](../results/README.md) to match a claim to its exact
implementation fingerprint, environment, arguments, backend counts, and raw
samples. Construction, first-use compilation, packed-QKV cache population, and
CUDA-graph capture are excluded from steady-state latency. Replay retains
additional private storage; input/mask copies, output cloning, and invalidation
checks are included. Use `execution_mode="eager"` to disable replay and
`clear_cuda_graph()` to release its cache. This is an inference benchmark, with
an explicit PyTorch attention baseline; it is not a claim to beat every SDPA
backend or to measure application/network latency.

To regenerate the README figure, install the optional plotting dependency
(`python -m pip install -r requirements-viz.txt`) and run:

```bash
python tools/plot_publication_results.py \
  --result docs/results/rtx-5070-ti-2026-10-18-c16-final.json \
  --confirmation docs/results/rtx-5070-ti-2026-10-18-c16-final-confirmation.json \
  --out docs/assets/final-speedups.svg
```

The whiskers show the two-run range, not statistical confidence intervals.
Matplotlib is figure-generation tooling, outside the kernel runtime dependencies.
