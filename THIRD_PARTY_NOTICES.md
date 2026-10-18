# Reference material and attribution

The project-owned implementation lives in `transformer_opt/`; its benchmark
runners and tests live in `benchmarks/` and `tests/`. The following files are
participant-supplied competition inputs, retained byte-for-byte for reference:

| File | Role |
| --- | --- |
| `benchmarks/torch_transformer_benchmark.py` | PyTorch baseline and executable comparison/timing contract |
| `benchmarks/tensorflow_transformer_benchmark.py` | Alternative framework and shape/resource-policy reference |

Their origin, receipt dates, sizes, and SHA-256 values are recorded in
[`benchmarks/reference/organizer_downloads.json`](benchmarks/reference/organizer_downloads.json).
The challenge snapshot is in [`hackathon-docs/`](hackathon-docs/). These inputs
are attributed to the TikTok TechJam 2026 organizers; this repository does not
claim authorship of them. The supplied files do not include a separate license
grant. No repository-wide license has been added that would relicense them.

PyTorch, Triton, NumPy, and pytest are external dependencies installed by the
user, rather than vendored implementations. The project uses online-softmax
attention tiling; it does not claim to have invented that algorithm. The
[kernel design](docs/KERNEL_DESIGN.md) describes this implementation's numerical,
layout, masking, dispatch, and fusion choices.

AI-assisted development is described in the
[technical report](deliverables/03_TECHNICAL_REPORT.md#14-ai-assisted-development-and-human-contribution).
Tests, raw samples, and profiler observations support the reported results;
generated prose is not measurement evidence.
