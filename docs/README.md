# Documentation map

Start with the [repository landing page](../README.md). For a technical interview
or portfolio review, the [case study](PORTFOLIO.md) explains the problem,
implementation, numerical trade-offs, evidence, and skills in one reading.

## Current implementation

| Question | Read |
| --- | --- |
| What changed in the latest optimization pass? | [Campaign 16 replay validation record](experiments/CAMPAIGN-016.md) |
| What do the numbers mean? | [Raw-result index](results/README.md) |
| How do I reproduce them? | [Setup and validation guide](guides/REPRODUCING.md) |
| What is the exact contract? | [Requirements](REQUIREMENTS.md) |
| How does the kernel work? | [Kernel design](KERNEL_DESIGN.md) |
| How does a command reach the GPU? | [Code flow](CODE_FLOW.md) |
| What belongs in a portfolio card? | [Portfolio copy and interview paths](PORTFOLIO.md#portfolio-copy) |
| How should a change be reviewed? | [Contributing](../CONTRIBUTING.md) |

## Historical competition record

These documents retain their original campaign measurements and submission
context. Campaign 16 supersedes their selected implementation; their old
fingerprints, branch states, external holds, and timings remain historical.

| Material | Purpose |
| --- | --- |
| [Campaign 15 component record](experiments/CAMPAIGN-015.md) | Accepted row-7 Welford normalization and rejected component candidates |
| [Campaign 14 rounding record](experiments/ROUNDED_ATTENTION_OPTIMIZATION.md) | Four accepted short attention shapes and rejected numerical stress cases |
| [Campaign 13 launch record](experiments/LAUNCH_OVERHEAD_OPTIMIZATION.md) | Prior bounded replay, precision routes, 3.744–4.187x final pair |
| [Campaign 12 publication record](experiments/PUBLICATION_OPTIMIZATION.md) | Prior causal-tile pruning, 2.422–2.776x final pair, and recorded variability |
| [Campaign run-through](experiments/CAMPAIGN_RUN_THROUGH.md) | Readable campaigns 1–11 narrative and rejected alternatives |
| [Optimization history](experiments/OPTIMIZATION_HISTORY.md) | Detailed chronology, attempt accounting, and decisions |
| [Implementation evidence](IMPLEMENTATION_EVIDENCE.md) | Campaign 11 engineering evidence reference |
| [Branch comparison](experiments/BRANCH_IMPLEMENTATION_COMPARISON.md) | Prior alternative implementation and failure evidence |
| [Submission validation](experiments/SUBMISSION_VALIDATION.md) | Prior repository-local release-gate record |
| [Competition inputs](../hackathon-docs/ORGANIZER_INPUTS.md) | Frozen references and unresolved organizer assumptions |
| [Submission bundle](../deliverables/) | Historical technical report, public copy, and demo script |
| [Demo runbook](guides/DEMO_RUNBOOK.md) | Detailed competition-era walkthrough |

Raw JSON remains immutable under [`results/`](results/) and historical command
records under [`experiments/attempts/`](experiments/attempts/). Scratch data,
profiler traces, local environments, and credentials stay outside the publication
set via `.gitignore`. Keeping the evidence preserves the rejected attempts and
prevents the showcase from becoming a collection of selected wins alone.
