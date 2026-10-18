"""Bind the current showcase to fresh source identities and raw measurements."""

import hashlib
import json
import statistics
from pathlib import Path

from tools.capture_environment import implementation_fingerprint


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/results"


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def test_current_final_pair_matches_current_source_and_complete_contract():
    fingerprint, _ = implementation_fingerprint()
    for suffix in ("final", "final-confirmation"):
        data = load(f"rtx-5070-ti-2026-10-18-c16-{suffix}.json")
        assert data["environment"]["git"]["implementation_sha256"] == fingerprint
        assert data["status"] == "PASS"
        assert data["summary"]["passed"] == 13
        assert data["summary"]["skipped_resource"] == 1
        assert data["summary"]["total_compared_elements"] == 938_885_120
        assert data["summary"]["total_failed_elements"] == 0
        assert len(data["results"]) == 14
        assert all(row["status"] in {"PASS", "SKIPPED_RESOURCE"} for row in data["results"])


def test_long_sequence_ab_claim_is_computed_from_retained_samples():
    data = load("rtx-5070-ti-2026-10-16-frontier-large-cases.json")
    result = next(row for row in data["results"]
                  if row.get("case_id") == "final-13-b64-d128-h4-s1024")
    medians = {}
    for name, samples in result["raw_ms"].items():
        assert len(samples) == 80
        medians[name] = statistics.median(samples)
        assert medians[name] == result["median_ms"][name]
    assert result["control_over_candidate"] == medians["control"] / medians["candidate"]
    assert result["failed_elements"] == 0
    assert result["accuracy_trials"] == 5
    assert data["control_source_sha256"] != data["candidate_source_sha256"]


def test_showcase_discloses_current_range_regression_and_unmet_target():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    runs = [load(f"rtx-5070-ti-2026-10-18-c16-{suffix}.json")
            for suffix in ("final", "final-confirmation")]
    geomeans = sorted(run["summary"]["geometric_mean_speedup"] for run in runs)
    assert f"{geomeans[0]:.3f}–{geomeans[1]:.3f}" in readme
    wide_ratios = sorted(next(row["parsed"]["speedup_median"]
                             for row in run["results"] if row["case_id"].startswith("final-08"))
                         for run in runs)
    assert f"{wide_ratios[0]:.3f}–{wide_ratios[1]:.3f}" in readme
    batch_ratios = sorted(next(row["parsed"]["speedup_median"]
                              for row in run["results"] if row["case_id"].startswith("final-06"))
                          for run in runs)
    assert f"{batch_ratios[0]:.3f}–{batch_ratios[1]:.3f}" in readme
    assert "10× aggregate tuning target was not reached" in readme
    assert "explicit PyTorch attention" in readme
    assert "rtx-5070-ti-2026-10-18-c16-final-confirmation.json" in readme
    assert "range" in (ROOT / "docs/assets/final-speedups.svg").read_text(encoding="utf-8")


def test_campaign12_artifacts_remain_immutable_historical_evidence():
    historical = load("publication-validation-2026-10-16.json")
    assert historical["implementation_sha256"] == (
        "abb639ce7fa68c9af9f21d776ca3d3afaa9e8f22be7e0ae8116ec4f71068c5ad"
    )
    for artifact in historical["evidence_artifacts"]:
        path = ROOT / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


def test_campaign13_artifacts_remain_immutable_historical_evidence():
    historical = load("optimization-validation-2026-10-18.json")
    assert historical["implementation_sha256"] == (
        "de8b6dd8831cb8637ac49cf8f2acfdcf69219e2cb9c303f60af81d123f34e868"
    )
    for artifact in historical["evidence_artifacts"]:
        assert hashlib.sha256((ROOT / artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]


def test_campaign14_artifacts_remain_immutable_historical_evidence():
    historical = load("optimization-validation-2026-10-18-c14.json")
    assert historical["implementation_sha256"] == (
        "ec46e346b8ab259de4da6d5db7bb90c7235984b6ef52bfecf5f5a89357a94992"
    )
    for artifact in historical["evidence_artifacts"] + historical["rejected_experimental_artifacts"]:
        assert hashlib.sha256((ROOT / artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]


def test_campaign15_artifacts_remain_immutable_historical_evidence():
    historical = load("optimization-validation-2026-10-18-c15.json")
    assert historical["implementation_sha256"] == (
        "eb4217f4a589e66d150cfe3ce36d1eacc4eded4d308984bf5b62e1ed7ed764f4"
    )
    for artifact in historical["evidence_artifacts"] + historical["research_artifacts"]:
        assert hashlib.sha256((ROOT / artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]


def test_graph_ab_proves_fresh_math_counts_and_raw_sample_ratios():
    data = load("rtx-5070-ti-2026-10-18-c16-graph-ab.json")
    fingerprint, _ = implementation_fingerprint()
    assert data["environment"]["git"]["implementation_sha256"] == fingerprint
    assert len(data["results"]) == 8
    assert data["parameters"]["input_copy_and_output_clone_included"]
    for row in data["results"]:
        assert row["failed_elements"] == 0
        assert row["accuracy_trials"] == 5
        assert row["bitwise_equal_to_eager"]
        assert row["candidate_backend_counts"] == row["control_backend_counts"]
        assert row["cuda_graph_replays"] == 330
        medians = {name: statistics.median(samples) for name, samples in row["raw_ms"].items()}
        assert all(len(samples) == 320 for samples in row["raw_ms"].values())
        assert row["median_ms"] == medians
        assert row["control_over_candidate"] == medians["control"] / medians["candidate"]


def test_historical_precision_ab_preserves_unchanged_shapes():
    data = load("rtx-5070-ti-2026-10-18-c13-precision-ab.json")
    assert data["environment"]["git"]["implementation_sha256"] == (
        "de8b6dd8831cb8637ac49cf8f2acfdcf69219e2cb9c303f60af81d123f34e868"
    )
    assert data["control_source_sha256"] != data["candidate_source_sha256"]
    assert data["parameters"]["execution_mode"] == "eager"
    assert len(data["results"]) == 6
    for row in data["results"]:
        assert all(len(samples) == 160 for samples in row["raw_ms"].values())
        if row["kind"] == "direct_attention":
            if row["shape_bshd"] != [64, 128, 1, 128]:
                assert row["bitwise_equal_to_control"]
        else:
            assert row["accuracy_trials"] == 5
            assert row["failed_elements"] == 0


def test_historical_rounded_ab_proves_samples_and_routes():
    # C15 changes normalization. Preserve the original C14 attention A/B;
    # fresh C15 model, replay, and normalization evidence covers current source.
    fingerprint = "ec46e346b8ab259de4da6d5db7bb90c7235984b6ef52bfecf5f5a89357a94992"
    for kind in ("rounded-ab", "rounded-auto-ab"):
        data = load(f"rtx-5070-ti-2026-10-18-c14-{kind}.json")
        assert data["environment"]["git"]["implementation_sha256"] == fingerprint
        assert {row["case_id"].split("-")[1] for row in data["results"]} == {"02", "03", "07", "12"}
        assert data["parameters"]["execution_mode"] == ("auto" if kind == "rounded-auto-ab" else "eager")
        for row in data["results"]:
            assert row["failed_elements"] == 0
            assert len(row["accuracy_trials"]) == 5
            assert sum(t["failed_elements"] for t in row["accuracy_trials"]) == 0
            assert row["compared_elements"] == sum(t["compared_elements"] for t in row["accuracy_trials"])
            assert all(len(samples) == 160 for samples in row["raw_ms"].values())
            medians = {name: statistics.median(samples) for name, samples in row["raw_ms"].items()}
            assert row["median_ms"] == medians
            assert row["control_over_candidate"] == medians["control"] / medians["candidate"]
            assert row["backend_counts_last_round"]["candidate"]["reference"] == 0
            if row["case_id"].startswith("final-07"):
                assert row["backend_counts_last_round"]["control"] == {
                    "triton": 120, "sdpa": 0, "reference": 40,
                }
                assert row["backend_counts_last_round"]["candidate"] == {
                    "triton": 160, "sdpa": 0, "reference": 0,
                }


def test_current_short_normalization_samples_accuracy_and_profiler_ownership():
    fingerprint, _ = implementation_fingerprint()
    data = load("rtx-5070-ti-2026-10-18-c16-norm-ab.json")
    assert data["environment"]["git"]["implementation_sha256"] == fingerprint
    assert data["status"] == "PASS" and len(data["trials"]) == 8
    assert all(t["failed_elements"] == 0 and t["bitwise_equal_to_control"] for t in data["trials"])
    assert data["backend_counts"]["control"] == data["backend_counts"]["candidate"]
    timing = data["timing"]
    assert all(len(values) == 320 for values in timing["raw_ms"].values())
    medians = {name: statistics.median(values) for name, values in timing["raw_ms"].items()}
    assert medians == timing["median_ms"]
    assert timing["control_over_candidate"] == medians["control"] / medians["candidate"]
    profile = load("rtx-5070-ti-2026-10-18-c16-row07-profile.json")
    assert profile["environment"]["git"]["implementation_sha256"] == fingerprint
    assert profile["validation_passed"]
    assert sum(e["count"] for e in profile["top_events"] if e["name"] == "_short_layer_norm_fwd") == 27


def test_current_replay_validation_preserves_state_math_and_timing_samples():
    data = load("rtx-5070-ti-2026-10-18-c16-replay-validation-ab.json")
    assert data["environment"]["git"]["implementation_sha256"] == implementation_fingerprint()[0]
    assert data["status"] == "PASS" and len(data["results"]) == 8
    assert data["parameters"]["validation_enabled_in_both"]
    for row in data["results"]:
        assert len(row["trials"]) == 8
        assert all(t["bitwise_equal_to_control"] and t["signature_equal_to_prior"]
                   and t["candidate_failed"] == t["control_failed"] for t in row["trials"])
        assert all(t["candidate_failed"] == 0 for t in row["trials"] if t["weight_seed"] == 1234)
        assert row["backend_counts"]["control"] == row["backend_counts"]["candidate"]
        assert row["graph_replays"] == {"control": 323, "candidate": 323}
        medians = {name: statistics.median(values) for name, values in row["timing"]["raw_ms"].items()}
        assert all(len(values) == 320 for values in row["timing"]["raw_ms"].values())
        assert row["timing"]["median_ms"] == medians
        assert row["timing"]["control_over_candidate"] == medians["control"] / medians["candidate"]


def test_campaign16_preserves_rejected_ffn_stress_and_exact_sources():
    data = load("rtx-5070-ti-2026-10-18-c16-component-screens.json")
    assert data["status"] == "COMPLETE" and len(data["records"]) == 8
    for archive in data["source_archives"].values():
        assert hashlib.sha256(archive["source_utf8"].encode()).hexdigest() == archive["sha256"]
    for record in data["records"]:
        value = record["payload"]["source_sha256"]
        hashes = set(value.values()) if isinstance(value, dict) else {value}
        assert hashes == {data["source_archives"][name]["sha256"] for name in record["source_versions"]}
    stress = next(r["payload"] for r in data["records"] if "ffn-row7-stress" in r["original_path"])
    assert stress["status"] == "FAIL" and len(stress["trials"]) == 32
    assert any(t["control_failed"] == 0 and t["candidate_failed"] > 0 for t in stress["trials"])
    assert any(t["control_failed"] > 0 for t in stress["trials"])
    diagnostic = data["records"][0]["payload"]
    assert diagnostic["unchecked_replay_is_selectable"] is False


def test_component_rejections_and_exact_experiment_sources_remain_visible():
    data = load("rtx-5070-ti-2026-10-18-c15-component-screens.json")
    assert data["status"] == "COMPLETE" and len(data["screens"]) == 9
    for version in data["source_archives"].values():
        assert hashlib.sha256(version["source_utf8"].encode("utf-8")).hexdigest() == version["sha256"]
    for screen in data["screens"]:
        payload = screen["payload"]
        for name, sha in payload["experiment_sources"].items():
            assert data["source_archives"][screen["source_versions"][name]]["sha256"] == sha
        if payload["parameters"]["kind"] in ("norm", "chunk"):
            assert all(row["status"] == "REJECTED_ACCURACY" and "timing" not in row for row in payload["results"])
        if payload["parameters"]["kind"] == "chunk":
            trial = payload["results"][0]["trials"][0]
            assert trial["candidate_failed"] == trial["control_failed"] == 560
            assert all(t["bitwise_equal_to_control"] for t in payload["results"][0]["trials"])


def test_rejected_broad_candidate_and_preexisting_stress_failures_remain_visible():
    data = load("rtx-5070-ti-2026-10-18-c14-rejected-stress.json")
    assert data["status"] == "FAIL"
    row6 = [r for r in data["results"] if r["row"] == 6 and r["scale"] == .25]
    assert {r["rounded"]: r["failed"] for r in row6} == {False: 560, True: 619}
    assert {r["row"] for r in data["results"] if r["rounded"] and r["failed"]} == {1, 4, 5, 6, 9, 10, 11}
    for suffix in ("final", "final-confirmation"):
        prototype = load(f"rtx-5070-ti-2026-10-18-c14-wide-prototype-{suffix}.json")
        assert prototype["environment"]["git"]["implementation_sha256"] == (
            "f91e17970cfb9172230c7c25b632d9693280c7d3a2435ba196993004a8c24f11"
        )
