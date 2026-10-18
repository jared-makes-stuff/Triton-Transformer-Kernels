"""CPU guards for the isolated optimization experiment."""

import pytest

from benchmarks.compare_causal_frontier import NEW_LOOP, OLD_LOOP, candidate_source, main


def test_candidate_changes_only_the_loop_bound():
    source = f"prefix\n{OLD_LOOP}\n            body\nsuffix\n"
    assert candidate_source(source) == source.replace(OLD_LOOP, NEW_LOOP)


@pytest.mark.parametrize("source", ["", OLD_LOOP + "\n" + OLD_LOOP])
def test_candidate_rejects_unknown_or_already_changed_source(source):
    with pytest.raises(ValueError, match="exactly one"):
        candidate_source(source)


def test_experiment_does_not_overwrite_evidence(tmp_path):
    out = tmp_path / "evidence.json"
    out.write_text("keep", encoding="utf-8")
    with pytest.raises(SystemExit, match="2"):
        main(["--out", str(out)])
    assert out.read_text() == "keep"
