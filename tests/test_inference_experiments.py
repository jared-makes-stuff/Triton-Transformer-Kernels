"""The new experiment CLIs reject invalid runs without overwriting evidence."""
import pytest

from benchmarks.compare_attention_precision import main as precision_main
from benchmarks.compare_cuda_graph import main as graph_main
from benchmarks.compare_rounded_attention import main as rounded_main
from benchmarks.compare_short_layer_norm import main as short_norm_main
from benchmarks.compare_replay_validation import main as replay_validation_main


@pytest.mark.parametrize("main", [precision_main, graph_main, rounded_main, short_norm_main, replay_validation_main])
def test_experiment_preserves_existing_output(main, tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text("existing evidence", encoding="utf-8")
    with pytest.raises(SystemExit, match="2"):
        main(["--out", str(path)])
    assert path.read_text() == "existing evidence"


@pytest.mark.parametrize("main", [precision_main, graph_main, rounded_main, short_norm_main, replay_validation_main])
@pytest.mark.parametrize("option,value", [("--repeats", "0"), ("--rounds", "1")])
def test_experiment_rejects_invalid_sampling(main, option, value, tmp_path):
    path = tmp_path / "evidence.json"
    with pytest.raises(SystemExit, match="2"):
        main(["--out", str(path), option, value])
    assert not path.exists()
