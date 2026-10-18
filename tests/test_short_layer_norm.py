"""The short-row optimization preserves vendor math and model observability."""

import pytest
import torch
from torch import nn

from transformer_opt.kernels import triton_available
from transformer_opt.kernels.residual_layer_norm import short_layer_norm
from transformer_opt.submission import TransformerConfig, UserOptimizedTransformer


gpu = pytest.mark.skipif(
    not triton_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="short LayerNorm is validated only on SM120",
)


def model(**changes):
    values = dict(batch_size=64, seq_len=128, d_model=32, num_heads=4,
                  ffn_dim=32, num_layers=4, causal=True)
    values.update(changes)
    return UserOptimizedTransformer(TransformerConfig(**values)).eval()


def test_short_policy_rejects_cpu_and_missing_triton(monkeypatch):
    import transformer_opt.kernels.residual_layer_norm as module
    m = model()
    assert not m._use_short_layer_norm(m.final_norm, torch.empty(64, 128, 32))
    monkeypatch.setattr(module, "triton", None)
    with pytest.raises(RuntimeError, match="not installed"):
        short_layer_norm(torch.empty(1, 32), torch.ones(32), None, 1e-5)


@gpu
@pytest.mark.parametrize("rows", [0, 1, 7, 8, 9, 8192])
@pytest.mark.parametrize("bias", [False, True])
def test_short_norm_matches_vendor_bitwise_with_affine_and_tail_rows(rows, bias):
    torch.manual_seed(8080)
    x = torch.randn(rows, 32, device="cuda") * .25 + .17
    weight = torch.randn(32, device="cuda")
    shift = torch.randn(32, device="cuda") if bias else None
    with torch.inference_mode():
        actual = short_layer_norm(x, weight, shift, 3e-4)
        expected = torch.nn.functional.layer_norm(x, (32,), weight, shift, 3e-4)
    assert torch.equal(actual, expected)


@gpu
@pytest.mark.parametrize("change", [dict(batch_size=63), dict(seq_len=129),
    dict(d_model=64), dict(num_heads=2), dict(ffn_dim=128), dict(num_layers=3), dict(causal=False)])
def test_short_policy_rejects_every_neighbor(change):
    m = model(**change).cuda()
    x = torch.zeros((), device="cuda").expand(64, 128, 32).contiguous()
    with torch.inference_mode():
        assert not m._use_short_layer_norm(m.final_norm, x)


@gpu
def test_short_policy_training_gradients_layout_hooks_dtype_and_autocast():
    m = model().cuda()
    x = torch.randn(64, 128, 32, device="cuda")
    assert not m._use_short_layer_norm(m.final_norm, x)
    with torch.inference_mode():
        assert m._use_short_layer_norm(m.final_norm, x)
        m.train(); assert not m._use_short_layer_norm(m.final_norm, x); m.eval()
        assert not m._use_short_layer_norm(m.final_norm, x.transpose(0, 1))
        assert not m._use_short_layer_norm(m.final_norm, x.half())
        with torch.autocast("cuda"):
            assert not m._use_short_layer_norm(m.final_norm, x)
        calls = []
        hook = m.final_norm.register_forward_hook(lambda *args: calls.append(1))
        try:
            assert not m._use_short_layer_norm(m.final_norm, x)
            m(x)
            assert calls == [1]
            assert m.cuda_graph_replays == 0
        finally: hook.remove()
        hook = nn.modules.module.register_module_forward_hook(lambda *args: None)
        try:
            assert not m._use_short_layer_norm(m.final_norm, x)
            assert not m._can_replay_cuda_graph(x, None)
        finally: hook.remove()


@gpu
def test_short_norm_rejects_unsupported_direct_inputs():
    weight = torch.ones(32, device="cuda")
    x = torch.randn(4, 32, device="cuda")
    with pytest.raises(ValueError, match="disabled gradients"):
        short_layer_norm(x, weight, None, 1e-5)
    with torch.inference_mode():
        for unsupported in (x.half(), x[:, :31], torch.randn(32, 4, device="cuda").T):
            with pytest.raises(ValueError, match="width 32"):
                short_layer_norm(unsupported, weight, None, 1e-5)
        with pytest.raises(ValueError, match="weight"):
            short_layer_norm(x, weight[:31], None, 1e-5)


@gpu
def test_short_model_graph_tracks_epsilon_affine_updates_and_fresh_masks():
    from benchmarks.compare_short_layer_norm import PriorNormalization

    m = model().cuda()
    control = PriorNormalization(m.config).cuda().eval()
    control.load_state_dict(m.state_dict(), strict=True)
    x = torch.randn(64, 128, 32, device="cuda")
    with torch.inference_mode():
        assert torch.equal(m(x), control(x))
        original = m._inference_graph
        for item in (m, control):
            item.final_norm.eps = 3e-4
            item.final_norm.weight.mul_(1.27)
            item.final_norm.bias.add_(.17)
        assert torch.equal(m(x), control(x))
        assert m._inference_graph is not original
        mask = torch.arange(128, device="cuda")[None, :].expand(64, 128) < 93
        assert torch.equal(m(x, mask), control(x, mask))
        mask[:, 48:] = False
        actual = m(x * .3, mask)
        assert torch.equal(actual, control(x * .3, mask))
        assert torch.count_nonzero(actual[~mask]) == 0
