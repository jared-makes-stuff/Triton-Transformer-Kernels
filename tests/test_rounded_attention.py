"""Rounded TF32 keeps normalized-probability, masking, and precision boundaries."""
import pytest
import torch
from types import SimpleNamespace

from transformer_opt.kernels.attention import (
    prefer_rounded_attention, triton_attention, triton_available,
)
from transformer_opt.submission import compare_outputs
from transformer_opt.submission import (
    BaselineTransformer, TransformerConfig, UserOptimizedTransformer,
    copy_model_weights, generate_random_case,
)
from transformer_opt.dispatch import _reference_attention


def test_rounded_attention_rejects_cpu():
    assert not prefer_rounded_attention(torch.zeros(1, 128, 4, 32), None, True)


def test_rounded_policy_requires_triton_even_when_cuda_shape_matches(monkeypatch):
    from transformer_opt.kernels import attention

    cuda = SimpleNamespace(
        get_device_capability=lambda device: (12, 0),
        matmul=SimpleNamespace(allow_tf32=True),
    )
    fake_torch = SimpleNamespace(float32="fp32", is_grad_enabled=lambda: False,
                                 cuda=cuda, backends=SimpleNamespace(cuda=cuda))
    q = SimpleNamespace(is_cuda=True, dtype="fp32", device="cuda",
                        shape=(1, 128, 4, 32))
    monkeypatch.setattr(attention, "torch", fake_torch)
    monkeypatch.setattr(attention, "triton", None)
    assert not attention.prefer_rounded_attention(q, None, True)


gpu = pytest.mark.skipif(
    not triton_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="rounded attention is validated only on SM120",
)


@gpu
@pytest.mark.parametrize("mask_kind", ["none", "prefix", "strided"])
@pytest.mark.parametrize("scale", [.25, 1., 3.])
def test_rounded_attention_matches_reference_with_packed_qkv(mask_kind, scale):
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        torch.manual_seed(1701)
        q, k, v = (torch.randn(1, 128, 3, 4, 32, device="cuda") * scale).unbind(2)
        mask = None
        if mask_kind != "none":
            mask = torch.arange(128, device="cuda")[None, :] < 89
            if mask_kind == "strided":
                storage = torch.zeros(1, 256, device="cuda", dtype=torch.bool)
                storage[:, ::2] = mask
                mask = storage[:, ::2]
        with torch.inference_mode():
            assert prefer_rounded_attention(q, mask, True)
            actual = triton_attention(q, k, v, mask, causal=True)
            expected = _reference_attention(q, k, v, mask, True, None)
            check = compare_outputs(expected, actual, rtol=.01, atol=.001)
        assert check.passed, check
    finally:
        torch.set_float32_matmul_precision(previous)


@gpu
@pytest.mark.parametrize("batch,seq,width,heads,ffn", [(1, 128, 128, 4, 128), (4, 128, 128, 4, 128), (64, 128, 32, 4, 32), (64, 32, 128, 4, 128), (64, 128, 32, 4, 128)])
def test_rounded_model_sensitive_routes_with_other_weights_scales_and_padding(batch, seq, width, heads, ffn):
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        torch.manual_seed(8080)
        config = TransformerConfig(batch, seq, width, heads, ffn, 4, True)
        baseline = BaselineTransformer(config).cuda().eval()
        model = UserOptimizedTransformer(config).cuda().eval()
        copy_model_weights(baseline, model, strict=True)
        with torch.inference_mode():
            for seed, scale, padding in ((2026, .25, 0.), (4096, 1., .25), (1701, 4., .25)):
                x, mask = generate_random_case(config, torch.device("cuda"), torch.float32,
                                               seed, padding, scale)
                expected = baseline(x, mask)
                actual = model(x, mask)
                check = compare_outputs(expected, actual, rtol=.01, atol=.001)
                assert check.passed, (seed, scale, padding, check)
                del x, mask, expected, actual
        expected_reference = 3 if width == 32 and ffn != width else 0
        assert model.attention_backend_counts == {"triton": 12 - expected_reference,
                                                  "sdpa": 0, "reference": expected_reference}
    finally:
        torch.set_float32_matmul_precision(previous)


@gpu
@pytest.mark.parametrize("shape", [(10000, 128, 4, 32), (64, 128, 16, 8), (64, 128, 4, 32), (16, 128, 4, 32), (128, 128, 4, 32), (64, 128, 1, 128), (64, 128, 2, 64)])
def test_rejected_stress_shapes_cannot_select_rounded_kernel(shape):
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        # The predicate uses metadata; expanding a single value avoids allocating
        # the very large rejected tensor just to test its exact shape guard.
        q = torch.zeros((), device="cuda").expand(shape)
        with torch.inference_mode():
            assert not prefer_rounded_attention(q, None, True)
    finally:
        torch.set_float32_matmul_precision(previous)


@gpu
def test_rounded_attention_zero_mask_causality_and_precision_guards():
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        q = torch.randn(1, 128, 4, 32, device="cuda")
        k, v = torch.randn_like(q), torch.randn_like(q)
        with torch.inference_mode():
            mask = torch.zeros(1, 128, device="cuda", dtype=torch.bool)
            assert torch.count_nonzero(triton_attention(q, k, v, mask, causal=True)) == 0
            before = triton_attention(q, k, v, causal=True)
            k[:, 64:] *= 100
            v[:, 64:] += 100
            after = triton_attention(q, k, v, causal=True)
            assert torch.equal(before[:, :64], after[:, :64])
            assert not prefer_rounded_attention(q, None, False)
            assert not prefer_rounded_attention(q[:, :127], None, True)
            assert not prefer_rounded_attention(q.half(), None, True)
            torch.set_float32_matmul_precision("highest")
            assert not prefer_rounded_attention(q, None, True)
        torch.set_float32_matmul_precision("high")
        assert not prefer_rounded_attention(q, None, True)
    finally:
        torch.set_float32_matmul_precision(previous)
