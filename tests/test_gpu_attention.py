"""Direct numerical and boundary tests for the Triton attention kernel."""

from __future__ import annotations

import pytest
import torch

from transformer_opt import attention_forward
from transformer_opt.kernels import triton_available


pytestmark = pytest.mark.skipif(
    not triton_available(),
    reason="CUDA Triton is unavailable",
)


CASES = [
    # (B, S, H, D, dtype, causal, mask mode, input scale)
    (2, 31, 4, 8, torch.float16, True, "all", 1.0),
    (1, 65, 16, 8, torch.float32, True, "prefix", 1.0),
    (1, 1, 1, 16, torch.float16, False, "none", 1.0),
    (2, 31, 4, 32, torch.float16, True, "all", 1.0),
    (1, 63, 4, 64, torch.float16, False, "prefix", 0.25),
    (2, 64, 2, 64, torch.float16, True, "prefix", 1.0),
    (1, 65, 2, 128, torch.float16, False, "minimum", 1.0),
    (1, 127, 4, 32, torch.float16, True, "none", 3.0),
    (1, 129, 4, 64, torch.float16, True, "prefix", 1.0),
    (1, 257, 2, 32, torch.float16, False, "prefix", 1.0),
    (1, 65, 2, 32, torch.float32, True, "prefix", 1.0),
]


def _mask(mode: str, batch: int, seq_len: int) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode == "all":
        return torch.ones(batch, seq_len, device="cuda", dtype=torch.bool)
    if mode == "minimum":
        lengths = torch.ones(batch, device="cuda", dtype=torch.long)
    elif mode == "prefix":
        lengths = torch.linspace(
            seq_len,
            max(1, seq_len // 2),
            batch,
            device="cuda",
        ).to(torch.long)
    else:  # pragma: no cover - test-data guard
        raise ValueError(mode)
    return torch.arange(seq_len, device="cuda")[None, :] < lengths[:, None]


def _assert_within_contract(actual: torch.Tensor, reference: torch.Tensor) -> None:
    actual32 = actual.float()
    reference32 = reference.float()
    error = (actual32 - reference32).abs()
    passed = (error <= 0.001) | (error <= 0.01 * reference32.abs())
    assert torch.isfinite(actual32).all()
    assert passed.all(), (
        f"failed={int((~passed).sum().item())}/{passed.numel()} "
        f"max_abs={error.max().item():.6g}"
    )


@pytest.mark.parametrize(
    "batch,seq_len,heads,head_dim,dtype,causal,mask_mode,input_scale",
    CASES,
)
def test_triton_attention_matches_explicit_reference(
    batch,
    seq_len,
    heads,
    head_dim,
    dtype,
    causal,
    mask_mode,
    input_scale,
):
    for seed in (7, 19):
        torch.manual_seed(seed)
        q = torch.randn(
            batch,
            seq_len,
            heads,
            head_dim,
            device="cuda",
            dtype=dtype,
        ) * input_scale
        k = torch.randn_like(q) * input_scale
        v = torch.randn_like(q)
        valid_mask = _mask(mask_mode, batch, seq_len)
        with torch.inference_mode():
            actual, custom = attention_forward(
                q,
                k,
                v,
                valid_mask,
                causal=causal,
                backend="triton",
            )
            reference, fallback = attention_forward(
                q,
                k,
                v,
                valid_mask,
                causal=causal,
                backend="reference",
            )
        assert custom.selected == "triton"
        assert fallback.selected == "reference"
        _assert_within_contract(actual, reference)


def test_all_keys_masked_is_finite_zero():
    q = torch.randn(2, 65, 2, 32, device="cuda", dtype=torch.float16)
    all_masked = torch.zeros(2, 65, device="cuda", dtype=torch.bool)
    with torch.inference_mode():
        output, decision = attention_forward(
            q,
            q,
            q,
            all_masked,
            causal=True,
            backend="triton",
        )
    assert decision.selected == "triton"
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == 0


def test_float32_ieee_path_follows_tf32_toggle():
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(29)
        q = torch.randn(1, 65, 2, 32, device="cuda", dtype=torch.float32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        with torch.inference_mode():
            actual, _ = attention_forward(q, k, v, causal=True, backend="triton")
            reference, _ = attention_forward(
                q,
                k,
                v,
                causal=True,
                backend="reference",
            )
        _assert_within_contract(actual, reference)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("seq_len", [31, 32, 33, 63, 64, 65, 127, 128, 129, 257])
@pytest.mark.parametrize("head_dim", [32, 128])
@pytest.mark.parametrize("mask_mode", ["none", "minimum"])
def test_causal_frontier_matches_reference_at_tile_boundaries(seq_len, head_dim, mask_mode):
    torch.manual_seed(41)
    q = torch.randn(1, seq_len, 2, head_dim, device="cuda")
    k, v = torch.randn_like(q), torch.randn_like(q)
    valid_mask = _mask(mask_mode, 1, seq_len)
    with torch.inference_mode():
        actual, decision = attention_forward(
            q, k, v, valid_mask, causal=True, backend="triton"
        )
        reference, _ = attention_forward(
            q, k, v, valid_mask, causal=True, backend="reference"
        )
    assert decision.selected == "triton"
    _assert_within_contract(actual, reference)


def test_causal_frontier_cannot_observe_future_key_value_tiles():
    torch.manual_seed(43)
    q = torch.randn(1, 129, 2, 32, device="cuda")
    k, v = torch.randn_like(q), torch.randn_like(q)
    with torch.inference_mode():
        before, _ = attention_forward(q, k, v, causal=True, backend="triton")
        k[:, 64:] *= 100
        v[:, 64:] += 100
        after, _ = attention_forward(q, k, v, causal=True, backend="triton")
    assert torch.equal(before[:, :64], after[:, :64])
