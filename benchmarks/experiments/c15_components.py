"""Campaign 15 research kernels; deliberately outside production dispatch."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice


@triton.jit
def _welford_norm(X, W, B, Y, ROWS: tl.constexpr, N: tl.constexpr,
                  EPS: tl.constexpr, BR: tl.constexpr, BIAS: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    lanes = tl.arange(0, 32)
    mean = tl.full((BR, 32), 0., tl.float32)
    m2 = tl.full((BR, 32), 0., tl.float32)
    count = tl.where(lanes[None, :] * 4 < N, 4., 0.) + tl.zeros((BR, 32), tl.float32)
    for i in tl.static_range(4):
        cols = lanes * 4 + i
        val = tl.load(X + rows[:, None] * N + cols[None, :],
                      (rows[:, None] < ROWS) & (cols[None, :] < N), 0.)
        delta = val - mean
        updated = tl.fma(delta, 1. / (i + 1), mean)
        m2 = tl.fma(delta, val - updated, m2)
        mean = updated
    # Each active lane consumes four adjacent values, followed by the same
    # descending warp tree as the installed PyTorch CUDA Welford path.
    for offset in tl.static_range(16, 0, -1):
        # static_range below uses powers of two, not all integer offsets.
        if offset == 16 or offset == 8 or offset == 4 or offset == 2 or offset == 1:
            indices = tl.broadcast_to(tl.minimum(lanes + offset, 31)[None, :], (BR, 32))
            other_mean = tl.gather(mean, indices, 1)
            other_m2 = tl.gather(m2, indices, 1)
            other_count = tl.gather(count, indices, 1)
            total = count + other_count
            coef = tl.div_rn(1., total)
            na, nb = other_count * coef, count * coef
            delta = mean - other_mean
            mean = tl.where(total > 0, tl.fma(na, other_mean, nb * mean), 0.)
            m2 = tl.where(total > 0, other_m2 + m2 + delta * delta * other_count * nb, 0.)
            count = total
    mean = tl.sum(tl.where(lanes[None, :] == 0, mean, 0.), 1)
    variance = tl.sum(tl.where(lanes[None, :] == 0, m2, 0.), 1) / N
    rstd = tl.rsqrt(variance + EPS)
    for i in tl.static_range(4):
        cols = lanes * 4 + i
        bounds = (rows[:, None] < ROWS) & (cols[None, :] < N)
        value = tl.load(X + rows[:, None] * N + cols[None, :], bounds, 0.)
        weight = tl.load(W + cols, cols < N, 0.)
        y = weight[None, :] * (rstd[:, None] * (value - mean[:, None]))
        if BIAS:
            y = tl.fma(weight[None, :], rstd[:, None] * (value - mean[:, None]),
                       tl.load(B + cols, cols < N, 0.)[None, :])
        tl.store(Y + rows[:, None] * N + cols[None, :], y, bounds)


def welford_norm(x, weight, bias, eps, rows_per_program=4):
    assert x.shape[-1] in (32, 128) and x.is_contiguous()
    y = torch.empty_like(x)
    rows = x.numel() // x.shape[-1]
    _welford_norm[(triton.cdiv(rows, rows_per_program),)](
        x, weight, bias if bias is not None else weight, y, rows, x.shape[-1],
        eps, rows_per_program, bias is not None, num_warps=4, num_stages=1,
    )
    return y


@triton.jit
def _norm(X, U, W, B, MASK, R, Y, ROWS: tl.constexpr, N: tl.constexpr,
          EPS: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
          RESIDUAL: tl.constexpr, BIAS: tl.constexpr, HAS_MASK: tl.constexpr,
          ZERO_R: tl.constexpr, ZERO_Y: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.arange(0, BC)
    offsets = rows[:, None] * N + cols[None, :]
    bounds = (rows[:, None] < ROWS) & (cols[None, :] < N)
    value = tl.load(X + offsets, bounds, 0)
    if RESIDUAL:
        value += tl.load(U + offsets, bounds, 0)
    valid = rows < ROWS
    if HAS_MASK:
        valid &= tl.load(MASK + rows, rows < ROWS, 0)
    if ZERO_R:
        value = tl.where(valid[:, None], value, 0.)
    if RESIDUAL:
        tl.store(R + offsets, value, bounds)
    mean = tl.sum(value, 1) / N
    centered = tl.where(cols[None, :] < N, value - mean[:, None], 0.)
    variance = tl.sum(centered * centered, 1) / N
    weight = tl.load(W + cols, cols < N, 0.)
    output = centered * tl.rsqrt(variance[:, None] + EPS) * weight[None, :]
    if BIAS:
        output += tl.load(B + cols, cols < N, 0.)[None, :]
    if ZERO_Y:
        output = tl.where(valid[:, None], output, 0.)
    tl.store(Y + offsets, output, bounds)


def norm(x, weight, bias, eps, *, rows_per_program=8, update=None, mask=None,
         zero_invalid_residual=False, zero_invalid_normalized=False):
    assert x.is_cuda and x.dtype == torch.float32 and x.is_contiguous()
    assert x.shape[-1] == weight.numel() and weight.is_contiguous()
    assert update is None or (update.shape == x.shape and update.is_contiguous())
    assert mask is None or (mask.shape == x.shape[:-1] and mask.is_contiguous())
    width = x.shape[-1]
    rows = x.numel() // width
    out = torch.empty_like(x)
    residual = torch.empty_like(x) if update is not None else out
    _norm[(triton.cdiv(rows, rows_per_program),)](
        x, update if update is not None else x, weight,
        bias if bias is not None else weight, mask if mask is not None else x,
        residual, out, rows, width, eps, rows_per_program,
        triton.next_power_of_2(width), update is not None, bias is not None,
        mask is not None, zero_invalid_residual, zero_invalid_normalized,
        num_warps=4, num_stages=1, enable_fp_fusion=False,
    )
    return (residual, out) if update is not None else out


@triton.jit
def _rne(x):
    bits = tl.inline_asm_elementwise(
        "cvt.rn.tf32.f32 $0, $1;", constraints="=r,f", args=[x],
        dtype=tl.int32, is_pure=True, pack=1,
    )
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _gemm_gelu(X, W, B, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               GROUP: tl.constexpr):
    pid = tl.program_id(0)
    pm, pn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    group_id = pid // (GROUP * pn)
    first_m = group_id * GROUP
    group_m = tl.minimum(pm - first_m, GROUP)
    mi = first_m + (pid % (GROUP * pn)) % group_m
    ni = (pid % (GROUP * pn)) // group_m
    m, n, k = mi * BM + tl.arange(0, BM), ni * BN + tl.arange(0, BN), tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + k
        a = tl.load(X + m[:, None] * K + kk[None, :],
                    (m[:, None] < M) & (kk[None, :] < K), 0.)
        b = tl.load(W + n[None, :] * K + kk[:, None],
                    (n[None, :] < N) & (kk[:, None] < K), 0.)
        acc = tl.dot(_rne(a), _rne(b), acc, input_precision="tf32")
    acc += tl.load(B + n, n < N, 0.)[None, :]
    # Match the baseline's erf GELU, retaining FP32 rounding boundaries.
    y = (0.5 * acc) * (1.0 + libdevice.erf(acc * 0.7071067811865475244))
    tl.store(Y + m[:, None] * N + n[None, :], y,
             (m[:, None] < M) & (n[None, :] < N))


def gemm_gelu(x, weight, bias, tile=(32, 64, 32, 4, 2)):
    assert x.is_cuda and x.dtype == torch.float32 and x.is_contiguous()
    assert weight.is_contiguous() and bias is not None
    n, k = weight.shape
    m = x.numel() // k
    out = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
    bm, bn, bk, warps, stages = tile
    _gemm_gelu[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        x, weight, bias, out, m, n, k, bm, bn, bk, 8,
        num_warps=warps, num_stages=stages, enable_fp_fusion=False,
    )
    return out
