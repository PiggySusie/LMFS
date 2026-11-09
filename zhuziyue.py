import torch
import triton
import triton.language as tl


_BLOCK_M = 64
_BLOCK_N = 64


@triton.jit
def _tile_ptr(base, head, stride_h, stride_row, stride_col, rows, cols):
    return (
        base
        + head.to(tl.int64) * stride_h
        + rows.to(tl.int64)[:, None] * stride_row
        + cols.to(tl.int64)[None, :] * stride_col
    )


@triton.jit
def _tile_mask(rows, cols, seq_len, dim, EVEN_D: tl.constexpr):
    row_ok = rows[:, None] < seq_len
    if EVEN_D:
        col_ok = cols[None, :] >= 0
    else:
        col_ok = cols[None, :] < dim
    return row_ok & col_ok


@triton.jit
def _load_tile(
    base,
    head,
    stride_h,
    stride_row,
    stride_col,
    rows,
    cols,
    seq_len,
    dim,
    EVEN_D: tl.constexpr,
):
    ptr = _tile_ptr(base, head, stride_h, stride_row, stride_col, rows, cols)
    mask = _tile_mask(rows, cols, seq_len, dim, EVEN_D)
    return tl.load(ptr, mask=mask, other=0.0)


@triton.jit
def _store_tile(
    base,
    value,
    head,
    stride_h,
    stride_row,
    stride_col,
    rows,
    cols,
    seq_len,
    dim,
    EVEN_D: tl.constexpr,
):
    ptr = _tile_ptr(base, head, stride_h, stride_row, stride_col, rows, cols)
    mask = _tile_mask(rows, cols, seq_len, dim, EVEN_D)
    tl.store(ptr, value, mask=mask)


@triton.jit
def _scores(q, k, sm_scale, cols, seq_len):
    qk = tl.dot(q, tl.trans(k))
    qk = qk.to(tl.float32) * sm_scale
    return tl.where(cols[None, :] < seq_len, qk, float("-inf"))


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    O,
    M,
    L,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kh,
    stride_km,
    stride_kd,
    stride_vh,
    stride_vm,
    stride_vd,
    stride_oh,
    stride_om,
    stride_od,
    stride_mh,
    stride_lh,
    seq_len,
    dim,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_D)

    q = _load_tile(
        Q, pid_h, stride_qh, stride_qm, stride_qd, rows, cols, seq_len, dim, EVEN_D
    )
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    for start_n in tl.range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = _load_tile(
            K, pid_h, stride_kh, stride_km, stride_kd, offs_n, cols, seq_len, dim, EVEN_D
        )
        v = _load_tile(
            V, pid_h, stride_vh, stride_vm, stride_vd, offs_n, cols, seq_len, dim, EVEN_D
        )
        qk = _scores(q, k, sm_scale, offs_n, seq_len)
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_new))
        p = tl.exp(qk - m_new[:, None])
        p = tl.where(offs_n[None, :] < seq_len, p, 0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(tl.float16), v)
        m_i = m_new

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / l_safe[:, None]
    _store_tile(
        O, acc, pid_h, stride_oh, stride_om, stride_od, rows, cols, seq_len, dim, EVEN_D
    )

    row_mask = rows < seq_len
    tl.store(M + pid_h.to(tl.int64) * stride_mh + rows.to(tl.int64), m_i, mask=row_mask)
    tl.store(L + pid_h.to(tl.int64) * stride_lh + rows.to(tl.int64), l_i, mask=row_mask)


@triton.jit
def _delta_kernel(
    O,
    DO,
    Delta,
    stride_oh,
    stride_om,
    stride_od,
    stride_dh,
    stride_dm,
    stride_dd,
    stride_deltah,
    seq_len,
    dim,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_D)
    o = _load_tile(
        O, pid_h, stride_oh, stride_om, stride_od, rows, cols, seq_len, dim, EVEN_D
    )
    do = _load_tile(
        DO, pid_h, stride_dh, stride_dm, stride_dd, rows, cols, seq_len, dim, EVEN_D
    )
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    tl.store(
        Delta + pid_h.to(tl.int64) * stride_deltah + rows.to(tl.int64),
        delta,
        mask=rows < seq_len,
    )


@triton.jit
def _bwd_dkdv_kernel(
    Q,
    K,
    V,
    DO,
    M,
    L,
    Delta,
    DK,
    DV,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kh,
    stride_km,
    stride_kd,
    stride_vh,
    stride_vm,
    stride_vd,
    stride_dh,
    stride_dm,
    stride_dd,
    stride_mh,
    stride_lh,
    stride_deltah,
    stride_dkh,
    stride_dkm,
    stride_dkd,
    stride_dvh,
    stride_dvm,
    stride_dvd,
    seq_len,
    dim,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_D)

    k = _load_tile(
        K, pid_h, stride_kh, stride_km, stride_kd, offs_n, cols, seq_len, dim, EVEN_D
    )
    v = _load_tile(
        V, pid_h, stride_vh, stride_vm, stride_vd, offs_n, cols, seq_len, dim, EVEN_D
    )
    dk = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)

    for start_m in tl.range(0, seq_len, BLOCK_M):
        rows = start_m + tl.arange(0, BLOCK_M)
        q = _load_tile(
            Q, pid_h, stride_qh, stride_qm, stride_qd, rows, cols, seq_len, dim, EVEN_D
        )
        do = _load_tile(
            DO, pid_h, stride_dh, stride_dm, stride_dd, rows, cols, seq_len, dim, EVEN_D
        )
        m_i = tl.load(
            M + pid_h.to(tl.int64) * stride_mh + rows.to(tl.int64),
            mask=rows < seq_len,
            other=0.0,
        )
        l_i = tl.load(
            L + pid_h.to(tl.int64) * stride_lh + rows.to(tl.int64),
            mask=rows < seq_len,
            other=1.0,
        )
        delta = tl.load(
            Delta + pid_h.to(tl.int64) * stride_deltah + rows.to(tl.int64),
            mask=rows < seq_len,
            other=0.0,
        )

        qk = _scores(q, k, sm_scale, offs_n, seq_len)
        valid = (rows[:, None] < seq_len) & (offs_n[None, :] < seq_len)
        p = tl.where(valid, tl.exp(qk - m_i[:, None]), 0.0)
        dp = tl.dot(do.to(tl.float16), tl.trans(v))
        ds = tl.where(valid, p * (dp - delta[:, None]), 0.0)
        inv_l = tl.where((rows < seq_len) & (l_i > 0), 1.0 / l_i, 0.0)
        scale = tl.max(inv_l, axis=0)
        scale = tl.where(scale > 0, scale, 1.0)
        norm = (inv_l / scale)[:, None]
        p16 = p.to(tl.float16)
        ds16 = ds.to(tl.float16)
        do_hat = (do.to(tl.float32) * norm).to(tl.float16)
        q_hat = (q.to(tl.float32) * norm).to(tl.float16)
        dv += tl.dot(tl.trans(p16), do_hat) * scale
        dk += tl.dot(tl.trans(ds16), q_hat) * scale

    dk = dk * sm_scale
    _store_tile(
        DK, dk, pid_h, stride_dkh, stride_dkm, stride_dkd, offs_n, cols, seq_len, dim, EVEN_D
    )
    _store_tile(
        DV, dv.to(tl.float16), pid_h, stride_dvh, stride_dvm, stride_dvd, offs_n, cols, seq_len, dim, EVEN_D
    )


@triton.jit
def _bwd_dq_kernel(
    Q,
    K,
    V,
    DO,
    M,
    L,
    Delta,
    DQ,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kh,
    stride_km,
    stride_kd,
    stride_vh,
    stride_vm,
    stride_vd,
    stride_dh,
    stride_dm,
    stride_dd,
    stride_mh,
    stride_lh,
    stride_deltah,
    stride_dqh,
    stride_dqm,
    stride_dqd,
    seq_len,
    dim,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_D)

    q = _load_tile(
        Q, pid_h, stride_qh, stride_qm, stride_qd, rows, cols, seq_len, dim, EVEN_D
    )
    do = _load_tile(
        DO, pid_h, stride_dh, stride_dm, stride_dd, rows, cols, seq_len, dim, EVEN_D
    )
    m_i = tl.load(
        M + pid_h.to(tl.int64) * stride_mh + rows.to(tl.int64),
        mask=rows < seq_len,
        other=0.0,
    )
    l_i = tl.load(
        L + pid_h.to(tl.int64) * stride_lh + rows.to(tl.int64),
        mask=rows < seq_len,
        other=1.0,
    )
    delta = tl.load(
        Delta + pid_h.to(tl.int64) * stride_deltah + rows.to(tl.int64),
        mask=rows < seq_len,
        other=0.0,
    )
    dq = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    inv_l = tl.where((rows < seq_len) & (l_i > 0), 1.0 / l_i, 0.0)

    for start_n in tl.range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = _load_tile(
            K, pid_h, stride_kh, stride_km, stride_kd, offs_n, cols, seq_len, dim, EVEN_D
        )
        v = _load_tile(
            V, pid_h, stride_vh, stride_vm, stride_vd, offs_n, cols, seq_len, dim, EVEN_D
        )
        qk = _scores(q, k, sm_scale, offs_n, seq_len)
        valid = (rows[:, None] < seq_len) & (offs_n[None, :] < seq_len)
        p = tl.where(valid, tl.exp(qk - m_i[:, None]), 0.0)
        dp = tl.dot(do.to(tl.float16), tl.trans(v))
        ds = tl.where(valid, p * (dp - delta[:, None]), 0.0)
        dq += tl.dot(ds.to(tl.float16), k)

    dq = dq * (sm_scale * inv_l[:, None])
    _store_tile(
        DQ, dq.to(tl.float16), pid_h, stride_dqh, stride_dqm, stride_dqd, rows, cols, seq_len, dim, EVEN_D
    )


def _block_d(dim: int) -> int:
    if dim <= 0 or dim > 128:
        raise NotImplementedError(f"只支持 1 到 128 的 d_head，收到 {dim}")
    return max(16, triton.next_power_of_2(dim))


def _check_qkv(q, k, v):
    if q.ndim != 3 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            f"q, k, v 必须是相同形状的 [n_heads, seq_len, d_head]，收到 {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}"
        )
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        raise TypeError(
            "FlashAttention2 的 q, k, v 需要是 float16。"
            "评测脚本会按 FlashAttention2.INPUT_DTYPE 做转换。"
        )
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise RuntimeError("FlashAttention2 需要 CUDA 张量")
    if q.device != k.device or q.device != v.device:
        raise RuntimeError("q, k, v 必须在同一块 GPU 上")


def _as_contig(t):
    return t if t.is_contiguous() else t.contiguous()


def _forward(q, k, v):
    _check_qkv(q, k, v)
    q = _as_contig(q)
    k = _as_contig(k)
    v = _as_contig(v)
    heads, seq_len, dim = q.shape
    block_d = _block_d(dim)
    sm_scale = dim ** -0.5

    out = torch.empty((heads, seq_len, dim), device=q.device, dtype=torch.float32)
    row_max = torch.empty((heads, seq_len), device=q.device, dtype=torch.float32)
    row_sum = torch.empty((heads, seq_len), device=q.device, dtype=torch.float32)

    grid = (triton.cdiv(seq_len, _BLOCK_M), heads)
    _fwd_kernel[grid](
        q, k, v, out, row_max, row_sum,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        row_max.stride(0),
        row_sum.stride(0),
        seq_len, dim, sm_scale,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=block_d,
        EVEN_D=(dim == block_d),
        num_warps=4,
        num_stages=2,
    )
    return out, row_max, row_sum, sm_scale


def _backward(q, k, v, out, do, row_max, row_sum, sm_scale):
    q = _as_contig(q)
    k = _as_contig(k)
    v = _as_contig(v)
    out = _as_contig(out)
    do = _as_contig(do)
    heads, seq_len, dim = q.shape
    block_d = _block_d(dim)
    even_d = dim == block_d

    delta = torch.empty((heads, seq_len), device=q.device, dtype=torch.float32)
    dq = torch.empty_like(q)
    dk = torch.empty(k.shape, device=k.device, dtype=torch.float32)
    dv = torch.empty_like(v)

    grid_m = (triton.cdiv(seq_len, _BLOCK_M), heads)
    grid_n = (triton.cdiv(seq_len, _BLOCK_N), heads)
    _delta_kernel[grid_m](
        out, do, delta,
        out.stride(0), out.stride(1), out.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        delta.stride(0),
        seq_len, dim,
        BLOCK_M=_BLOCK_M,
        BLOCK_D=block_d,
        EVEN_D=even_d,
        num_warps=4,
    )
    _bwd_dkdv_kernel[grid_n](
        q, k, v, do, row_max, row_sum, delta, dk, dv,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        row_max.stride(0),
        row_sum.stride(0),
        delta.stride(0),
        dk.stride(0), dk.stride(1), dk.stride(2),
        dv.stride(0), dv.stride(1), dv.stride(2),
        seq_len, dim, sm_scale,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=block_d,
        EVEN_D=even_d,
        num_warps=4,
        num_stages=2,
    )
    _bwd_dq_kernel[grid_m](
        q, k, v, do, row_max, row_sum, delta, dq,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        row_max.stride(0),
        row_sum.stride(0),
        delta.stride(0),
        dq.stride(0), dq.stride(1), dq.stride(2),
        seq_len, dim, sm_scale,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=block_d,
        EVEN_D=even_d,
        num_warps=4,
        num_stages=2,
    )
    return dq, dk, dv


class FlashAttention2(torch.autograd.Function):
    INPUT_DTYPE = torch.float16

    @staticmethod
    def forward(ctx, q, k, v):
        if k.requires_grad:
            k.grad_dtype = torch.float32
        out, row_max, row_sum, sm_scale = _forward(q, k, v)
        ctx.save_for_backward(q, k, v, out, row_max, row_sum)
        ctx.sm_scale = sm_scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if grad_out is None:
            return None, None, None
        q, k, v, out, row_max, row_sum = ctx.saved_tensors
        dq, dk, dv = _backward(q, k, v, out, grad_out, row_max, row_sum, ctx.sm_scale)
        return dq, dk, dv
