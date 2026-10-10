import triton
import triton.language as tl
import triton_viz


@triton_viz.trace("tracer")
@triton.jit
def gemm_kernel(
    x_ptr, b_ptr, o_ptr,
    ab_stride, am_stride,
    bn_stride,
    ob_stride, om_stride,
    m_dim: tl.constexpr, n_dim: tl.constexpr, k_dim: tl.constexpr,
    M_TILE_SIZE: tl.constexpr, 
    N_TILE_SIZE: tl.constexpr, 
    K_TILE_SIZE: tl.constexpr
):
    batch = tl.program_id(0)
    mk = tl.program_id(1)   # up to 16

    m = mk // tl.cdiv(k_dim, K_TILE_SIZE)
    k = mk % tl.cdiv(k_dim, K_TILE_SIZE)

    r = m * M_TILE_SIZE + tl.arange(0, M_TILE_SIZE)[:, None]
    c = tl.arange(0, K_TILE_SIZE)[None, :] + K_TILE_SIZE * k
    x_offset = batch*ab_stride + r*am_stride
    b_offset = c
    o_offset = batch*ob_stride + r*om_stride + c

    acc = tl.zeros((M_TILE_SIZE, K_TILE_SIZE), dtype=o_ptr.dtype.element_ty)

    for i in tl.range(tl.cdiv(n_dim, N_TILE_SIZE)):
        s = i * N_TILE_SIZE + tl.arange(0, N_TILE_SIZE)
        x_new_offsets = x_offset + s[None, :]
        b_new_offsets = b_offset + s[:, None]*bn_stride

        x_mask = (r < m_dim) & (s[None, :] < n_dim)
        b_mask = (s[:, None] < n_dim) & (c < k_dim)

        a = tl.load(x_ptr + x_new_offsets, x_mask, other=0)
        b = tl.load(b_ptr + b_new_offsets, b_mask, other=0)

        acc += tl.dot(a, b, input_precision="ieee")

    o_mask = (r < m_dim) & (c < k_dim)

    tl.store(o_ptr + o_offset, acc, o_mask)


@triton_viz.trace("tracer")
@triton.jit
def rms_norm_kernel(
    x_ptr, gammx_ptr, o_ptr,
    b_stride, n_stride,
    m_dim: tl.constexpr, m2_dim: tl.constexpr,
    eps: tl.constexpr
):
    B = tl.program_id(0)
    N = tl.program_id(1)

    x_offset = B * b_stride + N*n_stride + tl.arange(0, m2_dim)
    gammx_offset = tl.arange(0, m2_dim)
    x = tl.load(x_ptr + x_offset, mask=tl.arange(0, m2_dim) < m_dim).cast(tl.float32, other=0)
    gamma = tl.load(gammx_ptr + gammx_offset, mask=tl.arange(0, m2_dim) < m_dim, other=0)
    norm = tl.rsqrt(tl.sum(x*x) / m_dim + eps)
    x = x.cast(gammx_ptr.dtype.element_ty) * norm * gamma

    tl.store(o_ptr+x_offset, x, mask=tl.arange(0, m2_dim) < m_dim)


@triton_viz.trace("tracer")
@triton.jit
def gemm_res_rms_kernel(
    x_ptr, b_ptr, o_ptr, h0_ptr,
    partial_ptr, res_ptr, 
    ab_stride, am_stride,
    bn_stride,
    ob_stride, om_stride,
    partialb_stride, partialm_stride,
    m_dim: tl.constexpr, n_dim: tl.constexpr, k_dim: tl.constexpr,
    M_TILE_SIZE: tl.constexpr, 
    N_TILE_SIZE: tl.constexpr, 
    K_TILE_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    mk = tl.program_id(1)   # up to 16

    m = mk // tl.cdiv(k_dim, K_TILE_SIZE)
    k = mk % tl.cdiv(k_dim, K_TILE_SIZE)

    r = m * M_TILE_SIZE + tl.arange(0, M_TILE_SIZE)[:, None]
    c = tl.arange(0, K_TILE_SIZE)[None, :] + K_TILE_SIZE * k

    x_offset = batch*ab_stride + r*am_stride
    b_offset = c

    acc = tl.zeros((M_TILE_SIZE, K_TILE_SIZE), dtype=o_ptr.dtype.element_ty)

    for i in tl.range(tl.cdiv(n_dim, N_TILE_SIZE)):
        s = i * N_TILE_SIZE + tl.arange(0, N_TILE_SIZE)
        x_new_offsets = x_offset + s[None, :]
        b_new_offsets = b_offset + s[:, None]*bn_stride

        x_mask = (r < m_dim) & (s[None, :] < n_dim)
        b_mask = (s[:, None] < n_dim) & (c < k_dim)

        a = tl.load(x_ptr + x_new_offsets, x_mask, other=0)
        b = tl.load(b_ptr + b_new_offsets, b_mask, other=0)

        acc += tl.dot(a, b)


    o_mask = (r < m_dim) & (c < k_dim)
    o_offset = batch*ob_stride + r*om_stride + c

    res = tl.load(res_ptr + o_offset, o_mask, other=0)

    tl.store(h0_ptr + o_offset, acc, o_mask)
    acc += res

    partial = tl.where(o_mask, acc, 0.0)
    partial = tl.sum(partial*partial, axis=1, keep_dims=True)
    partial_offset = batch*partialb_stride + (m*M_TILE_SIZE + tl.arange(0, M_TILE_SIZE)[:, None]) * partialm_stride + k

    tl.store(o_ptr + o_offset, acc, o_mask)
    tl.store(partial_ptr + partial_offset, partial, r < m_dim)


@triton_viz.trace("tracer")
@triton.jit
def finish_gemm_res_rms_kernel(
    x_ptr, partial_ptr, gammx_ptr, o_ptr, new_partial_ptr,
    ab_stride, am_stride,
    partialb_stride, partialm_stride,
    newpartialb_stride,
    m_dim: tl.constexpr,
    n_dim: tl.constexpr,
    k_dim: tl.constexpr,
    m2_dim: tl.constexpr,
    n2_dim: tl.constexpr,
    k2_dim: tl.constexpr,
    eps: tl.constexpr,
):
    batch = tl.program_id(0)

    ar = tl.arange(0, m2_dim)[:, None]
    ac = tl.arange(0, n2_dim)[None, :]

    partialr = tl.arange(0, m2_dim)[:, None]
    partialc = tl.arange(0, k2_dim)[None, :]
    gammx_offset = tl.arange(0, n2_dim)

    x_offset = batch*ab_stride + ar*am_stride + ac
    partial_offset = batch*partialb_stride +partialr*partialm_stride + partialc
    new_partial_offset = batch* newpartialb_stride + partialr

    a = tl.load(x_ptr + x_offset, (ar < m_dim) & (ac < n_dim), other=0)
    partial = tl.load(partial_ptr + partial_offset, (partialr < m_dim) & (partialc < k_dim), other=0)
    gamma = tl.load(gammx_ptr + gammx_offset, gammx_offset < n_dim, other=0)

    partial = tl.rsqrt(tl.sum(partial, axis=1, keep_dims=True) / n_dim + eps)

    a *= gamma[None, :] * partial

    tl.store(new_partial_ptr + new_partial_offset, partial, partialr < m_dim)
    tl.store(o_ptr + x_offset, a, (ar < m_dim) & (ac < n_dim))


@triton_viz.trace("tracer")
@triton.jit
def backward_gemm_rms_res_kernel(
    grad_o_ptr, weight2_ptr, reduced_partial_ptr,
    h1_ptr, s_hat_ptr, gammx_ptr,
    grad_h2_ptr, grad_h1_ptr, grad_gammx_ptr,
    grad_o_bstride, grad_o_mstride,
    weight2_nstride,
    reduced_partial_bstride,
    h1_bstride, h1_mstride,
    s_hat_bstride,
    m_dim: tl.constexpr, n_dim: tl.constexpr, k_dim: tl.constexpr,
    M_TILE_SIZE: tl.constexpr, 
    N_TILE_SIZE: tl.constexpr, 
    K_TILE_SIZE: tl.constexpr
):
    batch = tl.program_id(0)
    mk = tl.program_id(1)

    m = mk // tl.cdiv(k_dim, K_TILE_SIZE)
    k = mk % tl.cdiv(k_dim, K_TILE_SIZE)

    grad_r = m*M_TILE_SIZE+tl.arange(0, M_TILE_SIZE)[:, None]
    weight_c = tl.arange(0, K_TILE_SIZE)[None, :]+k*K_TILE_SIZE
    grad_offset = batch*grad_o_bstride + grad_r*grad_o_mstride
    weight_offset = weight_c

    acc = tl.zeros((M_TILE_SIZE, K_TILE_SIZE), dtype=grad_o_ptr.dtype.element_ty)

    for i in tl.range(0, tl.cdiv(n_dim, N_TILE_SIZE)):
        n_offset = i*N_TILE_SIZE + tl.arange(0, N_TILE_SIZE) 
        grad_new_offset = grad_offset + n_offset[None, :]
        weight_new_offset = weight_offset + n_offset[:, None] * weight2_nstride

        grad = tl.load(grad_o_ptr + grad_new_offset, mask=(grad_r < m_dim) & (n_offset[None, :] < n_dim), other=0)
        weight = tl.load(weight2_ptr + weight_new_offset, mask=(weight_c < k_dim) & (n_offset[:, None] < n_dim))

        acc += tl.dot(grad, weight)

    m_offset = m*M_TILE_SIZE+tl.arange(0, M_TILE_SIZE)[:, None]

    reduced_offset = batch*reduced_partial_bstride + m_offset
    h1_offset = batch*h1_bstride + m_offset*h1_mstride + k*K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)[None, :]
    s_offset = batch*s_hat_bstride + m_offset
    gammx_offset = k*K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

    reduced_partial = tl.load(reduced_partial_ptr + reduced_offset, mask=(m_offset < m_dim), other=0)
    h1 = tl.load(h1_ptr + h1_offset, mask=(m_offset < m_dim) & (k*K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)[None, :] < k_dim), other=0)
    s_hat = tl.load(s_hat_ptr + s_offset, mask=(m_offset < m_dim), other=0)
    gamma = tl.load(gammx_ptr + gammx_offset, mask=gammx_offset < k_dim, other=0)

    grad_h1 = reduced_partial * (acc * gamma[None, :] - reduced_partial * h1 * s_hat)   
    grad_gamma = tl.sum(tl.sum(acc* h1 * reduced_partial, axis=0), axis=0)

    tl.store(grad_h2_ptr + h1_offset, acc, mask=(m_offset < m_dim)&(k*K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)[None, :] < k_dim))
    tl.store(grad_h1_ptr + h1_offset, grad_h1, mask=(m_offset < m_dim)&(k*K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)[None, :] < k_dim))
    tl.store(grad_gammx_ptr + gammx_offset, grad_gamma, mask=gammx_offset < k_dim)


@triton_viz.trace("tracer")
@triton.jit
def gemm_rope_kernel(
    x_ptr, weight_ptr, o_ptr,
    cos_ptr, sin_ptr,
    xb_stride, xm_stride,
    weightn_stride,
    ob_stride, om_stride,
    cosm_stride,
    m_dim: tl.constexpr, n_dim: tl.constexpr, k_dim: tl.constexpr,
    M_TILE_SIZE: tl.constexpr, 
    N_TILE_SIZE: tl.constexpr, 
    K_TILE_SIZE: tl.constexpr
):
    batch = tl.program_id(0)
    mk = tl.program_id(1)   # up to 16

    m = mk // tl.cdiv(k_dim, K_TILE_SIZE)
    k = mk % tl.cdiv(k_dim, K_TILE_SIZE)

    r = m * M_TILE_SIZE + tl.arange(0, M_TILE_SIZE)[:, None]
    c = tl.arange(0, K_TILE_SIZE)[None, :] + K_TILE_SIZE * k
    x_offset = batch*xb_stride + r*xm_stride
    weight_offset = c
    o_offset = batch*ob_stride + r*om_stride + c

    acc = tl.zeros((M_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)

    for i in tl.range(tl.cdiv(n_dim, N_TILE_SIZE)):
        s = i * N_TILE_SIZE + tl.arange(0, N_TILE_SIZE)
        x_new_offsets = x_offset + s[None, :]
        weight_new_offsets = weight_offset + s[:, None]*weightn_stride

        x_mask = (r < m_dim) & (s[None, :] < n_dim)
        weight_mask = (s[:, None] < n_dim) & (c < k_dim)

        a = tl.load(x_ptr + x_new_offsets, x_mask, other=0)
        b = tl.load(weight_ptr + weight_new_offsets, weight_mask, other=0)

        acc += tl.dot(a, b, input_precision="ieee")

    p = k * (K_TILE_SIZE // 2) + tl.arange(0, K_TILE_SIZE // 2)[None, :]
    cos_offset = r * cosm_stride + p
    mask = (r < m_dim) & (p < k_dim // 2)

    cos = tl.load(cos_ptr + cos_offset, mask=mask, other=0)
    sin = tl.load(sin_ptr + cos_offset, mask=mask, other=0)

    pairs = tl.reshape(
        acc,
        (M_TILE_SIZE, K_TILE_SIZE // 2, 2),
        can_reorder=False,
    )

    x1, x2 = tl.split(pairs)

    p = tl.arange(0, K_TILE_SIZE // 2)[None, :]
    even_c = k * K_TILE_SIZE + 2 * p
    odd_c = even_c + 1

    row_offset = batch * ob_stride + r * om_stride
    even_offset = row_offset + even_c
    odd_offset = row_offset + odd_c

    tl.store(
        o_ptr + even_offset,
        x1 * cos - x2 * sin,
        mask=(r < m_dim) & (even_c < k_dim),
    )
    tl.store(
        o_ptr + odd_offset,
        x1 * sin + x2 * cos,
        mask=(r < m_dim) & (odd_c < k_dim),
    )



@triton_viz.trace("tracer")
@triton.jit
def gemm_CE_kernel(
    x_ptr, weight_ptr, y_ptr, loss_ptr, logsumexp_ptr,
    xb_stride, xm_stride,
    weightn_stride,
    yb_stride, ym_stride,
    numb_stride,
    m_dim: tl.constexpr, n_dim: tl.constexpr, k_dim: tl.constexpr,
    M_TILE_SIZE: tl.constexpr, 
    N_TILE_SIZE: tl.constexpr, 
    K_TILE_SIZE: tl.constexpr
):
    batch = tl.program_id(0)
    m = tl.program_id(1)

    r = m * M_TILE_SIZE + tl.arange(0, M_TILE_SIZE)[:, None]
    x_offset = batch*xb_stride + r*xm_stride

    max_logit = tl.full((M_TILE_SIZE,), -float("inf"), dtype=tl.float32)
    sumexp = tl.zeros_like(max_logit).to(tl.float32)
    desired_logit = tl.zeros_like(max_logit).to(tl.float32)

    for k in tl.range(tl.cdiv(k_dim, K_TILE_SIZE)):
        c = tl.arange(0, K_TILE_SIZE)[None, :] + K_TILE_SIZE * k
        weight_offset = c
        y_offset = batch*yb_stride + r*ym_stride + c

        acc = tl.zeros((M_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)

        for i in tl.range(tl.cdiv(n_dim, N_TILE_SIZE)):
            s = i * N_TILE_SIZE + tl.arange(0, N_TILE_SIZE)
            x_new_offsets = x_offset + s[None, :]
            weight_new_offsets = weight_offset + s[:, None]*weightn_stride

            x_mask = (r < m_dim) & (s[None, :] < n_dim)
            weight_mask = (s[:, None] < n_dim) & (c < k_dim)

            a = tl.load(x_ptr + x_new_offsets, x_mask, other=0)
            b = tl.load(weight_ptr + weight_new_offsets, weight_mask, other=0)

            acc += tl.dot(a, b, input_precision="ieee")

        acc = tl.where(c < k_dim, acc, -float("inf"))
        new_max_logit = tl.maximum(max_logit, tl.max(acc, axis=-1))
        P = acc - new_max_logit[:, None]
        sumexp = tl.exp(max_logit - new_max_logit) * sumexp + tl.sum(tl.exp(P), axis=-1)

        y = tl.load(y_ptr + y_offset, mask=(r < m_dim) & (c < k_dim), other=0)
        desired_logit += tl.sum(tl.where(y != 0, acc, 0.0), axis=-1)
        max_logit = new_max_logit

    o_r = m * M_TILE_SIZE + tl.arange(0, M_TILE_SIZE)
    output_offset = batch*numb_stride + o_r

    logsumexp = tl.log(sumexp) + max_logit

    tl.store(loss_ptr + output_offset, logsumexp - desired_logit, mask=o_r < m_dim)
    tl.store(logsumexp_ptr + output_offset, logsumexp, mask=o_r < m_dim)

