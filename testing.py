import torch

SHARED_MEMORY_SIZE = 64*1024


def gemm(a: torch.Tensor, b: torch.Tensor, 
    M_TILE_SIZE: int = 32, K_TILE_SIZE: int = 32
):
    import triton
    from kernels import gemm_kernel
    if b.ndim == 3:
        assert a.shape[0] == b.shape[0]
        return torch.cat([gemm(a[i:i+1], b[i], M_TILE_SIZE, K_TILE_SIZE) for i in range(a.shape[0])])
    B, M, N = a.shape
    N2, K = b.shape
    assert N == N2

    o = torch.empty((B, M, K), device=a.device, dtype=a.dtype)

    N_TILE_SIZE = 16

    gemm_kernel[triton.cdiv(M, M_TILE_SIZE) * triton.cdiv(K, K_TILE_SIZE), B](  #256//32 * 64//32 == 8 * 2
        a, b, o,
        a.stride(0), a.stride(1),
        b.stride(0),
        o.stride(0), o.stride(1),
        M, N, K,
        M_TILE_SIZE=M_TILE_SIZE, 
        N_TILE_SIZE=N_TILE_SIZE, 
        K_TILE_SIZE=K_TILE_SIZE
    )

    return o


def my_rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6):
    import triton
    from kernels import rms_norm_kernel
    B, N, M = x.shape
    o = torch.empty_like(x)
    rms_norm_kernel[B, N](
        x, gamma, o,
        x.stride(0), x.stride(1),
        M, triton.next_power_of_2(M),
        eps
    )
    return o


def gem_res_rms(
    a, b, res, 
    M_TILE_SIZE: int = 32, K_TILE_SIZE: int = 32, 
):
    import triton
    from kernels import gemm_res_rms_kernel
    if b.ndim == 3:
        assert a.shape[0] == b.shape[0]
        pairs = [gem_res_rms(a[i:i+1], b[i], res[i:i+1], M_TILE_SIZE, K_TILE_SIZE) for i in range(a.shape[0])]
        return tuple(torch.cat(items) for items in zip(*pairs))
    B, M, N = a.shape
    N2, K = b.shape
    assert N == N2

    o = torch.empty((B, M, K), device=a.device, dtype=a.dtype)
    partial = torch.empty((B, M, triton.cdiv(K, K_TILE_SIZE)), device=a.device, dtype=torch.float32)
    h0 = torch.empty_like(o)

    N_TILE_SIZE = 16

    gemm_res_rms_kernel[triton.cdiv(M, M_TILE_SIZE) * triton.cdiv(K, K_TILE_SIZE), B](  #256//32 * 64//32 == 8 * 2
        a, b, o, h0, partial, res,
        a.stride(0), a.stride(1),
        b.stride(0),
        o.stride(0), o.stride(1),
        partial.stride(0), partial.stride(1),
        M, N, K,
        M_TILE_SIZE=M_TILE_SIZE, 
        N_TILE_SIZE=N_TILE_SIZE, 
        K_TILE_SIZE=K_TILE_SIZE
    )

    return o, partial


def finish_gemm_res_rms(a, partial, gamma, eps: float = 1e-6):
    import triton
    from kernels import finish_gemm_res_rms_kernel
    o = torch.empty_like(a)

    M, N = o.shape[1:]
    K = partial.shape[-1]

    m2_dim = 1
    reduced = torch.empty((*a.shape[:2], 1), device=a.device, dtype=torch.float32)
    n2_dim = triton.next_power_of_2(N)
    k2_dim = triton.next_power_of_2(K)

    finish_gemm_res_rms_kernel[a.shape[0], M](
        a, partial, gamma, o, reduced,
        a.stride(0), a.stride(1),
        partial.stride(0), partial.stride(1),
        reduced.stride(0),
        M, N, K,
        m2_dim, n2_dim, k2_dim,
        eps
    )
    return o


def reference_res_rms_gemm(a, b, res, gamma, eps=1e-6, K_TILE_SIZE: int = 32):
    from torch.nn.functional import rms_norm

    l = []
    for i in range(0, b.shape[-1], K_TILE_SIZE):
        intermediate = (a @ b[..., i:i+K_TILE_SIZE] + res[..., i:i+K_TILE_SIZE]).to(torch.float32)**2
        l.append(intermediate.sum(dim=-1))

    l = torch.stack(l, dim=-1)
    o = a @ b + res
    return o, l


def reference_finish_gemm_res_rms(a, partial, gamma, eps: float = 1e-6):
    partial = torch.sum(partial, dim=-1, keepdim=True)
    return a * gamma[None, None, :] * torch.rsqrt(partial / a.shape[-1] + eps)


def reference_gemm_res_rms_gemm(
    x: torch.Tensor,
    weight1: torch.Tensor,
    z: torch.Tensor,
    gamma: torch.Tensor,
    weight2: torch.Tensor,
    eps: float = 1e-6,
):
    from torch.nn.functional import rms_norm
    after_res =x@weight1 + z
    return rms_norm(after_res.float(), after_res.shape[-1:], gamma.float(), eps).to(x.dtype) @ weight2


def my_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos, sin
):
    h0 = x @ weight
    x1, x2 = h0[..., ::2].float(), h0[..., 1::2].float()
    cos, sin = cos.float(), sin.float()

    out = torch.empty_like(h0)
    out[..., ::2] = cos[None, :]*x1 - sin[None, :]*x2
    out[..., 1::2] = sin[None, :]*x1 + cos[None, :]*x2

    return out

def gemm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos, sin,
    M_TILE_SIZE: int = 32, K_TILE_SIZE: int = 32, 
):
    import triton
    from kernels import gemm_rope_kernel
    B, M, N = x.shape
    N2, K = weight.shape
    assert N == N2

    o = torch.empty((B, M, K), device=x.device, dtype=x.dtype)

    N_TILE_SIZE = 16

    gemm_rope_kernel[(triton.cdiv(M, M_TILE_SIZE) * triton.cdiv(K, K_TILE_SIZE)), B](
        x, weight, o,
        cos, sin,
        x.stride(0), x.stride(1),
        weight.stride(0),
        o.stride(0), o.stride(1),
        cos.stride(0),
        M, N, K,
        M_TILE_SIZE,
        N_TILE_SIZE,
        K_TILE_SIZE
    )

    return o
    


def precompute_rope_cache(
    seq_len: int,
    head_dim: int,
    *,
    base: float = 10000.0,
    device=None,
    dtype=torch.float32,
):
    assert head_dim % 2 == 0

    inv_freq = base ** (
        -torch.arange(
            0, head_dim, 2,
            device=device,
            dtype=torch.float32,
        ) / head_dim
    )  # [head_dim / 2]

    positions = torch.arange(
        seq_len,
        device=device,
        dtype=torch.float32,
    )

    angles = positions[:, None] * inv_freq[None, :]
    cos = angles.cos().to(dtype)
    sin = angles.sin().to(dtype)

    return cos, sin


def gemm_CE(x, weight, y, M_TILE_SIZE = 16, K_TILE_SIZE = 16):
    import triton
    B, M, N = x.shape
    K = weight.shape[-1]
    output = torch.empty(x.shape[:2], device=x.device, dtype=torch.float32)
    logsumexp_o = torch.empty_like(output)
    for b in range(x.shape[0]):
        for m in range(triton.cdiv(M, M_TILE_SIZE)):
            max_logit = torch.full((min(M_TILE_SIZE, M - m*M_TILE_SIZE),), -float("inf"), device=x.device)
            sum_exp = torch.zeros_like(max_logit, dtype=torch.float32)
            desired_logit = torch.zeros_like(max_logit, dtype=torch.float32)
            for k in range(triton.cdiv(K, K_TILE_SIZE)):
                S = x[b, m*M_TILE_SIZE:(m+1)*M_TILE_SIZE, :] @ \
                    weight[:, k*K_TILE_SIZE:(k+1)*K_TILE_SIZE]
                new_max_logit = torch.maximum(max_logit, torch.amax(S, dim=-1))
                sum_exp = torch.exp(max_logit - new_max_logit)*sum_exp + \
                    torch.exp(S - new_max_logit[:, None]).sum(dim=-1)
                labels = y[b, m*M_TILE_SIZE:(m+1)*M_TILE_SIZE]
                columns = torch.arange(k*K_TILE_SIZE, min((k+1)*K_TILE_SIZE, K), device=x.device)
                desired_logit += torch.where(labels[:, None] == columns, S.float(), 0).sum(-1)
                max_logit = new_max_logit
            output[b, m*M_TILE_SIZE:(m+1)*M_TILE_SIZE] = max_logit + torch.log(sum_exp) - desired_logit
            logsumexp_o[b, m*M_TILE_SIZE:(m+1)*M_TILE_SIZE] = max_logit + torch.log(sum_exp)

    return output, logsumexp_o

def reference_gemm_CE(x, weight, y):
    from torch.nn.functional import cross_entropy
    return cross_entropy(
        (x @ weight).float().transpose(1, 2),
        y,
        reduction="none",
    )



def reference_gemm_rope(x, weight, cos, sin, weight2):
    return my_rope(x, weight, cos, sin) @ weight2

def main():
    from coda import GemmCE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    x = torch.randn(2, 35, 19, device=device, requires_grad=True)
    weight1 = torch.nn.Parameter(torch.randn(19, 38, device=device) / 19**0.5)
    y = torch.randint(0, 38, (2, 35), device=device)

    result = GemmCE.apply(x, weight1, y)
    reference = reference_gemm_CE(x, weight1, y)
    torch.testing.assert_close(result, reference, atol=1e-5, rtol=1e-5)

    grad_output = torch.randn_like(result)
    inputs = (x, weight1)
    grads = torch.autograd.grad(result, inputs, grad_output)
    reference_grads = torch.autograd.grad(reference, inputs, grad_output)
    for grad, reference_grad in zip(grads, reference_grads):
        torch.testing.assert_close(grad, reference_grad, atol=1e-5, rtol=1e-5)

    print("GEMM + CE forward and backward match PyTorch")
    # launch_visualizer()


def coda_rms(*args):
    from coda import GemmResRmsGemm2
    return GemmResRmsGemm2.apply(*args)


def coda_rope(*args):
    from coda import GemmRope
    return GemmRope.apply(*args)


def coda_ce(*args):
    from coda import GemmCE
    return GemmCE.apply(*args)


def liger_rms(x, w, z, gamma, w2, eps):
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    return LigerRMSNormFunction.apply(x @ w + z, gamma, eps, 0.0, 'gemma', False, None) @ w2


def liger_rope(x, w, cos, sin, w2, *, cache=None):
    import math
    from liger_kernel.ops.rope import LigerRopeFunction
    h = x @ w
    B, M, K = h.shape
    H = math.gcd(K, 128)
    h = h.reshape(B, M, K // H, H)
    q = torch.cat((h[..., ::2], h[..., 1::2]), -1).transpose(1, 2).contiguous()
    c, s = cache if cache is not None else tuple(t[:, :H // 2].float().repeat(1, 2).unsqueeze(0).contiguous() for t in (cos, sin))
    a, _ = LigerRopeFunction.apply(q, torch.zeros_like(q[:, :1]), c, s, None, 1)
    h = a.transpose(1, 2)
    return torch.stack((h[..., :H // 2], h[..., H // 2:]), -1).reshape(B, M, K) @ w2


def liger_ce(x, w, labels):
    from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
    loss = LigerFusedLinearCrossEntropyFunction.apply(
        x.flatten(0, 1), w.T.contiguous(), labels.flatten(), None, None, -100, 0.0, 0.0,
        'none', None, False, torch.float32, False, False, False)
    return (loss[0] if isinstance(loss, tuple) else loss).reshape(labels.shape)


def liger_ce_unfused(x, w, labels):
    from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
    logits = (x @ w).float().flatten(0, 1)
    return LigerCrossEntropyFunction.apply(logits, labels.flatten(), None, -100, 0.0, 0.0, 'none')[0].reshape(labels.shape)


def flashinfer_rms(x, w, z, gamma, w2, eps):
    from flashinfer import rmsnorm
    h = x @ w + z
    return rmsnorm(h.flatten(0, 1), gamma, eps).reshape(h.shape) @ w2


def flashinfer_rope(x, w, cos, sin, w2, *, cache=None):
    import math
    from flashinfer.rope import apply_rope_with_cos_sin_cache
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError('FlashInfer RoPE requires FP16/BF16 inputs')
    h = x @ w
    B, M, K = h.shape
    H = math.gcd(K, 128)
    if cache is None:
        cache = (torch.arange(M, device=x.device).repeat(B),
                 torch.cat((cos[:, :H // 2], sin[:, :H // 2]), -1).float().contiguous())
    rotated, _ = apply_rope_with_cos_sin_cache(cache[0], h.flatten(0, 1),
                                               torch.zeros((B * M, H), device=x.device, dtype=x.dtype),
                                               H, cache[1], is_neox=False)
    return rotated.reshape(B, M, K) @ w2


def prepare_backend(fn, args):
    from functools import partial
    from importlib import import_module
    modules = {coda_rms: 'coda', coda_rope: 'coda', coda_ce: 'coda',
               liger_rms: 'liger_kernel.ops.rms_norm', liger_rope: 'liger_kernel.ops.rope',
               liger_ce: 'liger_kernel.ops.fused_linear_cross_entropy',
               liger_ce_unfused: 'liger_kernel.ops.cross_entropy',
               flashinfer_rms: 'flashinfer.norm', flashinfer_rope: 'flashinfer.rope',
               quack_rms: 'quack.linear', quack_rope: 'quack.linear', quack_ce: 'quack.linear'}
    if fn in modules:
        try:
            import_module(modules[fn])
        except ImportError:
            pass
    if fn is flashinfer_rope:
        import math
        x, w, cos, sin, _ = args
        H = math.gcd(w.shape[-1], 128)
        cache = (torch.arange(x.shape[1], device=x.device).repeat(x.shape[0]),
                 torch.cat((cos[:, :H // 2], sin[:, :H // 2]), -1).float().contiguous())
        prepared = partial(fn, cache=cache)
    elif fn is liger_rope:
        import math
        H = math.gcd(args[1].shape[-1], 128)
        cache = tuple(t[:, :H // 2].float().repeat(1, 2).unsqueeze(0).contiguous() for t in args[2:4])
        prepared = partial(fn, cache=cache)
    else:
        prepared = fn
    prepared.modes = getattr(fn, 'modes', ('fwd', 'bwd', 'fwd+bwd'))
    prepared.configs = getattr(fn, 'configs', lambda: '')
    return prepared


def _quack_mm(x, w):
    if not torch.compiler.is_compiling() and torch.cuda.get_device_capability()[0] < 8:
        raise NotImplementedError('QuACK requires SM80+')
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError('QuACK requires FP16/BF16 inputs')
    from quack.linear import linear_func
    return linear_func(x, w.T)


def quack_rms(x, w, z, gamma, w2, eps):
    h = _quack_mm(x, w) + z
    return _quack_mm(torch.nn.functional.rms_norm(h.float(), (h.shape[-1],), gamma.float(), eps).to(x.dtype), w2)


def quack_rope(x, w, cos, sin, w2):
    h = _quack_mm(x, w).float()
    cos, sin = cos.float(), sin.float()
    h = torch.stack((h[..., ::2] * cos - h[..., 1::2] * sin,
                     h[..., ::2] * sin + h[..., 1::2] * cos), -1).flatten(-2).to(x.dtype)
    return _quack_mm(h, w2)


def quack_ce(x, w, labels):
    return torch.nn.functional.cross_entropy(_quack_mm(x, w).float().transpose(1, 2), labels, reduction='none')


def _configs(*names):
    import kernels
    return '; '.join(f'{name}: {kernel.best_config}' for name in names
                     if hasattr(kernel := getattr(kernels, name), 'best_config'))


coda_rms.configs = lambda: _configs('gemm_res_rms_tuned', 'gemm_tuned', 'backward_gemm_rms_res_tuned')
coda_rope.configs = lambda: _configs('gemm_rope_tuned', 'gemm_tuned')
coda_ce.configs = lambda: _configs('gemm_CE_tuned')


for _fn in (liger_ce, flashinfer_rms, flashinfer_rope):
    _fn.modes = ('fwd',)


if __name__ == "__main__":
    main()
