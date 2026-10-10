import torch
import triton

from coda import GemmResRmsGemm, GemmResRmsGemm2, GemmRope
from kernels import (
    backward_gemm_rms_res_kernel,
    finish_gemm_res_rms_kernel,
    gemm_kernel,
    gemm_res_rms_kernel,
    rms_norm_kernel,
)
from utils import launch_visualizer

SHARED_MEMORY_SIZE = 64*1024


def gemm(a: torch.Tensor, b: torch.Tensor, 
    M_TILE_SIZE: int = 32, K_TILE_SIZE: int = 32
):
    B, M, N = a.shape
    B2, N2, K = b.shape
    assert B == B2
    assert N == N2

    o = torch.empty((B, M, K))

    N_TILE_SIZE = 16

    gemm_kernel[B, triton.cdiv(M, M_TILE_SIZE) * triton.cdiv(K, K_TILE_SIZE)](  #256//32 * 64//32 == 8 * 2
        a, b, o,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        o.stride(0), o.stride(1),
        M, N, K,
        M_TILE_SIZE=M_TILE_SIZE, 
        N_TILE_SIZE=N_TILE_SIZE, 
        K_TILE_SIZE=K_TILE_SIZE
    )

    return o


def my_rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6):
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
    B, M, N = a.shape
    B2, N2, K = b.shape
    assert B == B2
    assert N == N2

    o = torch.empty((B, M, K))
    partial = torch.empty((B, M, triton.cdiv(K, K_TILE_SIZE)), dtype=torch.float32)

    N_TILE_SIZE = 16

    gemm_res_rms_kernel[B, triton.cdiv(M, M_TILE_SIZE) * triton.cdiv(K, K_TILE_SIZE)](  #256//32 * 64//32 == 8 * 2
        a, b, o, partial, res,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        o.stride(0), o.stride(1),
        partial.stride(0), partial.stride(1),
        M, N, K,
        M_TILE_SIZE=M_TILE_SIZE, 
        N_TILE_SIZE=N_TILE_SIZE, 
        K_TILE_SIZE=K_TILE_SIZE
    )

    return o, partial


def finish_gemm_res_rms(a, partial, gamma, eps: float = 1e-6):
    o = torch.empty_like(a)

    M, N = o.shape[1:]
    K = partial.shape[-1]

    m2_dim = triton.next_power_of_2(M)
    n2_dim = triton.next_power_of_2(N)
    k2_dim = triton.next_power_of_2(K)

    finish_gemm_res_rms_kernel[a.shape[0],](
        a, partial, gamma, o,
        a.stride(0), a.stride(1),
        partial.stride(0), partial.stride(1),
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
    return rms_norm(after_res, after_res.shape[-1:], gamma, eps) @ weight2


def my_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos, sin
):
    h0 = x @ weight
    x1, x2 = h0[..., ::2], h0[..., 1::2]

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
    B, M, N = x.shape
    N2, K = weight.shape
    assert N == N2

    o = torch.empty((B, M, K), dtype=torch.float32)

    N_TILE_SIZE = 16

    from kernels import gemm_rope_kernel

    gemm_rope_kernel[B, (triton.cdiv(M, M_TILE_SIZE) * triton.cdiv(K, K_TILE_SIZE))](
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
    B, M, N = x.shape
    K = weight.shape[-1]
    output = torch.empty(x.shape[:2])
    logsumexp_o = torch.empty(x.shape[:2])
    for b in range(x.shape[0]):
        for m in range(triton.cdiv(M, M_TILE_SIZE)):
            max_logit = torch.full((min(M_TILE_SIZE, M - m*M_TILE_SIZE),), -1e9)
            sum_exp = torch.zeros_like(max_logit, dtype=torch.float32)
            desired_logit = torch.zeros_like(max_logit, dtype=torch.float32)
            for k in range(triton.cdiv(K, K_TILE_SIZE)):
                S = x[b, m*M_TILE_SIZE:(m+1)*M_TILE_SIZE, :] @ \
                    weight[:, k*K_TILE_SIZE:(k+1)*K_TILE_SIZE]
                new_max_logit = torch.maximum(max_logit, torch.amax(S, dim=-1))
                sum_exp = torch.exp(max_logit - new_max_logit)*sum_exp + \
                    torch.exp(S - new_max_logit[:, None]).sum(dim=-1)
                desired_logit = torch.exp(max_logit - new_max_logit) * desired_logit + \
                    torch.sum(torch.exp(S - new_max_logit[:, None])*y[b, m*M_TILE_SIZE:(m+1)*M_TILE_SIZE, k*K_TILE_SIZE:(k+1)*K_TILE_SIZE], dim=-1)
                max_logit = new_max_logit
            output[b, m*M_TILE_SIZE:(m+1)*M_TILE_SIZE] = desired_logit / sum_exp
            logsumexp_o[b, m*M_TILE_SIZE:(m+1)*M_TILE_SIZE] = max_logit + torch.log(sum_exp)

    return output, logsumexp_o

def reference_gemm_CE(x, weight, y):
    from torch.nn.functional import cross_entropy
    return cross_entropy(
        (x @ weight).transpose(1, 2),
        y,
        reduction="none",
    )


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


if __name__ == "__main__":
    main()
