import torch
import triton

from kernels import (
    backward_gemm_rms_res_kernel,
    gemm_rope_kernel,
    finish_gemm_res_rms_kernel,
    gemm_kernel,
    gemm_res_rms_kernel,
    rms_norm_kernel,
    gemm_CE_kernel
)


class GemmResRmsGemm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight1: torch.Tensor,
        z: torch.Tensor,
        gamma: torch.Tensor,
        weight2: torch.Tensor,
        eps: float = 1e-6,
    ):
        ctx.M_TILE_SIZE = 16
        ctx.N_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16
        
        B, M, N = x.shape
        N2, K = weight1.shape
        R = triton.cdiv(K, ctx.K_TILE_SIZE)

        assert N == N2
        assert x.is_contiguous()
        assert weight1.is_contiguous()
        assert z.is_contiguous()
        assert gamma.is_contiguous()
        assert weight2.is_contiguous()
    
        o1 = torch.empty((B, M, K))
        h0 = torch.empty((B, M, K))
        partial = torch.empty((B, M, R), dtype=torch.float32)
        
        gemm_res_rms_kernel[B, triton.cdiv(M, ctx.M_TILE_SIZE) * triton.cdiv(K, ctx.K_TILE_SIZE)](
            x, weight1, o1, h0, partial, z,
            x.stride(0), x.stride(1),
            weight1.stride(0),
            o1.stride(0), o1.stride(1),
            partial.stride(0), partial.stride(1),
            M, N, K,
            M_TILE_SIZE=ctx.M_TILE_SIZE, 
            N_TILE_SIZE=ctx.N_TILE_SIZE, 
            K_TILE_SIZE=ctx.K_TILE_SIZE
        )

        R = partial.shape[-1]
        m2_dim = triton.next_power_of_2(M)
        k2_dim = triton.next_power_of_2(K)
        r2_dim = triton.next_power_of_2(R)

        o2 = torch.empty_like(o1)
        reduced_partial = torch.empty((B, M, 1), dtype=torch.float32)

        finish_gemm_res_rms_kernel[o1.shape[0],](
            o1, partial, gamma, o2, reduced_partial,
            o1.stride(0), o1.stride(1),
            partial.stride(0), partial.stride(1),
            reduced_partial.stride(0),
            M, K, R,
            m2_dim, k2_dim, r2_dim,
            eps
        )

        N3, K2 = weight2.shape
        assert N3 == K
        o3 = torch.empty((B, M, K2))

        gemm_kernel[B, triton.cdiv(M, ctx.M_TILE_SIZE) * triton.cdiv(K2, ctx.K_TILE_SIZE)](
            o2, weight2, o3,
            o2.stride(0), o2.stride(1),
            weight2.stride(0),
            o3.stride(0), o3.stride(1),
            M, K, K2,
            M_TILE_SIZE=ctx.M_TILE_SIZE, 
            N_TILE_SIZE=ctx.N_TILE_SIZE, 
            K_TILE_SIZE=ctx.K_TILE_SIZE
        )

        ctx.save_for_backward(x, weight1, z, gamma, weight2, reduced_partial, h0, o1, o2, o3)

        return o3

    @staticmethod
    def backward(ctx, grad_output):
        x, weight1, z, gamma, weight2, reduced_partial, h0, h1, h2, y = ctx.saved_tensors

        s_hat = torch.sum(grad_output * y, dim=-1, keepdim=True) / h1.shape[-1]
        grad_h2 = grad_output @ weight2.T
        grad_weight2 = torch.sum(h2.transpose(2, 1) @ grad_output, dim=0)
        grad_h1 = reduced_partial * (grad_h2 * gamma[None, None, :] - reduced_partial * h1 * s_hat)
        grad_h0 = torch.clone(grad_h1)
        grad_gamma = torch.sum(grad_h2 * h1 * reduced_partial, dim=(0, 1))

        grad_x = grad_h0 @ weight1.T
        grad_weight1 = torch.sum(x.transpose(2, 1) @ grad_h0, dim=0)


        return grad_x, grad_weight1, torch.clone(grad_h1), grad_gamma, grad_weight2, None


class GemmResRmsGemm2(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight1: torch.Tensor,
        z: torch.Tensor,
        gamma: torch.Tensor,
        weight2: torch.Tensor,
        eps: float = 1e-6,
    ):
        ctx.M_TILE_SIZE = 16
        ctx.N_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16
        
        B, M, N = x.shape
        N2, K = weight1.shape
        R = triton.cdiv(K, ctx.K_TILE_SIZE)

        assert N == N2
        assert x.is_contiguous()
        assert weight1.is_contiguous()
        assert z.is_contiguous()
        assert gamma.is_contiguous()
        assert weight2.is_contiguous()
    
        o1 = torch.empty((B, M, K))
        h0 = torch.empty((B, M, K))
        partial = torch.empty((B, M, R), dtype=torch.float32)
        
        gemm_res_rms_kernel[B, triton.cdiv(M, ctx.M_TILE_SIZE) * triton.cdiv(K, ctx.K_TILE_SIZE)](
            x, weight1, o1, h0, partial, z,
            x.stride(0), x.stride(1),
            weight1.stride(0),
            o1.stride(0), o1.stride(1),
            partial.stride(0), partial.stride(1),
            M, N, K,
            M_TILE_SIZE=ctx.M_TILE_SIZE, 
            N_TILE_SIZE=ctx.N_TILE_SIZE, 
            K_TILE_SIZE=ctx.K_TILE_SIZE
        )

        R = partial.shape[-1]
        m2_dim = triton.next_power_of_2(M)
        k2_dim = triton.next_power_of_2(K)
        r2_dim = triton.next_power_of_2(R)

        o2 = torch.empty_like(o1)
        reduced_partial = torch.empty((B, M, 1), dtype=torch.float32)

        finish_gemm_res_rms_kernel[o1.shape[0],](
            o1, partial, gamma, o2, reduced_partial,
            o1.stride(0), o1.stride(1),
            partial.stride(0), partial.stride(1),
            reduced_partial.stride(0),
            M, K, R,
            m2_dim, k2_dim, r2_dim,
            eps
        )

        N3, K2 = weight2.shape
        assert N3 == K
        o3 = torch.empty((B, M, K2))

        gemm_kernel[B, triton.cdiv(M, ctx.M_TILE_SIZE) * triton.cdiv(K2, ctx.K_TILE_SIZE)](
            o2, weight2, o3,
            o2.stride(0), o2.stride(1),
            weight2.stride(0),
            o3.stride(0), o3.stride(1),
            M, K, K2,
            M_TILE_SIZE=ctx.M_TILE_SIZE, 
            N_TILE_SIZE=ctx.N_TILE_SIZE, 
            K_TILE_SIZE=ctx.K_TILE_SIZE
        )

        ctx.save_for_backward(x, weight1, gamma, weight2, reduced_partial, o1, o2, o3)

        return o3

    @staticmethod
    def backward(ctx, grad_output):
        M_TILE_SIZE = 16
        N_TILE_SIZE = 32
        K_TILE_SIZE = 16

        x, weight1, gamma, weight2, reduced_partial, h1, h2, y = ctx.saved_tensors
        s_hat = torch.sum(grad_output * y, dim=-1, keepdim=True) / h1.shape[-1]
        weight2 = weight2.T.contiguous()

        B, M, N = grad_output.shape
        N2, K = weight2.shape
        assert N2 == N

        B2, M2, K2 = h1.shape
        assert B == B2
        assert M == M2
        assert K == K2 

        assert reduced_partial.shape[-1] == 1
        assert s_hat.shape[-1] == 1
        assert reduced_partial.shape == s_hat.shape      

        grad_h2 = torch.empty((B, M, K))
        grad_h1 = torch.empty((B, M, K))
        grad_gamma = torch.empty_like(gamma)

        backward_gemm_rms_res_kernel[(B, triton.cdiv(M, M_TILE_SIZE) * triton.cdiv(K, K_TILE_SIZE))](
            grad_output, weight2, reduced_partial,
            h1, s_hat, gamma,
            grad_h2, grad_h1, grad_gamma,
            grad_output.stride(0), grad_output.stride(1),
            weight2.stride(0),
            reduced_partial.stride(0),
            h1.stride(0), h1.stride(1),
            s_hat.stride(0),
            M, N, K,
            M_TILE_SIZE,
            N_TILE_SIZE,
            K_TILE_SIZE,
        )

        grad_x = grad_h1 @ weight1.T
        grad_weight1 = torch.sum(x.transpose(2, 1) @ grad_h1, dim=0)
        grad_weight2 = torch.sum(h2.transpose(2, 1) @ grad_output, dim=0)

        return grad_x, grad_weight1, grad_h1, grad_gamma, grad_weight2, None


class GemmRope(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        weight2: torch.Tensor,
        M_TILE_SIZE: int = 32,
        K_TILE_SIZE: int = 32,
    ):
        B, M, N = x.shape
        N2, K = weight.shape
        N3, K2 = weight2.shape
        assert N == N2
        assert K == N3
        assert K % 2 == 0
        assert cos.shape == sin.shape == (M, K // 2)
        assert M_TILE_SIZE >= 16 and M_TILE_SIZE == triton.next_power_of_2(M_TILE_SIZE)
        assert K_TILE_SIZE >= 16 and K_TILE_SIZE == triton.next_power_of_2(K_TILE_SIZE)
        for tensor in (x, weight, cos, sin, weight2):
            assert tensor.is_contiguous()
            assert tensor.device == x.device
            assert tensor.dtype == x.dtype

        N_TILE_SIZE = 16
        o = torch.empty((B, M, K), device=x.device, dtype=x.dtype)
        y = torch.empty((B, M, K2), device=x.device, dtype=x.dtype)

        gemm_rope_kernel[B, triton.cdiv(M, M_TILE_SIZE) * triton.cdiv(K, K_TILE_SIZE)](
            x, weight, o,
            cos, sin,
            x.stride(0), x.stride(1),
            weight.stride(0),
            o.stride(0), o.stride(1),
            cos.stride(0),
            M, N, K,
            M_TILE_SIZE,
            N_TILE_SIZE,
            K_TILE_SIZE,
        )

        gemm_kernel[B, triton.cdiv(M, M_TILE_SIZE) * triton.cdiv(K2, K_TILE_SIZE)](
            o, weight2, y,
            o.stride(0), o.stride(1),
            weight2.stride(0),
            y.stride(0), y.stride(1),
            M, K, K2,
            M_TILE_SIZE,
            N_TILE_SIZE,
            K_TILE_SIZE,
        )

        ctx.M_TILE_SIZE = M_TILE_SIZE
        ctx.K_TILE_SIZE = K_TILE_SIZE
        ctx.save_for_backward(x, weight, cos, sin, weight2, o)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, cos, sin, weight2, o = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        weight2_t = weight2.T.contiguous()
        B, M, N = grad_output.shape
        K = weight.shape[-1]
        grad_h0 = torch.empty_like(o)

        gemm_rope_kernel[B, triton.cdiv(M, ctx.M_TILE_SIZE) * triton.cdiv(K, ctx.K_TILE_SIZE)](
            grad_output, weight2_t, grad_h0,
            cos, -sin,
            grad_output.stride(0), grad_output.stride(1),
            weight2_t.stride(0),
            grad_h0.stride(0), grad_h0.stride(1),
            cos.stride(0),
            M, N, K,
            ctx.M_TILE_SIZE,
            16,
            ctx.K_TILE_SIZE,
        )

        grad_x = grad_h0 @ weight.T
        grad_weight = torch.sum(x.transpose(2, 1) @ grad_h0, dim=0)
        grad_weight2 = torch.sum(o.transpose(2, 1) @ grad_output, dim=0)

        gradients = (grad_x, grad_weight, None, None, grad_weight2, None, None)
        return gradients[:len(ctx.needs_input_grad)]


class GemmCE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight1: torch.Tensor,
        y: torch.Tensor,
    ):
        ctx.M_TILE_SIZE = 16
        ctx.N_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16

        B, M, N = x.shape
        N1, K = weight1.shape

        assert N == N1
        assert y.shape == x.shape[:-1]

        assert x.is_contiguous() and weight1.is_contiguous()
        assert x.device == weight1.device == y.device

        cross_entropy = torch.empty((B, M), device=x.device, dtype=torch.float32)
        logsumexp = torch.empty_like(cross_entropy)

        one_hot = torch.nn.functional.one_hot(y, K)

        gemm_CE_kernel[B, triton.cdiv(M, ctx.M_TILE_SIZE)](
            x, weight1, one_hot, cross_entropy, logsumexp,
            x.stride(0), x.stride(1),
            weight1.stride(0),
            one_hot.stride(0), one_hot.stride(1),
            logsumexp.stride(0),
            M, N, K,
            ctx.M_TILE_SIZE,
            ctx.N_TILE_SIZE,
            ctx.K_TILE_SIZE,
        )

        ctx.save_for_backward(x, weight1, logsumexp, one_hot)

        return cross_entropy

    @staticmethod
    def backward(ctx, grad):
        x, weight1, logsumexp, one_hot = ctx.saved_tensors
        P = x @ weight1
        S = torch.exp(P - logsumexp[..., None])
        d_logits_tile = (S - one_hot) * grad[..., None]

        return d_logits_tile @ weight1.T, (x.transpose(2, 1) @ d_logits_tile).sum(dim=0), None
