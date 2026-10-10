from pathlib import Path
import re
import statistics
import time

import pandas as pd
import torch


def benchmark(functions, args, *, mode="fwd", warmup=10, repeats=30, trace_dir=None):
    if mode not in ("fwd", "bwd", "fwd+bwd") or repeats < 1 or warmup < 0:
        raise ValueError("Invalid mode, warmup or repeats")
    inputs = tuple(x for x in args if isinstance(x, torch.Tensor) and x.requires_grad)
    device, dtype = args[0].device, args[0].dtype
    cuda = device.type == "cuda"
    atol, rtol = (1e-5, 1e-4) if dtype == torch.float32 else (1e-2, 1e-2)
    forward_args = tuple(x.detach() if isinstance(x, torch.Tensor) else x for x in args)
    reference = functions[0][1]
    expected = reference(*args)
    generator = torch.Generator(device=device).manual_seed(0)
    upstream = torch.randn(expected.shape, device=device, dtype=expected.dtype, generator=generator)
    expected_grads = torch.autograd.grad(expected, inputs, upstream) if mode != "fwd" else ()
    expected = expected.detach()
    fp32_args = tuple(x.detach().float().requires_grad_(x.requires_grad)
                      if isinstance(x, torch.Tensor) and x.is_floating_point() else x for x in args)
    fp32_output = reference(*fp32_args)
    fp32_grads = torch.autograd.grad(fp32_output, tuple(fp32_args[i] for i, x in enumerate(args)
                                     if isinstance(x, torch.Tensor) and x.requires_grad), upstream.float()) if mode != "fwd" else ()
    fp32_output = fp32_output.detach()
    rows = []
    for name, fn in functions:
        row = dict(implementation=name, mode=mode, device=str(device), dtype=str(dtype),
                   status="ok", reason="", mean_ms=float("nan"), std_ms=float("nan"), median_ms=float("nan"))
        try:
            if mode not in getattr(fn, "modes", ("fwd", "bwd", "fwd+bwd")):
                raise NotImplementedError("Forward only")
            output = fn(*args) if mode != "fwd" else fn(*forward_args)
            row["output_max_abs"] = (output.float() - expected.float()).abs().max().item()
            row["output_fp32_max_abs"] = (output.float() - fp32_output).abs().max().item()
            torch.testing.assert_close(output.float(), expected.float(), atol=atol, rtol=rtol)
            if mode != "fwd":
                grads = torch.autograd.grad(output, inputs, upstream.clone())
                row["grad_max_abs"] = max((g.float() - r.float()).abs().max().item() for g, r in zip(grads, expected_grads))
                row["grad_fp32_max_abs"] = max((g.float() - r).abs().max().item() for g, r in zip(grads, fp32_grads))
                for grad, ref in zip(grads, expected_grads):
                    torch.testing.assert_close(grad.float(), ref.float(), atol=atol, rtol=rtol)

            @torch.no_grad()
            def forward():
                return fn(*forward_args)

            def both():
                return torch.autograd.grad(fn(*args), inputs, upstream.clone())

            def prepare():
                if mode != "bwd":
                    return forward if mode == "fwd" else both
                output, grad = fn(*args), upstream.clone()
                return lambda: torch.autograd.grad(output, inputs, grad)

            for _ in range(warmup):
                prepare()()
            samples = []
            for _ in range(repeats):
                run = prepare()
                if cuda:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize(device)
                    with torch.cuda.device(device):
                        start.record()
                        run()
                        end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end))
                else:
                    start = time.perf_counter()
                    run()
                    samples.append((time.perf_counter() - start) * 1000)
            row.update(mean_ms=statistics.mean(samples), std_ms=statistics.stdev(samples) if repeats > 1 else 0,
                       median_ms=statistics.median(samples))
            row["tiles"] = getattr(fn, "configs", lambda: "")()
            if trace_dir is not None and name == functions[-1][0]:
                run = prepare()
                activities = [torch.profiler.ProfilerActivity.CPU]
                if cuda:
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
                    with torch.profiler.record_function(f"{name}/{mode}"):
                        run()
                    if cuda:
                        torch.cuda.synchronize(device)
                folder = Path(trace_dir)
                folder.mkdir(parents=True, exist_ok=True)
                profiler.export_chrome_trace(str(folder / (re.sub(r"[^\w.-]", "_", f"{name}_{mode}") + ".json")))
        except Exception as error:
            row.update(status="incorrect" if isinstance(error, AssertionError) else "unsupported"
                       if isinstance(error, (ImportError, NotImplementedError)) else "error",
                       reason=f"{type(error).__name__}: {error}", mean_ms=float("nan"), std_ms=float("nan"), median_ms=float("nan"))
        rows.append(row)
    result = pd.DataFrame(rows)
    result["speedup_baseline"] = result.mean_ms.iloc[0] / result.mean_ms
    compiled = result[result.implementation.str.contains("torch.compile", regex=False)]
    result["speedup_compile"] = (compiled.mean_ms.iloc[0] if len(compiled) else float("nan")) / result.mean_ms
    return result
