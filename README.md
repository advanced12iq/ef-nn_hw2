# CODA: local GEMM + epilogue blocks

Project implementations in Triton: GEMM + residual + RMSNorm + GEMM,
GEMM + RoPE + GEMM, and GEMM + per-token cross entropy.
Benchmarks compare them with eager/compiled PyTorch, Liger, FlashInfer, and QuACK.
This is a partial reproduction of the CODA approach, rather than the authors' original kernels.

Paper: [CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs](https://arxiv.org/abs/2605.19269v1).

## Setup and execution

On a Linux server with an accessible NVIDIA GPU and an installed driver:

```bash
bash setup.sh
.venv/bin/jupyter lab test.ipynb --no-browser
```

Setup uses uv, creates a Python 3.12 environment, installs dependencies,
and registers the `CODA` notebook kernel. Installing a missing compiler or
CUDA Toolkit requires root or passwordless sudo. Automatic CUDA Toolkit
installation supports Ubuntu 22.04/24.04 and Debian 12 on x86_64.
The script does not install the NVIDIA driver.

Pinned versions: PyTorch 2.10.0 (CUDA 12.8), CuTeDSL 4.4.2, Liger 0.8.0,
FlashInfer 0.6.10.post1, and QuACK 0.4.1. Target benchmarks are intended for H100/A100;
large configurations require sufficient GPU memory (the saved run used an H100 80GB).
The GPU is selected automatically. To smoke-test the benchmark without a GPU:

```bash
bash setup.sh --cpu
.venv/bin/jupyter lab test.ipynb --no-browser
```

Select the `CODA` kernel and execute notebook cells in order.
Each target configuration and execution mode has its own cell.
For a batch run that preserves results even when assertions fail:

```bash
.venv/bin/jupyter nbconvert --execute --to notebook --inplace \
  --ExecutePreprocessor.kernel_name=coda \
  --ExecutePreprocessor.timeout=7200 \
  --ExecutePreprocessor.allow_errors=True test.ipynb
```

`allow_errors` lets the full grid finish and save the CSV. Correctness failures
remain visible in the outputs, and failed implementations receive no timing results.
Compilation, autotuning, and warmup are excluded from measurements.
By default, each run saves one CSV; plots remain embedded in the notebook.
Profiling is disabled by default and can be enabled for a selected experiment
through `trace_dir` (a Chrome trace for Perfetto).

## Saved run

- `test.ipynb`: executed notebook with tables, plots, and failed checks.
- `results/bench_20261010_192137.csv`: all results from the latest H100 run.
- `results/figures/coda_h100_speedup.png`: combined comparison plot for the report.
- `main.tex`: XeLaTeX report; a compiled PDF is not required in the repository.
