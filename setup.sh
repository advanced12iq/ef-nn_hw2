#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mode="${1:-gpu}"
if [[ $# -gt 1 || ( "$mode" != gpu && "$mode" != --cpu ) ]]; then
    echo 'Usage: bash setup.sh [--cpu]' >&2
    exit 2
fi

admin() {
    if [[ $EUID == 0 ]]; then
        "$@"
    elif command -v sudo >/dev/null && sudo -n true 2>/dev/null; then
        sudo -n "$@"
    else
        echo "Root or passwordless sudo required for: $*" >&2
        exit 1
    fi
}

export PATH="$HOME/.local/bin:$PATH"
export CXX="${CXX:-g++}"
if ! command -v curl >/dev/null || ! command -v "$CXX" >/dev/null; then
    command -v apt-get >/dev/null || { echo 'Install curl and a C++ compiler first.' >&2; exit 1; }
    admin apt-get update
    admin apt-get install -y build-essential curl ca-certificates git
fi
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi
printf '#include <cstdlib>\nint main() { return 0; }\n' | "$CXX" -x c++ -fsyntax-only -

if [[ "$mode" == gpu ]]; then
    if [[ -n "${CUDA_HOME:-}" ]]; then
        export PATH="$CUDA_HOME/bin:$PATH"
    else
        export PATH="/usr/local/cuda-12.8/bin:/usr/local/cuda/bin:$PATH"
    fi
    cuda_ready() {
        command -v nvcc >/dev/null || return 1
        local output
        output=$(nvcc --version)
        [[ "$output" =~ release\ ([0-9]+)\.([0-9]+) ]] || return 1
        (( BASH_REMATCH[1] > 12 || (BASH_REMATCH[1] == 12 && BASH_REMATCH[2] >= 8) ))
    }
    if ! cuda_ready; then
        source /etc/os-release
        case "$ID:$VERSION_ID:$(uname -m)" in
            ubuntu:22.04:x86_64|ubuntu:24.04:x86_64) distro="ubuntu${VERSION_ID//./}" ;;
            debian:12:x86_64) distro=debian12 ;;
            *) echo 'Automatic CUDA installation supports Ubuntu 22.04/24.04 or Debian 12 x86_64. Set CUDA_HOME on other systems.' >&2; exit 1 ;;
        esac
        keyring=$(mktemp --suffix=.deb)
        curl -fLsS "https://developer.download.nvidia.com/compute/cuda/repos/$distro/x86_64/cuda-keyring_1.1-1_all.deb" -o "$keyring"
        admin dpkg -i "$keyring"
        rm -f "$keyring"
        admin apt-get update
        admin apt-get install -y cuda-toolkit-12-8
        export PATH="/usr/local/cuda-12.8/bin:$PATH"
        export CUDA_HOME=/usr/local/cuda-12.8
    fi
    export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
fi

venv="${CODA_VENV:-$PWD/.venv}"
if [[ ! -x "$venv/bin/python" ]]; then
    uv venv --python "${CODA_PYTHON:-3.12}" "$venv"
fi
python="$venv/bin/python"
export PATH="$venv/bin:$PATH"
backend=cu128
packages=('torch==2.10.0' pandas matplotlib nbformat nbclient ipykernel jupyterlab packaging ninja setuptools wheel)
if [[ "$mode" == --cpu ]]; then
    backend=cpu
else
    packages+=('nvidia-cutlass-dsl==4.4.2' 'liger-kernel==0.8.0' 'flashinfer-python==0.6.10.post1' 'quack-kernels==0.4.1')
fi
uv pip install --python "$python" --torch-backend "$backend" "${packages[@]}"
uv pip check --python "$python"
"$python" - "$mode" <<'PY'
import sys
from importlib.metadata import version
import torch
import bench, testing
assert version('torch').split('+')[0] == '2.10.0'
x = torch.randn(8, requires_grad=True)
y = torch.compile(lambda x: x.sin(), fullgraph=True)(x)
torch.testing.assert_close(y, x.sin())
y.sum().backward()
if sys.argv[1] == 'gpu':
    expected = {'nvidia-cutlass-dsl': '4.4.2', 'liger-kernel': '0.8.0',
                'flashinfer-python': '0.6.10.post1', 'quack-kernels': '0.4.1'}
    for package, required in expected.items():
        assert version(package) == required, (package, version(package), required)
    assert torch.cuda.is_available(), 'CUDA unavailable: the server must expose a GPU and have an NVIDIA driver installed'
    from coda import GemmResRmsGemm2, GemmRope, GemmCE
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    from liger_kernel.ops.rope import LigerRopeFunction
    from flashinfer import rmsnorm
    from flashinfer.rope import apply_rope_with_cos_sin_cache
    from quack.linear import linear_func
    print('GPU:', torch.cuda.get_device_name(), '| versions:', {'torch': version('torch'), **expected})
else:
    print('CPU setup ready | PyTorch:', torch.__version__)
PY
"$python" -m ipykernel install --sys-prefix --name coda --display-name 'CODA'
"$python" - <<'PY'
import json, os, sys
from pathlib import Path
path = Path(sys.prefix) / 'share/jupyter/kernels/coda/kernel.json'
spec = json.loads(path.read_text())
spec['env'] = {key: os.environ[key] for key in ('CXX', 'CUDA_HOME', 'PATH') if key in os.environ}
path.write_text(json.dumps(spec, indent=2) + '\n')
PY
printf '\nReady. Run:\n  CXX=%q %q/bin/jupyter lab test.ipynb --no-browser\n' "$CXX" "$venv"
