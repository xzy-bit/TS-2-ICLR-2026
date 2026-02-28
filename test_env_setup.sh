#!/bin/bash
# ============================================================================
# TS-2-ICLR-2026 (Train) - Blackwell(sm_120) Environment Setup (Version A)
# ============================================================================
# Goal: Make training (transformers + deepspeed) run on Blackwell GPUs reliably.
# Strategy:
#   1) Install PyTorch nightly (cu128) FIRST to get sm_120 kernels.
#   2) Install project requirements EXCLUDING torch/vllm and using flash-attn from source.
#   3) Verify CUDA kernels actually execute (no "no kernel image").
# ============================================================================

set -euo pipefail

echo "=============================================="
echo "TS-2-ICLR-2026 - Train Env Setup (Blackwell)"
echo "Target GPU: NVIDIA RTX PRO 6000 Blackwell (sm_120)"
echo "=============================================="

# -----------------------------
# Config
# -----------------------------
ENV_NAME="${ENV_NAME:-ts2-test}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
REQ_FILE="${REQ_FILE:-requirements.txt}"  # use a cleaned req file for Version A
TORCH_NIGHTLY_INDEX_URL="https://download.pytorch.org/whl/nightly/cu128"

# -----------------------------
# Check conda
# -----------------------------
if ! command -v conda &> /dev/null; then
  echo "ERROR: conda is not installed or not in PATH"
  exit 1
fi
eval "$(conda shell.bash hook)"

# -----------------------------
# GPU info (optional)
# -----------------------------
echo ""
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
  echo "WARNING: nvidia-smi not found."
fi

# -----------------------------
# Create env (fresh)
# -----------------------------
echo ""
echo "Recreating conda env: ${ENV_NAME}"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env remove -n "${ENV_NAME}" -y
fi
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

echo ""
echo "Activating env..."
conda activate "${ENV_NAME}"

# -----------------------------
# Tooling
# -----------------------------
echo ""
echo "Upgrading pip tooling..."
python -m pip install --upgrade pip setuptools wheel


# -----------------------------
# NumPy (<2.0)
# -----------------------------
echo ""
echo "Installing NumPy < 2.0..."
pip install "numpy<2.0"

# -----------------------------
# PyTorch nightly (sm_120)
# -----------------------------
echo ""
echo "Installing PyTorch nightly (cu128) for Blackwell..."
pip install --pre torch torchvision torchaudio --index-url "${TORCH_NIGHTLY_INDEX_URL}"

echo ""
echo "Verifying torch CUDA + kernel execution..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available in torch")

print("torch.version.cuda:", torch.version.cuda)
print("arch list:", torch.cuda.get_arch_list())

# Must succeed on Blackwell:
x = torch.zeros(32, 32, device="cuda")
y = x + 1
z = (y @ y).sum()
print("CUDA basic ops: PASSED")
PY

# -----------------------------
# Requirements file (Version A)
# -----------------------------
echo ""
echo "Preparing requirements file: ${REQ_FILE}"
if [ ! -f "${REQ_FILE}" ]; then
  echo "ERROR: ${REQ_FILE} not found."
  echo "Create it next to this script (see below), or set REQ_FILE=/path/to/file"
  exit 1
fi

echo ""
echo "Installing requirements (no torch/vllm in this file)..."
pip install -r "${REQ_FILE}"

pip install vllm

echo ""
echo "Quick import checks..."
python - <<'PY'
import transformers, datasets, accelerate, deepspeed
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("accelerate:", accelerate.__version__)
print("deepspeed:", deepspeed.__version__)

import torch
x = torch.randn(128, 128, device="cuda", dtype=torch.float16)
y = (x @ x).sum()
print("CUDA matmul: PASSED")
PY

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo "Activate with:"
echo "  conda activate ${ENV_NAME}"
echo ""
