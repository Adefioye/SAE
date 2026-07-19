#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${SAE_ENV_NAME:-sae}"
PYTHON_VERSION="${SAE_PYTHON_VERSION:-3.11}"
PYPROJECT_FILE="${PROJECT_DIR}/pyproject.toml"

find_conda() {
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi

  if [[ -n "${MINICONDA_DIR:-}" && -x "${MINICONDA_DIR}/bin/conda" ]]; then
    printf '%s\n' "${MINICONDA_DIR}/bin/conda"
    return 0
  fi

  local candidate
  for candidate in \
    "/workspace/miniconda3/bin/conda" \
    "${HOME}/miniconda3/bin/conda" \
    "${HOME}/miniconda/bin/conda"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

if ! CONDA_BIN="$(find_conda)"; then
  echo "Error: Conda was not found." >&2
  echo "Run 'bash install_miniconda.sh' first, then rerun this script." >&2
  exit 1
fi

if [[ ! -f "${PYPROJECT_FILE}" ]]; then
  echo "Error: project metadata not found: ${PYPROJECT_FILE}" >&2
  exit 1
fi

# Make `conda activate` and all Conda subcommands available in this shell.
eval "$("${CONDA_BIN}" shell.bash hook)"

# RunPod's current Miniconda requires accepting the default-channel terms
# before `conda create` can access either Anaconda channel.
conda tos accept \
  --override-channels \
  --channel https://repo.anaconda.com/pkgs/main
conda tos accept \
  --override-channels \
  --channel https://repo.anaconda.com/pkgs/r

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  INSTALLED_PYTHON_VERSION="$(
    conda run -n "${ENV_NAME}" python -c \
      'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
  )"
  if [[ "${INSTALLED_PYTHON_VERSION}" != "${PYTHON_VERSION}" ]]; then
    echo "Error: Conda environment '${ENV_NAME}' uses Python" \
      "${INSTALLED_PYTHON_VERSION}; expected ${PYTHON_VERSION}." >&2
    echo "Remove or rename that environment, then rerun this script." >&2
    exit 1
  fi
  echo "Using existing Conda environment '${ENV_NAME}'."
else
  echo "Creating Conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
  conda create --yes --name "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

echo "Installing the CUDA 12.8 PyTorch build..."
conda run -n "${ENV_NAME}" \
  python -m pip install --upgrade pip setuptools wheel
conda run -n "${ENV_NAME}" \
  python -m pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.7.1+cu128"

echo "Installing the SAE project and development dependencies from pyproject.toml..."
conda run -n "${ENV_NAME}" \
  python -m pip install --editable "${PROJECT_DIR}[dev]"

echo "Verifying the SAELens runtime and training CLI..."
VERIFY_CODE='from importlib.metadata import version
import dotenv
import huggingface_hub
import sae_lens
import torch

sae_version = version("sae-lens")
print(f"Python dependencies OK: sae-lens={sae_version}, torch={torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA verification failed: PyTorch cannot access the RunPod GPU")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")'
conda run -n "${ENV_NAME}" python -c "${VERIFY_CODE}"
conda run -n "${ENV_NAME}" \
  python "${PROJECT_DIR}/scripts/train_two_seed_sae.py" --help >/dev/null

echo
echo "Environment '${ENV_NAME}' is ready."
echo "Activate it with:"
echo "  conda activate ${ENV_NAME}"
