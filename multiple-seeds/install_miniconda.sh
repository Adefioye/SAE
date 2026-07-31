#!/usr/bin/env bash

set -Eeuo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Error: this installer is intended for a Linux RunPod instance." >&2
  exit 1
fi

case "$(uname -m)" in
  x86_64)
    MINICONDA_ARCH="x86_64"
    ;;
  aarch64 | arm64)
    MINICONDA_ARCH="aarch64"
    ;;
  *)
    echo "Error: unsupported CPU architecture: $(uname -m)" >&2
    exit 1
    ;;
esac

if [[ -d /workspace && -w /workspace ]]; then
  DEFAULT_MINICONDA_DIR="/workspace/miniconda3"
else
  DEFAULT_MINICONDA_DIR="${HOME}/miniconda3"
fi
MINICONDA_INSTALL_DIR="${MINICONDA_DIR:-${DEFAULT_MINICONDA_DIR}}"

if [[ -x "${MINICONDA_INSTALL_DIR}/bin/conda" ]]; then
  echo "Miniconda is already installed at ${MINICONDA_INSTALL_DIR}."
  echo "Next step: bash setup_dev_env.sh"
  exit 0
fi

if [[ -e "${MINICONDA_INSTALL_DIR}" ]]; then
  echo "Error: ${MINICONDA_INSTALL_DIR} exists but is not a Conda install." >&2
  echo "Set MINICONDA_DIR to a different installation directory." >&2
  exit 1
fi

INSTALL_TEMP_DIR="$(mktemp -d)"
INSTALLER_PATH="${INSTALL_TEMP_DIR}/miniconda.sh"
INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${MINICONDA_ARCH}.sh"

cleanup() {
  rm -f "${INSTALLER_PATH}"
  rmdir "${INSTALL_TEMP_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Downloading Miniconda for Linux ${MINICONDA_ARCH}..."
if command -v curl >/dev/null 2>&1; then
  curl --fail --location --retry 3 "${INSTALLER_URL}" --output "${INSTALLER_PATH}"
elif command -v wget >/dev/null 2>&1; then
  wget --tries=3 "${INSTALLER_URL}" --output-document="${INSTALLER_PATH}"
else
  echo "Error: either curl or wget is required to download Miniconda." >&2
  exit 1
fi

echo "Installing Miniconda at ${MINICONDA_INSTALL_DIR}..."
bash "${INSTALLER_PATH}" -b -p "${MINICONDA_INSTALL_DIR}"

"${MINICONDA_INSTALL_DIR}/bin/conda" config --set auto_activate_base false
"${MINICONDA_INSTALL_DIR}/bin/conda" init bash >/dev/null

echo
echo "Miniconda installation complete."
echo "Next step: bash setup_dev_env.sh"
echo "To use conda interactively now, run:"
echo "  source ${MINICONDA_INSTALL_DIR}/etc/profile.d/conda.sh"
