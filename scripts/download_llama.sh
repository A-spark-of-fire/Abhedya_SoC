#!/usr/bin/env bash
# ==============================================================================
# Model Asset Downloader for Abhedya SoC / Trinetra AI
# Bharat AI-SoC Challenge — Problem Statement 3 (On-Device Security)
#
# Target Platform: Raspberry Pi 5 / Arm64 (Debian Bookworm 64-bit)
# Model Target: Meta Llama 3.2 1B-Instruct (4-bit XNNPACK Quantized for ExecuTorch)
# Hugging Face Repository: experimentalmachines/Llama-3.2-1B-Instruct-ExecuTorch
# ==============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Target local directory: models/llama or $HOME/models_llm or custom path argument
DEFAULT_DEST="${PROJECT_ROOT}/models/llama"
DEST_DIR="${1:-${DEFAULT_DEST}}"

HF_REPO="experimentalmachines/Llama-3.2-1B-Instruct-ExecuTorch"
BASE_URL="https://huggingface.co/${HF_REPO}/resolve/main"

# Core Asset Files
PTE_REMOTE_PATH="xnnpack/Llama-3.2-1B-Instruct-8da4w-gptq-4k.pte"
PTE_FILENAME="Llama-3.2-1B-Instruct-8da4w-gptq-4k.pte"
TOKENIZER_FILE="tokenizer.json"
CONFIG_REMOTE_PATH="xnnpack/config.json"
CONFIG_FILE="config.json"

log()  { printf '\n\033[1;36m[Abhedya-SoC]\033[0m %s\n' "$*"; }
info() { printf '  \033[1;32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[1;33m!\033[0m %s\n' "$*"; }
err()  { printf '  \033[1;31m✗\033[0m %s\n' "$*" >&2; }

log "Initializing ExecuTorch Llama 3.2 1B Model Asset Downloader..."
echo "  Target Directory: ${DEST_DIR}"
echo "  Source Repo:      huggingface.co/${HF_REPO}"

mkdir -p "${DEST_DIR}/xnnpack"

download_file() {
    local remote_subpath="$1"
    local local_file="$2"
    local url="${BASE_URL}/${remote_subpath}"

    if [ -f "${local_file}" ] && [ -s "${local_file}" ]; then
        info "File already exists: $(basename "${local_file}") ($(du -h "${local_file}" | cut -f1))"
        return 0
    fi

    log "Downloading $(basename "${local_file}") from ${url}..."
    if command -v curl >/dev/null 2>&1; then
        curl -fL --progress-bar "${url}" -o "${local_file}" || {
            err "curl download failed for ${url}"
            rm -f "${local_file}"
            return 1
        }
    elif command -v wget >/dev/null 2>&1; then
        wget -q --show-progress "${url}" -O "${local_file}" || {
            err "wget download failed for ${url}"
            rm -f "${local_file}"
            return 1
        }
    elif command -v python3 >/dev/null 2>&1; then
        python3 -c "
import urllib.request, sys
url = '${url}'
dest = '${local_file}'
print(f'Downloading via Python urllib: {url}')
urllib.request.urlretrieve(url, dest)
" || {
            err "python download failed for ${url}"
            rm -f "${local_file}"
            return 1
        }
    else
        err "Neither curl, wget, nor python3 found in PATH. Please install curl or wget."
        return 1
    fi

    info "Successfully downloaded $(basename "${local_file}")"
}

# 1. Download Tokenizer Configuration (~9 MB)
download_file "${TOKENIZER_FILE}" "${DEST_DIR}/${TOKENIZER_FILE}"

# 2. Download XNNPACK Window Config (~1 KB)
download_file "${CONFIG_REMOTE_PATH}" "${DEST_DIR}/xnnpack/${CONFIG_FILE}"

# 3. Download 4-bit Quantized ExecuTorch PTE Model (~750 MB - 1 GB)
# Also link/copy to both root of llama/ and xnnpack/ for runner flexibility
download_file "${PTE_REMOTE_PATH}" "${DEST_DIR}/xnnpack/${PTE_FILENAME}"
if [ -f "${DEST_DIR}/xnnpack/${PTE_FILENAME}" ] && [ ! -f "${DEST_DIR}/${PTE_FILENAME}" ]; then
    ln -s "${DEST_DIR}/xnnpack/${PTE_FILENAME}" "${DEST_DIR}/${PTE_FILENAME}" 2>/dev/null || \
    cp "${DEST_DIR}/xnnpack/${PTE_FILENAME}" "${DEST_DIR}/${PTE_FILENAME}"
fi

log "Download Summary for ExecuTorch Llama 3.2 1B:"
ls -lh "${DEST_DIR}"
ls -lh "${DEST_DIR}/xnnpack" || true

info "ExecuTorch Llama 3.2 1B assets ready for local on-device inference!"
