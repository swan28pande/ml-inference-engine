#!/usr/bin/env bash
# Downloads a prebuilt ONNX Runtime release into third_party/onnxruntime.
#
#   scripts/fetch_onnxruntime.sh            # CPU build for this platform
#   scripts/fetch_onnxruntime.sh --gpu      # Linux x64 CUDA 12 build
#   ORT_VERSION=1.22.0 scripts/fetch_onnxruntime.sh
set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.22.0}"
DEST="${DEST:-$(cd "$(dirname "$0")/.." && pwd)/third_party/onnxruntime}"
GPU=0
[[ "${1:-}" == "--gpu" ]] && GPU=1

os="$(uname -s)"
arch="$(uname -m)"
case "${os}-${arch}" in
  Linux-x86_64)  pkg="onnxruntime-linux-x64" ;;
  Linux-aarch64) pkg="onnxruntime-linux-aarch64" ;;
  Darwin-arm64)  pkg="onnxruntime-osx-arm64" ;;
  Darwin-x86_64) pkg="onnxruntime-osx-x86_64" ;;
  *) echo "unsupported platform ${os}-${arch}" >&2; exit 1 ;;
esac
if [[ "${GPU}" == 1 ]]; then
  [[ "${pkg}" == "onnxruntime-linux-x64" ]] || { echo "--gpu is only available for Linux x64" >&2; exit 1; }
  pkg="onnxruntime-linux-x64-gpu"
fi

url="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${pkg}-${ORT_VERSION}.tgz"
echo "Fetching ${url}"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
curl -fsSL "${url}" -o "${tmp}/ort.tgz"
tar -xzf "${tmp}/ort.tgz" -C "${tmp}"
rm -rf "${DEST}"
mkdir -p "$(dirname "${DEST}")"
mv "${tmp}/${pkg}-${ORT_VERSION}" "${DEST}"
echo "ONNX Runtime ${ORT_VERSION} installed to ${DEST}"
