#!/usr/bin/env bash
# Download a pinned, checksum-verified ONNX Runtime prebuilt for this machine's
# architecture into third_party/onnxruntime.
set -euo pipefail

ORT_VERSION="1.22.0"
SHA_X64="8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3"
SHA_AARCH64="bb76395092d150b52c7092dc6b8f2fe4d80f0f3bf0416d2f269193e347e24702"

case "$(uname -m)" in
  x86_64)        ARCH="x64";     EXPECTED="$SHA_X64" ;;
  aarch64|arm64) ARCH="aarch64"; EXPECTED="$SHA_AARCH64" ;;
  *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party"
NAME="onnxruntime-linux-${ARCH}-${ORT_VERSION}"
URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${NAME}.tgz"

mkdir -p "$DEST"
if [[ -e "$DEST/onnxruntime/lib" ]]; then
  echo "ONNX Runtime already present at $DEST/onnxruntime"; exit 0
fi

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
echo "Downloading $URL"
curl -fsSL --retry 3 -o "$TMP" "$URL"
ACTUAL="$(sha256sum "$TMP" | cut -d' ' -f1)"
if [[ "$ACTUAL" != "$EXPECTED" ]]; then
  echo "checksum mismatch for $NAME: expected $EXPECTED, got $ACTUAL" >&2; exit 1
fi
tar -xzf "$TMP" -C "$DEST"
ln -sfn "$NAME" "$DEST/onnxruntime"
echo "Installed $NAME -> $DEST/onnxruntime"
