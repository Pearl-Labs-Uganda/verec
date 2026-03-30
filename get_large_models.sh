#!/usr/bin/env bash
# Download large model files that exceed Git's 100 MB limit.
#
# Usage:
#   bash get_large_models.sh
#
# Downloads:
#   checkpoints/llava-fastvithd_0.5b_stage3/model.safetensors (1.4 GB)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CKPT_DIR="checkpoints/llava-fastvithd_0.5b_stage3"
SAFETENSORS="$CKPT_DIR/model.safetensors"
ZIP_URL="https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3.zip"

if [ -f "$SAFETENSORS" ]; then
    echo "✓ $SAFETENSORS already exists, skipping."
    exit 0
fi

echo "→ Downloading llava-fastvithd_0.5b_stage3.zip (~1.4 GB)..."
mkdir -p checkpoints

# Download zip to a temp location
TMP_ZIP="$(mktemp)"
trap 'rm -f "$TMP_ZIP"' EXIT

curl -L -o "$TMP_ZIP" "$ZIP_URL"

# Extract only the safetensors weights file
echo "→ Extracting model.safetensors..."
unzip -o -j "$TMP_ZIP" "llava-fastvithd_0.5b_stage3/model.safetensors" -d "$CKPT_DIR"

echo "✓ $SAFETENSORS downloaded successfully."
