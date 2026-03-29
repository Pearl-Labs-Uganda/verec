#!/usr/bin/env bash
# Download YOLO Pose + ST-GCN action recognition models for VEREC.
#
# Usage:
#   bash get_action_models.sh
#
# Models downloaded:
#   1. yolo11n-pose.onnx  — YOLO11n pose estimation (skeleton extraction)
#   2. checkpoints/stgcn_ntu60_joint.pth — ST-GCN (PYSKL, NTU60, 2D COCO skeleton, joint modality)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. YOLO11n-pose ONNX ──────────────────────────────────────────────────

POSE_ONNX="yolo11n-pose.onnx"

if [ -f "$POSE_ONNX" ]; then
    echo "✓ $POSE_ONNX already exists, skipping."
else
    echo "→ Exporting $POSE_ONNX from ultralytics..."
    # Install ultralytics temporarily if needed, export, then the .onnx stays
    python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11n-pose.pt')
model.export(format='onnx', imgsz=640, simplify=True)
print('Export complete.')
" 2>&1
    if [ -f "$POSE_ONNX" ]; then
        echo "✓ $POSE_ONNX exported successfully."
    else
        echo "✗ Failed to export $POSE_ONNX. Install ultralytics: pip install ultralytics"
        exit 1
    fi
fi

# ── 2. ST-GCN checkpoint (pyskl, NTU60, HRNet 2D, Joint) ─────────────────

mkdir -p checkpoints
STGCN_CKPT="checkpoints/stgcn_ntu60_joint.pth"
STGCN_URL="http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu60_xsub_hrnet/j.pth"

if [ -f "$STGCN_CKPT" ]; then
    echo "✓ $STGCN_CKPT already exists, skipping."
else
    echo "→ Downloading ST-GCN checkpoint from pyskl model zoo..."
    curl -L -o "$STGCN_CKPT" "$STGCN_URL"
    if [ -f "$STGCN_CKPT" ]; then
        echo "✓ $STGCN_CKPT downloaded successfully."
    else
        echo "✗ Failed to download ST-GCN checkpoint."
        exit 1
    fi
fi

echo ""
echo "All action recognition models are ready!"
echo "  • Pose:   $POSE_ONNX"
echo "  • Action: $STGCN_CKPT"
