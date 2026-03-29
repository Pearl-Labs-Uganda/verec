"""Real-time action recognizer: accumulate pose sequences → ST-GCN inference.

Usage:
    recognizer = ActionRecognizer("checkpoints/stgcn_ntu60_joint.pth", device="mps")
    # Per frame (called from the WS loop):
    result = recognizer.update(keypoints)
    # result is None while buffer fills, then a dict with predictions.
"""
from __future__ import annotations

import collections
import time
from typing import Any

import numpy as np
import torch

from .ntu60 import NTU60_ACTIONS
from .stgcn import RecognizerGCN


class ActionRecognizer:
    """Wraps ST-GCN with a sliding-window pose buffer for real-time use.

    Parameters
    ----------
    checkpoint : str       Path to pyskl-format .pth checkpoint.
    device     : str       'cpu', 'mps', or 'cuda'.
    clip_len   : int       Temporal window size (default 100 to match pyskl config).
    stride     : int       Run inference every `stride` frames (default 30).
    num_person : int       Max persons per frame (top-N by confidence; default 2).
    top_k      : int       Return top-k action predictions (default 3).
    """

    def __init__(self, checkpoint: str, device: str = "cpu",
                 clip_len: int = 100, stride: int = 30,
                 num_person: int = 2, top_k: int = 3):
        self.device = device
        self.clip_len = clip_len
        self.stride = stride
        self.num_person = num_person
        self.top_k = top_k
        self.labels = NTU60_ACTIONS

        self.model = RecognizerGCN.from_checkpoint(
            checkpoint, device=device, num_classes=len(self.labels))

        # Circular buffer: list of frames, each (M, 17, 3) — (persons, joints, xyc)
        self._buffer: collections.deque[np.ndarray] = collections.deque(maxlen=clip_len)
        self._frame_count = 0
        self._last_result: dict[str, Any] | None = None

    # ── public API ────────────────────────────────────────────────────────

    def update(self, keypoints: np.ndarray | None,
               scores: np.ndarray | None = None) -> dict[str, Any] | None:
        """Feed one frame of keypoints and optionally run inference.

        Parameters
        ----------
        keypoints : ndarray[P, 17, 3] or None
            Detected keypoints for P persons (from YOLOPoseDetector).
            Pass None if no persons detected (zero-padded frame is appended).
        scores : ndarray[P] or None
            Person confidence scores (used to pick top-M persons).

        Returns
        -------
        dict or None
            Returns prediction dict every `stride` frames once buffer is full,
            or the last cached prediction otherwise.
        """
        self._frame_count += 1

        # Pick top-M persons by confidence, zero-pad if fewer
        frame_kpts = np.zeros((self.num_person, 17, 3), dtype=np.float32)
        if keypoints is not None and len(keypoints) > 0:
            if scores is not None:
                order = np.argsort(-scores)
                keypoints = keypoints[order]
            n = min(len(keypoints), self.num_person)
            frame_kpts[:n] = keypoints[:n]

        self._buffer.append(frame_kpts)

        # Run inference periodically once buffer is full
        buf_full = len(self._buffer) >= self.clip_len
        if buf_full and (self._last_result is None
                         or self._frame_count % self.stride == 0):
            self._last_result = self._infer()

        return self._last_result

    # ── internals ─────────────────────────────────────────────────────────

    def _pre_normalize(self, data: np.ndarray) -> np.ndarray:
        """PreNormalize2D: centre skeleton around mean of valid keypoints."""
        # data: (T, M, V, C)  C=3(x, y, conf)
        T, M, V, C = data.shape

        # Gather all valid keypoints to compute center
        xy = data[..., :2]      # (T, M, V, 2)
        conf = data[..., 2:3]   # (T, M, V, 1)
        valid = conf > 0.3

        # Per-person, per-frame center (mean of valid keypoints)
        for t in range(T):
            for m in range(M):
                mask = valid[t, m, :, 0]
                if mask.any():
                    cx = xy[t, m, mask, 0].mean()
                    cy = xy[t, m, mask, 1].mean()
                    xy[t, m, :, 0] -= cx
                    xy[t, m, :, 1] -= cy

        # Scale to roughly [-1, 1] by the max absolute coordinate
        max_val = np.abs(xy[valid.repeat(2, axis=-1).reshape(xy.shape)]).max() + 1e-6
        xy /= max_val

        data[..., :2] = xy
        return data

    def _infer(self) -> dict[str, Any]:
        """Run ST-GCN on the current buffer."""
        # Stack buffer → (T, M, 17, 3)
        data = np.stack(list(self._buffer), axis=0).astype(np.float32)
        T, M, V, C = data.shape

        # Pre-normalize
        data = self._pre_normalize(data)

        # Model expects (N, M, T, V, C)
        x = torch.from_numpy(data).unsqueeze(0)  # (1, T, M, V, C)
        x = x.permute(0, 2, 1, 3, 4)             # (1, M, T, V, C)
        x = x.to(self.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(x)  # (1, num_classes)
        dt = time.perf_counter() - t0

        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        top_idx = np.argsort(-probs)[: self.top_k]

        actions = [
            {"label": self.labels[i], "confidence": round(float(probs[i]), 3)}
            for i in top_idx
        ]
        return {
            "actions": actions,
            "time_ms": round(dt * 1000, 1),
        }
