"""VEREC backend — shared model singletons and inference helpers.

All heavy models (YOLO, FastVLM, Pose, ST-GCN) are loaded once and reused.
"""
from __future__ import annotations

import os
import re
import time
import threading
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from camera import OpenCVCamera
from detectors import COCO_CLASSES, YOLODetector, YOLOPoseDetector
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
)

_lock = threading.Lock()


# ── VLM rolling text memory (cache) ──────────────────────────────────────

class VLMMemory:
    """Simple rolling text cache that gives the VLM short-term memory.

    Stores the last *maxlen* captions with timestamps.  When the prompt is
    built the memory is collapsed into a short "Previously observed:" block
    so the VLM can reference what it saw before.
    """

    def __init__(self, maxlen: int = 5):
        self._buf: list[dict[str, str]] = []  # [{timestamp, text}, ...]
        self._maxlen = maxlen
        self._lock = threading.Lock()

    def add(self, text: str, timestamp: str = ""):
        if not text:
            return
        with self._lock:
            self._buf.append({"timestamp": timestamp, "text": text})
            if len(self._buf) > self._maxlen:
                del self._buf[: len(self._buf) - self._maxlen]

    def get_context(self, max_entries: int | None = None) -> str:
        """Return a formatted string of recent observations for prompt injection."""
        with self._lock:
            entries = list(self._buf)
        if max_entries:
            entries = entries[-max_entries:]
        if not entries:
            return ""
        lines = []
        for i, e in enumerate(entries, 1):
            ts = f" ({e['timestamp']})" if e.get("timestamp") else ""
            lines.append(f"  {i}. {e['text']}{ts}")
        return "Previously observed:\n" + "\n".join(lines)

    def clear(self):
        with self._lock:
            self._buf.clear()

    def entries(self) -> list[dict[str, str]]:
        with self._lock:
            return list(self._buf)

    def __len__(self):
        with self._lock:
            return len(self._buf)


vlm_memory = VLMMemory(maxlen=5)


def build_composite_image(frames: list[np.ndarray], cols: int = 3,
                          target_h: int = 480) -> Image.Image:
    """Stitch N frames into a grid image for the VLM.

    E.g. 5 frames → 2 rows × 3 cols (last cell black if odd).
    Each cell is resized to keep aspect ratio within target_h per row.
    """
    if not frames:
        raise ValueError("No frames to composite")
    if len(frames) == 1:
        return Image.fromarray(frames[0])

    n = len(frames)
    rows = (n + cols - 1) // cols

    # Resize all frames to same height
    resized: list[np.ndarray] = []
    for f in frames:
        h, w = f.shape[:2]
        cell_h = target_h // rows
        scale = cell_h / h
        new_w = int(w * scale)
        resized.append(cv2.resize(f, (new_w, cell_h)))

    # Pad to fill grid
    cell_h = resized[0].shape[0]
    cell_w = max(r.shape[1] for r in resized)
    while len(resized) < rows * cols:
        resized.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

    # Pad each to same width
    padded = []
    for r in resized:
        if r.shape[1] < cell_w:
            pad = np.zeros((cell_h, cell_w - r.shape[1], 3), dtype=np.uint8)
            r = np.concatenate([r, pad], axis=1)
        padded.append(r)

    # Build grid
    row_imgs = []
    for r in range(rows):
        row_imgs.append(np.concatenate(padded[r * cols:(r + 1) * cols], axis=1))
    grid = np.concatenate(row_imgs, axis=0)
    return Image.fromarray(grid)


def _resolve_device() -> str:
    """Pick the best available torch device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ── Singletons ────────────────────────────────────────────────────────────

_yolo: YOLODetector | None = None
_yolo_pose: YOLOPoseDetector | None = None
_camera: OpenCVCamera | None = None
_action_recognizer = None  # ActionRecognizer | None

# VLM globals (set by init_vlm)
tokenizer: Any = None
vlm_model: Any = None
image_processor: Any = None
_vlm_config: dict[str, Any] | None = None  # stored for lazy init


def init_yolo() -> YOLODetector:
    global _yolo
    if _yolo is None:
        p = os.path.join(os.path.dirname(os.path.dirname(__file__)), "yolo11n.onnx")
        _yolo = YOLODetector(p)
    return _yolo




def init_yolo_pose() -> YOLOPoseDetector | None:
    """Initialise YOLO11n-pose. Returns None if ONNX file not found."""
    global _yolo_pose
    if _yolo_pose is None:
        p = os.path.join(os.path.dirname(os.path.dirname(__file__)), "yolo11n-pose.onnx")
        if not os.path.exists(p):
            return None
        _yolo_pose = YOLOPoseDetector(p)
    return _yolo_pose


def init_action(device: str = "cpu"):
    """Initialise ST-GCN action recogniser. Returns None if checkpoint not found."""
    global _action_recognizer
    if _action_recognizer is None:
        ckpt = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "checkpoints", "stgcn_ntu60_joint.pth")
        if not os.path.exists(ckpt):
            return None
        from action import ActionRecognizer
        _action_recognizer = ActionRecognizer(ckpt, device=device)
    return _action_recognizer


def init_camera() -> OpenCVCamera:
    global _camera
    if _camera is None:
        _camera = OpenCVCamera(device_id=0, width=640, height=480)
    return _camera


def configure_vlm(model_path: str, model_base: str | None = None):
    """Store VLM config for lazy loading. Does NOT load the model."""
    global _vlm_config
    _vlm_config = {"model_path": model_path, "model_base": model_base}


_vlm_warmup_scheduled = False

def schedule_vlm_warmup(delay: float = 10.0):
    """Start a background thread that loads the VLM after *delay* seconds.

    This keeps startup instant while ensuring the model is warm
    by the time the user actually needs a caption.
    Idempotent: only the first call schedules a thread.
    """
    global _vlm_warmup_scheduled
    if _vlm_warmup_scheduled:
        return
    _vlm_warmup_scheduled = True

    def _warmup():
        import time as _time
        _time.sleep(delay)
        print(f"[warmup] Loading VLM in background after {delay}s delay…")
        init_vlm()
    t = threading.Thread(target=_warmup, daemon=True)
    t.start()


def init_vlm():
    """Lazy-load VLM on first use. Returns True if model is ready."""
    global tokenizer, vlm_model, image_processor
    if vlm_model is not None:
        return True
    if _vlm_config is None:
        return False
    model_path = os.path.expanduser(_vlm_config["model_path"])
    model_base = _vlm_config["model_base"]
    gen_cfg = os.path.join(model_path, "generation_config.json")
    gen_cfg_hidden = os.path.join(model_path, ".generation_config.json")
    renamed = False
    if os.path.exists(gen_cfg):
        try:
            os.rename(gen_cfg, gen_cfg_hidden)
            renamed = True
        except OSError:
            pass  # read-only filesystem (e.g. Docker :ro volume) — skip rename

    print("Loading VLM (first use)…")
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    _device = _resolve_device()
    tokenizer, vlm_model, image_processor, _ = load_pretrained_model(
        model_path, model_base, model_name, device=_device
    )
    vlm_model.generation_config.pad_token_id = tokenizer.pad_token_id

    if renamed:
        os.rename(gen_cfg_hidden, gen_cfg)
    print("VLM loaded.")
    return True


# ── Inference helpers ─────────────────────────────────────────────────────

def get_camera_frame() -> np.ndarray | None:
    cam = init_camera()
    return cam.read() if cam.is_open else None


def run_detection(frame: np.ndarray, conf: float = 0.45, iou: float = 0.45) -> dict:
    yolo = init_yolo()
    t0 = time.perf_counter()
    boxes, scores, class_ids = yolo.detect(frame, conf=conf, iou=iou)
    dt = time.perf_counter() - t0
    objects = [
        {"class": COCO_CLASSES[c], "confidence": round(float(s), 3),
         "box": [int(x) for x in b]}
        for b, s, c in zip(boxes, scores, class_ids)
    ]
    annotated = YOLODetector.draw(frame, boxes, scores, class_ids)
    return {
        "objects": objects,
        "count": len(objects),
        "time_ms": round(dt * 1000, 1),
        "fps": round(1 / dt, 1) if dt > 0 else 0,
        "annotated": annotated,
        "boxes": boxes,
        "scores": scores,
        "class_ids": class_ids,
    }




def run_pose(frame: np.ndarray, conf: float = 0.45, iou: float = 0.45) -> dict | None:
    """Run YOLO Pose detection. Returns None if model not available."""
    pose = init_yolo_pose()
    if pose is None:
        return None
    t0 = time.perf_counter()
    boxes, scores, keypoints = pose.detect(frame, conf=conf, iou=iou)
    dt = time.perf_counter() - t0
    annotated = YOLOPoseDetector.draw(frame, boxes, scores, keypoints)
    persons = [
        {"confidence": round(float(s), 3), "box": [int(x) for x in b],
         "keypoints": kpts.tolist()}
        for b, s, kpts in zip(boxes, scores, keypoints)
    ]
    return {
        "persons": persons,
        "count": len(persons),
        "time_ms": round(dt * 1000, 1),
        "fps": round(1 / dt, 1) if dt > 0 else 0,
        "annotated": annotated,
        "keypoints": keypoints,   # raw ndarray for action recogniser
        "scores": scores,
    }


def run_action(keypoints: np.ndarray | None,
               scores: np.ndarray | None = None) -> dict | None:
    """Feed one frame of keypoints to ST-GCN action recogniser.

    Returns None if model not available or buffer not yet full.
    """
    rec = init_action()
    if rec is None:
        return None
    return rec.update(keypoints, scores)


def run_vlm(image: Image.Image | np.ndarray, prompt: str = "",
            temperature: float = 0.0, max_tokens: int = 64) -> dict:
    if vlm_model is None:
        if not init_vlm():
            return {"text": "", "error": "VLM not configured"}
    if not prompt:
        prompt = "Briefly describe what is happening."

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    qs = prompt
    if vlm_model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    _device = _resolve_device()
    input_ids = torch.as_tensor(
        tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    ).unsqueeze(0).to(torch.device(_device))

    image_tensor = process_images([image], image_processor, vlm_model.config)[0]

    with _lock, torch.inference_mode():
        t0 = time.perf_counter()
        output_ids = vlm_model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half(),
            image_sizes=[image.size],
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            max_new_tokens=int(max_tokens),
            use_cache=True,
        )
        dt = time.perf_counter() - t0

    text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    n_tokens = output_ids.shape[1] - input_ids.shape[1]
    tps = n_tokens / dt if dt > 0 else 0
    return {
        "text": text,
        "time_s": round(dt, 3),
        "tokens": n_tokens,
        "tokens_per_s": round(tps, 1),
    }


def resolve_stream_url(url: str) -> str:
    if not url:
        return url
    if re.match(r"https?://(www\.)?(youtube\.com|youtu\.be)/", url):
        try:
            import yt_dlp
            ydl_opts = {
                "format": "best[ext=mp4][height<=720]/best[height<=720]/best",
                "quiet": True, "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                resolved = info.get("url", "")
                if resolved:
                    return resolved
        except Exception:
            pass
    return url


IP_CAMERA_PRESETS = {
    "Walworth Road, London": "https://www.youtube.com/live/8JCk5M_xrBs",
    "Times Square, New York": "https://www.youtube.com/live/eJ7ZkQ5TC08",
    "Shibuya Crossing, Tokyo": "https://www.youtube.com/live/DjdUEyjx8GM",
    "Jackson Hole Town Square": "https://www.youtube.com/live/psfFJR3vZ78",
    "Big Buck Bunny (HLS test)": "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8",
    "Tears of Steel (HLS test)": "https://demo.unified-streaming.com/k8s/features/stable/video/tears-of-steel/tears-of-steel.ism/.m3u8",
}
