"""VEREC backend — shared model singletons and inference helpers.

All heavy models (YOLO, FastVLM, Pose, ST-GCN) are loaded once and reused.
"""
from __future__ import annotations

import os
import re
import time
import threading
from typing import Any, TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

# All heavy imports (camera, detectors, llava) are intentionally deferred
# to their respective init functions so the server can start instantly.
# When USE_QWEN_VL=true, none of these are needed at startup.

if TYPE_CHECKING:
    from camera import OpenCVCamera
    from detectors import YOLODetector, YOLOPoseDetector

_lock = threading.Lock()


def _resolve_device() -> str:
    """Pick the best available torch device: cuda > mps > cpu."""
    import torch
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

# Qwen Vision singleton
_qwen_analyzer = None  # QwenVisionAnalyzer | None


def init_yolo() -> YOLODetector:
    global _yolo
    if _yolo is None:
        print("[YOLO] Importing YOLODetector from detectors...")
        from detectors import YOLODetector
        print("[YOLO] Import done. Building path...")
        p = os.path.join(os.path.dirname(os.path.dirname(__file__)), "yolo11n.onnx")
        print(f"[YOLO] Loading ONNX from {p}...")
        _yolo = YOLODetector(p)
        print("[YOLO] YOLO initialized.")
    return _yolo




def init_yolo_pose() -> YOLOPoseDetector | None:
    """Initialise YOLO11n-pose. Returns None if ONNX file not found."""
    global _yolo_pose
    if _yolo_pose is None:
        from detectors import YOLOPoseDetector  # Lazy import
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
        from camera import OpenCVCamera  # Lazy import
        _camera = OpenCVCamera(device_id=0, width=640, height=480)
    return _camera


def init_qwen(model_name: str = "qwen2.5-vl:3b"):
    """Initialise Qwen2.5-VL analyzer via Ollama.
    
    Args:
        model_name: Ollama model name (e.g., "qwen2.5-vl:3b", "qwen2.5-vl:7b")
    
    Returns:
        QwenVisionAnalyzer instance
    """
    global _qwen_analyzer
    if _qwen_analyzer is None:
        from backend.qwen_vision import QwenVisionAnalyzer
        _qwen_analyzer = QwenVisionAnalyzer(model=model_name)
        print(f"Qwen Vision Analyzer initialized: {model_name}")
    return _qwen_analyzer


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
    """Lazy-load local LLaVA VLM on first use. Returns True if model is ready.

    All llava imports are deferred here so the server starts fine when
    USE_QWEN_VL=true (Qwen/Ollama path) without llava installed.
    """
    global tokenizer, vlm_model, image_processor
    if vlm_model is not None:
        return True
    if _vlm_config is None:
        return False

    # Lazy llava imports — only needed for the local LLaVA path
    try:
        import torch
        from llava.utils import disable_torch_init
        from llava.conversation import conv_templates  # noqa: F401 (used in run_vlm)
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
    except ImportError as exc:
        print(f"[VLM] llava not installed, cannot load local model: {exc}")
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
    from detectors import COCO_CLASSES, YOLODetector  # Lazy import
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
    from detectors import YOLOPoseDetector  # Lazy import
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
            temperature: float = 0.0, max_tokens: int = 64,
            use_qwen: bool | None = None) -> dict:
    """Run vision-language model inference.
    
    Automatically switches between local LLaVA and Qwen via Ollama based on
    the USE_QWEN_VL environment variable, or use the use_qwen parameter.
    
    Args:
        image: Input image (PIL Image or numpy array)
        prompt: Text prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        use_qwen: Force use of Qwen (True) or local (False), or None for auto-detect
    
    Returns:
        Dict with 'text', timing, and token statistics
    """
    # Determine which backend to use
    if use_qwen is None:
        use_qwen = os.getenv("USE_QWEN_VL", "false").lower() == "true"
    
    if use_qwen:
        # Use Qwen via Ollama
        qwen = init_qwen()
        if not prompt:
            prompt = "Briefly describe what is happening."
        
        import time
        t0 = time.perf_counter()
        try:
            text = qwen.analyze_image(image, prompt, temperature, max_tokens)
            dt = time.perf_counter() - t0
            return {
                "text": text,
                "time_s": round(dt, 3),
                "tokens": len(text.split()),  # rough token estimate
                "tokens_per_s": round(len(text.split()) / dt, 1) if dt > 0 else 0,
                "backend": "qwen-ollama",
            }
        except Exception as e:
            return {
                "text": "",
                "error": f"Qwen inference failed: {e}",
                "backend": "qwen-ollama",
            }
    
    # Use local LLaVA (original implementation)
    if vlm_model is None:
        if not init_vlm():
            return {"text": "", "error": "VLM not configured (llava not installed or no model path)", "backend": "local-llava"}
    if not prompt:
        prompt = "Briefly describe what is happening."

    # These imports are guaranteed to be available if init_vlm() returned True
    try:
        import torch
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token, process_images
        from llava.constants import (
            IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
        )
    except ImportError as exc:
        return {"text": "", "error": f"llava not installed: {exc}", "backend": "local-llava"}

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
        "backend": "local-llava",
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
