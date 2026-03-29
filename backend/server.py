"""VEREC FastAPI backend — REST + WebSocket API for the Next.js frontend.

Run:
    python -m backend.server --model-path checkpoints/llava-fastvithd_0.5b_stage3
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
import time
import argparse
import threading
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from backend.models import (
    init_yolo, init_yolo_seg, init_camera, init_vlm,
    get_camera_frame, run_detection, run_segmentation, run_vlm,
    resolve_stream_url, IP_CAMERA_PRESETS, COCO_CLASSES,
)

app = FastAPI(title="VEREC API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Data store — rolling logs for JSON export ─────────────────────────────

_export_lock = threading.Lock()
_detection_log: list[dict] = []
_vlm_log: list[dict] = []
_reports: list[dict] = []
MAX_LOG = 500


def _append_detection(entry: dict):
    with _export_lock:
        _detection_log.append(entry)
        if len(_detection_log) > MAX_LOG:
            del _detection_log[: len(_detection_log) - MAX_LOG]


def _append_vlm(entry: dict):
    with _export_lock:
        _vlm_log.append(entry)
        if len(_vlm_log) > MAX_LOG:
            del _vlm_log[: len(_vlm_log) - MAX_LOG]


def _append_report(entry: dict):
    with _export_lock:
        _reports.append(entry)


def _frame_to_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _frame_to_b64(frame: np.ndarray, quality: int = 80) -> str:
    return base64.b64encode(_frame_to_jpeg(frame, quality)).decode()


# ── REST endpoints ────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/api/presets")
def presets():
    return IP_CAMERA_PRESETS


@app.get("/api/frame")
def get_frame(source: str = "local", url: str = ""):
    """Get a single frame as JPEG."""
    if source == "local":
        frame = get_camera_frame()
    else:
        resolved = resolve_stream_url(url)
        cap = cv2.VideoCapture(resolved, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
        try:
            ok, bgr = cap.read()
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok and bgr is not None else None
        finally:
            cap.release()

    if frame is None:
        return JSONResponse({"error": "No frame available"}, status_code=503)

    jpeg = _frame_to_jpeg(frame)
    return StreamingResponse(io.BytesIO(jpeg), media_type="image/jpeg")


@app.post("/api/detect")
def detect(conf: float = 0.45, iou: float = 0.45, source: str = "local", url: str = ""):
    frame = _get_source_frame(source, url)
    if frame is None:
        return JSONResponse({"error": "No frame"}, status_code=503)
    result = run_detection(frame, conf=conf, iou=iou)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "objects": result["objects"],
        "count": result["count"],
        "time_ms": result["time_ms"],
    }
    _append_detection(entry)
    return {**entry, "fps": result["fps"]}


@app.post("/api/segment")
def segment(conf: float = 0.45, iou: float = 0.45, source: str = "local", url: str = ""):
    frame = _get_source_frame(source, url)
    if frame is None:
        return JSONResponse({"error": "No frame"}, status_code=503)
    result = run_segmentation(frame, conf=conf, iou=iou)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "objects": result["objects"],
        "count": result["count"],
        "time_ms": result["time_ms"],
    }
    return {**entry, "fps": result["fps"]}


@app.post("/api/vlm")
def vlm_caption(prompt: str = "Describe what is happening in one sentence.",
                temperature: float = 0.0, max_tokens: int = 64,
                source: str = "local", url: str = ""):
    frame = _get_source_frame(source, url)
    if frame is None:
        return JSONResponse({"error": "No frame"}, status_code=503)
    result = run_vlm(frame, prompt=prompt, temperature=temperature, max_tokens=max_tokens)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **result,
    }
    _append_vlm(entry)
    return entry


@app.post("/api/report")
def generate_report(
    model: str = "deepseek-r1:1.5b",
    ollama_url: str = "http://localhost:11434",
):
    """Generate an LLM report from collected detection + VLM data."""
    from openai import OpenAI

    with _export_lock:
        det_summary = _detection_log[-20:]
        vlm_summary = _vlm_log[-10:]

    det_text = "\n".join(
        f"[{d['timestamp']}] {', '.join(o['class'] for o in d['objects'])}" for d in det_summary
    ) or "(no detections)"
    vlm_text = "\n".join(
        f"- [{v['timestamp']}] {v.get('text', '')}" for v in vlm_summary
    ) or "(no captions)"

    prompt = (
        "You are VEREC, a video recognition and reporting system. "
        "Given the following detection log and scene captions from a live camera feed, "
        "write a concise surveillance report (3-6 sentences).\n\n"
        f"## Detection Log\n```\n{det_text}\n```\n\n"
        f"## Scene Captions\n{vlm_text}\n\n"
        "## Report"
    )
    try:
        client = OpenAI(base_url=f"{ollama_url.rstrip('/')}/v1", api_key="ollama")
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], max_tokens=512,
        )
        report_text = resp.choices[0].message.content or ""
    except Exception as e:
        report_text = f"Report generation failed: {e}"

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "report": report_text,
        "detection_count": len(det_summary),
        "caption_count": len(vlm_summary),
    }
    _append_report(entry)
    return entry


# ── JSON export endpoints ─────────────────────────────────────────────────

@app.get("/api/export/detections")
def export_detections():
    with _export_lock:
        data = list(_detection_log)
    return {"count": len(data), "detections": data}


@app.get("/api/export/vlm")
def export_vlm():
    with _export_lock:
        data = list(_vlm_log)
    return {"count": len(data), "captions": data}


@app.get("/api/export/reports")
def export_reports():
    with _export_lock:
        data = list(_reports)
    return {"count": len(data), "reports": data}


@app.get("/api/export/all")
def export_all():
    with _export_lock:
        return {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "detections": list(_detection_log),
            "vlm_captions": list(_vlm_log),
            "reports": list(_reports),
        }


# ── WebSocket — live feed with AI ─────────────────────────────────────────

@app.websocket("/ws/feed")
async def ws_feed(
    websocket: WebSocket,
    source: str = "local",
    url: str = "",
    conf: float = 0.45,
    iou: float = 0.45,
    vlm_interval: int = 5,
    enable_det: bool = True,
    enable_seg: bool = True,
    enable_vlm: bool = True,
):
    await websocket.accept()
    cap = None
    use_local = source == "local"

    if not use_local:
        resolved = resolve_stream_url(url)
        cap = cv2.VideoCapture(resolved, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        if not cap.isOpened():
            await websocket.send_json({"error": "Failed to open stream"})
            await websocket.close()
            return

    last_vlm_time = 0.0
    last_seg_time = 0.0
    seg_overlay = None
    last_caption = ""
    SEG_INTERVAL = 3.0

    try:
        while True:
            # Check for control messages (non-blocking)
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                ctrl = json.loads(msg)
                if ctrl.get("action") == "stop":
                    break
                # Allow live toggle of settings
                enable_det = ctrl.get("enable_det", enable_det)
                enable_seg = ctrl.get("enable_seg", enable_seg)
                enable_vlm = ctrl.get("enable_vlm", enable_vlm)
                conf = ctrl.get("conf", conf)
                iou = ctrl.get("iou", iou)
                vlm_interval = ctrl.get("vlm_interval", vlm_interval)
            except (asyncio.TimeoutError, Exception):
                pass

            # Get frame
            if use_local:
                frame = get_camera_frame()
            else:
                ok, bgr = cap.read()
                frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok and bgr is not None else None

            if frame is None:
                await asyncio.sleep(0.05)
                continue

            now = time.perf_counter()
            ts = datetime.now(timezone.utc).isoformat()
            payload: dict[str, Any] = {"timestamp": ts}

            # Detection
            det_objects = []
            if enable_det:
                det_result = run_detection(frame, conf=conf, iou=iou)
                det_objects = det_result["objects"]
                payload["detection"] = {
                    "objects": det_objects,
                    "count": det_result["count"],
                    "time_ms": det_result["time_ms"],
                    "fps": det_result["fps"],
                }
                _append_detection({"timestamp": ts, "objects": det_objects,
                                   "count": det_result["count"], "time_ms": det_result["time_ms"]})

            # Segmentation (throttled)
            if enable_seg and now - last_seg_time >= SEG_INTERVAL:
                last_seg_time = now
                seg_result = run_segmentation(frame, conf=conf, iou=iou)
                seg_overlay = seg_result["annotated"]
                payload["segmentation"] = {
                    "objects": seg_result["objects"],
                    "count": seg_result["count"],
                    "time_ms": seg_result["time_ms"],
                }

            # Build AI frame
            if seg_overlay is not None and seg_overlay.shape == frame.shape:
                ai_frame = seg_overlay.copy()
            else:
                ai_frame = frame.copy()
            if enable_det:
                from detectors import YOLODetector
                ai_frame = YOLODetector.draw(
                    ai_frame, det_result["boxes"], det_result["scores"], det_result["class_ids"]
                )

            # VLM (throttled)
            if enable_vlm and now - last_vlm_time >= vlm_interval:
                last_vlm_time = now
                obj_hint = ", ".join(o["class"] for o in det_objects) if det_objects else "nothing specific"
                vlm_result = run_vlm(
                    frame,
                    prompt=f"Objects detected: {obj_hint}. Describe what is happening in one sentence.",
                    max_tokens=64,
                )
                last_caption = vlm_result.get("text", "")
                payload["vlm"] = vlm_result
                _append_vlm({"timestamp": ts, **vlm_result})

            # Encode frames
            payload["raw_frame"] = _frame_to_b64(frame, quality=70)
            payload["ai_frame"] = _frame_to_b64(ai_frame, quality=70)
            if last_caption:
                payload["caption"] = last_caption

            await websocket.send_json(payload)
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        pass
    finally:
        if cap is not None:
            cap.release()


def _get_source_frame(source: str, url: str) -> np.ndarray | None:
    if source == "local":
        return get_camera_frame()
    resolved = resolve_stream_url(url)
    cap = cv2.VideoCapture(resolved, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
    try:
        ok, bgr = cap.read()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok and bgr is not None else None
    finally:
        cap.release()


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description="VEREC API Server")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print("Loading models…")
    init_vlm(args.model_path, args.model_base)
    init_yolo()
    init_yolo_seg()
    init_camera()
    print("Models loaded. Starting API server…")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
