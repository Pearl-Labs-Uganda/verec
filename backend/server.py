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
import platform
import sys
import time
import argparse
import threading
from datetime import datetime, timezone
from pathlib import Path
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
    init_yolo, init_camera, configure_vlm, schedule_vlm_warmup,
    init_yolo_pose, init_action,
    get_camera_frame, run_detection, run_vlm,
    run_pose, run_action,
    resolve_stream_url, IP_CAMERA_PRESETS, COCO_CLASSES,
    build_composite_image,
)
import backend.models as _models

app = FastAPI(title="VEREC API", version="1.0.0")

# ── Default model path (used when running via uvicorn --reload / dev mode) ──
_DEFAULT_MODEL_PATH = os.environ.get(
    "VLM_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)),
                 "checkpoints", "llava-fastvithd_0.5b_stage3"),
)

@app.on_event("startup")
def _auto_init():
    """Initialise models when uvicorn imports the app (e.g. --reload mode).

    When run via `python -m backend.server`, main() calls these first and
    this is a harmless no-op because init functions are idempotent.
    """
    init_yolo()
    if _models._vlm_config is None:
        configure_vlm(_DEFAULT_MODEL_PATH)
    schedule_vlm_warmup(delay=0)

_ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://frontend:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Data store — rolling logs + JSON disk persistence ─────────────────────

_PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(__file__)))
_LOG_DIR = _PROJECT_ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_export_lock = threading.Lock()
_detection_log: list[dict] = []
_vlm_log: list[dict] = []
_reports: list[dict] = []
_scene_index: list[dict] = []  # structured scene index for search
_pose_log: list[dict] = []     # per-frame pose keypoints log
_action_log: list[dict] = []   # action recognition log
_frame_log: list[dict] = []    # every frame's combined data
_object_counts: dict[str, int] = {}  # cumulative class → count
MAX_LOG = 5000  # keep more in memory for accuracy testing

# Session timestamp for file names
_session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _flush_json(filename: str, data: list[dict]):
    """Write a log list to disk as JSON."""
    path = _LOG_DIR / filename
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def _append_jsonl(filename: str, entry: dict):
    """Append a single JSON line to a .jsonl file (efficient for large logs)."""
    path = _LOG_DIR / filename
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


_frame_counter = 0


def _append_detection(entry: dict):
    with _export_lock:
        _detection_log.append(entry)
        if len(_detection_log) > MAX_LOG:
            del _detection_log[: len(_detection_log) - MAX_LOG]
        _flush_json(f"detections_{_session_id}.json", _detection_log[-500:])
        # Tally object class frequencies
        for obj in entry.get("objects", []):
            cls = obj.get("class", "unknown")
            _object_counts[cls] = _object_counts.get(cls, 0) + 1


def _append_pose(entry: dict):
    with _export_lock:
        _pose_log.append(entry)
        if len(_pose_log) > MAX_LOG:
            del _pose_log[: len(_pose_log) - MAX_LOG]
        _append_jsonl(f"poses_{_session_id}.jsonl", entry)


def _append_action(entry: dict):
    with _export_lock:
        _action_log.append(entry)
        if len(_action_log) > MAX_LOG:
            del _action_log[: len(_action_log) - MAX_LOG]
        _append_jsonl(f"actions_{_session_id}.jsonl", entry)


def _append_frame(entry: dict):
    """Log every frame's combined data to JSONL for accuracy testing."""
    global _frame_counter
    with _export_lock:
        _frame_counter += 1
        entry["frame_id"] = _frame_counter
        _frame_log.append(entry)
        if len(_frame_log) > MAX_LOG:
            del _frame_log[: len(_frame_log) - MAX_LOG]
        _append_jsonl(f"frames_{_session_id}.jsonl", entry)


def _append_vlm(entry: dict):
    with _export_lock:
        _vlm_log.append(entry)
        if len(_vlm_log) > MAX_LOG:
            del _vlm_log[: len(_vlm_log) - MAX_LOG]
        _flush_json(f"captions_{_session_id}.json", _vlm_log)


def _append_report(entry: dict):
    with _export_lock:
        _reports.append(entry)
        _flush_json(f"reports_{_session_id}.json", _reports)


def _append_scene_index(entry: dict):
    """Append a structured scene snapshot to the index log."""
    with _export_lock:
        _scene_index.append(entry)
        if len(_scene_index) > MAX_LOG:
            del _scene_index[: len(_scene_index) - MAX_LOG]
        _flush_json(f"scene_index_{_session_id}.json", _scene_index)


def _frame_to_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _frame_to_b64(frame: np.ndarray, quality: int = 80) -> str:
    return base64.b64encode(_frame_to_jpeg(frame, quality)).decode()


# ── REST endpoints ────────────────────────────────────────────────────────

_start_time = time.time()


@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


def _get_local_ip() -> str:
    """Best-effort local LAN IP."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@app.get("/api/system")
def system_info():
    """Return system details for the UI status bar."""
    import torch
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    with _export_lock:
        det_count = len(_detection_log)
        vlm_count = len(_vlm_log)
        report_count = len(_reports)
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "device": device,
        "local_ip": _get_local_ip(),
        "models": {
            "detector": "YOLOv11n (ONNX)",
            "pose": "YOLOv11n-Pose (ONNX)",
            "action": "ST-GCN (NTU-60)",
            "vlm": "FastVLM 0.5B",
            "llm": "DeepSeek-R1:1.5B (Ollama)",
        },
        "uptime_s": round(time.time() - _start_time),
        "log_counts": {
            "detections": det_count,
            "captions": vlm_count,
            "reports": report_count,
        },
        "log_dir": str(_LOG_DIR),
    }


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
    ollama_url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434"),
):
    """Generate an LLM report from collected detection + VLM data."""
    from openai import OpenAI

    with _export_lock:
        det_summary = _detection_log[-20:]
        vlm_summary = _vlm_log[-10:]

    def _fmt_det(d: dict) -> str:
        objs = ", ".join(
            f"{o['class']} ({o.get('confidence', '?')})" for o in d["objects"]
        )
        return f"[{d['timestamp']}] {objs} — {d['count']} total"

    det_text = "\n".join(_fmt_det(d) for d in det_summary) or "(no detections)"
    vlm_text = "\n".join(
        f"- [{v['timestamp']}] {v.get('text', '')}" for v in vlm_summary
    ) or "(no captions)"

    prompt = (
        "You are VEREC, a video recognition and reporting system. "
        "Given the following detection log and scene captions from a live camera feed, "
        "write a concise surveillance report.\n\n"
        "Your report MUST include:\n"
        "1. **Summary** (2-3 sentences describing the scene)\n"
        "2. **Key Observations** (bullet points of notable detections)\n"
        "3. **Recommended Actions** (specific actionable items based on what was observed, "
        "e.g. \"Monitor crowd density at entrance\", \"Investigate unattended object\", "
        "\"Alert: person in restricted zone\")\n\n"
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


@app.get("/api/export/scene_index")
def export_scene_index():
    with _export_lock:
        data = list(_scene_index)
    return {"count": len(data), "scenes": data}


@app.get("/api/export/all")
def export_all():
    with _export_lock:
        return {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "detections": list(_detection_log),
            "vlm_captions": list(_vlm_log),
            "reports": list(_reports),
            "scene_index": list(_scene_index),
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
    enable_vlm: bool = True,
    enable_pose: bool = True,
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
    last_caption = ""
    last_action: dict | None = None
    # Frame buffer for multi-frame VLM captioning
    _vlm_frame_buf: list[np.ndarray] = []
    _vlm_context_buf: list[dict] = []  # per-frame detection/pose/action context
    VLM_FRAME_COUNT = 5  # collect N frames before running VLM

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
                enable_vlm = ctrl.get("enable_vlm", enable_vlm)
                enable_pose = ctrl.get("enable_pose", enable_pose)
                conf = ctrl.get("conf", conf)
                iou = ctrl.get("iou", iou)
                vlm_interval = ctrl.get("vlm_interval", vlm_interval)
            except (asyncio.TimeoutError, Exception):
                pass

            # --- frame processing wrapped so one bad frame won't kill the socket ---
            try:
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
                det_entry = {}
                if enable_det:
                    det_result = run_detection(frame, conf=conf, iou=iou)
                    det_objects = det_result["objects"]
                    payload["detection"] = {
                        "objects": det_objects,
                        "count": det_result["count"],
                        "time_ms": det_result["time_ms"],
                        "fps": det_result["fps"],
                    }
                    det_entry = {
                        "timestamp": ts,
                        "objects": det_objects,
                        "count": det_result["count"],
                        "time_ms": det_result["time_ms"],
                        "conf_threshold": conf,
                        "iou_threshold": iou,
                    }
                    _append_detection(det_entry)
                    # Include cumulative object frequency counts
                    with _export_lock:
                        payload["object_counts"] = dict(_object_counts)

                # Pose estimation + action recognition
                pose_keypoints = None
                pose_scores = None
                pose_entry = {}
                action_entry = {}
                if enable_pose:
                    pose_result = run_pose(frame, conf=conf, iou=iou)
                    if pose_result is not None:
                        payload["pose"] = {
                            "persons": pose_result["persons"],
                            "count": pose_result["count"],
                            "time_ms": pose_result["time_ms"],
                            "fps": pose_result["fps"],
                        }
                        pose_keypoints = pose_result["keypoints"]
                        pose_scores = pose_result["scores"]

                        # Log full pose data with keypoints for accuracy testing
                        pose_entry = {
                            "timestamp": ts,
                            "person_count": pose_result["count"],
                            "time_ms": pose_result["time_ms"],
                            "persons": [
                                {
                                    "id": i + 1,
                                    "confidence": p["confidence"],
                                    "box": p["box"],
                                    "keypoints": p["keypoints"],
                                }
                                for i, p in enumerate(pose_result["persons"])
                            ],
                        }
                        _append_pose(pose_entry)

                        # Feed to ST-GCN action recogniser
                        action_result = run_action(pose_keypoints, pose_scores)
                        if action_result is not None:
                            last_action = action_result
                            action_entry = {
                                "timestamp": ts,
                                "actions": last_action.get("actions", []),
                                "inference_time_ms": last_action.get("time_ms", 0),
                                "buffer_frames": last_action.get("buffer_size", 0),
                            }
                            _append_action(action_entry)
                        if last_action is not None:
                            payload["action"] = last_action

                # --- Log combined frame data for accuracy testing ---
                frame_entry: dict[str, Any] = {"timestamp": ts}
                if det_entry:
                    frame_entry["detection"] = det_entry
                if pose_entry:
                    frame_entry["pose"] = pose_entry
                if action_entry:
                    frame_entry["action"] = action_entry
                if last_caption:
                    frame_entry["caption"] = last_caption
                _append_frame(frame_entry)

                # Build AI frame with detection overlay
                ai_frame = frame.copy()
                if enable_det:
                    from detectors import YOLODetector
                    ai_frame = YOLODetector.draw(
                        ai_frame, det_result["boxes"], det_result["scores"], det_result["class_ids"]
                    )

                # Overlay pose skeleton on AI frame
                if enable_pose and pose_keypoints is not None and len(pose_keypoints) > 0:
                    from detectors import YOLOPoseDetector
                    pose_r = pose_result  # type: ignore[possibly-undefined]
                    ai_frame = YOLOPoseDetector.draw(
                        ai_frame,
                        np.array([p["box"] for p in pose_r["persons"]]),
                        pose_scores, pose_keypoints,
                    )

                # VLM (throttled) — multi-frame composite with enriched context
                if enable_vlm and now - last_vlm_time >= vlm_interval:
                    # Build per-frame context snapshot
                    frame_ctx: dict[str, Any] = {}
                    if det_objects:
                        # Label each person: Person 1, Person 2, etc.
                        person_count = 0
                        labeled_objects = []
                        for o in det_objects:
                            if o["class"] == "person":
                                person_count += 1
                                labeled_objects.append(
                                    f"Person {person_count} ({o['confidence']:.0%}, box {o.get('box', '?')})"
                                )
                            else:
                                labeled_objects.append(f"{o['class']} ({o['confidence']:.0%})")
                        frame_ctx["objects"] = labeled_objects
                        frame_ctx["person_count"] = person_count
                        frame_ctx["total_objects"] = len(det_objects)

                    if enable_pose and pose_keypoints is not None and len(pose_keypoints) > 0:
                        frame_ctx["pose_count"] = len(pose_keypoints)

                    if last_action and "actions" in last_action:
                        frame_ctx["actions"] = [
                            f"{a['label']} ({a['confidence']:.0%})"
                            for a in last_action["actions"]
                        ]

                    _vlm_frame_buf.append(frame.copy())
                    _vlm_context_buf.append(frame_ctx)

                    # Once we have enough frames, run VLM on composite
                    if len(_vlm_frame_buf) >= VLM_FRAME_COUNT:
                        last_vlm_time = now
                        composite = build_composite_image(_vlm_frame_buf, cols=3)

                        # Build enriched prompt from accumulated context
                        all_objects: set[str] = set()
                        total_persons = 0
                        all_actions: set[str] = set()
                        for ctx in _vlm_context_buf:
                            for obj_str in ctx.get("objects", []):
                                all_objects.add(obj_str)
                            total_persons = max(total_persons, ctx.get("person_count", 0))
                            for act in ctx.get("actions", []):
                                all_actions.add(act)

                        prompt_parts = [
                            "Ignore the grid layout. Treat this as a continuous video clip.",
                        ]
                        if all_objects:
                            prompt_parts.append(f"Objects detected: {', '.join(sorted(all_objects))}.")
                        if total_persons > 0:
                            prompt_parts.append(f"People tracked: {total_persons}.")
                        if all_actions:
                            prompt_parts.append(f"Actions detected: {', '.join(sorted(all_actions))}.")

                        prompt_parts.append(
                            "Describe ONLY the unique and important details: "
                            "Who is doing what? Where are they? What are they wearing? "
                            "What vehicles, signs, or objects are visible? "
                            "What direction are people/vehicles moving? "
                            "Any unusual behaviour, interactions between people, "
                            "or notable changes between frames? "
                            "Be specific and concise. Do NOT describe the image format or grid."
                        )
                        vlm_prompt = " ".join(prompt_parts)

                        vlm_result = run_vlm(
                            composite,
                            prompt=vlm_prompt,
                            max_tokens=200,
                        )
                        last_caption = vlm_result.get("text", "")
                        payload["vlm"] = vlm_result
                        _append_vlm({"timestamp": ts, **vlm_result})

                        # Build structured scene index entry
                        scene_entry = {
                            "timestamp": ts,
                            "frame_count": len(_vlm_frame_buf),
                            "caption": last_caption,
                            "objects": sorted(all_objects),
                            "person_count": total_persons,
                            "actions": sorted(all_actions),
                            "object_counts": dict(_object_counts) if _object_counts else {},
                        }
                        _append_scene_index(scene_entry)

                        # Reset buffers
                        _vlm_frame_buf.clear()
                        _vlm_context_buf.clear()

                # Encode frames
                raw_b64 = _frame_to_b64(frame, quality=70)
                ai_b64 = _frame_to_b64(ai_frame, quality=70)
                payload["raw_frame"] = raw_b64
                payload["ai_frame"] = ai_b64
                payload["frame_bytes"] = len(raw_b64) + len(ai_b64)  # approx payload size
                if last_caption:
                    payload["caption"] = last_caption

                await websocket.send_json(payload)
            except Exception as frame_exc:
                # Log but don't kill the connection for a single frame failure
                import logging
                logging.warning("Frame processing error: %s", frame_exc)
                await asyncio.sleep(0.1)
                continue
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        # Send a meaningful error to the client before closing
        try:
            await websocket.send_json({"error": str(exc)})
            await websocket.close(code=1011)
        except Exception:
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

    print("Loading detector…")
    init_yolo()
    # VLM, camera, pose, and action models are all lazy-loaded on first
    # use to keep startup fast and avoid memory spikes.
    configure_vlm(args.model_path, args.model_base)
    # Warm up VLM in a background thread after 10s so the first
    # caption request doesn't stall.
    schedule_vlm_warmup(delay=0)  # load immediately
    print("Ready (VLM loading in background; camera/pose/action load on first use). Starting API server…")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
