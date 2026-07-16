"""VEREC FastAPI backend — REST + WebSocket API for the Next.js frontend."""
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
from fastapi import Request

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from backend.models import (
    init_yolo, init_camera, configure_vlm, schedule_vlm_warmup,
    init_yolo_pose, init_action,
    get_camera_frame, run_detection, run_vlm,
    run_pose, run_action,
    resolve_stream_url, IP_CAMERA_PRESETS,
)
import backend.models as _models
from backend.memory_indexer import build_memory_node, search_memory
from backend.query_engine import answer_query
from backend.auto_memory import start_auto_memory_thread

app = FastAPI(title="VEREC API", version="1.0.0")

# ── Default model path ──────────────────────────────────────────────────────
_DEFAULT_MODEL_PATH = os.environ.get(
    "VLM_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)),
                 "checkpoints", "llava-fastvithd_0.5b_stage3"),
)

# ── Global variables ──────────────────────────────────────────────────────
_current_source = "local"
_current_url = ""
_vlm_running = False

# ── Startup event ──────────────────────────────────────────────────────────
@app.on_event("startup")
def _auto_init():
    print("[Startup] Entering _auto_init...")
    # Check Ollama
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            print(f"[Startup] ✅ Ollama reachable. Models: {models}")
            if "qwen2.5-vl:3b" in models:
                print("[Startup] ⏳ Pre-warming qwen2.5-vl:3b vision path (this may take 1-2 minutes)...")
                try:
                    _dummy_buf = io.BytesIO()
                    Image.new("RGB", (224, 224), color=(0, 0, 0)).save(_dummy_buf, format="JPEG")
                    _dummy_b64 = base64.b64encode(_dummy_buf.getvalue()).decode("utf-8")
                    warmup_resp = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "qwen2.5-vl:3b",
                            "prompt": "Hello",
                            "images": [_dummy_b64],
                            "stream": False,
                            "options": {"num_predict": 1}
                        },
                        timeout=60
                    )
                    if warmup_resp.status_code == 200:
                        print("[Startup] ✅ Model warmed up.")
                    else:
                        print(f"[Startup] ⚠️ Warmup returned status {warmup_resp.status_code}")
                except requests.Timeout:
                    print("[Startup] ⚠️ Warmup timed out (model may load on first request).")
                except Exception as e:
                    print(f"[Startup] ⚠️ Warmup error: {e}")
            else:
                print("[Startup] ❌ qwen2.5-vl:3b not found.")
        else:
            print("[Startup] ❌ Ollama error")
    except Exception as e:
        print(f"[Startup] ❌ Could not reach Ollama: {e}")

    if _models._vlm_config is None:
        print("[Startup] Configuring VLM...")
        configure_vlm(_DEFAULT_MODEL_PATH)
        print("[Startup] VLM configured.")
    if os.getenv("USE_QWEN_VL", "false").lower() != "true":
        print("[Startup] Scheduling VLM warmup...")
        schedule_vlm_warmup(delay=0)
        print("[Startup] Warmup scheduled.")
    print("[Startup] Starting automatic memory compressor thread...")
    start_auto_memory_thread(
        _LOG_DIR / f"detections_{_session_id}.json",
        _LOG_DIR / f"captions_{_session_id}.json",
        camera_id="default",
    )
    print("[Startup] _auto_init() complete.")

# ── CORS ────────────────────────────────────────────────────────────────────
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

# ── Data store ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(__file__)))
_LOG_DIR = _PROJECT_ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_export_lock = threading.Lock()
_detection_log: list[dict] = []
_vlm_log: list[dict] = []
_reports: list[dict] = []
_object_counts: dict[str, int] = {}
MAX_LOG = 500
_session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def _flush_json(filename: str, data: list[dict]):
    path = _LOG_DIR / filename
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass

def _append_detection(entry: dict):
    with _export_lock:
        _detection_log.append(entry)
        if len(_detection_log) > MAX_LOG:
            del _detection_log[: len(_detection_log) - MAX_LOG]
        for obj in entry.get("objects", []):
            cls = obj.get("class", "unknown")
            _object_counts[cls] = _object_counts.get(cls, 0) + 1
        _flush_json(f"detections_{_session_id}.json", _detection_log)

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
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

@app.get("/api/system")
def system_info():
    print("[SYSTEM] Entered")
    ip = _get_local_ip()
    print(f"[SYSTEM] IP: {ip}")
    use_qwen = os.getenv("USE_QWEN_VL", "false").lower() == "true"
    vlm_name = "Qwen2.5-VL-3B (Ollama)" if use_qwen else "FastVLM 0.5B"
    data = {
        "status": "ok",
        "ip": ip,
        "vlm_backend": vlm_name,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    print("[SYSTEM] Returning")
    return JSONResponse(content=data)

@app.get("/api/presets")
def presets():
    return IP_CAMERA_PRESETS

@app.get("/api/frame")
def get_frame(source: str = "local", url: str = ""):
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
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **result}
    _append_vlm(entry)
    return entry

@app.post("/api/chat")
async def chat(request: Request,
               prompt: str = "",
               temperature: float = 0.0,
               max_tokens: int = 100,
               source: str = "",
               url: str = ""):
    global _current_source, _current_url
    if not prompt:
        try:
            body = await request.json()
            prompt = body.get("prompt") or body.get("message") or body.get("question") or ""
        except Exception:
            pass
    if not source:
        source = _current_source
    if not url and _current_url:
        url = _current_url
    if not source:
        source = "local"
    frame = _get_source_frame(source, url)
    if frame is None:
        return JSONResponse({"error": "No frame available"}, status_code=503)
    result = run_vlm(frame, prompt=prompt, temperature=temperature, max_tokens=max_tokens)
    return {
        "text": result.get("text", ""),
        "answer": result.get("text", ""),
        "reply": result.get("text", ""),
        "response": result.get("text", ""),
        "time_s": result.get("time_s", 0),
        "tokens": result.get("tokens", 0),
        "tokens_per_s": result.get("tokens_per_s", 0),
        "backend": result.get("backend", "qwen-ollama"),
    }

@app.post("/api/report")
def generate_report(
    model: str = "llama3.2:3b",  # Changed from deepseek-r1:1.5b
    ollama_url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434"),
):
    from openai import OpenAI
    with _export_lock:
        det_summary = _detection_log[-20:]
        vlm_summary = _vlm_log[-10:]
    def _fmt_det(d):
        objs = ", ".join(f"{o['class']} ({o.get('confidence', '?')})" for o in d["objects"])
        return f"[{d['timestamp']}] {objs} — {d['count']} total"
    det_text = "\n".join(_fmt_det(d) for d in det_summary) or "(no detections)"
    vlm_text = "\n".join(f"- [{v['timestamp']}] {v.get('text', '')}" for v in vlm_summary) or "(no captions)"
    prompt = (
        "You are VEREC, a video recognition and reporting system. "
        "Given the following detection log and scene captions from a live camera feed, "
        "write a concise surveillance report.\n\n"
        "Your report MUST include:\n"
        "1. **Summary** (2-3 sentences describing the scene)\n"
        "2. **Key Observations** (bullet points of notable detections)\n"
        "3. **Recommended Actions** (specific actionable items based on what was observed)\n\n"
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

@app.post("/api/memory/build")
def memory_build(camera_id: str = "default", model: str = "llama3.2:3b"):
    """Compress the current session's detection/caption log into a memory node (VR-3)."""
    with _export_lock:
        detections = list(_detection_log)
        captions = list(_vlm_log)
    if not detections and not captions:
        return JSONResponse({"error": "No detections or captions to index"}, status_code=400)
    node = build_memory_node(detections, captions, camera_id=camera_id, model=model)
    return node

@app.get("/api/memory/search")
def memory_search(q: str, camera_id: str = "", limit: int = 10):
    """Search stored memory nodes by keyword, without re-processing frames."""
    results = search_memory(q, camera_id=camera_id or None, limit=limit)
    return {"query": q, "count": len(results), "results": results}

@app.get("/api/memory/query")
def memory_query(q: str, camera_id: str = "", model: str = "llama3.2:3b", limit: int = 5):
    """Answer a natural-language question, grounded strictly in stored memory nodes (VR-4)."""
    return answer_query(q, camera_id=camera_id or None, model=model, limit=limit)

# ── WebSocket ──────────────────────────────────────────────────────────────

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
    enable_pose: bool = False,
):
    global _current_source, _current_url, _vlm_running
    _current_source = source
    _current_url = url

    await websocket.accept()
    await websocket.send_json({"status": "connected", "message": "Connection established, initializing..."})
    
    cap = None
    use_local = source == "local"

    if not use_local:
        resolved = resolve_stream_url(url)
        print(f"[WS] Resolved URL: {resolved}")
        cap = cv2.VideoCapture(resolved, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        if not cap.isOpened():
            print("[WS] Failed to open stream")
            await websocket.send_json({"error": "Failed to open stream"})
            await websocket.close()
            return
        print("[WS] Stream opened successfully")

    last_vlm_time = 0.0
    last_caption = ""
    last_action: dict | None = None
    last_vlm_signature: tuple = ()
    MIN_VLM_GAP = 2.0  # seconds; debounce so detection flicker can't spam VLM calls
    consecutive_read_failures = 0

    # ── Inner VLM processor (captures last_caption) ──
    async def process_vlm_inner(frame: np.ndarray, prompt: str, ts: str):
        nonlocal last_caption
        global _vlm_running
        if _vlm_running:
            print("[WS] VLM already running, skipping.")
            return
        _vlm_running = True
        try:
            h, w = frame.shape[:2]
            if h > 336 or w > 336:
                frame = cv2.resize(frame, (336, 336), interpolation=cv2.INTER_LINEAR)
                print("[WS] Frame resized to 336x336 for VLM")

            print("[WS] Calling run_vlm (timeout=300s)...")
            vlm_result = await asyncio.wait_for(
                asyncio.to_thread(run_vlm, frame, prompt=prompt, max_tokens=80),
                timeout=300.0
            )
            print("[WS] run_vlm completed")
            new_caption = vlm_result.get("text", "")
            if not new_caption and not vlm_result.get("error"):
                # Empty-but-not-erroring response — often a cold-start hiccup on the
                # first vision call. Retry once before giving up.
                print("[WS] ⚠️ VLM returned empty text, retrying once...")
                vlm_result = await asyncio.wait_for(
                    asyncio.to_thread(run_vlm, frame, prompt=prompt, max_tokens=80),
                    timeout=300.0
                )
                new_caption = vlm_result.get("text", "")
            if new_caption:
                last_caption = new_caption
                await websocket.send_json({
                    "timestamp": ts,
                    "vlm": vlm_result,
                    "caption": new_caption,
                    "status": "vlm_done"
                })
                _append_vlm({"timestamp": ts, **vlm_result})
            else:
                err = vlm_result.get("error", "no error reported")
                print(f"[WS] ⚠️ VLM returned empty text after retry: {err}")
                await websocket.send_json({
                    "timestamp": ts,
                    "vlm": vlm_result,
                    "caption": "",
                    "status": "vlm_error",
                    "message": f"VLM returned no text: {err}",
                })
        except asyncio.TimeoutError:
            print("[WS] VLM timed out after 300 seconds")
            await websocket.send_json({
                "timestamp": ts,
                "vlm": {"text": "⚠️ VLM timed out. Check Ollama.", "backend": "qwen-ollama"},
                "caption": "⚠️ VLM timed out. Check Ollama.",
                "status": "vlm_error"
            })
        except Exception as e:
            print(f"[WS] VLM error: {e}")
            await websocket.send_json({
                "timestamp": ts,
                "vlm": {"text": f"⚠️ VLM error: {str(e)}", "backend": "qwen-ollama"},
                "caption": f"⚠️ VLM error: {str(e)}",
                "status": "vlm_error"
            })
        finally:
            _vlm_running = False

    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                ctrl = json.loads(msg)
                if ctrl.get("action") == "stop":
                    break
                enable_det = ctrl.get("enable_det", enable_det)
                enable_vlm = ctrl.get("enable_vlm", enable_vlm)
                enable_pose = ctrl.get("enable_pose", enable_pose)
                conf = ctrl.get("conf", conf)
                iou = ctrl.get("iou", iou)
                vlm_interval = ctrl.get("vlm_interval", vlm_interval)
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                raise
            except Exception:
                pass

            try:
                if use_local:
                    frame = get_camera_frame()
                else:
                    assert cap is not None
                    ok, bgr = cap.read()
                    if not ok:
                        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        if frame_count and frame_count > 0:
                            # Finite file source hit EOF — loop back to the start.
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            consecutive_read_failures = 0
                            await asyncio.sleep(0.05)
                            continue
                        consecutive_read_failures += 1
                        if consecutive_read_failures == 1 or consecutive_read_failures % 100 == 0:
                            print(f"[WS] Frame read failed ({consecutive_read_failures}x) – retrying")
                        if consecutive_read_failures > 300:
                            print("[WS] Stream appears dead after repeated read failures, closing")
                            break
                        await asyncio.sleep(0.05)
                        continue
                    consecutive_read_failures = 0
                    frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None

                if frame is None:
                    await asyncio.sleep(0.05)
                    continue

                now = time.perf_counter()
                ts = datetime.now(timezone.utc).isoformat()
                payload: dict[str, Any] = {"timestamp": ts}

                det_objects = []
                if enable_det:
                    await websocket.send_json({"status": "loading_detector", "message": "Loading YOLO model..."})
                    print("[WS] Calling run_detection...")
                    det_result = run_detection(frame, conf=conf, iou=iou)
                    print("[WS] run_detection completed")
                    det_objects = det_result["objects"]
                    payload["detection"] = {
                        "objects": det_objects,
                        "count": det_result["count"],
                        "time_ms": det_result["time_ms"],
                        "fps": det_result["fps"],
                    }
                    _append_detection({"timestamp": ts, "objects": det_objects,
                                       "count": det_result["count"], "time_ms": det_result["time_ms"]})
                    with _export_lock:
                        payload["object_counts"] = dict(_object_counts)

                pose_keypoints = None
                pose_scores = None
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
                        action_result = run_action(pose_keypoints, pose_scores)
                        if action_result is not None:
                            last_action = action_result
                        if last_action is not None:
                            payload["action"] = last_action

                ai_frame = frame.copy()
                if enable_det:
                    from detectors import YOLODetector
                    ai_frame = YOLODetector.draw(
                        ai_frame, det_result["boxes"], det_result["scores"], det_result["class_ids"]
                    )

                if enable_pose and pose_keypoints is not None and len(pose_keypoints) > 0:
                    from detectors import YOLOPoseDetector
                    pose_r = pose_result
                    ai_frame = YOLOPoseDetector.draw(
                        ai_frame,
                        np.array([p["box"] for p in pose_r["persons"]]),
                        pose_scores, pose_keypoints,
                    )

                raw_b64 = _frame_to_b64(frame, quality=70)
                ai_b64 = _frame_to_b64(ai_frame, quality=70)
                payload["raw_frame"] = raw_b64
                payload["ai_frame"] = ai_b64
                payload["frame_bytes"] = len(raw_b64) + len(ai_b64)

                # Include latest caption in every frame
                if last_caption:
                    payload["caption"] = last_caption

                await websocket.send_json(payload)

                # Trigger VLM when the detected environment changes (new/gone objects,
                # count shifts), falling back to the plain interval for static scenes.
                class_counts: dict[str, int] = {}
                for o in det_objects:
                    class_counts[o["class"]] = class_counts.get(o["class"], 0) + 1
                current_signature = tuple(sorted(class_counts.items()))
                env_changed = enable_det and current_signature != last_vlm_signature

                trigger_vlm = (
                    enable_vlm
                    and not _vlm_running
                    and now - last_vlm_time >= MIN_VLM_GAP
                    and (env_changed or now - last_vlm_time >= vlm_interval)
                )

                if trigger_vlm:
                    last_vlm_time = now
                    last_vlm_signature = current_signature
                    await websocket.send_json({"status": "vlm_loading", "message": "Generating caption..."})
                    if det_objects:
                        obj_detail = "; ".join(
                            f"{o['class']} ({o['confidence']:.0%})" for o in det_objects
                        )
                        vlm_prompt = (
                            f"Objects detected: {obj_detail}. "
                            f"Total: {len(det_objects)} object(s). "
                            "Describe the scene and any notable activity in one sentence."
                        )
                    else:
                        vlm_prompt = "Describe what is happening in this scene in one sentence."
                    await websocket.send_json({"status": "vlm_started", "message": "VLM processing started (may take a few minutes)"})
                    asyncio.create_task(process_vlm_inner(frame, vlm_prompt, ts))

            except Exception as frame_exc:
                import logging
                if (
                    websocket.client_state != WebSocketState.CONNECTED
                    or websocket.application_state != WebSocketState.CONNECTED
                ):
                    logging.warning("Socket no longer connected, stopping feed loop: %s", frame_exc)
                    break
                logging.warning("Frame processing error: %s", frame_exc)
                await asyncio.sleep(0.1)
                continue
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"error": str(exc)})
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        if cap is not None:
            cap.release()

# ── Helper ──────────────────────────────────────────────────────────────────

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

# ── CLI entry ──────────────────────────────────────────────────────────────

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
    configure_vlm(args.model_path, args.model_base)
    schedule_vlm_warmup(delay=0)
    print("Ready. Starting API server…")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()