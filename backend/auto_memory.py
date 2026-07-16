"""VEREC automatic rule-based memory compressor.

Unlike the LLM-driven Data Indexer (memory_indexer.build_memory_node, used by
the manual "Build Memory Node" button), this module is a fully deterministic
compressor — no LLM call, so it keeps working even when Ollama is slow or
down. It runs on a background timer and periodically collapses the raw
detection/caption log into a compact memory node, following four rules:

1. TIME-BLOCK GENERALIZATION — an object that stays in roughly the same
   place for 2+ minutes is collapsed into a single stationary summary line
   instead of being logged frame by frame.
2. DYNAMIC LOGGING TRIGGER — the moment an object's position changes
   enough to count as movement, or a brand-new object appears, that event
   is recorded individually rather than absorbed into a stationary block.
3. CONFIDENCE FILTERING — objects are grouped into High (>0.8) and Medium
   (0.5-0.8) confidence buckets, plain object names with no quote marks;
   each unmoving object collapses to one Max Confidence instance per block.
4. AUTOMATIC STORAGE — a node is saved every 5 minutes, or immediately if a
   brand-new object class enters the scene (a "major" shift), whichever
   comes first.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from backend.memory_indexer import _consolidate_tracks, _iou, _save_node

MOVE_IOU_THRESHOLD = 0.5   # below this, a track's position counts as "moved"
QUIET_GAP_S = 120.0        # minimum unchanging duration to collapse into one block
HIGH_CONF = 0.8
MED_CONF = 0.5

_COLOR_WORDS = ("white", "black", "red", "blue", "yellow", "silver", "green", "gray", "grey", "orange")


# ── Confidence grouping (rule 3) ─────────────────────────────────────────

def _bucket(conf: float) -> str | None:
    if conf > HIGH_CONF:
        return "High"
    if conf >= MED_CONF:
        return "Medium"
    return None


def _color_hint(captions: list[dict], cls: str, start_ts: str, end_ts: str) -> str:
    """Reuse a color word already present in a caption covering this block —
    never invent one. E.g. "White car parked" only if some caption in this
    window actually said "white car"."""
    pattern = re.compile(rf"\b({'|'.join(_COLOR_WORDS)})\s+{re.escape(cls)}\b", re.IGNORECASE)
    for c in captions:
        if start_ts <= c["timestamp"] <= end_ts:
            m = pattern.search(c.get("text", ""))
            if m:
                return f"{m.group(1).lower()} {cls}"
    return cls


# ── Track classification (rules 1 & 2) ───────────────────────────────────

def _classify_tracks(detections: list[dict]) -> tuple[list[dict], list[dict]]:
    """Splits consolidated tracks into stationary (long, unmoving) vs
    dynamic (moved, or too brief to have "settled") groups."""
    tracks = _consolidate_tracks(detections, iou_threshold=0.3)
    stationary, dynamic = [], []
    for t in tracks:
        duration_s = (datetime.fromisoformat(t["last_ts"]) - datetime.fromisoformat(t["first_ts"])).total_seconds()
        net_iou = _iou(t["first_box"], t["box"])
        if duration_s >= QUIET_GAP_S and net_iou >= MOVE_IOU_THRESHOLD:
            stationary.append(t)
        else:
            t["moved"] = net_iou < MOVE_IOU_THRESHOLD
            dynamic.append(t)
    return stationary, dynamic


def _fmt_ts(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%H:%M:%S")
    except ValueError:
        return ts


def _stationary_block_text(tracks: list[dict], start_ts: str, end_ts: str, captions: list[dict]) -> str:
    high = sorted({_color_hint(captions, t["class"], start_ts, end_ts) for t in tracks if _bucket(t["max_conf"]) == "High"})
    med = sorted({_color_hint(captions, t["class"], start_ts, end_ts) for t in tracks if _bucket(t["max_conf"]) == "Medium"})
    if not high and not med:
        body = "Scene stationary. No significant activity."
    else:
        parts = []
        if high:
            parts.append(f"{', '.join(high)} present (High confidence)")
        if med:
            parts.append(f"{', '.join(med)} present (Medium confidence)")
        body = "Scene stationary. " + "; ".join(parts) + ". No significant activity."
    return f"[{_fmt_ts(start_ts)} to {_fmt_ts(end_ts)}] {body}"


def _dynamic_block_text(tracks: list[dict]) -> str:
    lines = []
    for t in sorted(tracks, key=lambda t: t["first_ts"]):
        bucket = _bucket(t["max_conf"])
        if bucket is None:
            continue
        verb = "moved through the scene" if t.get("moved") else "appeared briefly"
        lines.append(
            f"[{_fmt_ts(t['first_ts'])} to {_fmt_ts(t['last_ts'])}] {t['class']} {verb} "
            f"({bucket} confidence, max {t['max_conf']:.2f})."
        )
    return "\n".join(lines)


# ── Node assembly ─────────────────────────────────────────────────────────

def build_auto_node(detections: list[dict], captions: list[dict], camera_id: str) -> dict | None:
    """Deterministically compress one window of raw log data into a memory
    node — no LLM call. Returns None if there's nothing to index."""
    if not detections and not captions:
        return None
    all_ts = [d["timestamp"] for d in detections] + [c["timestamp"] for c in captions]
    start_ts, end_ts = min(all_ts), max(all_ts)

    stationary, dynamic = _classify_tracks(detections)

    blocks: list[str] = []
    if stationary:
        # Multiple stationary tracks can share overlapping windows; report
        # the union span so rule 1's "collapse the quiet stretch" holds even
        # when several unmoving objects coexist.
        s_start = min(t["first_ts"] for t in stationary)
        s_end = max(t["last_ts"] for t in stationary)
        blocks.append(_stationary_block_text(stationary, s_start, s_end, captions))
    if dynamic:
        dyn_text = _dynamic_block_text(dynamic)
        if dyn_text:
            blocks.append(dyn_text)
    if not blocks:
        blocks.append(f"[{_fmt_ts(start_ts)} to {_fmt_ts(end_ts)}] No significant activity.")

    text = "\n".join(blocks)

    tags = sorted({t["class"] for t in stationary} | {t["class"] for t in dynamic})
    node_id = f"{camera_id}_auto_{start_ts.replace(':', '').replace('-', '')}"

    node = {
        "id": node_id,
        "camera_id": camera_id,
        "start": start_ts,
        "end": end_ts,
        "tags": tags,
        "tags_source": "auto-rule",
        "text": text,
        "model": "rule-based",
        "source_frame_count": len(detections),
        "source_caption_count": len(captions),
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_node(node)
    return node


# ── Background scheduler (rule 4) ────────────────────────────────────────

def _read_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def start_auto_memory_thread(
    detections_path: Path,
    captions_path: Path,
    camera_id: str = "default",
    interval_s: float = 300.0,
    poll_s: float = 10.0,
    min_gap_s: float = 30.0,
) -> threading.Thread:
    """Starts a daemon thread that saves a new auto memory node every
    interval_s seconds, or as soon as a brand-new object class shows up
    (bounded by min_gap_s so a burst of new classes can't spam saves)."""

    def _loop():
        checkpoint: str | None = None
        last_save = time.monotonic()
        seen_classes: set[str] = set()
        while True:
            time.sleep(poll_s)
            detections = _read_json(detections_path)
            captions = _read_json(captions_path)
            if checkpoint is not None:
                detections = [d for d in detections if d["timestamp"] > checkpoint]
                captions = [c for c in captions if c["timestamp"] > checkpoint]
            if not detections and not captions:
                continue

            new_classes = {o["class"] for d in detections for o in d.get("objects", [])} - seen_classes
            elapsed = time.monotonic() - last_save
            major_shift = bool(new_classes) and elapsed >= min_gap_s

            if elapsed >= interval_s or major_shift:
                try:
                    node = build_auto_node(detections, captions, camera_id)
                except Exception as e:
                    print(f"[auto_memory] build failed: {e}")
                    node = None
                if node is not None:
                    checkpoint = node["end"]
                    last_save = time.monotonic()
                    seen_classes |= new_classes

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t
