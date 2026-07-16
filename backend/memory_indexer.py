"""VEREC hierarchical memory indexer (VR-3).

Compresses a raw per-frame detection/caption timeline for one camera/time
block into a single structured, searchable "memory node". Runs once per
camera per time block; later queries search the stored nodes (by tag/text
keyword match) instead of re-processing the source frames.

Raw per-frame detections carry no persistent track ID, so object tracks are
first consolidated with a simple greedy IoU tracker. The consolidated
timeline (tracks + VLM captions) is then handed to an LLM via Ollama, using
the Data Indexer prompt, to produce the structured memory node text.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(__file__)))
_MEMORY_DIR = _PROJECT_ROOT / "logs" / "memory"
_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
_INDEX_PATH = _MEMORY_DIR / "index.json"

DEFAULT_MODEL = "llama3.2:3b"

_INDEXER_SYSTEM_PROMPT = """You are a Data Indexer for a Hierarchical Security Memory System. Your job is to compress raw timeline logs into a single high-level, searchable memory node. Remove redundant details while preserving all critical tracking data.

INPUT: A raw timeline log (timestamped detection events, object tracks, or frame annotations) for one camera over one time block.

TASK: Analyze the raw log and output EXACTLY the structured format below. Do not add commentary before or after the structure. If a field has no data, write "None" - do not omit the field.

[MEMORY NODE ID]
- Time Range: [start-end, YYYY-MM-DD HH:MM to HH:MM]
- Location/Camera: [camera name/ID]

[SEMANTIC INDEX TAGS]
Comma-separated keywords for fast filtering. Include all vehicle types, colors, clothing colors, and actions observed.
(Example: white delivery van, blue jacket, package delivery, red sedan, pedestrian)

[HIERARCHICAL SCENE SUMMARY]
A 3-sentence narrative of what happened in this time block. Focus on the core story, not frame-by-frame detail.

[ENTITY REGISTRY]
One line per unique person/vehicle tracking event. Do not repeat entries for the same object across frames - consolidate into a single entry.
- Entity 1: [description] | [total duration seen] | [key actions performed]
- Entity 2: [description] | [total duration seen] | [key actions performed]

[CRITICAL ANOMALIES]
List loitering, unexpected nighttime activity, or safety flags. If none, write "None".

RULES:
- Never invent details not present in the raw log.
- If object identity is ambiguous across frames, note it as "possible re-entry" rather than merging into one entity.
- Keep the summary factual and neutral - no speculation about intent."""


# ── Object track consolidation ───────────────────────────────────────────

def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _consolidate_tracks(
    detections: list[dict], iou_threshold: float = 0.3, max_gap_frames: int = 10
) -> list[dict]:
    """Greedy IoU tracker consolidating per-frame boxes into object tracks.

    Each frame's detections are matched to the nearest still-active track of
    the same class (by IoU); unmatched detections start new tracks, and
    tracks not matched for more than max_gap_frames are closed out.
    """
    active: list[dict] = []
    finished: list[dict] = []

    for entry in detections:
        ts = entry["timestamp"]
        matched = [False] * len(active)
        for obj in entry.get("objects", []):
            cls, box, conf = obj["class"], obj["box"], obj["confidence"]
            best_idx, best_iou = -1, iou_threshold
            for i, tr in enumerate(active):
                if tr["class"] != cls or matched[i]:
                    continue
                score = _iou(tr["box"], box)
                if score > best_iou:
                    best_idx, best_iou = i, score
            if best_idx >= 0:
                tr = active[best_idx]
                tr["box"] = box
                tr["last_ts"] = ts
                tr["frames"] += 1
                tr["max_conf"] = max(tr["max_conf"], conf)
                tr["gap"] = 0
                matched[best_idx] = True
            else:
                active.append({
                    "class": cls, "box": box, "first_box": box, "first_ts": ts, "last_ts": ts,
                    "frames": 1, "max_conf": conf, "gap": 0,
                })
                matched.append(True)

        still_active = []
        for i, tr in enumerate(active):
            if not matched[i]:
                tr["gap"] += 1
            if tr["gap"] > max_gap_frames:
                finished.append(tr)
            else:
                still_active.append(tr)
        active = still_active

    finished.extend(active)
    return finished


def _format_duration(first_ts: str, last_ts: str) -> str:
    try:
        t0 = datetime.fromisoformat(first_ts)
        t1 = datetime.fromisoformat(last_ts)
    except ValueError:
        return "unknown"
    secs = max(0.0, (t1 - t0).total_seconds())
    if secs < 1:
        return "<1s"
    m, s = divmod(int(secs), 60)
    return f"{m}m {s}s" if m else f"{s}s"


def _build_raw_log_text(tracks: list[dict], captions: list[dict]) -> str:
    """Format precomputed tracks + captions into the LLM's raw-log input."""
    lines = ["## Detected object tracks"]
    if tracks:
        for t in tracks:
            lines.append(
                f"- {t['class']} | first seen {t['first_ts']} | last seen {t['last_ts']} "
                f"| duration {_format_duration(t['first_ts'], t['last_ts'])} "
                f"| {t['frames']} frames | max confidence {t['max_conf']:.2f}"
            )
    else:
        lines.append("(no sustained object tracks)")

    lines.append("\n## Scene captions")
    if captions:
        for c in captions:
            lines.append(f"- [{c['timestamp']}] {c.get('text', '')}")
    else:
        lines.append("(no captions)")

    return "\n".join(lines)


def _extract_section(text: str, header: str) -> str:
    # Tolerate stray whitespace inside the brackets (e.g. "[ ANSWER ]"),
    # which small/CPU-bound models occasionally emit.
    pattern = rf"\[\s*{re.escape(header)}\s*\]\s*(.*?)(?=\n\s*\[|\Z)"
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _has_headers(text: str, headers: tuple[str, ...]) -> bool:
    """Small/CPU-bound models sometimes ignore a required output template and
    free-associate instead — check the response actually has the required
    section headers before trusting it, rather than silently trusting
    unparseable output."""
    return all(h in text for h in headers)


_REQUIRED_HEADERS = ("[MEMORY NODE ID]", "[SEMANTIC INDEX TAGS]", "[HIERARCHICAL SCENE SUMMARY]", "[ENTITY REGISTRY]")


def _is_well_formed(text: str) -> bool:
    return _has_headers(text, _REQUIRED_HEADERS)


# ── Memory node build / store ────────────────────────────────────────────

def build_memory_node(
    detections: list[dict],
    captions: list[dict],
    camera_id: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str | None = None,
) -> dict:
    """Compress one camera's raw timeline into a stored, searchable memory node."""
    all_ts = [d["timestamp"] for d in detections] + [c["timestamp"] for c in captions]
    if not all_ts:
        raise ValueError("No detections or captions to index")
    start_ts, end_ts = min(all_ts), max(all_ts)

    tracks = sorted(
        (t for t in _consolidate_tracks(detections) if t["frames"] >= 3),
        key=lambda t: t["first_ts"],
    )
    raw_log_text = _build_raw_log_text(tracks, captions)
    node_id = f"{camera_id}_{start_ts.replace(':', '').replace('-', '')}"

    user_prompt = (
        f"Camera: {camera_id}\n"
        f"Time Range: {start_ts} to {end_ts}\n\n"
        f"{raw_log_text}"
    )

    from openai import OpenAI
    ollama_url = ollama_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
    node_text = ""
    tags_source = "llm"
    try:
        client = OpenAI(base_url=f"{ollama_url.rstrip('/')}/v1", api_key="ollama")
        messages = [
            {"role": "system", "content": _INDEXER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        resp = client.chat.completions.create(model=model, messages=messages, max_tokens=700)
        node_text = resp.choices[0].message.content or ""

        # Small/CPU-bound models sometimes ignore the required template. Give
        # it one corrective retry before falling back to deterministic tags,
        # rather than silently saving an unsearchable, empty-tag node.
        if not _is_well_formed(node_text):
            messages += [
                {"role": "assistant", "content": node_text},
                {"role": "user", "content": (
                    "Your previous reply did not follow the required format. "
                    "Reply again using EXACTLY these section headers, in order: "
                    "[MEMORY NODE ID], [SEMANTIC INDEX TAGS], [HIERARCHICAL SCENE SUMMARY], "
                    "[ENTITY REGISTRY], [CRITICAL ANOMALIES]. No other text."
                )},
            ]
            retry_resp = client.chat.completions.create(model=model, messages=messages, max_tokens=700)
            retry_text = retry_resp.choices[0].message.content or ""
            if _is_well_formed(retry_text):
                node_text = retry_text
    except Exception as e:
        node_text = f"[MEMORY NODE ID]\n- Time Range: {start_ts} to {end_ts}\n- Location/Camera: {camera_id}\n\n(Indexing failed: {e})"

    tags_raw = _extract_section(node_text, "SEMANTIC INDEX TAGS")
    tags = [t.strip().lower() for t in tags_raw.split(",") if t.strip()]
    if not tags:
        # LLM never produced a usable tags section — fall back to the object
        # classes actually tracked, so the node stays searchable rather than
        # silently indexing as empty.
        tags = sorted({t["class"] for t in tracks})
        tags_source = "fallback"

    node = {
        "id": node_id,
        "camera_id": camera_id,
        "start": start_ts,
        "end": end_ts,
        "tags": tags,
        "tags_source": tags_source,
        "text": node_text,
        "model": model,
        "source_frame_count": len(detections),
        "source_caption_count": len(captions),
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_node(node)
    return node


def _node_path(node_id: str) -> Path:
    return _MEMORY_DIR / f"{node_id}.json"


def _save_node(node: dict) -> None:
    _node_path(node["id"]).write_text(json.dumps(node, indent=2), encoding="utf-8")

    index = _load_index()
    index = [e for e in index if e["id"] != node["id"]]
    index.append({
        "id": node["id"], "camera_id": node["camera_id"],
        "start": node["start"], "end": node["end"], "tags": node["tags"],
    })
    index.sort(key=lambda e: e["start"])
    _INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")


def _load_index() -> list[dict]:
    if not _INDEX_PATH.exists():
        return []
    try:
        return json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _load_node(node_id: str) -> dict | None:
    path = _node_path(node_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ── Search ────────────────────────────────────────────────────────────────

def search_memory(query: str, camera_id: str | None = None, limit: int = 10) -> list[dict]:
    """Search stored memory nodes by keyword — reads the index/nodes, never raw frames.

    Ranks nodes by how many query terms match their semantic index tags,
    falling back to a full-text match against the node body for a lower score.
    """
    terms = [t.strip().lower() for t in re.split(r"[,\s]+", query) if t.strip()]
    if not terms:
        return []

    scored: list[tuple[float, dict]] = []
    for entry in _load_index():
        if camera_id and entry["camera_id"] != camera_id:
            continue
        tag_text = " ".join(entry.get("tags", []))
        tag_hits = sum(1 for t in terms if t in tag_text)
        if tag_hits > 0:
            scored.append((float(tag_hits), entry))
            continue
        node = _load_node(entry["id"])
        if node and any(t in node["text"].lower() for t in terms):
            scored.append((0.5, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for _, entry in scored[:limit]:
        node = _load_node(entry["id"])
        if node:
            results.append(node)
    return results
