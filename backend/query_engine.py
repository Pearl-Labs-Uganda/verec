"""VEREC natural-language query interface over the memory store (VR-4).

Runs at query time: given a user's question, retrieves the memory node(s)
relevant to its time window, tags, or entities, then asks an LLM (via Ollama)
to answer strictly from that stored context using the Query Assistant prompt.
No raw frames are touched — answers are grounded only in memory node text
produced by the Data Indexer (see memory_indexer.py, VR-3).
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone

from backend.memory_indexer import DEFAULT_MODEL, _extract_section, _has_headers, _load_index, _load_node

_ANSWER_REQUIRED_HEADERS = ("[ANSWER]", "[SUPPORTING EVIDENCE]", "[CONFIDENCE]")

_QUERY_SYSTEM_PROMPT = """You are a Query Assistant for a Hierarchical Security Memory System. You answer natural language questions using ONLY the memory nodes provided as context - you never have access to raw video and must not assume facts beyond what's given.

INPUT:
1. One or more [MEMORY NODE] blocks (produced by the Data Indexer).
2. A user question in natural language.

TASK:
1. Identify which memory node(s) are relevant to the time range, location, or entities mentioned in the question.
2. Answer the question directly and concisely first.
3. Follow with a brief justification citing which memory node(s) and which fields (Entity Registry, Semantic Tags, etc.) support the answer.
4. If the memory nodes don't contain enough information to answer confidently, say so explicitly rather than guessing - do not fabricate counts, times, or details.

OUTPUT FORMAT:
[ANSWER]
[one direct sentence]

[SUPPORTING EVIDENCE]
- Memory Node(s) referenced: [Time Range / Camera]
- Relevant fields: [what was used]

[CONFIDENCE]
High / Medium / Low - Low if the question requires details not present in memory.

RULES:
- Never re-derive answers from imagined frame data; use only what's in the memory nodes.
- For counting questions (e.g. "how many vehicles"), count Entity Registry entries that match the query type - do not double-count re-entries flagged as ambiguous.
- For time-based questions, cross-reference the Time Range field before answering."""

_TIME_WINDOW_RE = re.compile(r"last\s+(?:(\d+)\s*)?(hour|hours|minute|minutes|day|days)", re.IGNORECASE)
_UNIT_TO_KWARG = {
    "hour": "hours", "hours": "hours",
    "minute": "minutes", "minutes": "minutes",
    "day": "days", "days": "days",
}

_VEHICLE_TERMS = {
    "vehicle", "vehicles", "car", "cars", "bus", "buses", "truck", "trucks",
    "van", "vans", "taxi", "taxis", "motorcycle", "motorcycles",
    "bicycle", "bicycles", "sedan",
}
_PERSON_TERMS = {"person", "people", "pedestrian", "pedestrians"}


def _parse_time_window(question: str) -> tuple[datetime, datetime] | None:
    m = _TIME_WINDOW_RE.search(question)
    if not m:
        return None
    n = int(m.group(1)) if m.group(1) else 1
    kwarg = _UNIT_TO_KWARG[m.group(2).lower()]
    now = datetime.now(timezone.utc)
    return now - timedelta(**{kwarg: n}), now


def _node_end_dt(entry: dict) -> datetime | None:
    try:
        return datetime.fromisoformat(entry["end"])
    except (ValueError, KeyError):
        return None


def _retrieve_nodes(question: str, camera_id: str | None, limit: int) -> list[dict]:
    """Rank stored memory-node index entries by relevance to the question.

    Generic vehicle/person words are expanded to the concrete classes the
    Data Indexer tags with, so "vehicles" also matches "car"/"bus"/etc.

    For "last N hours"-style questions, retrieval is restricted to nodes
    whose Time Range actually falls in that window — mixing in older nodes
    would let the LLM answer a time-scoped question from stale context, and
    the RULES require the Time Range field to be cross-referenced first. If
    the window matches no nodes, an empty list is returned so the caller
    reports insufficient information rather than guessing from old data.
    Otherwise (no time window in the question), falls back to the most
    recent nodes when no tags match, so there's still grounded context.
    """
    terms = {t for t in re.split(r"[^\w]+", question.lower()) if t}
    if terms & _VEHICLE_TERMS:
        terms |= _VEHICLE_TERMS
    if terms & _PERSON_TERMS:
        terms |= _PERSON_TERMS

    pool = [e for e in _load_index() if not camera_id or e["camera_id"] == camera_id]

    window = _parse_time_window(question)
    if window is not None:
        pool = [e for e in pool if (dt := _node_end_dt(e)) is not None and window[0] <= dt <= window[1]]
        if not pool:
            return []

    def _tag_score(entry: dict) -> int:
        tag_text = " ".join(entry.get("tags", []))
        return sum(1 for t in terms if t in tag_text)

    scored = [(_tag_score(e), e) for e in pool]
    if window is None and not any(s > 0 for s, _ in scored):
        pool.sort(key=lambda e: e["end"], reverse=True)
        return pool[:limit]

    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored[:limit]]


_NO_CONTEXT_RESULT_TEXT = (
    "[ANSWER]\nNo memory nodes are available to answer this question.\n\n"
    "[SUPPORTING EVIDENCE]\n- Memory Node(s) referenced: None\n- Relevant fields: None\n\n"
    "[CONFIDENCE]\nLow"
)


def answer_query(
    question: str,
    camera_id: str | None = None,
    model: str = DEFAULT_MODEL,
    limit: int = 5,
    ollama_url: str | None = None,
) -> dict:
    """Answer a natural-language question grounded strictly in stored memory nodes."""
    entries = _retrieve_nodes(question, camera_id, limit)
    nodes = [n for n in (_load_node(e["id"]) for e in entries) if n is not None]

    if not nodes:
        return {
            "question": question,
            "answer": "No memory nodes are available to answer this question.",
            "evidence": "",
            "confidence": "Low",
            "text": _NO_CONTEXT_RESULT_TEXT,
            "nodes_used": [],
        }

    context = "\n\n".join(f"[MEMORY NODE]\n{n['text']}" for n in nodes)
    user_prompt = f"{context}\n\n[QUESTION]\n{question}"

    from openai import OpenAI
    ollama_url = ollama_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
    answer_text = ""
    try:
        client = OpenAI(base_url=f"{ollama_url.rstrip('/')}/v1", api_key="ollama")
        messages = [
            {"role": "system", "content": _QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        resp = client.chat.completions.create(model=model, messages=messages, max_tokens=400)
        answer_text = resp.choices[0].message.content or ""

        # Small/CPU-bound models sometimes skip the [ANSWER]/[SUPPORTING
        # EVIDENCE]/[CONFIDENCE] headers entirely. Give it one corrective
        # retry before falling back, rather than silently returning blanks.
        if not _has_headers(answer_text, _ANSWER_REQUIRED_HEADERS):
            messages += [
                {"role": "assistant", "content": answer_text},
                {"role": "user", "content": (
                    "Your previous reply did not follow the required format. "
                    "Reply again using EXACTLY these section headers, in order: "
                    "[ANSWER], [SUPPORTING EVIDENCE], [CONFIDENCE]. No other text."
                )},
            ]
            retry_resp = client.chat.completions.create(model=model, messages=messages, max_tokens=400)
            retry_text = retry_resp.choices[0].message.content or ""
            if _has_headers(retry_text, _ANSWER_REQUIRED_HEADERS):
                answer_text = retry_text
    except Exception as e:
        answer_text = (
            f"[ANSWER]\nQuery failed: {e}\n\n"
            "[SUPPORTING EVIDENCE]\n- Memory Node(s) referenced: None\n- Relevant fields: None\n\n"
            "[CONFIDENCE]\nLow"
        )

    answer = _extract_section(answer_text, "ANSWER")
    evidence = _extract_section(answer_text, "SUPPORTING EVIDENCE")
    confidence = _extract_section(answer_text, "CONFIDENCE")
    if not answer:
        # Still no [ANSWER] header after the retry — show the model's raw
        # reply rather than an empty answer bubble, since it usually does
        # contain a real (if unlabeled) answer.
        answer = answer_text.strip() or "The model did not return a usable answer."
        confidence = confidence or "Low"

    return {
        "question": question,
        "answer": answer,
        "evidence": evidence,
        "confidence": confidence,
        "text": answer_text,
        "nodes_used": [n["id"] for n in nodes],
    }
