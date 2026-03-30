#!/usr/bin/env bash
# ── VEREC native dev mode ─────────────────────────────────────────────────
# Runs backend (with MPS GPU + camera) and frontend (Next.js dev server)
# side by side. Use this instead of Docker for local Mac development.
#
# Usage:  ./dev.sh
# Stop:   Ctrl-C (kills both processes)
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── Backend ───────────────────────────────────────────────────────────────
echo "▶ Starting backend (MPS GPU + camera)…"
cd "$ROOT"
python -m backend.server \
  --model-path checkpoints/llava-fastvithd_0.5b_stage3 \
  --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# ── Frontend ──────────────────────────────────────────────────────────────
echo "▶ Starting frontend dev server…"
cd "$ROOT/frontend"
npm run dev &
FRONTEND_PID=$!

# ── Cleanup on exit ──────────────────────────────────────────────────────
cleanup() {
  echo ""
  echo "Shutting down…"
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
  wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo "  GPU:      MPS (Apple Silicon)"
echo "  Camera:   native access"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

wait
