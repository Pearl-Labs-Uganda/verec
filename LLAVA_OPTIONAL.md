# LLaVA is Optional When Using Qwen

## Summary

When `USE_QWEN_VL=true` in your `.env` file, the backend uses **Qwen2.5-VL via Ollama** for all vision tasks. LLaVA (FastVLM) is never loaded, so you don't need to install it.

## What Changed

The `backend/models.py` file now has **fully lazy imports** for LLaVA:

1. **No LLaVA imports at module load** — the server starts instantly
2. **LLaVA only loads when needed** — imports happen inside `init_vlm()` and `run_vlm()` 
3. **Graceful fallback** — if LLaVA isn't installed, it returns a clear error message

## Quick Start (Qwen-only Setup)

```bash
# 1. Ensure Qwen is running
ollama list  # Should show qwen2.5-vl:3b or similar

# 2. Check your .env
USE_QWEN_VL=true  # Must be set

# 3. Start the backend (no LLaVA needed!)
python -m uvicorn backend.server:app --host 0.0.0.0 --port 8000
```

The server will start **immediately** without loading any LLaVA dependencies.

## When to Install LLaVA

Only install LLaVA if you want to use the **local FastVLM fallback**:

```bash
# Optional: Install LLaVA for local inference
pip install -e .
python get_models.sh  # Download model weights
```

Then set `USE_QWEN_VL=false` in `.env` to use the local model.

## Architecture

```
User Request
    ↓
run_vlm(use_qwen=auto)
    ↓
    ├─ USE_QWEN_VL=true  → Qwen via Ollama (no LLaVA imports)
    └─ USE_QWEN_VL=false → Local LLaVA (lazy imports)
```

## Benefits

✅ **Fast startup** — no heavyweight model loading  
✅ **Clean environment** — no conflicting torch/CUDA dependencies  
✅ **Easy deployment** — just Python + Ollama  
✅ **Flexible** — can enable LLaVA later without code changes  

## Verification

Start the server and check for these signs:

```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
Qwen Vision Analyzer initialized: qwen2.5-vl:3b
INFO:     Application startup complete.
```

No LLaVA loading messages = ✅ working correctly!

## Troubleshooting

**If server hangs on startup:**
- Check that `USE_QWEN_VL=true` in `.env`
- Ensure no LLaVA imports are at the top of `backend/models.py`
- Verify Ollama is running: `ollama list`

**If you get "llava not installed" errors:**
- This is expected when `USE_QWEN_VL=true` — ignore it
- The backend will never call the local VLM path

**To switch back to LLaVA:**
1. Install LLaVA: `pip install -e .`
2. Set `USE_QWEN_VL=false` in `.env`
3. Restart the server
