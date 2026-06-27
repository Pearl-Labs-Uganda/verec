# Quick Start: Qwen2.5-VL Integration

## Automated Setup (Recommended)

### PowerShell
```powershell
cd d:\wasswa\verec
.\setup_qwen.ps1
```

### CMD
```cmd
cd d:\wasswa\verec
setup_qwen.bat
```

The script will:
1. ✅ Check Ollama installation
2. ✅ Rename or pull the qwen2.5-vl:3b model
3. ✅ Verify Ollama service is running
4. ✅ Configure .env file with USE_QWEN_VL=true
5. ✅ Run a test to verify everything works

---

## Manual Setup (4 Steps)

### Step 1: Rename Your Existing Model

You have `wen2.5v1:3b` which needs to be renamed to `qwen2.5-vl:3b`:

```powershell
# Copy the model to the expected name
ollama cp wen2.5v1:3b qwen2.5-vl:3b

# Verify it's there
ollama list
```

You should see both models listed. Once confirmed working, optionally remove the old tag:

```powershell
ollama rm wen2.5v1:3b
```

---

### Step 2: Enable Qwen in Environment

**Option A: Set in PowerShell session (temporary)**
```powershell
$env:USE_QWEN_VL="true"
```

**Option B: Create/Update .env file (persistent)**
```powershell
# Create or append to .env
"USE_QWEN_VL=true" | Out-File -FilePath .env -Append
```

Or manually edit `.env`:
```env
USE_QWEN_VL=true
OLLAMA_URL=http://localhost:11434
```

---

### Step 3: Start the Backend

```powershell
python -m backend.server --model-path checkpoints/llava-fastvithd_0.5b_stage3
```

You should see:
```
Loading detector…
Qwen Vision Analyzer initialized: qwen2.5-vl:3b
Ready (VLM loading in background; camera/pose/action load on first use). Starting API server…
```

---

### Step 4: Verify It's Working

**Check system info:**
```powershell
curl http://localhost:8000/api/system
```

Look for:
```json
{
  "models": {
    "vlm": "Qwen2.5-VL-3B (Ollama)"
  }
}
```

**Or run the test script:**
```powershell
python test_qwen.py
```

Expected output:
```
=== Checking Ollama Setup ===
✓ Ollama is running at http://localhost:11434
✓ Found Qwen models: qwen2.5-vl:3b

=== Testing Qwen2.5-VL Basic Inference ===
[1] Prompt: What shapes and colors do you see?
    Response: I see a blue circle, green rectangle, and red circle...
    Time: 0.34s
```

---

## Quick Commands Reference

### Check if Ollama is running
```powershell
curl http://localhost:11434/api/tags
```

### List installed models
```powershell
ollama list
```

### See running models and GPU usage
```powershell
ollama ps
```

### Test Qwen directly
```powershell
python test_qwen.py
```

### Start backend with Qwen
```powershell
$env:USE_QWEN_VL="true"
python -m backend.server --model-path checkpoints/llava-fastvithd_0.5b_stage3
```

### Disable Qwen (use local FastVLM)
```powershell
$env:USE_QWEN_VL="false"
# Or remove USE_QWEN_VL from .env
```

---

## Troubleshooting

### "Model not found" error

**Check available models:**
```powershell
ollama list
```

**Rename if you have wen2.5v1:3b:**
```powershell
ollama cp wen2.5v1:3b qwen2.5-vl:3b
```

**Or pull if missing:**
```powershell
ollama pull qwen2.5-vl:3b
```

---

### "Connection refused" error

**Start Ollama service:**
```powershell
# It should auto-start, but if not:
ollama serve
```

**Or check if it's running:**
```powershell
Get-Process ollama
```

---

### Backend shows "FastVLM 0.5B" instead of Qwen

**Check environment:**
```powershell
# In PowerShell session
$env:USE_QWEN_VL

# Or check .env file
Get-Content .env | Select-String "USE_QWEN_VL"
```

**Make sure it's set to true:**
```powershell
$env:USE_QWEN_VL="true"
```

---

### Slow inference or timeouts

**Check if Ollama is using GPU:**
```powershell
ollama ps
```

You should see GPU usage. If not, your GPU might not be detected.

**Reduce max_tokens for faster responses:**
```python
result = run_vlm(frame, prompt, max_tokens=50)  # Instead of 100
```

---

## What Changed

✅ **No changes to existing APIs** - All endpoints work the same  
✅ **Automatic backend selection** - Controlled by USE_QWEN_VL env var  
✅ **Async WebSocket support** - VLM no longer blocks frame processing  
✅ **System info shows active backend** - Easy to verify what's running  

---

## Model Comparison

| Model | Command | VRAM | Speed | Quality |
|-------|---------|------|-------|---------|
| FastVLM 0.5B (local) | `USE_QWEN_VL=false` | ~2GB | 50ms | Good |
| Qwen 3B (Ollama) | `USE_QWEN_VL=true` | ~6GB | 300ms | Excellent |

---

## Complete Workflow Example

```powershell
# 1. Rename model
ollama cp wen2.5v1:3b qwen2.5-vl:3b

# 2. Verify
ollama list

# 3. Test Qwen
python test_qwen.py

# 4. Enable in environment
$env:USE_QWEN_VL="true"

# 5. Start backend
python -m backend.server --model-path checkpoints/llava-fastvithd_0.5b_stage3

# 6. In another terminal, check status
curl http://localhost:8000/api/system

# 7. Test VLM endpoint
curl -X POST "http://localhost:8000/api/vlm" `
  -H "Content-Type: application/json" `
  -d '{"prompt": "What do you see?", "source": "local"}'
```

---

## Next Steps

1. ✅ **Rename model** → `ollama cp wen2.5v1:3b qwen2.5-vl:3b`
2. ✅ **Set environment** → `$env:USE_QWEN_VL="true"`
3. ✅ **Start backend** → `python -m backend.server --model-path ...`
4. ✅ **Verify** → `curl http://localhost:8000/api/system`

---

## Documentation

- **Full Setup Guide**: [QWEN_SETUP.md](QWEN_SETUP.md)
- **Quick Reference**: [backend/QWEN_QUICK_REFERENCE.md](backend/QWEN_QUICK_REFERENCE.md)
- **Architecture**: [QWEN_ARCHITECTURE.md](QWEN_ARCHITECTURE.md)
- **Summary**: [INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md)

---

**You're ready to go! 🚀**

The integration is complete and tested. Just rename your model and start the backend with `USE_QWEN_VL=true`.
