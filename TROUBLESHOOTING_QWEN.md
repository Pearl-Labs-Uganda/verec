# Qwen Integration Troubleshooting Guide

## Quick Diagnostic Commands

Run these commands to check your setup:

```powershell
# 1. Is Ollama installed?
ollama --version

# 2. Is Ollama running?
curl http://localhost:11434/api/tags

# 3. What models are available?
ollama list

# 4. Is the backend configured correctly?
Get-Content .env | Select-String "USE_QWEN_VL"

# 5. Test Qwen directly
python test_qwen.py

# 6. Check backend status (if running)
curl http://localhost:8000/api/system
```

---

## Common Issues & Solutions

### 1. "Model 'qwen2.5-vl:3b' not found"

**Symptoms:**
- Error when starting backend or calling VLM
- Backend shows "VLM not configured"

**Diagnosis:**
```powershell
ollama list
```

**Solution A: You have `wen2.5v1:3b`**
```powershell
# Rename it
ollama cp wen2.5v1:3b qwen2.5-vl:3b

# Verify
ollama list
```

**Solution B: Model is missing**
```powershell
# Pull the model (~6GB download)
ollama pull qwen2.5-vl:3b
```

**Solution C: Use the rename script**
```powershell
.\rename_model.ps1
```

---

### 2. Backend still uses FastVLM instead of Qwen

**Symptoms:**
- `/api/system` shows "vlm": "FastVLM 0.5B"
- No "Qwen Vision Analyzer initialized" message in logs

**Diagnosis:**
```powershell
# Check environment variable
$env:USE_QWEN_VL

# Check .env file
Get-Content .env | Select-String "USE_QWEN_VL"
```

**Solution A: Set in PowerShell session**
```powershell
$env:USE_QWEN_VL="true"
python -m backend.server --model-path checkpoints/llava-fastvithd_0.5b_stage3
```

**Solution B: Update .env file**
```powershell
# Add or update in .env
"USE_QWEN_VL=true" | Out-File -FilePath .env -Append

# Then restart backend
```

**Solution C: Verify the setting**
```python
# Quick Python check
import os
print(os.getenv("USE_QWEN_VL"))  # Should print: true
```

---

### 3. "Connection refused to localhost:11434"

**Symptoms:**
- Backend starts but VLM calls fail
- Error: "Cannot connect to Ollama"

**Diagnosis:**
```powershell
# Test Ollama connectivity
curl http://localhost:11434/api/tags
```

**Solution A: Start Ollama (if not running)**
```powershell
# Check if running
Get-Process ollama

# If not, Ollama should auto-start on Windows
# Try restarting your computer or reinstalling Ollama
```

**Solution B: Check firewall**
```powershell
# Make sure localhost:11434 is not blocked
# Add firewall rule if needed:
New-NetFirewallRule -DisplayName "Ollama" -Direction Inbound -LocalPort 11434 -Protocol TCP -Action Allow
```

**Solution C: Check Ollama service status**
```powershell
# Windows Service
Get-Service ollama

# If service doesn't exist, Ollama runs differently
# Just ensure the process is running
Start-Process "ollama" -ArgumentList "serve"
```

---

### 4. Very slow inference (>5 seconds per image)

**Symptoms:**
- VLM calls take 5-10+ seconds
- `ollama ps` shows high CPU usage but low/no GPU usage

**Diagnosis:**
```powershell
# Check if Ollama is using GPU
ollama ps
```

**Expected output (GPU):**
```
NAME                  SIZE    GPU     CPU
qwen2.5-vl:3b        6.1 GB  100%    5%
```

**Bad output (CPU only):**
```
NAME                  SIZE    GPU     CPU
qwen2.5-vl:3b        6.1 GB  0%      100%
```

**Solution A: Ensure GPU drivers are installed**
- NVIDIA: Install latest drivers from nvidia.com
- AMD: Install ROCm (Linux only)
- Check: `nvidia-smi` (NVIDIA) or `rocm-smi` (AMD)

**Solution B: Reinstall Ollama with GPU support**
- Download latest version from ollama.com
- Reinstall (will detect GPU automatically)

**Solution C: Reduce model size**
```powershell
# Use smaller model if GPU memory is limited
ollama pull qwen2.5-vl:3b  # Instead of 7b or 72b
```

**Solution D: Optimize parameters**
```python
# In your code, reduce max_tokens
result = run_vlm(frame, prompt, max_tokens=50)  # Instead of 100+
```

---

### 5. Out of memory errors

**Symptoms:**
- Ollama crashes or becomes unresponsive
- Error: "CUDA out of memory" or similar

**Diagnosis:**
```powershell
# Check GPU memory usage
nvidia-smi  # Windows/Linux NVIDIA
```

**Solution A: Free up GPU memory**
```powershell
# Close other GPU-intensive applications
# Restart Ollama
Get-Process ollama | Stop-Process
Start-Process "ollama" -ArgumentList "serve"
```

**Solution B: Use smaller model**
```powershell
# qwen2.5-vl:3b uses ~6GB
# If you have qwen2.5-vl:7b (~14GB), switch to 3b
ollama pull qwen2.5-vl:3b

# Update .env or code to use 3b model
```

**Solution C: Run Ollama on CPU (slower but works)**
```powershell
# Set environment variable before starting Ollama
$env:OLLAMA_NUM_GPU=0
ollama serve
```

**Solution D: Move Ollama to separate machine**
```env
# In .env
USE_QWEN_VL=true
OLLAMA_URL=http://other-machine:11434
```

---

### 6. Test script fails

**Symptoms:**
- `python test_qwen.py` errors
- ImportError or ModuleNotFoundError

**Diagnosis:**
```powershell
# Check Python dependencies
python -c "import openai; print(openai.__version__)"
python -c "import numpy; print(numpy.__version__)"
python -c "import PIL; print(PIL.__version__)"
```

**Solution A: Install missing dependencies**
```powershell
pip install openai numpy pillow opencv-python
```

**Solution B: Install full requirements**
```powershell
pip install -r requirements.txt
```

**Solution C: Check Python version**
```powershell
python --version  # Should be 3.10+
```

---

### 7. WebSocket drops frames when VLM is running

**Symptoms:**
- Video feed stutters every 5 seconds (when VLM runs)
- Frame rate drops significantly

**Diagnosis:**
Check if async support is working:
```python
# In backend/server.py, VLM call should look like:
vlm_result = await asyncio.to_thread(run_vlm, ...)
# NOT:
vlm_result = run_vlm(...)  # This blocks!
```

**Solution:**
✅ Already fixed in the integration! The WebSocket handler uses `asyncio.to_thread()`.

If you still see issues:
- Reduce `vlm_interval` to less frequent (e.g., every 10s instead of 5s)
- Lower `max_tokens` in VLM calls
- Check Ollama GPU usage with `ollama ps`

---

### 8. Backend won't start - ImportError

**Symptoms:**
```
ModuleNotFoundError: No module named 'backend.qwen_vision'
```

**Solution:**
Ensure you're running from the project root:
```powershell
cd d:\wasswa\verec
python -m backend.server --model-path checkpoints/llava-fastvithd_0.5b_stage3
```

Not from inside the `backend` folder!

---

### 9. VLM returns empty responses

**Symptoms:**
- API calls succeed but `"text": ""`
- No error message

**Diagnosis:**
```powershell
# Test Ollama directly
curl http://localhost:11434/api/generate `
  -d '{"model": "qwen2.5-vl:3b", "prompt": "Hello"}'
```

**Solution A: Model might be corrupted**
```powershell
# Re-pull the model
ollama rm qwen2.5-vl:3b
ollama pull qwen2.5-vl:3b
```

**Solution B: Increase timeout**
Qwen might be taking longer than expected. Check `qwen_vision.py` for timeout settings.

---

### 10. CORS errors in browser console

**Symptoms:**
- Frontend can't connect to backend
- Browser console shows CORS errors

**Solution:**
Update CORS_ORIGINS in `.env`:
```env
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://frontend:3000
```

Or set in environment:
```powershell
$env:CORS_ORIGINS="http://localhost:3000"
```

---

## Verification Checklist

Run through this checklist to verify everything is working:

### ☑ Installation Check
```powershell
# 1. Ollama installed
ollama --version  # Should show version

# 2. Model available
ollama list | Select-String "qwen2.5-vl:3b"  # Should find it

# 3. Ollama running
curl http://localhost:11434/api/tags  # Should return JSON

# 4. Python dependencies
pip list | Select-String "openai"  # Should show openai package
```

### ☑ Configuration Check
```powershell
# 5. Environment variable
$env:USE_QWEN_VL  # Should be "true"

# 6. .env file (optional but recommended)
Get-Content .env | Select-String "USE_QWEN_VL=true"
```

### ☑ Integration Test
```powershell
# 7. Direct Qwen test
python test_qwen.py  # Should complete successfully

# 8. Backend health
curl http://localhost:8000/api/health  # Should return {"status":"ok"}

# 9. System info
curl http://localhost:8000/api/system  # Should show "Qwen2.5-VL-3B (Ollama)"

# 10. VLM endpoint
curl -X POST "http://localhost:8000/api/vlm" `
  -H "Content-Type: application/json" `
  -d '{"prompt":"Test","source":"local"}'
```

If all checks pass: ✅ **Your integration is working correctly!**

---

## Getting Help

### 1. Collect Diagnostic Info
```powershell
# Save diagnostic info to file
"=== Ollama Status ===" | Out-File diagnostics.txt
ollama list | Out-File -Append diagnostics.txt
""  | Out-File -Append diagnostics.txt

"=== Environment ===" | Out-File -Append diagnostics.txt
$env:USE_QWEN_VL | Out-File -Append diagnostics.txt
Get-Content .env | Out-File -Append diagnostics.txt
""  | Out-File -Append diagnostics.txt

"=== Connectivity ===" | Out-File -Append diagnostics.txt
curl http://localhost:11434/api/tags | Out-File -Append diagnostics.txt
```

### 2. Enable Debug Logging
```python
# In backend/server.py or models.py, add:
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 3. Test in Isolation
```python
# Create test_isolated.py
from backend.qwen_vision import QwenVisionAnalyzer
import numpy as np

analyzer = QwenVisionAnalyzer()
img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
result = analyzer.analyze_image(img, "What do you see?")
print(f"Result: {result}")
```

### 4. Check Documentation
- **Setup**: [QWEN_SETUP.md](QWEN_SETUP.md)
- **Quick Start**: [QUICK_START_QWEN.md](QUICK_START_QWEN.md)
- **Architecture**: [QWEN_ARCHITECTURE.md](QWEN_ARCHITECTURE.md)
- **Code Reference**: [backend/QWEN_QUICK_REFERENCE.md](backend/QWEN_QUICK_REFERENCE.md)

---

## Still Having Issues?

If you've tried all the above and still having problems:

1. **Fallback to local FastVLM** (temporarily):
   ```powershell
   $env:USE_QWEN_VL="false"
   ```

2. **Check Ollama logs** (if available):
   ```powershell
   # Ollama logs location varies by OS
   # Usually in: %LOCALAPPDATA%\Ollama\logs (Windows)
   ```

3. **Verify file integrity**:
   ```powershell
   # Make sure all files are in place
   Test-Path backend\qwen_vision.py
   Test-Path backend\models.py
   Test-Path test_qwen.py
   ```

4. **Try the automated setup script**:
   ```powershell
   .\setup_qwen.ps1
   ```

The integration has been thoroughly tested and should work out of the box once Ollama is properly set up with the correct model name. Most issues are related to model naming or environment configuration.
