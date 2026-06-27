# Qwen2.5-VL Integration Guide

This guide shows how to integrate **Qwen2.5-VL** via **Ollama** as a drop-in upgrade for the local LLaVA vision-language model in VEREC.

## Why Qwen2.5-VL via Ollama?

### Benefits
- **Memory Efficiency** – Runs in a separate process, freeing GPU memory for YOLO detectors
- **Better Vision Understanding** – State-of-the-art vision-language capabilities
- **Easy Deployment** – Standard OpenAI-compatible API
- **Scalability** – Can run on a different machine or cloud endpoint
- **Backward Compatible** – Keep LLaVA as fallback; switch via environment variable

## Prerequisites

### 1. Install Ollama

**Windows:**
```powershell
# Download from https://ollama.com/download/windows
# Or use winget:
winget install Ollama.Ollama
```

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull Qwen2.5-VL Model

```bash
# 3B model (recommended for most use cases)
ollama pull qwen2.5-vl:3b

# Or 7B model for better quality (requires more VRAM)
ollama pull qwen2.5-vl:7b

# Or 72B model for production (requires significant resources)
ollama pull qwen2.5-vl:72b
```

### 3. Start Ollama Service

**Windows/macOS:**
Ollama runs as a service automatically after installation.

**Linux:**
```bash
ollama serve
```

Verify it's running:
```bash
curl http://localhost:11434/api/tags
```

## Configuration

### Option 1: Environment Variable (Recommended)

Set the `USE_QWEN_VL` environment variable to enable Qwen:

**Windows (PowerShell):**
```powershell
$env:USE_QWEN_VL="true"
python -m backend.server --model-path checkpoints/llava-fastvithd_0.5b_stage3
```

**Windows (CMD):**
```cmd
set USE_QWEN_VL=true
python -m backend.server --model-path checkpoints/llava-fastvithd_0.5b_stage3
```

**Linux/macOS:**
```bash
export USE_QWEN_VL=true
python -m backend.server --model-path checkpoints/llava-fastvithd_0.5b_stage3
```

### Option 2: .env File

Create or update `.env` in the project root:

```env
USE_QWEN_VL=true
OLLAMA_URL=http://localhost:11434
VLM_MODEL_PATH=checkpoints/llava-fastvithd_0.5b_stage3
```

### Option 3: Docker Compose

Update `docker-compose.yml`:

```yaml
services:
  backend:
    environment:
      - USE_QWEN_VL=true
      - OLLAMA_URL=http://ollama:11434
    depends_on:
      - ollama

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  ollama_data:
```

## Usage

### Automatic Switching

The system automatically detects `USE_QWEN_VL` and switches backends:

- `USE_QWEN_VL=true` → Uses Qwen via Ollama
- `USE_QWEN_VL=false` or not set → Uses local LLaVA

### Manual Override

In code, you can force a specific backend:

```python
from backend.models import run_vlm

# Force Qwen
result = run_vlm(frame, prompt="Describe this scene", use_qwen=True)

# Force local LLaVA
result = run_vlm(frame, prompt="Describe this scene", use_qwen=False)
```

### WebSocket API

The WebSocket feed (`/ws/feed`) automatically uses the configured backend:

```javascript
const ws = new WebSocket(
  "ws://localhost:8000/ws/feed?" +
  "source=local&enable_vlm=true&vlm_interval=5"
);
```

The VLM captions will use Qwen if `USE_QWEN_VL=true`.

### REST API

The `/api/vlm` endpoint also respects the environment setting:

```bash
curl -X POST "http://localhost:8000/api/vlm" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What objects are visible?",
    "temperature": 0.0,
    "max_tokens": 100,
    "source": "local"
  }'
```

## Advanced: Multi-Frame Analysis

Qwen2.5-VL can process multiple frames for temporal understanding:

```python
from backend.models import init_qwen

qwen = init_qwen()

# Analyze 4 consecutive frames
frames = [frame1, frame2, frame3, frame4]
description = qwen.analyze_multiple_frames(
    frames,
    prompt="Describe the motion and activity across these frames",
    max_tokens=150
)
```

## Performance Tuning

### Model Selection

| Model | VRAM | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| qwen2.5-vl:3b | ~6GB | Fast | Good | Real-time feeds, development |
| qwen2.5-vl:7b | ~14GB | Medium | Better | Production, high accuracy |
| qwen2.5-vl:72b | ~80GB | Slow | Best | Critical applications, cloud |

### Ollama Configuration

Optimize Ollama settings in `~/.ollama/config.json`:

```json
{
  "num_gpu": 1,
  "num_thread": 8,
  "num_ctx": 2048
}
```

### Concurrent Requests

Qwen via Ollama can handle concurrent requests:

```python
import asyncio
from backend.models import init_qwen

async def analyze_batch(frames, prompts):
    qwen = init_qwen()
    tasks = [
        asyncio.to_thread(qwen.analyze_image, frame, prompt)
        for frame, prompt in zip(frames, prompts)
    ]
    return await asyncio.gather(*tasks)
```

## Troubleshooting

### Ollama Not Running

**Error:** `Connection refused to localhost:11434`

**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama (Linux)
ollama serve

# Check service status (Windows)
Get-Service ollama
```

### Model Not Found

**Error:** `model 'qwen2.5-vl:3b' not found`

**Solution:**
```bash
ollama pull qwen2.5-vl:3b
ollama list  # Verify it's installed
```

### Out of Memory

**Error:** GPU out of memory when running Qwen

**Solutions:**
1. Use smaller model: `qwen2.5-vl:3b` instead of `7b`
2. Run Ollama on CPU: `OLLAMA_NUM_GPU=0 ollama serve`
3. Move Ollama to separate machine and set `OLLAMA_URL`

### Slow Inference

**Optimization tips:**
- Ensure Ollama uses GPU: Check with `ollama ps`
- Reduce `max_tokens` in prompts
- Use smaller image sizes (resize before sending)
- Enable Ollama's flash attention (automatic in newer versions)

## Fallback Strategy

The system automatically falls back to local LLaVA if Qwen fails:

```python
# In models.py, run_vlm() handles exceptions gracefully
try:
    text = qwen.analyze_image(image, prompt)
except Exception as e:
    return {"error": f"Qwen failed: {e}", "text": ""}
```

You can implement automatic fallback:

```python
result = run_vlm(frame, prompt, use_qwen=True)
if result.get("error"):
    print("Qwen failed, falling back to local LLaVA")
    result = run_vlm(frame, prompt, use_qwen=False)
```

## Monitoring

Check which backend is active via the system info endpoint:

```bash
curl http://localhost:8000/api/system | jq '.models.vlm'
# Returns: "Qwen2.5-VL-3B (Ollama)" or "FastVLM 0.5B"
```

## Migration Checklist

- [ ] Install Ollama
- [ ] Pull Qwen model (`ollama pull qwen2.5-vl:3b`)
- [ ] Verify Ollama is running (`curl localhost:11434/api/tags`)
- [ ] Set `USE_QWEN_VL=true` in environment or `.env`
- [ ] Restart backend server
- [ ] Test via `/api/vlm` endpoint
- [ ] Verify WebSocket feed shows Qwen captions
- [ ] Check `/api/system` confirms Qwen is active
- [ ] Monitor performance and adjust model size if needed

## Next Steps

- **Production**: Move Ollama to dedicated GPU server
- **Scale**: Use load balancer for multiple Ollama instances
- **Optimize**: Fine-tune Qwen on your specific use case
- **Extend**: Implement multi-frame temporal analysis for action recognition

## References

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Qwen2.5-VL Model Card](https://huggingface.co/Qwen/Qwen2.5-VL)
- [OpenAI Vision API Spec](https://platform.openai.com/docs/guides/vision)
