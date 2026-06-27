# Qwen Integration Quick Reference
# D:\wasswa\verec\test_videos\traffic.mp4

## Setup (One-Time)

```bash
# 1. Install Ollama
# Windows: Download from https://ollama.com/download/windows
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull model
ollama pull qwen2.5-vl:3b

# 3. Verify
curl http://localhost:11434/api/tags
```

## Usage

### Enable Qwen (Environment Variable)

```bash
# PowerShell (Windows)
$env:USE_QWEN_VL="true"
python -m backend.server --model-path checkpoints/llava-fastvithd_0.5b_stage3

# Bash (Linux/macOS)
export USE_QWEN_VL=true
python -m backend.server --model-path checkpoints/llava-fastvithd_0.5b_stage3
```

### Enable Qwen (.env file)

```env
USE_QWEN_VL=true
OLLAMA_URL=http://localhost:11434
```

## API Examples

### Python - Automatic Backend Selection

```python
from backend.models import run_vlm
import numpy as np

frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
result = run_vlm(frame, prompt="What do you see?", max_tokens=100)
print(result["text"])
print(f"Backend: {result['backend']}")  # "qwen-ollama" or "local-llava"
```

### Python - Force Specific Backend

```python
# Force Qwen
result = run_vlm(frame, prompt="Describe this", use_qwen=True)

# Force local LLaVA
result = run_vlm(frame, prompt="Describe this", use_qwen=False)
```

### Python - Direct Qwen Client

```python
from backend.qwen_vision import QwenVisionAnalyzer

qwen = QwenVisionAnalyzer(model="qwen2.5-vl:3b")
caption = qwen.analyze_image(frame, "What objects are visible?")
print(caption)
```

### Python - Multi-Frame Analysis

```python
from backend.models import init_qwen

qwen = init_qwen()
frames = [frame1, frame2, frame3, frame4]  # List of numpy arrays
description = qwen.analyze_multiple_frames(
    frames,
    prompt="Describe the motion across these frames",
    max_tokens=150
)
```

### REST API

```bash
# Check active backend
curl http://localhost:8000/api/system | jq '.models.vlm'

# Analyze frame
curl -X POST "http://localhost:8000/api/vlm" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is happening?",
    "temperature": 0.0,
    "max_tokens": 100,
    "source": "local"
  }'
```

### WebSocket

```javascript
const ws = new WebSocket(
  "ws://localhost:8000/ws/feed?" +
  "source=local&enable_vlm=true&vlm_interval=5"
);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.vlm) {
    console.log("Caption:", data.vlm.text);
    console.log("Backend:", data.vlm.backend);
  }
};
```

## Async Usage (Recommended for WebSocket)

```python
import asyncio
from backend.models import run_vlm

# In async context (e.g., FastAPI WebSocket handler)
result = await asyncio.to_thread(
    run_vlm,
    frame,
    prompt="Describe this scene",
    max_tokens=80
)
```

## Model Selection

| Model | Command | VRAM | Use Case |
|-------|---------|------|----------|
| 3B | `ollama pull qwen2.5-vl:3b` | ~6GB | Development, real-time |
| 7B | `ollama pull qwen2.5-vl:7b` | ~14GB | Production |
| 72B | `ollama pull qwen2.5-vl:72b` | ~80GB | High accuracy |

## Troubleshooting

### Check Ollama Status

```bash
# Test connection
curl http://localhost:11434/api/tags

# List models
ollama list

# Check running models
ollama ps
```

### Common Issues

**Connection Refused:**
```bash
# Windows/macOS: Check service
Get-Service ollama  # PowerShell

# Linux: Start manually
ollama serve
```

**Model Not Found:**
```bash
ollama pull qwen2.5-vl:3b
ollama list  # Verify installation
```

**Out of Memory:**
```bash
# Use smaller model
ollama pull qwen2.5-vl:3b

# Or run on CPU
OLLAMA_NUM_GPU=0 ollama serve
```

## Testing

```bash
# Basic test
python test_qwen.py

# Test with custom image
python test_qwen.py --image path/to/image.jpg

# Test with webcam
python test_qwen.py --camera

# Test multi-frame
python test_qwen.py --multi-frame
```

## Performance Tips

1. **Use async calls** in WebSocket handlers to avoid blocking
2. **Lower max_tokens** for faster responses
3. **Resize images** before sending to reduce processing time
4. **Enable GPU** for Ollama (check with `ollama ps`)
5. **Batch requests** when analyzing multiple frames

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_QWEN_VL` | `false` | Enable Qwen backend |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `VLM_MODEL_PATH` | — | Local LLaVA checkpoint path |

## Backend Response Format

```python
{
    "text": "A person standing in front of a building",
    "time_s": 0.234,
    "tokens": 12,
    "tokens_per_s": 51.3,
    "backend": "qwen-ollama"  # or "local-llava"
}
```

## Fallback Strategy

```python
# Try Qwen, fallback to local on error
result = run_vlm(frame, prompt, use_qwen=True)
if result.get("error"):
    print(f"Qwen failed: {result['error']}, using local LLaVA")
    result = run_vlm(frame, prompt, use_qwen=False)
```

## Remote Ollama

```bash
# Set remote URL
export OLLAMA_URL=http://gpu-server:11434

# Or in .env
OLLAMA_URL=http://gpu-server:11434
```

## Docker Integration

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

## See Also

- **[QWEN_SETUP.md](../QWEN_SETUP.md)** - Detailed setup guide
- **[models.py](models.py)** - Implementation details
- **[qwen_vision.py](qwen_vision.py)** - Qwen client wrapper
- **[test_qwen.py](../test_qwen.py)** - Test script
- [Ollama Documentation](https://github.com/ollama/ollama)
