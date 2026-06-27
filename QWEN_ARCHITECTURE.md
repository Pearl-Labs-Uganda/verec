# Qwen Integration Architecture

## System Overview

### Before: Monolithic Local VLM

```
┌────────────────────────────────────────────────────────────────┐
│                      VEREC Backend Process                      │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐      │
│  │   YOLO11n    │   │  FastVLM     │   │  ST-GCN      │      │
│  │  Detection   │   │   0.5B       │   │  Action      │      │
│  │   (ONNX)     │   │  (PyTorch)   │   │  (PyTorch)   │      │
│  │              │   │              │   │              │      │
│  │  ~200MB      │   │  ~2GB VRAM   │   │  ~500MB      │      │
│  └──────────────┘   └──────────────┘   └──────────────┘      │
│                                                                 │
│  Total GPU Memory: ~3GB                                        │
└────────────────────────────────────────────────────────────────┘
```

**Issues:**
- All models compete for GPU memory
- VLM memory not released when idle
- Upgrading VLM requires restarting entire service
- Cannot scale VLM independently

---

### After: Distributed Architecture with Ollama

```
┌──────────────────────────────┐      ┌──────────────────────────┐
│   VEREC Backend Process      │      │    Ollama Service        │
│                              │      │                          │
│  ┌────────────────────────┐  │      │  ┌──────────────────┐   │
│  │     YOLO11n            │  │      │  │  Qwen2.5-VL 3B   │   │
│  │     Detection          │  │      │  │                  │   │
│  │     (ONNX)             │  │      │  │  - Vision        │   │
│  │                        │  │      │  │    Encoder       │   │
│  │     ~200MB VRAM        │  │      │  │  - LLM           │   │
│  └────────────────────────┘  │      │  │                  │   │
│                              │      │  │  ~6GB VRAM       │   │
│  ┌────────────────────────┐  │      │  └──────────────────┘   │
│  │     ST-GCN             │  │      │                          │
│  │     Action             │  │ HTTP │  OpenAI-Compatible       │
│  │     Recognition        │  │◄────►│  REST API                │
│  │                        │  │      │                          │
│  │     ~500MB VRAM        │  │      │  /v1/chat/completions    │
│  └────────────────────────┘  │      │                          │
│                              │      └──────────────────────────┘
│  ┌────────────────────────┐  │
│  │  QwenVisionAnalyzer    │  │      - Can run on same machine
│  │  (HTTP Client)         │  │      - Or different GPU server
│  │  ~1MB                  │  │      - Or cloud endpoint
│  └────────────────────────┘  │
│                              │
│  Total GPU Memory: ~700MB    │
└──────────────────────────────┘
```

**Benefits:**
- ✅ Freed 2GB GPU memory in main process
- ✅ VLM runs independently, can be restarted without affecting detectors
- ✅ Can scale to multiple Ollama instances
- ✅ Ollama can run on dedicated GPU machine

---

## Request Flow Comparison

### Local FastVLM Flow

```
User Request
    │
    ↓
┌────────────────────────────┐
│  FastAPI Endpoint          │
│  /api/vlm                  │
└───────────┬────────────────┘
            │
            ↓
┌────────────────────────────┐
│  run_vlm()                 │
│  - Lock acquired           │  ← Blocks other requests
│  - Load image              │
│  - Tokenize                │
│  - Model inference         │  ← 50-100ms on GPU
│  - Decode output           │
└───────────┬────────────────┘
            │
            ↓
    JSON Response
```

**Characteristics:**
- Synchronous blocking
- Holds GPU lock during inference
- Fast but limited to local GPU
- No concurrency

---

### Qwen via Ollama Flow

```
User Request
    │
    ↓
┌────────────────────────────┐
│  FastAPI Endpoint          │
│  /api/vlm                  │
└───────────┬────────────────┘
            │
            ↓
┌────────────────────────────┐
│  asyncio.to_thread()       │  ← Non-blocking async wrapper
│  └─► run_vlm(use_qwen=True)│
└───────────┬────────────────┘
            │
            ↓
┌────────────────────────────┐
│  QwenVisionAnalyzer        │
│  - Convert image to JPEG   │
│  - Base64 encode           │
│  - HTTP POST to Ollama     │  ← Network call (async)
└───────────┬────────────────┘
            │
            ↓
┌────────────────────────────┐
│  Ollama Service            │
│  - Decode image            │
│  - Vision encoding         │  ← 100-200ms
│  - LLM generation          │  ← 100-300ms
│  - Return JSON             │
└───────────┬────────────────┘
            │
            ↓
    JSON Response
```

**Characteristics:**
- Asynchronous non-blocking
- Parallel requests possible
- Slightly higher latency (network + processing)
- Better quality output
- Scalable across machines

---

## WebSocket Feed Architecture

### With Local FastVLM

```
WebSocket Connection
        │
        ↓
    ┌───────────────────────┐
    │  Frame Loop (33ms)    │
    └───────┬───────────────┘
            │
            ├──► Object Detection (10ms) ────┐
            │                                 │
            ├──► Pose Estimation (15ms) ─────┤
            │                                 │
            └──► VLM Caption (every 5s)      │
                 └─► run_vlm() [BLOCKS!]     ├──► Combine Results
                     - 50-100ms              │
                     - Holds GPU lock        │
                     - Frame rate drops      │
                                             ↓
                                     Send JSON to Client
```

**Problem:** VLM blocks the entire loop, causing frame drops

---

### With Qwen via Ollama (Async)

```
WebSocket Connection
        │
        ↓
    ┌───────────────────────┐
    │  Frame Loop (33ms)    │
    └───────┬───────────────┘
            │
            ├──► Object Detection (10ms) ────────┐
            │                                     │
            ├──► Pose Estimation (15ms) ─────────┤
            │                                     │
            └──► VLM Caption (every 5s)          │
                 └─► await asyncio.to_thread()   ├──► Combine Results
                     - Runs in background        │
                     - Loop continues            │
                     - No frame drops!           │
                                                 ↓
                                         Send JSON to Client
                                                 │
    ┌────────────────────────────────────────────┘
    │  (Later, when Qwen responds)
    ↓
    Update caption in next frame
```

**Benefit:** VLM runs in background, main loop stays responsive

---

## Data Flow Diagram

```
                            ┌──────────────────────────┐
                            │   Frontend (Next.js)     │
                            │   WebSocket Client       │
                            └────────┬─────────────────┘
                                     │
                                     │ WebSocket
                                     │
                    ┌────────────────▼────────────────┐
                    │    FastAPI Server               │
                    │    /ws/feed                     │
                    └────────┬────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼─────┐  ┌────▼─────┐  ┌────▼─────────┐
    │ Detection     │  │  Pose    │  │  VLM         │
    │ run_detection │  │ run_pose │  │  run_vlm     │
    └────────┬──────┘  └────┬─────┘  └────┬─────────┘
             │               │             │
             │               │             │
    ┌────────▼─────┐  ┌──────▼──────┐   ┌─▼──────────┐
    │  YOLO11n     │  │  YOLO Pose  │   │  Backend   │
    │  (ONNX)      │  │  (ONNX)     │   │  Switch    │
    └──────────────┘  └─────────────┘   └─┬──────────┘
                                           │
                              ┌────────────┴────────────┐
                              │                         │
                    ┌─────────▼────────┐     ┌─────────▼────────┐
                    │ USE_QWEN_VL=false│     │ USE_QWEN_VL=true │
                    │                  │     │                  │
                    │  FastVLM 0.5B    │     │  Qwen via Ollama │
                    │  (Local GPU)     │     │  (HTTP API)      │
                    └──────────────────┘     └──────────────────┘
```

---

## Deployment Scenarios

### Scenario 1: Development (All Local)

```
┌─────────────────────────────────────────────┐
│          Your Development Machine           │
│                                             │
│  ┌─────────────┐    ┌─────────────────┐   │
│  │   Backend   │───►│  Ollama Service │   │
│  │   :8000     │    │  :11434         │   │
│  └─────────────┘    └─────────────────┘   │
│         ▲                                   │
│         │                                   │
│  ┌──────┴──────┐                           │
│  │   Frontend  │                           │
│  │   :3000     │                           │
│  └─────────────┘                           │
│                                             │
│  GPU: Shared between YOLO + Ollama         │
└─────────────────────────────────────────────┘
```

---

### Scenario 2: Production (Dedicated GPU for VLM)

```
┌──────────────────────┐        ┌──────────────────────┐
│   App Server         │        │   GPU Server         │
│   (CPU only)         │        │   (High-end GPU)     │
│                      │        │                      │
│  ┌────────────────┐  │        │  ┌────────────────┐ │
│  │    Backend     │  │  HTTP  │  │     Ollama     │ │
│  │    FastAPI     │◄─┼────────┼─►│  Qwen2.5-VL 7B │ │
│  │                │  │        │  │                │ │
│  │  YOLO (ONNX)   │  │        │  │  Multiple      │ │
│  │  Pose (ONNX)   │  │        │  │  Instances     │ │
│  └────────────────┘  │        │  └────────────────┘ │
│         ▲            │        │                      │
│         │            │        └──────────────────────┘
│  ┌──────┴──────┐     │
│  │   Frontend  │     │
│  │   Next.js   │     │
│  └─────────────┘     │
│                      │
└──────────────────────┘
```

**Configuration:**
```env
OLLAMA_URL=http://gpu-server:11434
USE_QWEN_VL=true
```

---

### Scenario 3: Cloud Scale (Load Balanced)

```
                    ┌───────────────┐
                    │ Load Balancer │
                    └───────┬───────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
    ┌─────▼─────┐     ┌─────▼─────┐    ┌─────▼─────┐
    │  Ollama 1 │     │  Ollama 2 │    │  Ollama 3 │
    │  Qwen 3B  │     │  Qwen 7B  │    │  Qwen 3B  │
    └───────────┘     └───────────┘    └───────────┘
          ▲                 ▲                ▲
          │                 │                │
          └─────────────────┴────────────────┘
                            │
                    ┌───────┴───────┐
                    │  Backend      │
                    │  Cluster      │
                    └───────────────┘
```

---

## Memory Layout Comparison

### Before (Single Process)

```
GPU Memory (8GB total)
├─ System Reserved         : 500MB
├─ YOLO Detection         : 200MB
├─ YOLO Pose              : 300MB
├─ FastVLM                : 2000MB  ← Large footprint!
├─ ST-GCN                 : 500MB
└─ Available              : 4500MB
```

---

### After (Distributed)

**Main Process GPU:**
```
GPU Memory (8GB total)
├─ System Reserved         : 500MB
├─ YOLO Detection         : 200MB
├─ YOLO Pose              : 300MB
├─ ST-GCN                 : 500MB
└─ Available              : 6500MB  ← 2GB freed!
```

**Ollama Process GPU (can be same or different GPU):**
```
GPU Memory (8GB or 16GB)
├─ System Reserved         : 500MB
├─ Qwen2.5-VL 3B          : 6000MB
└─ Available              : 1500MB (or 9500MB on 16GB GPU)
```

---

## Configuration Matrix

| Environment | Backend | OLLAMA_URL | GPU Memory | Latency | Quality |
|-------------|---------|------------|------------|---------|---------|
| Dev (Default) | FastVLM | n/a | 3GB | 50ms | Good |
| Dev (Qwen) | Qwen 3B | localhost:11434 | 0.7GB + 6GB | 300ms | Excellent |
| Prod (Local) | FastVLM | n/a | 3GB | 50ms | Good |
| Prod (Dedicated) | Qwen 7B | gpu-server:11434 | 0.7GB | 350ms | Best |
| Cloud | Qwen 3B | lb.example.com | 0.7GB | 400ms | Excellent |

---

## Switching Logic

```python
def run_vlm(image, prompt, use_qwen=None):
    # Auto-detect from environment
    if use_qwen is None:
        use_qwen = os.getenv("USE_QWEN_VL", "false").lower() == "true"
    
    if use_qwen:
        # Route to Ollama
        qwen = init_qwen()
        return qwen.analyze_image(image, prompt)
    else:
        # Use local FastVLM
        return _local_vlm_inference(image, prompt)
```

**Decision Tree:**
```
                    run_vlm() called
                          │
                ┌─────────┴─────────┐
                │ use_qwen param?   │
                └─────────┬─────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
    use_qwen=True   use_qwen=False   use_qwen=None
         │                │                │
         ▼                ▼                ▼
    Use Qwen        Use FastVLM    Check USE_QWEN_VL env
                                           │
                                ┌──────────┴──────────┐
                                │                     │
                          "true" or "1"         "false" or unset
                                │                     │
                                ▼                     ▼
                           Use Qwen             Use FastVLM
```

---

## Error Handling Flow

```
┌─────────────────────┐
│  Client Request     │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  run_vlm()          │
│  use_qwen=True      │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  Try Ollama         │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
   OK           Error
    │             │
    ↓             ↓
┌────────┐  ┌──────────────────┐
│Return  │  │ Return error     │
│Result  │  │ with "error" key │
└────────┘  └──────────────────┘
                   │
                   ↓
         Client can decide to:
         - Show error message
         - Retry with FastVLM
         - Fail gracefully
```

**Resilient Pattern:**
```python
# Try Qwen, fallback to FastVLM
result = run_vlm(frame, prompt, use_qwen=True)
if result.get("error"):
    logger.warning(f"Qwen failed: {result['error']}, using FastVLM")
    result = run_vlm(frame, prompt, use_qwen=False)
```

---

## Conclusion

The Qwen integration transforms VEREC from a **monolithic** architecture to a **microservices-style** architecture for vision-language understanding, providing:

1. **Better Resource Management** - VLM isolated from detection pipeline
2. **Improved Scalability** - Can scale VLM independently
3. **Higher Quality** - State-of-the-art Qwen model
4. **Operational Flexibility** - Easy to upgrade/downgrade models
5. **Backward Compatibility** - Zero changes for existing deployments

The architecture supports **graceful degradation** and **horizontal scaling**, making it production-ready for both small deployments and enterprise scale.
