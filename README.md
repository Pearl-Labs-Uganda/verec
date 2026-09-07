# VEREC — a CCTV camera that describes what it sees

**VEREC turns a normal camera feed into plain-English commentary.**

A security camera can record for twelve hours and still tell you nothing. Someone has to sit and watch it. VEREC watches instead, and writes down what happened — in sentences a person can read, not just boxes on a screen.

Point it at a webcam, a video file, or an IP/RTSP camera. It tells you things like:

> *"Three people are standing near the entrance. One of them just set a bag down and walked away."*

Think of it as a **commentator for CCTV** — the way a sports commentator narrates a match, VEREC narrates a camera feed.

---

## Why this is hard (and what makes VEREC different)

Ordinary object detection gives you labels and boxes: `person 0.91`, `backpack 0.78`. That tells you *what is in frame*, but not *what is going on*.

Knowing "what is going on" needs several different kinds of seeing, layered together:

| Layer | Question it answers | What we use |
|---|---|---|
| **Object detection** | *What things are here?* | YOLO11n — finds and boxes objects |
| **Pose detection** | *How are people's bodies positioned?* | YOLO11n-pose — 17 skeleton keypoints per person |
| **Action recognition** | *What are they physically doing?* | ST-GCN — reads skeletons over time (sitting down, falling, throwing…) |
| **Vision-language model** | *What does this scene mean?* | FastVLM — looks at the frame and writes a sentence |
| **Reasoning model** | *So what should we do about it?* | An LLM turns the log into a report |

Each layer alone is limited. A detector sees a `person` and a `bag`. Pose sees the person bending. The VLM sees *"a person leaving a bag unattended near a doorway."* **Stacking them is the whole idea.**

---

## How it works

```
   camera / video / RTSP stream
              │
              ▼
        ┌───────────┐
        │   frame   │
        └─────┬─────┘
              │  (every frame)
      ┌───────┼────────┬──────────────┐
      ▼       ▼        ▼              │
  objects   poses   skeletons         │ (every N seconds)
  (YOLO)   (YOLO)   over time         ▼
      │       │        │         ┌─────────┐
      │       │        ▼         │ FastVLM │
      │       │    ST-GCN        │ caption │
      │       │    action        └────┬────┘
      └───────┴────────┴──────────────┘
                       │
                       ▼
              running event log
                       │
                       ▼
              ┌─────────────────┐
              │  LLM report     │
              │  summary +      │
              │  observations + │
              │  what to do     │
              └─────────────────┘
```

Detection and pose run on **every frame** because they're fast. The vision-language model is slower and richer, so it runs **on an interval** (every 5 seconds by default) and writes a caption. Everything lands in a timestamped log, and at any point you can ask for a report.

### The report

The final step feeds the whole log to a reasoning model, which returns:

1. **Summary** — 2–3 sentences on what happened
2. **Key Observations** — the notable moments as bullets
3. **Recommended Actions** — e.g. *"Monitor crowd density at entrance"*, *"Investigate unattended object"*

Everything is exportable as JSON — detections, captions, and reports.

---

## Quick start

**Requirements:** Python 3.10 recommended (3.8+ supported), Node.js (for the web UI), and a camera or video file. Apple Silicon (MPS), CUDA, and CPU are all supported.

```bash
# 1. Install
conda create -n verec python=3.10 && conda activate verec
pip install -e .

# 2. Get the models
bash get_models.sh          # FastVLM vision-language checkpoints
bash get_action_models.sh   # ST-GCN action recognition

# 3. Set your keys (never commit this file — it is gitignored)
cp .env.example .env

# 4. Run backend + web UI together
./dev.sh
```

Then open **http://localhost:3000**. The API runs on port 8000.

### Just want to poke at it?

```bash
python demo.py
```

A Gradio app with tabs for Live Feed, Camera, Upload, IP Camera, Object Detection, AI Chat, and offline Reasoning.

---

## The API

The backend is FastAPI. The important parts:

| Endpoint | What it does |
|---|---|
| `WS /ws/feed` | **The main one.** Live stream — frames in, detections + poses + captions out |
| `POST /api/detect` | Run object detection on one frame |
| `POST /api/vlm` | Caption one frame with a custom prompt |
| `POST /api/report` | Generate the written report from everything logged so far |
| `GET /api/export/all` | Download detections, captions, and reports as JSON |
| `GET /api/system` | Device info — which GPU/accelerator is in use |

The websocket takes `enable_det`, `enable_pose`, `enable_vlm`, and `vlm_interval` so you can trade speed against detail.

---

## Repo map

```
backend/      FastAPI server + model loading/warmup
detectors/    YOLO object detection, pose detection, segmentation
action/       ST-GCN skeleton-based action recognition (NTU-60 classes)
llava/        FastVLM / LLaVA vision-language model code
frontend/     Next.js web interface
model_export/ Export models for Apple Silicon
app/          iOS / macOS demo app
demo.py       All-in-one Gradio demo
dev.sh        Runs backend + frontend together
```

---

## Project status

This is **active research work**, not a finished product. Honest picture of where things stand:

**Working now**
- Live object detection, pose detection, and action recognition
- FastVLM scene captioning on an interval
- LLM report generation, JSON export
- Web UI and Gradio demo, with Apple Silicon / CUDA / CPU support

**Not there yet**
- **Open-vocabulary detection.** Right now detection is *closed-set* — YOLO11n over the 80 standard COCO classes. It cannot find "a red delivery van" or "a person holding a crowbar" unless that's already a COCO class. Moving to open-vocabulary detection (so you can describe any object in words and have it found) is the main thing on the roadmap.
- Multi-camera support, and person re-identification across cameras
- Long-horizon memory, so the system can say *"this is the third time today"*

---

## Built on FastVLM

VEREC is built on top of Apple's **[FastVLM: Efficient Vision Encoding for Vision Language Models](https://www.arxiv.org/abs/2412.13303)** (CVPR 2025). FastVLM is what makes live captioning practical — its FastViTHD encoder produces far fewer tokens than comparable encoders, so time-to-first-token is low enough to caption a moving video feed rather than a still photo.

For FastVLM's own documentation, model zoo, and inference instructions, see **[docs/FASTVLM.md](docs/FASTVLM.md)**.

```bibtex
@InProceedings{fastvlm2025,
  author = {Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokul Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel, Hadi Pouransari},
  title = {FastVLM: Efficient Vision Encoding for Vision Language Models},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2025},
}
```

## Acknowledgements

Built on multiple open-source projects — see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS). Also uses [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) for detection and pose, and ST-GCN for action recognition.

## License

The FastVLM code and models in this repository are Apple's and remain under Apple's terms — see [LICENSE](LICENSE) and [LICENSE_MODEL](LICENSE_MODEL) before using them.
