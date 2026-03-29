"""
FastVLM Speed Test — Browser webcam + upload UI.

Uses Gradio's built-in webcam streaming for camera access.

Tabs:
  • Camera — capture a frame from your browser webcam, then analyse
  • Upload — drag-and-drop an image, then analyse
  • Object Detection — YOLO11n via ONNX Runtime (live webcam stream)
  • Segmentation — YOLO11n-seg via ONNX Runtime (live webcam stream)
  • AI Chat — OpenAI gpt-4o-mini (requires OPENAI_API_KEY in .env)
  • Reasoning (Offline) — DeepSeek-R1 via Ollama (local, no API key)

Usage:
    python demo.py --model-path checkpoints/llava-fastvithd_0.5b_stage3
"""
import os
import time
import argparse
import threading

from dotenv import load_dotenv
load_dotenv()  # load .env before anything reads os.environ

import numpy as np
import torch
import gradio as gr
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from PIL import Image

from detectors import COCO_CLASSES, YOLODetector, YOLOSegDetector
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
)

_lock = threading.Lock()


# ── YOLO singletons ──────────────────────────────────────────────────────

_yolo = None
_yolo_seg = None


def _ensure_yolo():
    global _yolo
    if _yolo is None:
        onnx_path = os.path.join(os.path.dirname(__file__), 'yolo11n.onnx')
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f'yolo11n.onnx not found at {onnx_path}. '
                'Export it with: python -c "from ultralytics import YOLO; '
                "YOLO('yolo11n.pt').export(format='onnx',imgsz=640)\"")
        _yolo = YOLODetector(onnx_path)
        print(f'YOLO loaded from {onnx_path}')
    return _yolo


def _ensure_yolo_seg():
    global _yolo_seg
    if _yolo_seg is None:
        onnx_path = os.path.join(os.path.dirname(__file__), 'yolo11n-seg.onnx')
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f'yolo11n-seg.onnx not found at {onnx_path}. '
                'Export it with: python -c "from ultralytics import YOLO; '
                "YOLO('yolo11n-seg.pt').export(format='onnx',imgsz=640)\"")
        _yolo_seg = YOLOSegDetector(onnx_path)
        print(f'YOLO-seg loaded from {onnx_path}')
    return _yolo_seg


# ── Model helpers ─────────────────────────────────────────────────────────

def load_model(model_path, model_base=None):
    model_path = os.path.expanduser(model_path)
    gen_cfg = os.path.join(model_path, "generation_config.json")
    gen_cfg_hidden = os.path.join(model_path, ".generation_config.json")
    renamed = False
    if os.path.exists(gen_cfg):
        os.rename(gen_cfg, gen_cfg_hidden)
        renamed = True

    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, device="mps"
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    if renamed:
        os.rename(gen_cfg_hidden, gen_cfg)
    return tokenizer, model, image_processor


def run_inference(image, prompt, temperature, max_tokens):
    """Run model on a PIL Image.  Returns (text, stats_str, status_str)."""
    if image is None:
        return "", "", "No image provided"
    if not prompt or not prompt.strip():
        prompt = "Briefly describe what is happening."

    try:
        qs = prompt
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = torch.as_tensor(
            tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        ).unsqueeze(0).to(torch.device("mps"))

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)[0]

        with _lock, torch.inference_mode():
            t0 = time.perf_counter()
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half(),
                image_sizes=[image.size],
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                max_new_tokens=int(max_tokens),
                use_cache=True,
            )
            t1 = time.perf_counter()

        text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        dt = t1 - t0
        n = output_ids.shape[1] - input_ids.shape[1]
        tps = n / dt if dt > 0 else 0
        stats = f"Time: {dt:.2f}s  |  Tokens: {n}  |  Speed: {tps:.1f} tok/s"
        return text, stats, "Done"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return "", "", f"Error: {e}"


def upload_analyse(image, prompt, temp, tokens):
    """Analyse an uploaded / webcam-captured image."""
    if image is None:
        yield "", "", "No image provided"
        return
    yield "", "", "Processing…"
    text, stats, status = run_inference(image, prompt, temp, tokens)
    yield text, stats, status


# ── YOLO helpers ──────────────────────────────────────────────────────────

def _run_yolo(frame, conf, iou_thresh):
    """Run YOLO detection on one RGB frame."""
    yolo = _ensure_yolo()
    t0 = time.perf_counter()
    boxes, scores, class_ids = yolo.detect(frame, conf=conf, iou=iou_thresh)
    t1 = time.perf_counter()
    annotated = YOLODetector.draw(frame, boxes, scores, class_ids)
    dt = t1 - t0
    n = len(boxes)
    stats = f"Time: {dt*1000:.0f}ms  |  Objects: {n}  |  FPS: {1/dt:.1f}"
    labels = ", ".join(
        f"{COCO_CLASSES[c]} ({s:.0%})" for c, s in zip(class_ids, scores)
    ) if n else "No objects detected"
    return annotated, stats, labels


def _run_seg(frame, conf, iou_thresh):
    """Run YOLO-seg on one RGB frame."""
    seg = _ensure_yolo_seg()
    t0 = time.perf_counter()
    boxes, scores, class_ids, masks = seg.detect(frame, conf=conf, iou=iou_thresh)
    t1 = time.perf_counter()
    annotated = YOLOSegDetector.draw(frame, boxes, scores, class_ids, masks)
    dt = t1 - t0
    n = len(boxes)
    stats = f"Time: {dt*1000:.0f}ms  |  Objects: {n}  |  FPS: {1/dt:.1f}"
    labels = ", ".join(
        f"{COCO_CLASSES[c]} ({s:.0%})" for c, s in zip(class_ids, scores)
    ) if n else "No objects detected"
    return annotated, stats, labels


# ── Gradio UI ─────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="FastVLM Speed Test") as demo:
        gr.Markdown("# FastVLM Speed Test")
        gr.Markdown(f"**Model:** `{args.model_path}`  |  **Device:** Apple Silicon (MPS)")

        with gr.Tabs():

            # ── Tab 1: Camera (Browser Webcam) ─────────────────────────
            with gr.Tab("Camera"):
                gr.Markdown(
                    "**Browser webcam** — your browser will ask for camera "
                    "permission.  Click **Capture & Analyse** to run the VLM "
                    "on the current frame."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        live_img = gr.Image(
                            sources=["webcam"], streaming=True,
                            label="Live Camera", type="numpy",
                        )
                        cam_prompt = gr.Textbox(
                            value="What is happening? Answer in one short sentence.",
                            label="Prompt", lines=2,
                        )
                        with gr.Row():
                            cam_temp = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temperature")
                            cam_tokens = gr.Slider(8, 128, value=32, step=8, label="Max Tokens")
                        snap_btn = gr.Button("📷 Capture & Analyse", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        cam_status = gr.Textbox(label="Status", lines=1, value="Idle")
                        analysed_img = gr.Image(label="Analysed Frame", interactive=False)
                        cam_output = gr.Textbox(label="Model Output", lines=5)
                        cam_stats = gr.Textbox(label="Speed", lines=1)

                def cam_analyse(frame, prompt, temp, tokens):
                    if frame is None:
                        yield None, "", "", "No frame — enable webcam first"
                        return
                    yield frame, "", "", "Processing…"
                    image = Image.fromarray(frame)
                    text, stats, status = run_inference(image, prompt, temp, tokens)
                    yield frame, text, stats, status

                snap_btn.click(
                    fn=cam_analyse,
                    inputs=[live_img, cam_prompt, cam_temp, cam_tokens],
                    outputs=[analysed_img, cam_output, cam_stats, cam_status],
                )

            # ── Tab 2: Upload ─────────────────────────────────────────
            with gr.Tab("Upload"):
                gr.Markdown("Upload an image (or use the browser webcam) and click **Analyse**.")
                with gr.Row():
                    with gr.Column(scale=1):
                        upload_img = gr.Image(
                            sources=["upload", "webcam"],
                            type="pil",
                            label="Image",
                        )
                        up_prompt = gr.Textbox(
                            value="Describe the image.",
                            label="Prompt", lines=2,
                        )
                        with gr.Row():
                            up_temp = gr.Slider(0.0, 1.0, value=0.2, step=0.1, label="Temperature")
                            up_tokens = gr.Slider(16, 512, value=128, step=16, label="Max Tokens")
                        up_btn = gr.Button("Analyse", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        up_status = gr.Textbox(label="Status", lines=1, value="Ready")
                        up_output = gr.Textbox(label="Model Output", lines=10)
                        up_stats = gr.Textbox(label="Speed Stats", lines=1)

                up_btn.click(
                    fn=upload_analyse,
                    inputs=[upload_img, up_prompt, up_temp, up_tokens],
                    outputs=[up_output, up_stats, up_status],
                )

            # ── Tab 3: Object Detection (YOLO11n ONNX) ───────────────
            with gr.Tab("Object Detection"):
                gr.Markdown(
                    "**YOLO11n** via ONNX Runtime (CPU).  "
                    "Enable your webcam for live detection, "
                    "or upload a single image."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        det_live = gr.Image(
                            sources=["webcam"], streaming=True,
                            label="Live Camera", type="numpy",
                        )
                        with gr.Row():
                            det_conf = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="Confidence")
                            det_iou = gr.Slider(0.1, 1.0, value=0.45, step=0.05, label="IoU (NMS)")
                        gr.Markdown("---")
                        det_upload = gr.Image(
                            sources=["upload"], type="numpy",
                            label="Or upload an image",
                        )
                        det_upload_btn = gr.Button("Detect on Image", variant="secondary")

                    with gr.Column(scale=1):
                        det_status = gr.Textbox(label="Status", lines=1, value="Idle")
                        det_output = gr.Image(label="Detections", interactive=False)
                        det_stats = gr.Textbox(label="Stats", lines=1)
                        det_labels = gr.Textbox(label="Detected Objects", lines=5)

                def yolo_stream(frame, conf, iou_thresh):
                    if frame is None:
                        return None, "", "", "Waiting for camera…"
                    try:
                        annotated, stats, labels = _run_yolo(frame, conf, iou_thresh)
                        return annotated, stats, labels, "Detecting…"
                    except Exception as e:
                        return None, "", "", f"Error: {e}"

                det_live.stream(
                    fn=yolo_stream,
                    inputs=[det_live, det_conf, det_iou],
                    outputs=[det_output, det_stats, det_labels, det_status],
                    stream_every=0.1,
                )

                def yolo_on_upload(image, conf, iou_thresh):
                    if image is None:
                        return None, "", "", "No image provided"
                    try:
                        annotated, stats, labels = _run_yolo(image, conf, iou_thresh)
                        return annotated, stats, labels, "Done"
                    except Exception as e:
                        return None, "", "", f"Error: {e}"

                det_upload_btn.click(
                    fn=yolo_on_upload, inputs=[det_upload, det_conf, det_iou],
                    outputs=[det_output, det_stats, det_labels, det_status],
                )

            # ── Tab 4: Segmentation (YOLO11n-seg ONNX) ───────────────
            with gr.Tab("Segmentation"):
                gr.Markdown(
                    "**YOLO11n-seg** via ONNX Runtime (CPU).  "
                    "Instance segmentation with per-object masks.  "
                    "Enable your webcam for live segmentation, "
                    "or upload a single image."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        seg_live = gr.Image(
                            sources=["webcam"], streaming=True,
                            label="Live Camera", type="numpy",
                        )
                        with gr.Row():
                            seg_conf = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="Confidence")
                            seg_iou = gr.Slider(0.1, 1.0, value=0.45, step=0.05, label="IoU (NMS)")
                        gr.Markdown("---")
                        seg_upload = gr.Image(
                            sources=["upload"], type="numpy",
                            label="Or upload an image",
                        )
                        seg_upload_btn = gr.Button("Segment Image", variant="secondary")

                    with gr.Column(scale=1):
                        seg_status = gr.Textbox(label="Status", lines=1, value="Idle")
                        seg_output = gr.Image(label="Segmentation", interactive=False)
                        seg_stats = gr.Textbox(label="Stats", lines=1)
                        seg_labels = gr.Textbox(label="Detected Objects", lines=5)

                def seg_stream(frame, conf, iou_thresh):
                    if frame is None:
                        return None, "", "", "Waiting for camera…"
                    try:
                        annotated, stats, labels = _run_seg(frame, conf, iou_thresh)
                        return annotated, stats, labels, "Segmenting…"
                    except Exception as e:
                        return None, "", "", f"Error: {e}"

                seg_live.stream(
                    fn=seg_stream,
                    inputs=[seg_live, seg_conf, seg_iou],
                    outputs=[seg_output, seg_stats, seg_labels, seg_status],
                    stream_every=0.1,
                )

                def seg_on_upload(image, conf, iou_thresh):
                    if image is None:
                        return None, "", "", "No image provided"
                    try:
                        annotated, stats, labels = _run_seg(image, conf, iou_thresh)
                        return annotated, stats, labels, "Done"
                    except Exception as e:
                        return None, "", "", f"Error: {e}"

                seg_upload_btn.click(
                    fn=seg_on_upload, inputs=[seg_upload, seg_conf, seg_iou],
                    outputs=[seg_output, seg_stats, seg_labels, seg_status],
                )

            # ── Tab 5: AI Chat (OpenAI gpt-4o-mini) ──────────────────
            with gr.Tab("AI Chat"):
                gr.Markdown(
                    "**AI Chat** powered by OpenAI `gpt-4o-mini`.  "
                    "Set `OPENAI_API_KEY` in your `.env` file."
                )

                chat_model = gr.Dropdown(
                    choices=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                    value="gpt-4o-mini", label="Model",
                )

                chatbot = gr.Chatbot(label="Conversation", height=520, autoscroll=True)
                with gr.Row():
                    chat_input = gr.Textbox(
                        label="Message", placeholder="Type a message…",
                        scale=4, lines=1,
                    )
                    chat_send = gr.Button("Send", variant="primary", scale=1)
                chat_clear = gr.Button("🗑 Clear Chat")

                def chat_respond(message, history, model_name):
                    api_key = os.environ.get("OPENAI_API_KEY", "")
                    if not api_key:
                        history = history + [
                            {"role": "user", "content": message},
                            {"role": "assistant", "content": "⚠️ Set OPENAI_API_KEY in your .env file and restart."},
                        ]
                        return history, ""
                    if not message or not message.strip():
                        return history, ""

                    history = history + [{"role": "user", "content": message}]

                    try:
                        client = OpenAI(api_key=api_key)
                        messages: list[ChatCompletionMessageParam] = [
                            {"role": "system", "content": "You are a helpful assistant."},
                        ] + [
                            {"role": m["role"], "content": m["content"]}  # type: ignore[typeddict-item]
                            for m in history
                        ]
                        resp = client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            max_tokens=1024,
                        )
                        reply = resp.choices[0].message.content
                    except Exception as e:
                        reply = f"⚠️ Error: {e}"

                    history = history + [{"role": "assistant", "content": reply}]
                    return history, ""

                chat_send.click(
                    fn=chat_respond,
                    inputs=[chat_input, chatbot, chat_model],
                    outputs=[chatbot, chat_input],
                )
                chat_input.submit(
                    fn=chat_respond,
                    inputs=[chat_input, chatbot, chat_model],
                    outputs=[chatbot, chat_input],
                )
                chat_clear.click(fn=lambda: ([], ""), outputs=[chatbot, chat_input])

            # ── Tab 6: Offline Reasoning (DeepSeek-R1 via Ollama) ─────
            with gr.Tab("Reasoning (Offline)"):
                gr.Markdown(
                    "**DeepSeek-R1-Distill-Qwen-1.5B** — local reasoning model "
                    "via Ollama.  Runs entirely on your machine (no API key needed).  \n"
                    "The model thinks step-by-step before answering."
                )

                with gr.Row():
                    r1_model = gr.Dropdown(
                        choices=["deepseek-r1:1.5b"],
                        value="deepseek-r1:1.5b", label="Model", scale=1,
                    )
                    r1_url = gr.Textbox(
                        value=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
                        label="Ollama URL", scale=2,
                    )

                r1_chatbot = gr.Chatbot(label="Conversation", height=520, autoscroll=True)
                with gr.Row():
                    r1_input = gr.Textbox(
                        label="Message", placeholder="Ask the reasoning model…",
                        scale=4, lines=1,
                    )
                    r1_send = gr.Button("Send", variant="primary", scale=1)
                r1_clear = gr.Button("🗑 Clear Chat")
                r1_status = gr.Textbox(label="Status", lines=1, value="Ready")

                def _parse_r1_response(text):
                    """Split DeepSeek-R1 output into (thinking, answer)."""
                    import re
                    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
                    if think_match:
                        thinking = think_match.group(1).strip()
                        answer = text[think_match.end():].strip()
                        return thinking, answer
                    return "", text.strip()

                def r1_respond(message, history, model_name, base_url):
                    if not message or not message.strip():
                        return history, "", "Ready"

                    history = history + [{"role": "user", "content": message}]

                    try:
                        client = OpenAI(
                            base_url=f"{base_url.rstrip('/')}/v1",
                            api_key="ollama",
                        )

                        api_messages: list[ChatCompletionMessageParam] = [
                            {"role": "system", "content": "You are a helpful reasoning assistant. Think step by step."},
                        ]
                        for m in history:
                            api_messages.append({"role": m["role"], "content": m["content"]})  # type: ignore[typeddict-item]

                        resp = client.chat.completions.create(
                            model=model_name,
                            messages=api_messages,
                            max_tokens=2048,
                        )
                        raw = resp.choices[0].message.content
                        thinking, answer = _parse_r1_response(raw)

                        if thinking:
                            reply = f"💭 **Thinking:**\n{thinking}\n\n---\n\n**Answer:**\n{answer}"
                        else:
                            reply = answer

                        status = "Done"
                    except Exception as e:
                        err = str(e)
                        if "Connection refused" in err or "ConnectError" in err:
                            reply = (
                                "⚠️ Cannot connect to Ollama. Make sure it's running:\n"
                                "```\nollama serve\n```\n"
                                "Then pull the model:\n"
                                "```\nollama pull deepseek-r1:1.5b\n```"
                            )
                        else:
                            reply = f"⚠️ Error: {e}"
                        status = "Error"

                    history = history + [{"role": "assistant", "content": reply}]
                    return history, "", status

                r1_send.click(
                    fn=r1_respond,
                    inputs=[r1_input, r1_chatbot, r1_model, r1_url],
                    outputs=[r1_chatbot, r1_input, r1_status],
                )
                r1_input.submit(
                    fn=r1_respond,
                    inputs=[r1_input, r1_chatbot, r1_model, r1_url],
                    outputs=[r1_chatbot, r1_input, r1_status],
                )
                r1_clear.click(fn=lambda: ([], "", "Ready"), outputs=[r1_chatbot, r1_input, r1_status])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print("Loading model…")
    tokenizer, model, image_processor = load_model(args.model_path, args.model_base)
    print("Model loaded.  Starting UI…")

    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.port)
