"""
FastVLM Speed Test — OpenCV camera + upload UI.

Uses OpenCV (cv2) for direct camera access instead of the browser's
MediaDevices API.  This mirrors the iOS app's architecture:

  iOS CameraController (AVCaptureSession)  →  OpenCVCamera (cv2.VideoCapture)
  AsyncStream .bufferingNewest(1)          →  background thread + latest-frame
  ContentView dual-stream distribution     →  gr.Timer live preview + inference loop

Two modes, same as the iOS app:
  • Continuous — every frame is analysed automatically
  • Single Capture — freeze one frame (or upload), then analyse

Usage:
    python demo.py --model-path checkpoints/llava-fastvithd_0.5b_stage3
"""
import os
import time
import argparse
import threading

from dotenv import load_dotenv
load_dotenv()  # load .env before anything reads os.environ

import cv2
import numpy as np
import torch
import gradio as gr
import onnxruntime as ort
from openai import OpenAI
from PIL import Image

# macOS: skip the AVFoundation auth dialog from a background thread;
# the user must grant camera access to Terminal / their terminal app
# in System Settings → Privacy & Security → Camera.
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")


# ── YOLO11n ONNX Object Detection ────────────────────────────────────────
# Works with the ultralytics-exported yolo11n.onnx (input: 'images' [1,3,640,640],
# output: 'output0' [1,84,8400] — 4 box coords + 80 class scores × 8400 anchors).

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
]

np.random.seed(42)
COCO_COLORS = np.random.uniform(0, 255, size=(len(COCO_CLASSES), 3)).astype(int)


class YOLODetector:
    """YOLO11n via ONNX Runtime (CPU).  Ultralytics export format."""

    def __init__(self, model_path, conf=0.5, iou=0.45, img_size=640):
        self.session = ort.InferenceSession(model_path)
        self.conf = conf
        self.iou = iou
        self.img_size = img_size

    def _preprocess(self, img_rgb):
        """Letterbox-resize to img_size × img_size, normalise to 0-1."""
        h, w = img_rgb.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img_rgb, (nw, nh))
        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        top, left = (self.img_size - nh) // 2, (self.img_size - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # NCHW
        return blob, scale, top, left

    def detect(self, img_rgb, conf=None, iou=None):
        """Run detection on an RGB numpy array.
        Returns (boxes, scores, class_ids) in original-image coordinates."""
        conf = conf if conf is not None else self.conf
        iou = iou if iou is not None else self.iou

        blob, scale, pad_top, pad_left = self._preprocess(img_rgb)
        raw = self.session.run(None, {'images': blob})[0]  # [1, 84, 8400]
        preds = raw[0].T  # [8400, 84]

        # columns: cx, cy, w, h, cls0..cls79
        cls_scores = preds[:, 4:]
        max_scores = cls_scores.max(axis=1)
        keep = max_scores > conf
        preds = preds[keep]
        max_scores = max_scores[keep]
        class_ids = cls_scores[keep].argmax(axis=1)

        # Convert cx,cy,w,h → x1,y1,x2,y2 in letterboxed space
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Undo letterbox → original image coords
        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top) / scale

        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(int)

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), max_scores.tolist(), conf, iou,
        )
        if len(indices) == 0:
            return np.empty((0, 4), int), np.array([]), np.array([], int)
        indices = np.array(indices).flatten()
        return boxes[indices], max_scores[indices], class_ids[indices]

    @staticmethod
    def draw(img_rgb, boxes, scores, class_ids, alpha=0.3):
        """Draw boxes + labels on a copy of the image. Returns RGB."""
        det = img_rgb.copy()
        mask = img_rgb.copy()
        h, w = img_rgb.shape[:2]
        font_scale = min(h, w) * 0.0006
        thickness = max(1, int(min(h, w) * 0.001))

        for cid, box, score in zip(class_ids, boxes, scores):
            color = COCO_COLORS[cid].tolist()
            x1, y1, x2, y2 = box
            cv2.rectangle(mask, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(det, (x1, y1), (x2, y2), color, 2)
            label = f'{COCO_CLASSES[cid]} {int(score * 100)}%'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          font_scale, thickness)
            cv2.rectangle(det, (x1, y1 - int(th * 1.4)), (x1 + tw, y1), color, -1)
            cv2.putText(det, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return cv2.addWeighted(mask, alpha, det, 1 - alpha, 0)


_yolo = None


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


class YOLOSegDetector:
    """YOLO11n-seg via ONNX Runtime (CPU).  Ultralytics export format.

    Outputs:
      output0 [1, 116, 8400] — 4 box + 80 classes + 32 mask coefficients
      output1 [1, 32, 160, 160] — 32 mask prototypes
    """

    def __init__(self, model_path, conf=0.5, iou=0.45, img_size=640):
        self.session = ort.InferenceSession(model_path)
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.mask_h = 160
        self.mask_w = 160

    def _preprocess(self, img_rgb):
        """Letterbox-resize to img_size × img_size, normalise to 0-1."""
        h, w = img_rgb.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img_rgb, (nw, nh))
        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        top, left = (self.img_size - nh) // 2, (self.img_size - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # NCHW
        return blob, scale, top, left

    def detect(self, img_rgb, conf=None, iou=None):
        """Run segmentation on an RGB numpy array.
        Returns (boxes, scores, class_ids, masks) where masks is a list of
        boolean masks in original-image coordinates."""
        conf = conf if conf is not None else self.conf
        iou = iou if iou is not None else self.iou
        h_orig, w_orig = img_rgb.shape[:2]

        blob, scale, pad_top, pad_left = self._preprocess(img_rgb)
        outputs = self.session.run(None, {'images': blob})
        raw_det = outputs[0][0]    # [116, 8400]
        protos = outputs[1][0]     # [32, 160, 160]

        preds = raw_det.T  # [8400, 116]

        # columns: cx, cy, w, h, cls0..cls79, mask0..mask31
        cls_scores = preds[:, 4:84]
        max_scores = cls_scores.max(axis=1)
        keep = max_scores > conf
        preds = preds[keep]
        max_scores = max_scores[keep]
        class_ids = cls_scores[keep].argmax(axis=1)

        if len(preds) == 0:
            return np.empty((0, 4), int), np.array([]), np.array([], int), []

        # mask coefficients
        mask_coeffs = preds[:, 84:]  # [N, 32]

        # Convert cx,cy,w,h → x1,y1,x2,y2 in letterboxed space
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Undo letterbox → original image coords
        x1_orig = (x1 - pad_left) / scale
        y1_orig = (y1 - pad_top) / scale
        x2_orig = (x2 - pad_left) / scale
        y2_orig = (y2 - pad_top) / scale

        boxes = np.stack([x1_orig, y1_orig, x2_orig, y2_orig], axis=1).astype(int)

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), max_scores.tolist(), conf, iou,
        )
        if len(indices) == 0:
            return np.empty((0, 4), int), np.array([]), np.array([], int), []
        indices = np.array(indices).flatten()
        boxes = boxes[indices]
        max_scores = max_scores[indices]
        class_ids = class_ids[indices]
        mask_coeffs = mask_coeffs[indices]

        # Compute per-instance masks: coefficients @ prototypes
        # mask_coeffs [N, 32] @ protos [32, 160*160] → [N, 160*160]
        raw_masks = mask_coeffs @ protos.reshape(32, -1)  # [N, 25600]
        raw_masks = raw_masks.reshape(-1, self.mask_h, self.mask_w)  # [N, 160, 160]

        # Sigmoid
        raw_masks = 1.0 / (1.0 + np.exp(-raw_masks))

        # Map mask coordinates back from letterboxed 640 → original image
        # The mask is at 160×160 for 640×640 input (4× downsampled)
        mask_scale = self.mask_h / self.img_size  # 0.25
        pad_top_m = int(pad_top * mask_scale)
        pad_left_m = int(pad_left * mask_scale)
        nh_m = int(int(h_orig * scale) * mask_scale)
        nw_m = int(int(w_orig * scale) * mask_scale)

        # Crop to the non-padded region, then resize to original image size
        masks = []
        for i in range(len(boxes)):
            m = raw_masks[i]
            # Crop out padding
            m_crop = m[pad_top_m:pad_top_m + nh_m, pad_left_m:pad_left_m + nw_m]
            # Resize to original image size
            m_resized = cv2.resize(m_crop, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            # Clip to bounding box for cleaner masks
            bx1, by1, bx2, by2 = boxes[i]
            bx1 = max(0, bx1); by1 = max(0, by1)
            bx2 = min(w_orig, bx2); by2 = min(h_orig, by2)
            box_mask = np.zeros_like(m_resized)
            box_mask[by1:by2, bx1:bx2] = 1.0
            m_resized = m_resized * box_mask
            masks.append(m_resized > 0.5)

        return boxes, max_scores, class_ids, masks

    @staticmethod
    def draw(img_rgb, boxes, scores, class_ids, masks, alpha=0.5):
        """Draw masks + boxes + labels. Returns RGB."""
        out = img_rgb.copy()
        overlay = img_rgb.copy()
        h, w = img_rgb.shape[:2]
        font_scale = min(h, w) * 0.0006
        thickness = max(1, int(min(h, w) * 0.001))

        for i, (cid, box, score) in enumerate(zip(class_ids, boxes, scores)):
            color = COCO_COLORS[cid].tolist()
            mask = masks[i] if i < len(masks) else None

            # Fill mask region with color
            if mask is not None:
                overlay[mask] = color

            # Draw bounding box
            x1, y1, x2, y2 = box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f'{COCO_CLASSES[cid]} {int(score * 100)}%'
            (tw, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                            font_scale, thickness)
            cv2.rectangle(out, (x1, y1 - int(th_t * 1.4)), (x1 + tw, y1), color, -1)
            cv2.putText(out, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)


_yolo_seg = None


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


from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
)

_lock = threading.Lock()


# ── OpenCV Camera ─────────────────────────────────────────────────────────
# Mirrors iOS CameraController + AsyncStream(.bufferingNewest(1)).
# A daemon thread captures frames continuously; callers always get the
# most recent frame, older ones are silently dropped.

class OpenCVCamera:
    def __init__(self, device_id=0, width=640, height=480):
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            print("[OpenCVCamera] WARNING: could not open camera device", device_id)
            print("  → On macOS, grant camera access to your terminal app in")
            print("    System Settings → Privacy & Security → Camera.")
            self._running = False
            self._frame = None
            self._lock = threading.Lock()
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def is_open(self):
        return self._running

    def _loop(self):
        while self._running:
            ok, bgr = self.cap.read()
            if ok:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._frame = rgb
            else:
                time.sleep(0.01)

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def release(self):
        self._running = False
        if self.cap.isOpened():
            self.cap.release()


_camera = None
_camera_error = None


def _ensure_camera():
    global _camera, _camera_error
    if _camera is None:
        _camera = OpenCVCamera()
        if not _camera.is_open:
            _camera_error = (
                "Camera not available. On macOS, grant camera access to your "
                "terminal app in System Settings → Privacy & Security → Camera, "
                "then restart the demo."
            )
    return _camera


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

        input_ids = (
            tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(torch.device("mps"))
        )

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


# ── Camera callbacks ──────────────────────────────────────────────────────

def get_live_frame():
    """Return the latest camera frame for the live preview (called by Timer)."""
    cam = _ensure_camera()
    if not cam.is_open:
        return None
    return cam.read()


def continuous_loop(prompt, temp, tokens):
    """Generator: grab latest frame → run inference → yield → repeat.
    Mirrors the iOS continuous mode.  Stale frames are automatically
    dropped because we always call cam.read() which returns the newest."""
    cam = _ensure_camera()
    if not cam.is_open:
        yield None, "", "", _camera_error or "Camera not available"
        return
    while True:
        frame = cam.read()
        if frame is None:
            yield None, "", "", "Waiting for camera…"
            time.sleep(0.1)
            continue
        image = Image.fromarray(frame)
        text, stats, status = run_inference(image, prompt, temp, tokens)
        yield frame, text, stats, status


_captured_frame = None


def capture_and_analyse(prompt, temp, tokens):
    """Grab one frame from the live camera, then analyse it."""
    global _captured_frame
    cam = _ensure_camera()
    if not cam.is_open:
        yield None, "", "", _camera_error or "Camera not available"
        return
    _captured_frame = cam.read()
    if _captured_frame is None:
        yield None, "", "", "No frame from camera"
        return
    yield _captured_frame, "", "", "Processing…"
    image = Image.fromarray(_captured_frame)
    text, stats, status = run_inference(image, prompt, temp, tokens)
    yield _captured_frame, text, stats, status


def upload_analyse(image, prompt, temp, tokens):
    """Analyse an uploaded / webcam-captured image."""
    if image is None:
        yield "", "", "No image provided"
        return
    yield "", "", "Processing…"
    text, stats, status = run_inference(image, prompt, temp, tokens)
    yield text, stats, status


# ── Gradio UI ─────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="FastVLM Speed Test") as demo:
        gr.Markdown("# FastVLM Speed Test")
        gr.Markdown(f"**Model:** `{args.model_path}`  |  **Device:** Apple Silicon (MPS)")

        with gr.Tabs():

            # ── Tab 1: Camera (OpenCV) ────────────────────────────────
            with gr.Tab("Camera"):
                gr.Markdown(
                    "**OpenCV** captures your local camera.  "
                    "Use *Start Continuous* to analyse every frame, "
                    "or *Capture & Analyse* for a single shot."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        live_img = gr.Image(label="Live Camera", interactive=False)
                        cam_prompt = gr.Textbox(
                            value="What is happening? Answer in one short sentence.",
                            label="Prompt", lines=2,
                        )
                        with gr.Row():
                            cam_temp = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temperature")
                            cam_tokens = gr.Slider(8, 128, value=32, step=8, label="Max Tokens")
                        with gr.Row():
                            start_btn = gr.Button("▶ Start Continuous", variant="primary")
                            stop_btn = gr.Button("⏹ Stop", variant="stop")
                            snap_btn = gr.Button("📷 Capture & Analyse")

                    with gr.Column(scale=1):
                        cam_status = gr.Textbox(label="Status", lines=1, value="Idle")
                        analysed_img = gr.Image(label="Analysed Frame", interactive=False)
                        cam_output = gr.Textbox(label="Model Output", lines=5)
                        cam_stats = gr.Textbox(label="Speed", lines=1)

                # Live preview — refreshes at ~10 fps via Timer
                preview_timer = gr.Timer(value=0.1, active=True)
                preview_timer.tick(fn=get_live_frame, outputs=live_img)

                # Continuous analysis — generator yields forever until cancelled
                cont_event = start_btn.click(
                    fn=continuous_loop,
                    inputs=[cam_prompt, cam_temp, cam_tokens],
                    outputs=[analysed_img, cam_output, cam_stats, cam_status],
                )
                stop_btn.click(fn=None, cancels=[cont_event])

                # Single capture & analyse
                snap_btn.click(
                    fn=capture_and_analyse,
                    inputs=[cam_prompt, cam_temp, cam_tokens],
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
                    "Use *Start Live Detection* for continuous video, "
                    "or upload/capture a single image."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        det_live = gr.Image(label="Live Camera", interactive=False)
                        with gr.Row():
                            det_conf = gr.Slider(
                                0.1, 1.0, value=0.5, step=0.05,
                                label="Confidence",
                            )
                            det_iou = gr.Slider(
                                0.1, 1.0, value=0.45, step=0.05,
                                label="IoU (NMS)",
                            )
                        with gr.Row():
                            det_start = gr.Button("▶ Start Live Detection", variant="primary")
                            det_stop = gr.Button("⏹ Stop", variant="stop")
                            det_snap = gr.Button("📷 Detect Single Frame")
                        gr.Markdown("---")
                        det_upload = gr.Image(
                            sources=["upload", "webcam"],
                            type="numpy",
                            label="Or upload / capture an image",
                        )
                        det_upload_btn = gr.Button("Detect on Image", variant="secondary")

                    with gr.Column(scale=1):
                        det_status = gr.Textbox(label="Status", lines=1, value="Idle")
                        det_output = gr.Image(label="Detections", interactive=False)
                        det_stats = gr.Textbox(label="Stats", lines=1)
                        det_labels = gr.Textbox(label="Detected Objects", lines=5)

                # Live preview for detection tab (same camera)
                det_preview_timer = gr.Timer(value=0.1, active=True)
                det_preview_timer.tick(fn=get_live_frame, outputs=det_live)

                def _run_yolo(frame, conf, iou_thresh):
                    """Shared helper: run YOLO on one RGB frame."""
                    yolo = _ensure_yolo()
                    t0 = time.perf_counter()
                    boxes, scores, class_ids = yolo.detect(
                        frame, conf=conf, iou=iou_thresh,
                    )
                    t1 = time.perf_counter()
                    annotated = YOLODetector.draw(frame, boxes, scores, class_ids)
                    dt = t1 - t0
                    n = len(boxes)
                    stats = f"Time: {dt*1000:.0f}ms  |  Objects: {n}  |  FPS: {1/dt:.1f}"
                    labels = ", ".join(
                        f"{COCO_CLASSES[c]} ({s:.0%})" for c, s in zip(class_ids, scores)
                    ) if n else "No objects detected"
                    return annotated, stats, labels

                def yolo_continuous(conf, iou_thresh):
                    """Generator: grab latest frame → detect → yield → repeat."""
                    cam = _ensure_camera()
                    if not cam.is_open:
                        yield None, "", "", _camera_error or "Camera not available"
                        return
                    try:
                        yolo = _ensure_yolo()
                    except Exception as e:
                        yield None, "", "", f"Error: {e}"
                        return
                    while True:
                        frame = cam.read()
                        if frame is None:
                            yield None, "", "", "Waiting for camera…"
                            time.sleep(0.05)
                            continue
                        try:
                            annotated, stats, labels = _run_yolo(frame, conf, iou_thresh)
                            yield annotated, stats, labels, "Detecting…"
                        except Exception as e:
                            yield None, "", "", f"Error: {e}"

                def yolo_single_frame(conf, iou_thresh):
                    """Grab one frame, detect, return."""
                    cam = _ensure_camera()
                    if not cam.is_open:
                        yield None, "", "", _camera_error or "Camera not available"
                        return
                    frame = cam.read()
                    if frame is None:
                        yield None, "", "", "No frame from camera"
                        return
                    yield None, "", "", "Detecting…"
                    try:
                        annotated, stats, labels = _run_yolo(frame, conf, iou_thresh)
                        yield annotated, stats, labels, "Done"
                    except Exception as e:
                        yield None, "", "", f"Error: {e}"

                def yolo_on_upload(image, conf, iou_thresh):
                    """Detect on an uploaded image."""
                    if image is None:
                        yield None, "", "", "No image provided"
                        return
                    yield None, "", "", "Detecting…"
                    try:
                        annotated, stats, labels = _run_yolo(image, conf, iou_thresh)
                        yield annotated, stats, labels, "Done"
                    except Exception as e:
                        yield None, "", "", f"Error: {e}"

                det_cont_event = det_start.click(
                    fn=yolo_continuous,
                    inputs=[det_conf, det_iou],
                    outputs=[det_output, det_stats, det_labels, det_status],
                )
                det_stop.click(fn=None, cancels=[det_cont_event])

                det_snap.click(
                    fn=yolo_single_frame,
                    inputs=[det_conf, det_iou],
                    outputs=[det_output, det_stats, det_labels, det_status],
                )
                det_upload_btn.click(
                    fn=yolo_on_upload,
                    inputs=[det_upload, det_conf, det_iou],
                    outputs=[det_output, det_stats, det_labels, det_status],
                )

            # ── Tab 4: Segmentation (YOLO11n-seg ONNX) ───────────────
            with gr.Tab("Segmentation"):
                gr.Markdown(
                    "**YOLO11n-seg** via ONNX Runtime (CPU).  "
                    "Instance segmentation with per-object masks.  "
                    "Use *Start Live Segmentation* for continuous video, "
                    "or upload/capture a single image."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        seg_live = gr.Image(label="Live Camera", interactive=False)
                        with gr.Row():
                            seg_conf = gr.Slider(
                                0.1, 1.0, value=0.5, step=0.05,
                                label="Confidence",
                            )
                            seg_iou = gr.Slider(
                                0.1, 1.0, value=0.45, step=0.05,
                                label="IoU (NMS)",
                            )
                        with gr.Row():
                            seg_start = gr.Button("▶ Start Live Segmentation", variant="primary")
                            seg_stop = gr.Button("⏹ Stop", variant="stop")
                            seg_snap = gr.Button("📷 Segment Single Frame")
                        gr.Markdown("---")
                        seg_upload = gr.Image(
                            sources=["upload", "webcam"],
                            type="numpy",
                            label="Or upload / capture an image",
                        )
                        seg_upload_btn = gr.Button("Segment Image", variant="secondary")

                    with gr.Column(scale=1):
                        seg_status = gr.Textbox(label="Status", lines=1, value="Idle")
                        seg_output = gr.Image(label="Segmentation", interactive=False)
                        seg_stats = gr.Textbox(label="Stats", lines=1)
                        seg_labels = gr.Textbox(label="Detected Objects", lines=5)

                # Live preview for segmentation tab (same camera)
                seg_preview_timer = gr.Timer(value=0.1, active=True)
                seg_preview_timer.tick(fn=get_live_frame, outputs=seg_live)

                def _run_seg(frame, conf, iou_thresh):
                    """Shared helper: run YOLO-seg on one RGB frame."""
                    seg = _ensure_yolo_seg()
                    t0 = time.perf_counter()
                    boxes, scores, class_ids, masks = seg.detect(
                        frame, conf=conf, iou=iou_thresh,
                    )
                    t1 = time.perf_counter()
                    annotated = YOLOSegDetector.draw(
                        frame, boxes, scores, class_ids, masks,
                    )
                    dt = t1 - t0
                    n = len(boxes)
                    stats = f"Time: {dt*1000:.0f}ms  |  Objects: {n}  |  FPS: {1/dt:.1f}"
                    labels = ", ".join(
                        f"{COCO_CLASSES[c]} ({s:.0%})" for c, s in zip(class_ids, scores)
                    ) if n else "No objects detected"
                    return annotated, stats, labels

                def seg_continuous(conf, iou_thresh):
                    """Generator: grab latest frame → segment → yield → repeat."""
                    cam = _ensure_camera()
                    if not cam.is_open:
                        yield None, "", "", _camera_error or "Camera not available"
                        return
                    try:
                        _ensure_yolo_seg()
                    except Exception as e:
                        yield None, "", "", f"Error: {e}"
                        return
                    while True:
                        frame = cam.read()
                        if frame is None:
                            yield None, "", "", "Waiting for camera…"
                            time.sleep(0.05)
                            continue
                        try:
                            annotated, stats, labels = _run_seg(frame, conf, iou_thresh)
                            yield annotated, stats, labels, "Segmenting…"
                        except Exception as e:
                            yield None, "", "", f"Error: {e}"

                def seg_single_frame(conf, iou_thresh):
                    """Grab one frame, segment, return."""
                    cam = _ensure_camera()
                    if not cam.is_open:
                        yield None, "", "", _camera_error or "Camera not available"
                        return
                    frame = cam.read()
                    if frame is None:
                        yield None, "", "", "No frame from camera"
                        return
                    yield None, "", "", "Segmenting…"
                    try:
                        annotated, stats, labels = _run_seg(frame, conf, iou_thresh)
                        yield annotated, stats, labels, "Done"
                    except Exception as e:
                        yield None, "", "", f"Error: {e}"

                def seg_on_upload(image, conf, iou_thresh):
                    """Segment an uploaded image."""
                    if image is None:
                        yield None, "", "", "No image provided"
                        return
                    yield None, "", "", "Segmenting…"
                    try:
                        annotated, stats, labels = _run_seg(image, conf, iou_thresh)
                        yield annotated, stats, labels, "Done"
                    except Exception as e:
                        yield None, "", "", f"Error: {e}"

                seg_cont_event = seg_start.click(
                    fn=seg_continuous,
                    inputs=[seg_conf, seg_iou],
                    outputs=[seg_output, seg_stats, seg_labels, seg_status],
                )
                seg_stop.click(fn=None, cancels=[seg_cont_event])

                seg_snap.click(
                    fn=seg_single_frame,
                    inputs=[seg_conf, seg_iou],
                    outputs=[seg_output, seg_stats, seg_labels, seg_status],
                )
                seg_upload_btn.click(
                    fn=seg_on_upload,
                    inputs=[seg_upload, seg_conf, seg_iou],
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
                    value="gpt-4o-mini",
                    label="Model",
                )

                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=480,
                )
                with gr.Row():
                    chat_input = gr.Textbox(
                        label="Message",
                        placeholder="Type a message…",
                        scale=4,
                        lines=1,
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
                        resp = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                            ] + [
                                {"role": m["role"], "content": m["content"]}
                                for m in history
                            ],
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
                        value="deepseek-r1:1.5b",
                        label="Model",
                        scale=1,
                    )
                    r1_url = gr.Textbox(
                        value=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
                        label="Ollama URL",
                        scale=2,
                    )

                r1_chatbot = gr.Chatbot(
                    label="Conversation",
                    height=480,
                )
                with gr.Row():
                    r1_input = gr.Textbox(
                        label="Message",
                        placeholder="Ask the reasoning model…",
                        scale=4,
                        lines=1,
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

                        # Build messages from history
                        api_messages = [
                            {"role": "system", "content": "You are a helpful reasoning assistant. Think step by step."},
                        ]
                        for m in history:
                            api_messages.append({"role": m["role"], "content": m["content"]})

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
