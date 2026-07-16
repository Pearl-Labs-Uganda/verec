"""Phase 1 — VLM call: image in, text out (batch version).

Loops over >=5 sample photos, sends each to the VLM (Qwen2.5-VL via Ollama),
and prints a text description per image. Confirms the VLM is reachable and
returns sane descriptions, not just error strings.

By default this reads frames produced by frame_extractor.py (Phase 2). Run
that first for a realistic traffic-scene batch:
    python frame_extractor.py --max-frames 5
    python test_qwen_batch.py

If sample_frames/ doesn't exist yet or has fewer than 5 images, this script
falls back to generating 5 synthetic images so it still runs standalone.
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import cv2
import numpy as np
import requests

from backend.qwen_vision import QwenVisionAnalyzer

_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _synthetic_images(n: int) -> list[np.ndarray]:
    """Zero-dependency fallback: n images with different colored shapes."""
    images = []
    palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]
    for i in range(n):
        img = np.full((480, 640, 3), 240, dtype=np.uint8)
        color = palette[i % len(palette)]
        cv2.circle(img, (200 + i * 20, 240), 80, color, -1)
        cv2.rectangle(img, (350, 160), (520, 320), (0, 128, 128), -1)
        cv2.putText(img, f"Sample {i + 1}", (200, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        images.append(img)
    return images


def _load_sample_images(sample_dir: str, min_count: int) -> list[tuple[str, np.ndarray]]:
    paths = sorted(
        p for p in glob.glob(os.path.join(sample_dir, "*"))
        if p.lower().endswith(_IMAGE_EXTS)
    )
    if len(paths) < min_count:
        print(f"[!] {sample_dir}/ has {len(paths)} image(s), need >= {min_count}.")
        print(f"    Falling back to {min_count} synthetic test images.")
        print(f"    (Run 'python frame_extractor.py --max-frames {min_count}' for real frames instead.)\n")
        return [(f"synthetic_{i}", img) for i, img in enumerate(_synthetic_images(min_count))]

    loaded = []
    for p in paths:
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            continue
        loaded.append((os.path.basename(p), cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
    return loaded


def check_ollama(ollama_url: str) -> bool:
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if resp.status_code != 200:
            print(f"[X] Ollama returned status {resp.status_code}")
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
        qwen_models = [m for m in models if "qwen2.5-vl" in m.lower()]
        if not qwen_models:
            print("[X] No Qwen2.5-VL models found. Run: ollama pull qwen2.5-vl:3b")
            return False
        print(f"[OK] Ollama reachable at {ollama_url}, found: {', '.join(qwen_models)}\n")
        return True
    except requests.exceptions.ConnectionError:
        print(f"[X] Cannot connect to Ollama at {ollama_url} — is it running?")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch-test VLM image-in/text-out over >=5 sample photos")
    parser.add_argument("--dir", default="sample_frames", help="Directory of sample images")
    parser.add_argument("--min-count", type=int, default=5, help="Minimum number of images to test")
    parser.add_argument("--model", default="qwen2.5-vl:3b", help="Ollama model name")
    parser.add_argument("--prompt", default="Describe what is happening in this scene in one sentence.")
    args = parser.parse_args()

    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    print("=== Phase 1: VLM batch test ===\n")
    if not check_ollama(ollama_url):
        print("\nSetup incomplete — fix Ollama above and re-run.")
        return

    images = _load_sample_images(args.dir, args.min_count)
    analyzer = QwenVisionAnalyzer(model=args.model, base_url=ollama_url)

    results = []
    for name, img in images:
        print(f"[{name}]")
        t0 = time.perf_counter()
        try:
            text = analyzer.analyze_image(img, args.prompt, max_tokens=100)
            dt = time.perf_counter() - t0
            ok = bool(text) and not text.strip().lower().startswith(("error", "exception"))
            print(f"  -> {text}")
            print(f"  ({dt:.1f}s, {'sane' if ok else 'SUSPECT — looks like an error string'})\n")
            results.append(ok)
        except Exception as e:
            dt = time.perf_counter() - t0
            print(f"  -> ERROR: {e} ({dt:.1f}s)\n")
            results.append(False)

    passed = sum(results)
    print("=== Summary ===")
    print(f"{passed}/{len(results)} images returned a sane, non-error description.")
    if passed == len(results) and len(results) >= args.min_count:
        print("VLM reachability + sane-output check: PASSED")
    else:
        print("VLM reachability + sane-output check: FAILED — see errors above")


if __name__ == "__main__":
    main()
