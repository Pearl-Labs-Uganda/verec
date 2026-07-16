"""Phase 2 — Video as a stack of images.

Standalone frame extractor: opens a video file, pulls out every Nth frame,
and saves each as a JPEG. Deliberately decoupled from any VLM call — this
only proves frame extraction works, nothing here describes what's in a frame.

Usage:
    python frame_extractor.py
    python frame_extractor.py --video test_videos/sample.mp4 --every-n 15
    python frame_extractor.py --max-frames 5
"""
from __future__ import annotations

import argparse
import os

import cv2


def extract_frames(video_path: str, every_n: int, out_dir: str, max_frames: int | None = None) -> int:
    """Extract every Nth frame from video_path into out_dir as JPEGs.

    Returns the number of frames written.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_idx = 0
    saved = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break  # end of file
            if frame_idx % every_n == 0:
                out_path = os.path.join(out_dir, f"frame_{saved:04d}.jpg")
                cv2.imwrite(out_path, frame)
                saved += 1
                if max_frames is not None and saved >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap.release()

    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract every Nth frame from a video file")
    parser.add_argument("--video", default="test_videos/traffic.mp4", help="Path to source video")
    parser.add_argument("--every-n", type=int, default=30, help="Save one frame every N frames")
    parser.add_argument("--out-dir", default="sample_frames", help="Output directory for extracted JPEGs")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after this many saved frames")
    args = parser.parse_args()

    print(f"=== Extracting frames from {args.video} (every {args.every_n} frames) ===\n")
    saved = extract_frames(args.video, args.every_n, args.out_dir, args.max_frames)
    print(f"Total frames extracted: {saved}")
    print(f"Saved to: {os.path.abspath(args.out_dir)}/")


if __name__ == "__main__":
    main()
