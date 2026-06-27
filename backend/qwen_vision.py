"""Qwen2.5-VL integration via Ollama — drop-in replacement for local VLM.

This module provides a clean async-friendly wrapper around Ollama's
OpenAI-compatible vision API, allowing Qwen2.5-VL to run as a separate
service while keeping your main process lean.
"""
from __future__ import annotations

import base64
import io
import os
from typing import Optional

import numpy as np
from PIL import Image
from openai import OpenAI


class QwenVisionAnalyzer:
    """Client for Qwen2.5-VL running via Ollama's OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str = "qwen2.5-vl:3b",
        base_url: Optional[str] = None,
    ):
        """Initialize the Qwen vision analyzer.

        Args:
            model: Ollama model name (e.g., "qwen2.5-vl:3b", "qwen2.5-vl:7b")
            base_url: Ollama server URL (defaults to OLLAMA_URL env or localhost:11434)
        """
        self.model = model
        self.base_url = base_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
        # Ollama's OpenAI-compatible endpoint is at /v1
        self.client = OpenAI(base_url=f"{self.base_url}/v1", api_key="ollama")

    def analyze_image(
        self,
        image: np.ndarray | Image.Image,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 100,
    ) -> str:
        """Send an image and prompt to Qwen via Ollama.

        Args:
            image: RGB image as numpy array or PIL Image
            prompt: Text prompt describing what to analyze
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated caption/description text
        """
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_img = Image.fromarray(image).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        # Encode as JPEG base64
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Call Ollama's OpenAI-compatible vision endpoint
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                    ],
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def analyze_multiple_frames(
        self,
        frames: list[np.ndarray | Image.Image],
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 150,
    ) -> str:
        """Analyze multiple frames together (for temporal understanding).

        Args:
            frames: List of RGB images (numpy arrays or PIL Images)
            prompt: Text prompt describing what to analyze across frames
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated description considering all frames
        """
        # Build content with multiple images
        content: list[dict] = [{"type": "text", "text": prompt}]

        for frame in frames:
            # Convert to PIL and encode
            if isinstance(frame, np.ndarray):
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                pil_img = Image.fromarray(frame).convert("RGB")
            else:
                pil_img = frame.convert("RGB")

            buffered = io.BytesIO()
            pil_img.save(buffered, format="JPEG", quality=80)
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                }
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
