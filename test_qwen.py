"""Test script for Qwen2.5-VL integration via Ollama.

Usage:
    python test_qwen.py                          # Test with sample image
    python test_qwen.py --image path/to/img.jpg  # Test with custom image
    python test_qwen.py --camera                 # Test with webcam
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from backend.qwen_vision import QwenVisionAnalyzer


def create_test_image() -> np.ndarray:
    """Create a simple test image with colored shapes."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:, :] = [240, 240, 240]  # Light gray background
    
    # Draw shapes
    cv2.circle(img, (160, 240), 80, (255, 0, 0), -1)  # Blue circle
    cv2.rectangle(img, (280, 160), (440, 320), (0, 255, 0), -1)  # Green rectangle
    cv2.circle(img, (480, 120), 60, (0, 0, 255), -1)  # Red circle
    
    # Add text
    cv2.putText(img, "Test Image", (200, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    return img


def test_qwen_basic():
    """Test basic Qwen inference."""
    print("=== Testing Qwen2.5-VL Basic Inference ===\n")
    
    # Create analyzer
    analyzer = QwenVisionAnalyzer(model="qwen2.5-vl:3b")
    
    # Create test image
    img = create_test_image()
    print("Created test image (640x480 with shapes and text)")
    
    # Test prompts
    prompts = [
        "What shapes and colors do you see?",
        "Read any text visible in the image.",
        "Describe this image in one sentence.",
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}] Prompt: {prompt}")
        start = time.perf_counter()
        try:
            response = analyzer.analyze_image(img, prompt, max_tokens=100)
            elapsed = time.perf_counter() - start
            print(f"    Response: {response}")
            print(f"    Time: {elapsed:.2f}s")
        except Exception as e:
            print(f"    Error: {e}")


def test_qwen_from_file(image_path: str):
    """Test Qwen with an image file."""
    print(f"=== Testing Qwen with {image_path} ===\n")
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Load image
    img = Image.open(image_path).convert("RGB")
    print(f"Loaded image: {img.size[0]}x{img.size[1]}")
    
    # Create analyzer
    analyzer = QwenVisionAnalyzer(model="qwen2.5-vl:3b")
    
    # Analyze
    prompt = "Describe everything you see in this image in detail."
    print(f"\nPrompt: {prompt}")
    
    start = time.perf_counter()
    try:
        response = analyzer.analyze_image(img, prompt, max_tokens=200)
        elapsed = time.perf_counter() - start
        print(f"\nResponse:\n{response}")
        print(f"\nTime: {elapsed:.2f}s")
    except Exception as e:
        print(f"Error: {e}")


def test_qwen_camera():
    """Test Qwen with webcam feed."""
    print("=== Testing Qwen with Webcam ===\n")
    print("Press 'c' to capture and analyze, 'q' to quit")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Create analyzer
    analyzer = QwenVisionAnalyzer(model="qwen2.5-vl:3b")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break
        
        # Display frame
        cv2.imshow("Qwen Test - Press 'c' to analyze, 'q' to quit", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Analyze
            prompt = "Describe what you see in this camera frame."
            print(f"\n[Analyzing frame...]")
            start = time.perf_counter()
            try:
                response = analyzer.analyze_image(rgb_frame, prompt, max_tokens=100)
                elapsed = time.perf_counter() - start
                print(f"Response: {response}")
                print(f"Time: {elapsed:.2f}s\n")
            except Exception as e:
                print(f"Error: {e}\n")
    
    cap.release()
    cv2.destroyAllWindows()


def test_multi_frame():
    """Test multi-frame analysis."""
    print("=== Testing Multi-Frame Analysis ===\n")
    
    # Create 4 frames with progressive changes
    frames = []
    for i in range(4):
        img = create_test_image()
        # Add frame number
        cv2.putText(img, f"Frame {i+1}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # Move circle slightly
        cv2.circle(img, (160 + i*50, 240), 80, (255, 128, 0), -1)
        frames.append(img)
    
    print("Created 4 frames with progressive motion")
    
    # Analyze
    analyzer = QwenVisionAnalyzer(model="qwen2.5-vl:3b")
    prompt = "Describe how the scene changes across these frames."
    
    print(f"\nPrompt: {prompt}")
    start = time.perf_counter()
    try:
        response = analyzer.analyze_multiple_frames(frames, prompt, max_tokens=150)
        elapsed = time.perf_counter() - start
        print(f"\nResponse:\n{response}")
        print(f"\nTime: {elapsed:.2f}s")
    except Exception as e:
        print(f"Error: {e}")


def check_ollama():
    """Check if Ollama is running and has the required model."""
    import requests
    
    print("=== Checking Ollama Setup ===\n")
    
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    
    # Check if Ollama is running
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if resp.status_code == 200:
            print(f"✓ Ollama is running at {ollama_url}")
            
            # Check for Qwen models
            models = resp.json().get("models", [])
            qwen_models = [m["name"] for m in models if "qwen2.5-vl" in m["name"].lower()]
            
            if qwen_models:
                print(f"✓ Found Qwen models: {', '.join(qwen_models)}")
            else:
                print("✗ No Qwen2.5-VL models found")
                print("  Run: ollama pull qwen2.5-vl:3b")
                return False
        else:
            print(f"✗ Ollama returned status {resp.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to Ollama at {ollama_url}")
        print("  Make sure Ollama is running:")
        print("    - Windows/macOS: Should auto-start after installation")
        print("    - Linux: Run 'ollama serve'")
        return False
    except Exception as e:
        print(f"✗ Error checking Ollama: {e}")
        return False
    
    print()
    return True


def main():
    parser = argparse.ArgumentParser(description="Test Qwen2.5-VL integration")
    parser.add_argument("--image", help="Path to image file to analyze")
    parser.add_argument("--camera", action="store_true", help="Use webcam")
    parser.add_argument("--multi-frame", action="store_true", help="Test multi-frame analysis")
    parser.add_argument("--skip-check", action="store_true", help="Skip Ollama check")
    
    args = parser.parse_args()
    
    # Check Ollama setup
    if not args.skip_check:
        if not check_ollama():
            print("\nSetup incomplete. Please fix the issues above and try again.")
            return
    
    # Run requested test
    if args.image:
        test_qwen_from_file(args.image)
    elif args.camera:
        test_qwen_camera()
    elif args.multi_frame:
        test_multi_frame()
    else:
        test_qwen_basic()
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
