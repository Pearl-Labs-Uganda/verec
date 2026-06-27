import time
print("Importing cv2...")
start = time.perf_counter()
import cv2
print(f"cv2: {time.perf_counter() - start:.2f}s")

print("Importing numpy...")
start = time.perf_counter()
import numpy as np
print(f"numpy: {time.perf_counter() - start:.2f}s")

print("Importing PIL...")
start = time.perf_counter()
from PIL import Image
print(f"PIL: {time.perf_counter() - start:.2f}s")

print("Importing camera...")
start = time.perf_counter()
from camera import OpenCVCamera
print(f"camera: {time.perf_counter() - start:.2f}s")

print("Importing detectors...")
start = time.perf_counter()
from detectors import COCO_CLASSES, YOLODetector, YOLOPoseDetector
print(f"detectors: {time.perf_counter() - start:.2f}s")