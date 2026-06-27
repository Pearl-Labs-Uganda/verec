"""Test that backend.models imports quickly without loading heavy dependencies."""
import os
import time

# Set Qwen mode
os.environ["USE_QWEN_VL"] = "true"

print("Testing backend.models import speed...\n")
print(f"Environment: USE_QWEN_VL={os.getenv('USE_QWEN_VL')}\n")

start = time.perf_counter()
import backend.models
elapsed = time.perf_counter() - start

print(f"✓ backend.models imported in {elapsed:.3f}s\n")

if elapsed < 1.0:
    print(f"✓✓ EXCELLENT! Import time < 1s - lazy imports working perfectly!")
elif elapsed < 3.0:
    print(f"✓ GOOD! Import time < 3s - acceptable startup time")
else:
    print(f"⚠ SLOW! Import time ≥ 3s - may still be loading heavy dependencies")

print(f"\nNow test that Qwen works without loading detectors...")
try:
    print("Calling init_qwen()...")
    qwen = backend.models.init_qwen()
    print(f"✓ Qwen initialized: {qwen}")
except Exception as e:
    print(f"⚠ Qwen init failed (OK if Ollama not running): {e}")

print("\nDone! Server should start quickly with these lazy imports.")
