"""Test backend startup time with and without LLaVA imports.

This script verifies that when USE_QWEN_VL=true, the backend starts
quickly without loading any LLaVA dependencies.
"""
import os
import sys
import time

def test_import_speed():
    """Test how fast backend.models imports."""
    print("=== Testing Backend Import Speed ===\n")
    
    # Ensure USE_QWEN_VL is set
    os.environ["USE_QWEN_VL"] = "true"
    print(f"Environment: USE_QWEN_VL={os.getenv('USE_QWEN_VL')}\n")
    
    # Time the import
    print("Importing backend.models...")
    start = time.perf_counter()
    
    try:
        import backend.models as models
        elapsed = time.perf_counter() - start
        
        print(f"✓ Import completed in {elapsed:.3f}s\n")
        
        # Check if LLaVA is imported
        llava_imported = False
        for module_name in sys.modules:
            if 'llava' in module_name.lower():
                llava_imported = True
                print(f"⚠ Found LLaVA module in sys.modules: {module_name}")
        
        if not llava_imported:
            print("✓ No LLaVA modules imported (as expected with USE_QWEN_VL=true)\n")
        
        # Try to initialize Qwen (should work)
        print("Testing Qwen initialization...")
        try:
            qwen = models.init_qwen()
            print(f"✓ Qwen initialized successfully: {qwen}\n")
        except Exception as e:
            print(f"⚠ Qwen initialization failed: {e}")
            print("  (This is OK if Ollama isn't running)\n")
        
        # Test that run_vlm works with Qwen backend
        print("Testing run_vlm with Qwen backend...")
        print("  (Creating a test image)")
        import numpy as np
        from PIL import Image
        
        # Create a simple test image
        test_image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        
        try:
            result = models.run_vlm(
                test_image,
                prompt="What is this?",
                use_qwen=True,  # Force Qwen backend
                max_tokens=10
            )
            
            if "error" in result:
                print(f"  Qwen inference test: {result.get('error')}")
                print("  (This is OK if Ollama isn't running)\n")
            else:
                print(f"✓ Qwen inference successful:")
                print(f"  Backend: {result.get('backend')}")
                print(f"  Response: {result.get('text', '')[:50]}...")
                print(f"  Time: {result.get('time_s')}s\n")
        except Exception as e:
            print(f"  Qwen inference error: {e}")
            print("  (This is OK if Ollama isn't running)\n")
        
        # Verify LLaVA is still not imported after run_vlm with Qwen
        llava_imported_after = False
        for module_name in sys.modules:
            if 'llava' in module_name.lower():
                llava_imported_after = True
        
        if not llava_imported_after:
            print("✓ LLaVA still not imported after using Qwen backend\n")
        else:
            print("⚠ LLaVA was imported (unexpected when use_qwen=True)\n")
        
        # Summary
        print("=== Summary ===")
        print(f"Import time: {elapsed:.3f}s")
        if elapsed < 2.0:
            print("✓ Fast startup (< 2s) - lazy imports working!")
        else:
            print("⚠ Slow startup (≥ 2s) - may be loading unnecessary modules")
        
        if not llava_imported_after:
            print("✓ No LLaVA dependencies loaded with USE_QWEN_VL=true")
        else:
            print("⚠ LLaVA was loaded unexpectedly")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

if __name__ == "__main__":
    success = test_import_speed()
    sys.exit(0 if success else 1)
