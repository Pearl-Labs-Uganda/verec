import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU devices: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # Quick tensor test
    x = torch.randn(1000, 1000).cuda()
    print("✅ Successfully moved tensor to GPU")
else:
    print("❌ CUDA not detected - reinstalling PyTorch with CUDA support")
