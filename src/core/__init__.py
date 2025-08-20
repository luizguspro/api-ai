"""Core utilities for AI Core"""
from .hdfs_adapter import HDFSAdapter

# Verificar GPU com PyTorch (funciona!)
GPU_AVAILABLE = False
GPU_COUNT = 0
GPU_INFO = []

try:
    import torch
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        GPU_COUNT = torch.cuda.device_count()
        for i in range(GPU_COUNT):
            GPU_INFO.append({
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_gb': torch.cuda.get_device_properties(i).total_memory / 1024**3
            })
        print(f"✅ {GPU_COUNT} GPU(s) detectada(s) via PyTorch")
except ImportError:
    print("⚠️ PyTorch não instalado - GPU desabilitada")

__all__ = ['HDFSAdapter', 'GPU_AVAILABLE', 'GPU_COUNT', 'GPU_INFO']
