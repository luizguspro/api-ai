#!/usr/bin/env python3
"""Benchmark das GPUs"""
import torch
import time

print("🏃 Benchmark GPU - NVIDIA L40S")
print("-" * 40)

# Configurar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

if device.type == 'cuda':
    print(f"GPUs disponíveis: {torch.cuda.device_count()}")
    
    # Testar cada GPU
    for gpu_id in range(torch.cuda.device_count()):
        torch.cuda.set_device(gpu_id)
        print(f"\n📊 GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        
        # Criar matriz grande
        size = 10000
        a = torch.randn(size, size, device=f'cuda:{gpu_id}')
        b = torch.randn(size, size, device=f'cuda:{gpu_id}')
        
        # Aquecer GPU
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"   Tempo para 10 multiplicações {size}x{size}: {elapsed:.2f}s")
        print(f"   TFLOPS: {(2 * size**3 * 10 / elapsed) / 1e12:.2f}")
        print(f"   Memória usada: {torch.cuda.memory_allocated(gpu_id) / 1024**3:.2f} GB")

print("\n✅ Benchmark concluído!")
