#!/usr/bin/env python3
"""Benchmark seguro das GPUs"""
import torch
import time

print("🏃 Benchmark GPU - NVIDIA L40S")
print("-" * 40)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

if device.type == 'cuda':
    print(f"GPUs disponíveis: {torch.cuda.device_count()}")
    
    for gpu_id in range(torch.cuda.device_count()):
        print(f"\n📊 GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        
        # Verificar memória disponível
        torch.cuda.set_device(gpu_id)
        mem_free = torch.cuda.mem_get_info(gpu_id)[0] / 1024**3
        mem_total = torch.cuda.mem_get_info(gpu_id)[1] / 1024**3
        print(f"   Memória: {mem_free:.1f} GB livre de {mem_total:.1f} GB total")
        
        if mem_free < 2.0:
            print(f"   ⚠️  Memória insuficiente, pulando GPU {gpu_id}")
            continue
        
        # Usar matriz menor para teste seguro
        size = 5000  # Reduzido de 10000
        try:
            a = torch.randn(size, size, device=f'cuda:{gpu_id}', dtype=torch.float32)
            b = torch.randn(size, size, device=f'cuda:{gpu_id}', dtype=torch.float32)
            
            # Aquecer GPU
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            
            # Benchmark
            start = time.time()
            iterations = 5
            for _ in range(iterations):
                c = torch.matmul(a, b)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            print(f"   ✅ Tempo para {iterations} multiplicações {size}x{size}: {elapsed:.2f}s")
            print(f"   📈 TFLOPS: {(2 * size**3 * iterations / elapsed) / 1e12:.2f}")
            
            # Limpar memória
            del a, b, c
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"   ❌ Sem memória suficiente nesta GPU")
            torch.cuda.empty_cache()

print("\n✅ Benchmark concluído!")
