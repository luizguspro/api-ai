#!/usr/bin/env python3
"""Teste selecionando GPU específica"""
import torch
import sys
sys.path.append('src')

print("🎮 Teste de seleção de GPU")
print("-" * 40)

# Verificar GPUs disponíveis
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"GPUs detectadas: {num_gpus}")
    
    # Escolher GPU com mais memória livre
    best_gpu = 0
    max_free_mem = 0
    
    for i in range(num_gpus):
        free_mem = torch.cuda.mem_get_info(i)[0]
        print(f"GPU {i}: {free_mem/1024**3:.1f} GB livre")
        if free_mem > max_free_mem:
            max_free_mem = free_mem
            best_gpu = i
    
    print(f"\n✅ Usando GPU {best_gpu} (mais memória livre)")
    
    # Configurar para usar GPU específica
    torch.cuda.set_device(best_gpu)
    device = torch.device(f'cuda:{best_gpu}')
    
    # Teste simples
    print("\n📊 Teste de processamento:")
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = torch.matmul(x, y)
    print(f"   Multiplicação de matrizes 1000x1000: ✅")
    print(f"   Device: {z.device}")
    print(f"   Shape: {z.shape}")
    
    # Limpar memória
    del x, y, z
    torch.cuda.empty_cache()
    
else:
    print("❌ GPU não disponível")

print("\n✅ Teste concluído!")
