#!/usr/bin/env python3
"""Limpa cache de memória das GPUs"""
import torch
import gc

print("🧹 Limpando memória das GPUs...")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        free = torch.cuda.mem_get_info(i)[0] / 1024**3
        total = torch.cuda.mem_get_info(i)[1] / 1024**3
        print(f"GPU {i}: {free:.1f} GB livre de {total:.1f} GB")

# Coletor de lixo Python
gc.collect()
print("✅ Limpeza concluída!")
