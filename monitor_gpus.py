#!/usr/bin/env python3
"""Monitor de GPUs em tempo real"""
import torch
import subprocess
import time

def get_gpu_processes():
    """Obtém processos usando GPU via nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,name,used_memory', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        return result.stdout.strip()
    except:
        return ""

print("🖥️  MONITOR DE GPUs - NVIDIA L40S")
print("=" * 60)

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Total de GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        print(f"\n📊 GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Memória
        mem_free, mem_total = torch.cuda.mem_get_info(i)
        mem_used = mem_total - mem_free
        
        print(f"   Memória Total: {mem_total/1024**3:.1f} GB")
        print(f"   Memória Usada: {mem_used/1024**3:.1f} GB")
        print(f"   Memória Livre: {mem_free/1024**3:.1f} GB")
        print(f"   Uso: {(mem_used/mem_total)*100:.1f}%")
    
    print("\n📝 Processos usando GPU:")
    processes = get_gpu_processes()
    if processes:
        for line in processes.split('\n'):
            if line:
                print(f"   {line}")
    else:
        print("   Nenhum processo detectado")
else:
    print("❌ Nenhuma GPU disponível")

print("\n" + "=" * 60)
print("💡 Dica: GPU 1 geralmente tem mais memória livre!")
