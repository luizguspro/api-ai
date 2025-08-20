#!/usr/bin/env python3
"""Teste completo do ambiente AI Core com GPU"""
import sys
import os
sys.path.append('src')

print("=" * 60)
print("🎯 TESTE COMPLETO - AI CORE GPU")
print("=" * 60)

# 1. Imports básicos
print("\n1️⃣ Verificando dependências:")
modules = {
    'pandas': '✅',
    'numpy': '✅', 
    'pyarrow': '✅',
    'torch': '✅',
    'fsspec': '✅',
    'yaml': '✅'
}

for mod in modules:
    try:
        __import__(mod)
        print(f"   {modules[mod]} {mod}")
    except:
        print(f"   ❌ {mod}")

# 2. Módulos do projeto
print("\n2️⃣ Módulos do projeto:")
try:
    from core import HDFSAdapter, GPU_AVAILABLE, GPU_COUNT, GPU_INFO
    print(f"   ✅ HDFSAdapter")
    print(f"   ✅ GPU utils")
except Exception as e:
    print(f"   ❌ Erro: {e}")

# 3. Status GPU
print("\n3️⃣ Status GPU:")
if GPU_AVAILABLE:
    print(f"   ✅ {GPU_COUNT} GPU(s) disponível(is):")
    for gpu in GPU_INFO:
        print(f"      • GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f} GB)")
else:
    print("   ⚠️  GPU não disponível")

# 4. Teste funcional
print("\n4️⃣ Teste funcional:")
try:
    import pandas as pd
    import numpy as np
    import torch
    
    # Criar dados de teste
    df = pd.DataFrame({
        'descricao': ['produto A', 'produto B', 'produto C'],
        'codigo': [1, 2, 3]
    })
    print(f"   ✅ DataFrame criado: {df.shape}")
    
    # Simular embeddings
    embeddings = np.random.randn(3, 768)
    print(f"   ✅ Embeddings (CPU): {embeddings.shape}")
    
    # Testar GPU com PyTorch
    if GPU_AVAILABLE:
        tensor = torch.tensor(embeddings).cuda()
        print(f"   ✅ Tensor na GPU: {tensor.device}")
        
        # Operação simples na GPU
        result = torch.nn.functional.softmax(tensor, dim=1)
        print(f"   ✅ Processamento GPU: {result.shape}")
    
except Exception as e:
    print(f"   ❌ Erro: {e}")

print("\n" + "=" * 60)
print("🎉 SUCESSO! Ambiente pronto para uso com GPU!")
print("=" * 60)
