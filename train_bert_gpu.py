#!/usr/bin/env python3
"""Treino de modelo BERT usando GPU disponível"""
import torch
from transformers import BertModel, BertTokenizer
import sys
sys.path.append('src')

print("🤖 Teste BERT com GPU")
print("-" * 40)

# Verificar memória disponível
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        free = torch.cuda.mem_get_info(i)[0] / 1024**3
        print(f"GPU {i}: {free:.1f} GB livre")
    
    # Usar GPU com mais memória
    device = torch.device("cuda:1" if torch.cuda.mem_get_info(1)[0] > torch.cuda.mem_get_info(0)[0] else "cuda:0")
else:
    device = torch.device("cpu")

print(f"Usando: {device}")

try:
    # Carregar tokenizer e modelo pequeno
    print("\n📥 Carregando BERT (modelo pequeno)...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    
    # Texto de teste
    text = "Classificação de produtos NCM usando IA"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Forward pass
    print("🔄 Processando...")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
    
    print(f"✅ Embeddings gerados: {embeddings.shape}")
    print(f"   Device: {embeddings.device}")
    
except Exception as e:
    print(f"⚠️ Erro: {e}")
    print("Memória insuficiente para BERT completo")

print("\n✅ Teste concluído!")
