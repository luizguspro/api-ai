
#!/usr/bin/env python3
"""
Setup do Orquestrador - Cria toda estrutura de arquivos
"""
import os
from pathlib import Path

def create_file(filepath, content):
    """Cria arquivo com conteúdo."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    print(f"✓ Criado: {filepath}")

# Criar diretório base
base = Path("orchestrator")
base.mkdir(exist_ok=True)

print("="*50)
print("CRIANDO ESTRUTURA DO ORQUESTRADOR")
print("="*50)

# 1. Criar app.py principal
create_file("orchestrator/app.py", """#!/usr/bin/env python3
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="AI Training Orchestrator")

@app.get("/")
def root():
    return {"message": "AI Orchestrator Running"}

@app.post("/api/v1/training/train")
async def train(request: dict):
    return {
        "job_id": "job-123",
        "status": "queued",
        "message": "Job submitted successfully"
    }

@app.get("/api/v1/training/status/{job_id}")
async def get_status(job_id: str):
    return {
        "job_id": job_id,
        "status": "running",
        "progress": 45.0
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
""")

# 2. Criar requirements.txt
create_file("orchestrator/requirements.txt", """fastapi==0.104.1
uvicorn==0.24.0
redis==5.0.1
pydantic==2.4.2
pandas==2.1.3
requests==2.31.0
""")

# 3. Criar cliente Trainee
create_file("orchestrator/trainee.py", """import requests

class Trainee:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
    
    def train(self, input_path, output_path, git_hash):
        response = requests.post(
            f"{self.api_url}/api/v1/training/train",
            json={
                "input_path": input_path,
                "output_path": output_path,
                "git_hash": git_hash
            }
        )
        result = response.json()
        print(f"Job submitted: {result['job_id']}")
        return result['job_id']
    
    def get_status(self, job_id):
        response = requests.get(
            f"{self.api_url}/api/v1/training/status/{job_id}"
        )
        return response.json()
""")

# 4. Criar script de execução
create_file("orchestrator/run.sh", """#!/bin/bash
echo "Starting AI Orchestrator..."
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
""")

os.chmod("orchestrator/run.sh", 0o755)

print("\n" + "="*50)
print("✓ ESTRUTURA CRIADA COM SUCESSO!")
print("="*50)
print("\nPara executar o orquestrador:")
print("  cd orchestrator")
print("  ./run.sh")
print("\nOu manualmente:")
print("  cd orchestrator")
print("  python3 -m venv venv")
print("  source venv/bin/activate")
print("  pip install -r requirements.txt")
print("  python app.py")
