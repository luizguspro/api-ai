
#!/usr/bin/env python3
from fastapi import FastAPI, BackgroundTasks
import uvicorn
import uuid
from datetime import datetime
from typing import Dict
from ai_core_integration import AICoreBridge

app = FastAPI(title="AI Training Orchestrator")
bridge = AICoreBridge()

# Armazenamento simples em memória
jobs_db = {}

@app.get("/")
def root():
    return {
        "service": "AI Training Orchestrator",
        "status": "running",
        "version": "2.0",
        "ai_core_connected": True
    }

@app.post("/api/v1/training/train")
async def train(request: Dict, background_tasks: BackgroundTasks):
    # Criar job
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    
    job = {
        "id": job_id,
        "input_path": request.get("input_path"),
        "output_path": request.get("output_path"),
        "git_hash": request.get("git_hash"),
        "config": request.get("config", {}),
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "progress": 0
    }
    
    jobs_db[job_id] = job
    
    # Executar em background
    background_tasks.add_task(execute_training, job_id)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Job submitted successfully"
    }

def execute_training(job_id: str):
    """Executa treino em background."""
    job = jobs_db[job_id]
    job["status"] = "running"
    
    # Chamar AI Core
    result = bridge.submit_training(
        job_id,
        job["input_path"],
        job["output_path"],
        job["config"]
    )
    
    if result["success"]:
        job["status"] = "completed"
        job["progress"] = 100
    else:
        job["status"] = "failed"
        job["error"] = result["stderr"]
    
    job["completed_at"] = datetime.now().isoformat()

@app.get("/api/v1/training/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs_db:
        return {"error": "Job not found"}
    
    job = jobs_db[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "created_at": job["created_at"],
        "completed_at": job.get("completed_at")
    }

@app.get("/api/v1/jobs/list")
async def list_jobs():
    return list(jobs_db.values())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
