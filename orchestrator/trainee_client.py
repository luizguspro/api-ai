
"""
Cliente Trainee para usar no Jupyter
"""
import requests
import time
import json

class Trainee:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        
    def train(self, input_path, output_path, git_hash, config=None, wait=False):
        """Submete job de treino."""
        
        response = requests.post(
            f"{self.api_url}/api/v1/training/train",
            json={
                "input_path": input_path,
                "output_path": output_path,
                "git_hash": git_hash,
                "config": config or {}
            }
        )
        
        result = response.json()
        job_id = result["job_id"]
        
        print(f"✓ Job submetido: {job_id}")
        
        if wait:
            return self.wait_for_completion(job_id)
        return job_id
    
    def get_status(self, job_id):
        """Verifica status do job."""
        response = requests.get(
            f"{self.api_url}/api/v1/training/status/{job_id}"
        )
        return response.json()
    
    def wait_for_completion(self, job_id):
        """Aguarda job completar."""
        print(f"Aguardando job {job_id}...")
        
        while True:
            status = self.get_status(job_id)
            
            if status["status"] == "completed":
                print("✓ Job concluído!")
                return status
            elif status["status"] == "failed":
                print("✗ Job falhou!")
                return status
            
            print(f"  Status: {status['status']}, Progress: {status['progress']}%")
            time.sleep(5)
    
    def list_jobs(self):
        """Lista todos os jobs."""
        response = requests.get(f"{self.api_url}/api/v1/jobs/list")
        return response.json()
