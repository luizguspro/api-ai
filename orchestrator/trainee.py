import requests

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
