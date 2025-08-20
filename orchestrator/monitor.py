
#!/usr/bin/env python3
"""
Monitor de Jobs do Orquestrador
"""
import requests
import time
import json
from datetime import datetime

def monitor_jobs():
    """Monitora status dos jobs."""
    api_url = "http://localhost:8000"
    
    while True:
        try:
            # Buscar lista de jobs
            response = requests.get(f"{api_url}/api/v1/jobs/list")
            jobs = response.json()
            
            # Limpar tela
            print("\033[2J\033[H")  # Clear screen
            print("="*60)
            print(f"MONITOR DE JOBS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*60)
            print(f"Total de Jobs: {len(jobs)}")
            print("")
            
            # Status summary
            status_count = {}
            for job in jobs:
                status = job['status']
                status_count[status] = status_count.get(status, 0) + 1
            
            print("Resumo por Status:")
            for status, count in status_count.items():
                print(f"  {status}: {count}")
            
            print("\nJobs Detalhados:")
            print("-"*60)
            
            for job in jobs:
                print(f"ID: {job['id']}")
                print(f"  Status: {job['status']}")
                print(f"  Progress: {job['progress']}%")
                print(f"  Input: {job['input_path']}")
                print(f"  Created: {job['created_at']}")
                print("")
            
            time.sleep(5)  # Atualizar a cada 5 segundos
            
        except KeyboardInterrupt:
            print("\nMonitor encerrado")
            break
        except Exception as e:
            print(f"Erro: {e}")
            time.sleep(10)

if __name__ == "__main__":
    monitor_jobs()
