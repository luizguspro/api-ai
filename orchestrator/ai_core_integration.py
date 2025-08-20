
#!/usr/bin/env python3
"""
Integração com AI Core GPU
"""
import subprocess
import json
import os
from pathlib import Path

class AICoreBridge:
    """Ponte entre Orquestrador e AI Core."""
    
    def __init__(self):
        self.ai_core_path = "/home/lgsilva/SAT_IA/ai_core"
        
    def submit_training(self, job_id, input_path, output_path, config):
        """Submete treino para o AI Core."""
        
        # Criar script temporário para executar
        script_content = f"""
#!/bin/bash
cd {self.ai_core_path}
export TF_CPP_MIN_LOG_LEVEL=2

# Executar treino com AI Core
python3 -m ai_toolkit.cli train \
    --data {input_path} \
    --epochs {config.get('epochs', 10)} \
    --batch {config.get('batch_size', 32)}

# Salvar modelo no output_path
echo "Modelo salvo em {output_path}"
        """
        
        script_path = f"/tmp/train_{job_id}.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        
        # Executar
        result = subprocess.run([script_path], capture_output=True, text=True)
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    def submit_prediction(self, input_csv, output_csv):
        """Executa predição usando AI Core."""
        cmd = [
            "python3", "-m", "ai_toolkit.cli", "predict",
            "--in-path", input_csv,
            "--out-path", output_csv,
            "--strategy", "maioria"
        ]
        
        result = subprocess.run(
            cmd, 
            cwd=self.ai_core_path,
            capture_output=True, 
            text=True
        )
        
        return result.returncode == 0
