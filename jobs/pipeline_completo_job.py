"""
Job Spark para Pipeline Completo GPU
"""
import argparse
import subprocess
import sys
from pathlib import Path

def run_phase(phase: str, args: list):
    """Executa uma fase do pipeline"""
    script = f"fase{phase}_*_job.py"
    script_path = list(Path(__file__).parent.glob(script))[0]
    
    cmd = ["python", str(script_path)] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Erro na fase {phase}: {result.stderr}")
        sys.exit(1)
    
    print(f"Fase {phase} concluída")
    return result.stdout

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    
    print("=== Pipeline AI Core GPU ===")
    
    # Fase 1: Preparação
    print("\n[1/3] Preparação de dados...")
    run_phase("1", [
        "--input", args.input,
        "--output", f"{args.output}/preparacao",
        "--config", args.config
    ])
    
    # Fase 2: Treinamento
    print("\n[2/3] Treinamento com GPU...")
    run_phase("2", [
        "--input", f"{args.output}/preparacao",
        "--output", f"{args.output}/modelos",
        "--config", args.config
    ])
    
    # Fase 3: Predição
    print("\n[3/3] Predição e ensemble...")
    run_phase("3", [
        "--input", args.input,
        "--models", f"{args.output}/modelos",
        "--output", f"{args.output}/predicoes",
        "--config", args.config,
        "--strategy", "maioria"
    ])
    
    print("\n=== Pipeline Completo! ===")
    print(f"Resultados em: {args.output}")

if __name__ == "__main__":
    main()
