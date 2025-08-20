#!/usr/bin/env python3
"""
Script para criar o Orquestrador de Treino (lado esquerdo).
Gera API, fila, integração YARN e controle de jobs.
"""

import os
import json
from pathlib import Path

def create_file(path, content):
    """Cria arquivo com conteúdo."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"✓ Criado: {path}")

def main():
    print("="*60)
    print("CRIANDO ORQUESTRADOR DE TREINO - LADO ESQUERDO")
    print("="*60)
    
    base_dir = Path("orchestrator")
    
    # 1. ESTRUTURA DE DIRETÓRIOS
    dirs = [
        "api", "api/routes", "api/models", "api/services",
        "queue", "queue/backends", 
        "executor", "executor/yarn",
        "storage", "storage/hdfs",
        "monitor", "monitor/metrics",
        "config", "tests", "scripts", "logs"
    ]
    
    for d in dirs:
        (base_dir / d).mkdir(parents=True, exist_ok=True)
    
    # 2. ARQUIVO PRINCIPAL - APP.PY
    create_file(base_dir / "app.py", '''#!/usr/bin/env python3
"""
Aplicação principal do Orquestrador.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import training, jobs, models, health
from queue.manager import QueueManager
from executor.scheduler import JobScheduler
from monitor.tracker import MetricsTracker
import uvicorn
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/orchestrator.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Criar aplicação FastAPI
app = FastAPI(
    title="AI Training Orchestrator",
    description="Orquestrador de treino com integração YARN e GPU",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar rotas
app.include_router(training.router, prefix="/api/v1/training", tags=["training"])
app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["jobs"])
app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
app.include_router(health.router, prefix="/api/v1/health", tags=["health"])

# Inicializar componentes
queue_manager = QueueManager()
scheduler = JobScheduler(queue_manager)
metrics = MetricsTracker()

@app.on_event("startup")
async def startup_event():
    """Inicialização da aplicação."""
    logger.info("Iniciando Orquestrador...")
    await queue_manager.initialize()
    await scheduler.start()
    await metrics.start()
    logger.info("✓ Orquestrador iniciado")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown da aplicação."""
    logger.info("Parando Orquestrador...")
    await scheduler.stop()
    await queue_manager.close()
    await metrics.stop()
    logger.info("✓ Orquestrador parado")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
''')

    # 3. API - ROTAS DE TRAINING
    create_file(base_dir / "api/routes/training.py", '''"""
Rotas de treinamento.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
import uuid
import logging
from ..services.job_service import JobService
from ..models.job import Job, JobStatus, JobPriority

router = APIRouter()
logger = logging.getLogger(__name__)
job_service = JobService()

class TrainRequest(BaseModel):
    """Request para treino."""
    input_path: str  # Caminho HDFS dos dados
    output_path: str  # Onde salvar modelo
    git_hash: str  # Hash do commit do modelo
    config: Dict[str, Any] = {}  # Configurações do treino
    priority: str = "normal"  # low, normal, high, critical
    user: str = "default"
    description: Optional[str] = None

class TrainResponse(BaseModel):
    """Response do treino."""
    job_id: str
    status: str
    message: str
    created_at: datetime
    estimated_wait: Optional[int] = None  # segundos

@router.post("/train", response_model=TrainResponse)
async def train(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Endpoint principal para submeter job de treino.
    Chamado pelo Jupyter via Trainee.train().
    """
    try:
        # Criar job
        job = Job(
            id=str(uuid.uuid4()),
            user=request.user,
            type="training",
            input_path=request.input_path,
            output_path=request.output_path,
            git_hash=request.git_hash,
            config=request.config,
            priority=JobPriority[request.priority.upper()],
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            description=request.description
        )
        
        # Adicionar à fila
        position = await job_service.submit_job(job)
        
        # Estimar tempo de espera
        estimated_wait = await job_service.estimate_wait_time(job.id)
        
        logger.info(f"Job {job.id} submetido por {request.user}")
        
        return TrainResponse(
            job_id=job.id,
            status="queued",
            message=f"Job adicionado à fila (posição {position})",
            created_at=job.created_at,
            estimated_wait=estimated_wait
        )
        
    except Exception as e:
        logger.error(f"Erro ao submeter job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Retorna status de um job."""
    try:
        job = await job_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job não encontrado")
        
        return {
            "job_id": job.id,
            "status": job.status.value,
            "progress": job.progress,
            "logs": job.logs[-10:] if job.logs else [],  # últimas 10 linhas
            "metrics": job.metrics,
            "error": job.error
        }
    except Exception as e:
        logger.error(f"Erro ao buscar status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Cancela um job."""
    try:
        success = await job_service.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail="Job não encontrado ou já finalizado")
        
        return {"message": f"Job {job_id} cancelado"}
    except Exception as e:
        logger.error(f"Erro ao cancelar job: {e}")
        raise HTTPException(status_code=500, detail=str(e))
''')

    # 4. MODELOS DE DADOS
    create_file(base_dir / "api/models/job.py", '''"""
Modelos de dados para jobs.
"""
from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

class JobStatus(Enum):
    """Status possíveis de um job."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobPriority(Enum):
    """Prioridades de execução."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

class Job(BaseModel):
    """Modelo de Job."""
    id: str
    user: str
    type: str  # training, evaluation, prediction
    input_path: str
    output_path: str
    git_hash: str
    config: Dict[str, Any]
    priority: JobPriority
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    description: Optional[str] = None
    
    # Recursos
    gpu_required: bool = True
    cpu_cores: int = 4
    memory_gb: int = 16
    
    # Progresso
    progress: float = 0.0  # 0-100
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    
    # Resultados
    metrics: Dict[str, Any] = {}
    logs: List[str] = []
    error: Optional[str] = None
    
    # YARN
    yarn_app_id: Optional[str] = None
    yarn_tracking_url: Optional[str] = None
    
    class Config:
        use_enum_values = True
''')

    # 5. GERENCIADOR DE FILA
    create_file(base_dir / "queue/manager.py", '''"""
Gerenciador de fila de jobs.
"""
import asyncio
from typing import List, Optional, Dict
from datetime import datetime
import redis
import json
import logging
from api.models.job import Job, JobStatus, JobPriority

logger = logging.getLogger(__name__)

class QueueManager:
    """Gerencia fila de jobs com Redis."""
    
    def __init__(self):
        self.redis_client = None
        self.queues = {
            JobPriority.CRITICAL: "queue:critical",
            JobPriority.HIGH: "queue:high",
            JobPriority.NORMAL: "queue:normal",
            JobPriority.LOW: "queue:low"
        }
        
    async def initialize(self):
        """Inicializa conexão com Redis."""
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("✓ Conectado ao Redis")
        except Exception as e:
            logger.error(f"Erro ao conectar Redis: {e}")
            # Fallback para fila em memória
            self.redis_client = None
            self.memory_queue = []
    
    async def add_job(self, job: Job) -> int:
        """Adiciona job à fila apropriada."""
        queue_name = self.queues[job.priority]
        job_data = job.json()
        
        if self.redis_client:
            # Adiciona à fila do Redis
            position = self.redis_client.rpush(queue_name, job_data)
            # Salva metadados
            self.redis_client.hset(f"job:{job.id}", mapping={
                "data": job_data,
                "status": job.status.value,
                "queue": queue_name
            })
        else:
            # Fila em memória
            self.memory_queue.append(job)
            position = len(self.memory_queue)
        
        logger.info(f"Job {job.id} adicionado à fila {queue_name} (posição {position})")
        return position
    
    async def get_next_job(self) -> Optional[Job]:
        """Retorna próximo job a executar (por prioridade)."""
        # Verifica filas por ordem de prioridade
        for priority in [JobPriority.CRITICAL, JobPriority.HIGH, 
                         JobPriority.NORMAL, JobPriority.LOW]:
            queue_name = self.queues[priority]
            
            if self.redis_client:
                job_data = self.redis_client.lpop(queue_name)
                if job_data:
                    job = Job.parse_raw(job_data)
                    # Atualiza status
                    self.redis_client.hset(f"job:{job.id}", "status", JobStatus.RUNNING.value)
                    return job
            else:
                # Fila em memória
                if self.memory_queue:
                    # Ordena por prioridade
                    self.memory_queue.sort(key=lambda j: j.priority.value, reverse=True)
                    return self.memory_queue.pop(0)
        
        return None
    
    async def get_queue_status(self) -> Dict:
        """Retorna status das filas."""
        status = {}
        
        if self.redis_client:
            for priority, queue_name in self.queues.items():
                status[priority.name] = self.redis_client.llen(queue_name)
        else:
            # Conta jobs por prioridade na memória
            for priority in JobPriority:
                count = sum(1 for j in self.memory_queue if j.priority == priority)
                status[priority.name] = count
        
        return status
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Busca job por ID."""
        if self.redis_client:
            job_data = self.redis_client.hget(f"job:{job_id}", "data")
            if job_data:
                return Job.parse_raw(job_data)
        else:
            # Busca na memória
            for job in self.memory_queue:
                if job.id == job_id:
                    return job
        return None
    
    async def update_job(self, job: Job):
        """Atualiza job."""
        if self.redis_client:
            self.redis_client.hset(f"job:{job.id}", mapping={
                "data": job.json(),
                "status": job.status.value
            })
        # Em memória, o objeto já é atualizado por referência
    
    async def remove_job(self, job_id: str) -> bool:
        """Remove job da fila."""
        if self.redis_client:
            # Remove de todas as filas
            for queue_name in self.queues.values():
                # Busca e remove
                jobs = self.redis_client.lrange(queue_name, 0, -1)
                for job_data in jobs:
                    job = Job.parse_raw(job_data)
                    if job.id == job_id:
                        self.redis_client.lrem(queue_name, 1, job_data)
                        self.redis_client.delete(f"job:{job_id}")
                        return True
        else:
            # Remove da memória
            self.memory_queue = [j for j in self.memory_queue if j.id != job_id]
            return True
        
        return False
    
    async def close(self):
        """Fecha conexões."""
        if self.redis_client:
            self.redis_client.close()
''')

    # 6. SCHEDULER DE JOBS
    create_file(base_dir / "executor/scheduler.py", '''"""
Scheduler de jobs com integração YARN.
"""
import asyncio
import logging
from datetime import datetime
from typing import Optional
from queue.manager import QueueManager
from api.models.job import Job, JobStatus
from .yarn.yarn_client import YarnClient
from .gpu_executor import GPUExecutor

logger = logging.getLogger(__name__)

class JobScheduler:
    """Agenda e executa jobs."""
    
    def __init__(self, queue_manager: QueueManager):
        self.queue_manager = queue_manager
        self.yarn_client = YarnClient()
        self.gpu_executor = GPUExecutor()
        self.running = False
        self.current_job: Optional[Job] = None
        self.max_concurrent_jobs = 2  # Máximo de jobs simultâneos
        self.running_jobs = []
        
    async def start(self):
        """Inicia scheduler."""
        self.running = True
        logger.info("Scheduler iniciado")
        # Inicia loop de processamento
        asyncio.create_task(self._process_loop())
        
    async def stop(self):
        """Para scheduler."""
        self.running = False
        logger.info("Scheduler parado")
        
    async def _process_loop(self):
        """Loop principal de processamento."""
        while self.running:
            try:
                # Verifica recursos disponíveis
                if len(self.running_jobs) < self.max_concurrent_jobs:
                    if await self._has_available_resources():
                        # Pega próximo job
                        job = await self.queue_manager.get_next_job()
                        if job:
                            # Executa job
                            asyncio.create_task(self._execute_job(job))
                
                # Aguarda antes de verificar novamente
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Erro no scheduler: {e}")
                await asyncio.sleep(10)
    
    async def _has_available_resources(self) -> bool:
        """Verifica se há recursos disponíveis."""
        try:
            # Verifica GPU disponível
            gpu_available = await self.gpu_executor.check_gpu_available()
            
            # Verifica recursos YARN
            yarn_resources = await self.yarn_client.get_cluster_metrics()
            
            # Precisa ter pelo menos 8GB RAM e 4 cores livres
            memory_available = yarn_resources.get("availableMemoryMB", 0) > 8192
            cores_available = yarn_resources.get("availableVirtualCores", 0) > 4
            
            return gpu_available and memory_available and cores_available
            
        except Exception as e:
            logger.error(f"Erro ao verificar recursos: {e}")
            return False
    
    async def _execute_job(self, job: Job):
        """Executa um job."""
        try:
            logger.info(f"Iniciando execução do job {job.id}")
            self.running_jobs.append(job)
            
            # Atualiza status
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            await self.queue_manager.update_job(job)
            
            # Submete ao YARN
            yarn_app_id = await self.yarn_client.submit_job(job)
            job.yarn_app_id = yarn_app_id
            
            # Monitora execução
            success = await self._monitor_job(job)
            
            if success:
                job.status = JobStatus.COMPLETED
                logger.info(f"Job {job.id} completado com sucesso")
            else:
                job.status = JobStatus.FAILED
                logger.error(f"Job {job.id} falhou")
            
            job.completed_at = datetime.now()
            await self.queue_manager.update_job(job)
            
        except Exception as e:
            logger.error(f"Erro ao executar job {job.id}: {e}")
            job.status = JobStatus.FAILED
            job.error = str(e)
            await self.queue_manager.update_job(job)
            
        finally:
            self.running_jobs.remove(job)
    
    async def _monitor_job(self, job: Job) -> bool:
        """Monitora execução do job."""
        max_wait = 3600 * 6  # 6 horas máximo
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < max_wait:
            try:
                # Verifica status no YARN
                status = await self.yarn_client.get_job_status(job.yarn_app_id)
                
                if status["state"] == "FINISHED":
                    return status["finalStatus"] == "SUCCEEDED"
                elif status["state"] == "FAILED":
                    job.error = status.get("diagnostics", "Unknown error")
                    return False
                
                # Atualiza progresso
                job.progress = status.get("progress", 0)
                
                # Busca logs
                logs = await self.yarn_client.get_job_logs(job.yarn_app_id)
                if logs:
                    job.logs = logs.split("\\n")[-100:]  # últimas 100 linhas
                
                await self.queue_manager.update_job(job)
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Erro ao monitorar job: {e}")
                await asyncio.sleep(30)
        
        # Timeout
        job.error = "Timeout na execução"
        return False
''')

    # 7. CLIENTE YARN
    create_file(base_dir / "executor/yarn/yarn_client.py", '''"""
Cliente YARN para submissão de jobs.
"""
import subprocess
import json
import logging
import os
from typing import Dict, Optional
from api.models.job import Job

logger = logging.getLogger(__name__)

class YarnClient:
    """Cliente para interagir com YARN."""
    
    def __init__(self):
        self.yarn_cmd = "yarn"
        self.spark_submit = "spark-submit"
        
    async def get_cluster_metrics(self) -> Dict:
        """Obtém métricas do cluster."""
        try:
            # Chama API REST do ResourceManager
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:8088/ws/v1/cluster/metrics"
                ) as resp:
                    data = await resp.json()
                    return data["clusterMetrics"]
        except Exception as e:
            logger.error(f"Erro ao obter métricas: {e}")
            return {}
    
    async def submit_job(self, job: Job) -> str:
        """Submete job ao YARN."""
        try:
            # Monta comando spark-submit
            cmd = [
                self.spark_submit,
                "--master", "yarn",
                "--deploy-mode", "cluster",
                "--name", f"ai-training-{job.id}",
                "--queue", self._get_queue_name(job),
                "--driver-memory", "4g",
                "--executor-memory", f"{job.memory_gb}g",
                "--executor-cores", str(job.cpu_cores),
                "--num-executors", "1",
            ]
            
            # Adiciona configuração de GPU se necessário
            if job.gpu_required:
                cmd.extend([
                    "--conf", "spark.executor.resource.gpu.amount=1",
                    "--conf", "spark.task.resource.gpu.amount=1",
                    "--conf", "spark.executor.resource.gpu.discoveryScript=/opt/find_gpu.sh",
                    "--conf", "spark.yarn.executor.nodeLabelExpression=gpu-node"
                ])
            
            # Adiciona arquivo Python e argumentos
            cmd.extend([
                "/home/lgsilva/SAT_IA/ai_core/train_wrapper.py",
                "--job-id", job.id,
                "--input-path", job.input_path,
                "--output-path", job.output_path,
                "--git-hash", job.git_hash,
                "--config", json.dumps(job.config)
            ])
            
            # Executa comando
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            # Extrai application ID
            for line in result.stdout.split("\\n"):
                if "application_" in line:
                    parts = line.split()
                    for part in parts:
                        if part.startswith("application_"):
                            logger.info(f"Job {job.id} submetido: {part}")
                            return part
            
            raise Exception("Não foi possível obter application ID")
            
        except Exception as e:
            logger.error(f"Erro ao submeter job: {e}")
            raise
    
    async def get_job_status(self, app_id: str) -> Dict:
        """Obtém status de um job no YARN."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://localhost:8088/ws/v1/cluster/apps/{app_id}"
                ) as resp:
                    data = await resp.json()
                    return data["app"]
        except Exception as e:
            logger.error(f"Erro ao obter status: {e}")
            return {"state": "UNKNOWN"}
    
    async def get_job_logs(self, app_id: str) -> str:
        """Obtém logs de um job."""
        try:
            cmd = [self.yarn_cmd, "logs", "-applicationId", app_id]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.stdout
        except Exception as e:
            logger.error(f"Erro ao obter logs: {e}")
            return ""
    
    async def kill_job(self, app_id: str) -> bool:
        """Cancela um job no YARN."""
        try:
            cmd = [self.yarn_cmd, "application", "-kill", app_id]
            result = subprocess.run(cmd)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Erro ao cancelar job: {e}")
            return False
    
    def _get_queue_name(self, job: Job) -> str:
        """Determina fila YARN baseada na prioridade."""
        priority_map = {
            "CRITICAL": "high-priority",
            "HIGH": "high-priority",
            "NORMAL": "default",
            "LOW": "low-priority"
        }
        return priority_map.get(job.priority.name, "default")
''')

    # 8. EXECUTOR GPU
    create_file(base_dir / "executor/gpu_executor.py", '''"""
Executor para jobs com GPU.
"""
import subprocess
import logging
import asyncio
from typing import Dict, List

logger = logging.getLogger(__name__)

class GPUExecutor:
    """Gerencia execução em GPU."""
    
    async def check_gpu_available(self) -> bool:
        """Verifica se há GPU disponível."""
        try:
            # Checa com nvidia-smi
            cmd = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Verifica se tem pelo menos 4GB livre
                memory_free = int(result.stdout.strip().split("\\n")[0])
                return memory_free > 4096
                
        except Exception as e:
            logger.warning(f"GPU não disponível: {e}")
            
        return False
    
    async def get_gpu_info(self) -> List[Dict]:
        """Retorna informações das GPUs."""
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            gpus = []
            for line in result.stdout.strip().split("\\n"):
                parts = line.split(", ")
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total": parts[2],
                    "memory_free": parts[3],
                    "utilization": parts[4]
                })
            
            return gpus
            
        except Exception as e:
            logger.error(f"Erro ao obter info GPU: {e}")
            return []
    
    async def allocate_gpu(self, job_id: str) -> int:
        """Aloca GPU para um job."""
        # Por enquanto, retorna GPU 0
        # Em produção, implementar lógica de alocação
        return 0
    
    async def release_gpu(self, job_id: str, gpu_id: int):
        """Libera GPU após uso."""
        logger.info(f"GPU {gpu_id} liberada pelo job {job_id}")
''')

    # 9. SERVIÇO DE JOBS
    create_file(base_dir / "api/services/job_service.py", '''"""
Serviço de gerenciamento de jobs.
"""
import logging
from typing import Optional, List
from datetime import datetime, timedelta
from api.models.job import Job, JobStatus
from queue.manager import QueueManager

logger = logging.getLogger(__name__)

class JobService:
    """Serviço para gerenciar jobs."""
    
    def __init__(self):
        self.queue_manager = QueueManager()
        
    async def submit_job(self, job: Job) -> int:
        """Submete job para execução."""
        # Valida quotas do usuário
        if not await self._check_user_quota(job.user):
            raise Exception(f"Usuário {job.user} excedeu quota")
        
        # Adiciona à fila
        position = await self.queue_manager.add_job(job)
        
        # Registra no histórico
        await self._log_job_submission(job)
        
        return position
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Busca job por ID."""
        return await self.queue_manager.get_job(job_id)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancela um job."""
        job = await self.get_job(job_id)
        if not job:
            return False
        
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            return False
        
        # Remove da fila ou cancela execução
        if job.status == JobStatus.PENDING:
            await self.queue_manager.remove_job(job_id)
        elif job.status == JobStatus.RUNNING and job.yarn_app_id:
            # Cancela no YARN
            from executor.yarn.yarn_client import YarnClient
            yarn = YarnClient()
            await yarn.kill_job(job.yarn_app_id)
        
        # Atualiza status
        job.status = JobStatus.CANCELLED
        await self.queue_manager.update_job(job)
        
        return True
    
    async def list_jobs(self, user: Optional[str] = None, 
                       status: Optional[JobStatus] = None) -> List[Job]:
        """Lista jobs com filtros."""
        # Implementar busca com filtros
        # Por ora, retorna lista vazia
        return []
    
    async def estimate_wait_time(self, job_id: str) -> int:
        """Estima tempo de espera em segundos."""
        # Simplificado: 5 minutos por job na frente
        queue_status = await self.queue_manager.get_queue_status()
        total_ahead = sum(queue_status.values())
        return total_ahead * 300  # 5 min por job
    
    async def _check_user_quota(self, user: str) -> bool:
        """Verifica quota do usuário."""
        # Por ora, sempre permite
        # Em produção: verificar limite de jobs, uso de GPU, etc.
        return True
    
    async def _log_job_submission(self, job: Job):
        """Registra submissão no histórico."""
        logger.info(f"Job submetido: {job.id} por {job.user}")
        # Em produção: salvar em banco de dados
''')

    # 10. CLIENTE TRAINEE (PARA JUPYTER)
    create_file(base_dir / "client/trainee.py", '''"""
Cliente Trainee para uso no Jupyter.
"""
import requests
import time
from typing import Optional, Dict, Any
import json

class Trainee:
    """Cliente para interagir com o Orquestrador."""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.session = requests.Session()
        
    def train(self, 
             input_path: str, 
             output_path: str, 
             git_hash: str,
             config: Optional[Dict[str, Any]] = None,
             priority: str = "normal",
             wait: bool = False) -> str:
        """
        Submete job de treino.
        
        Args:
            input_path: Caminho HDFS dos dados
            output_path: Onde salvar o modelo
            git_hash: Hash do commit do modelo
            config: Configurações do treino
            priority: low, normal, high, critical
            wait: Se deve aguardar conclusão
            
        Returns:
            job_id se wait=False, caminho do modelo se wait=True
        """
        # Prepara request
        payload = {
            "input_path": input_path,
            "output_path": output_path,
            "git_hash": git_hash,
            "config": config or {},
            "priority": priority,
            "user": self._get_user()
        }
        
        # Submete job
        response = self.session.post(
            f"{self.api_url}/api/v1/training/train",
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        job_id = result["job_id"]
        
        print(f"✓ Job submetido: {job_id}")
        print(f"  Status: {result['status']}")
        print(f"  Tempo estimado: {result.get('estimated_wait', 0) // 60} minutos")
        
        if wait:
            # Aguarda conclusão
            return self._wait_for_completion(job_id)
        else:
            return job_id
    
    def get_status(self, job_id: str) -> Dict:
        """Obtém status de um job."""
        response = self.session.get(
            f"{self.api_url}/api/v1/training/status/{job_id}"
        )
        response.raise_for_status()
        return response.json()
    
    def get_model(self, output_path: str) -> str:
        """
        Baixa modelo do HDFS.
        
        Args:
            output_path: Caminho HDFS do modelo
            
        Returns:
            Caminho local do modelo baixado
        """
        # Implementar download do HDFS
        local_path = f"/tmp/model_{int(time.time())}.keras"
        
        # hdfs dfs -get output_path local_path
        import subprocess
        cmd = ["hdfs", "dfs", "-get", output_path, local_path]
        subprocess.run(cmd, check=True)
        
        print(f"✓ Modelo baixado: {local_path}")
        return local_path
    
    def cancel(self, job_id: str) -> bool:
        """Cancela um job."""
        response = self.session.delete(
            f"{self.api_url}/api/v1/training/cancel/{job_id}"
        )
        if response.status_code == 200:
            print(f"✓ Job {job_id} cancelado")
            return True
        else:
            print(f"✗ Erro ao cancelar: {response.text}")
            return False
    
    def list_jobs(self) -> List[Dict]:
        """Lista jobs do usuário."""
        response = self.session.get(
            f"{self.api_url}/api/v1/jobs/list",
            params={"user": self._get_user()}
        )
        response.raise_for_status()
        return response.json()
    
    def _wait_for_completion(self, job_id: str) -> str:
        """Aguarda job completar."""
        print(f"Aguardando conclusão do job {job_id}...")
        
        while True:
            status = self.get_status(job_id)
            
            if status["status"] == "completed":
                print(f"✓ Job concluído!")
                # Retorna caminho do modelo
                job = self.session.get(
                    f"{self.api_url}/api/v1/jobs/{job_id}"
                ).json()
                return job["output_path"]
                
            elif status["status"] == "failed":
                error = status.get("error", "Unknown error")
                raise Exception(f"Job falhou: {error}")
                
            elif status["status"] == "cancelled":
                raise Exception("Job foi cancelado")
            
            # Mostra progresso
            progress = status.get("progress", 0)
            print(f"  Progresso: {progress:.1f}%", end="\\r")
            
            time.sleep(10)
    
    def _get_user(self) -> str:
        """Obtém usuário atual."""
        import os
        return os.getenv("USER", "default")

# Exemplo de uso no Jupyter:
# from trainee import Trainee
# client = Trainee()
# job_id = client.train(
#     input_path="hdfs:///data/train.csv",
#     output_path="hdfs:///models/my_model",
#     git_hash="abc123",
#     priority="high"
# )
''')

    # 11. DOCKER COMPOSE
    create_file(base_dir / "docker-compose.yml", '''version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    
  orchestrator:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
      - YARN_RESOURCE_MANAGER=http://resourcemanager:8088
    volumes:
      - ./logs:/app/logs
      - ./config:/app/config
    depends_on:
      - redis
    restart: unless-stopped

volumes:
  redis_data:
''')

    # 12. DOCKERFILE
    create_file(base_dir / "Dockerfile", '''FROM python:3.9-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Criar diretórios
RUN mkdir -p logs config

# Expor porta
EXPOSE 8000

# Comando de inicialização
CMD ["python", "app.py"]
''')

    # 13. REQUIREMENTS
    create_file(base_dir / "requirements.txt", '''fastapi==0.104.1
uvicorn==0.24.0
redis==5.0.1
pydantic==2.4.2
aiohttp==3.9.0
pandas==2.1.3
numpy==1.24.3
pyyaml==6.0.1
requests==2.31.0
python-multipart==0.0.6
''')

    # 14. SCRIPT DE INSTALAÇÃO
    create_file(base_dir / "install.sh", '''#!/bin/bash
# Script de instalação do Orquestrador

echo "======================================"
echo "INSTALAÇÃO DO ORQUESTRADOR DE TREINO"
echo "======================================"

# 1. Instalar Redis (se não tiver)
if ! command -v redis-server &> /dev/null; then
    echo "Instalando Redis..."
    sudo apt-get update
    sudo apt-get install -y redis-server
    sudo systemctl enable redis-server
    sudo systemctl start redis-server
fi

# 2. Criar ambiente Python
echo "Criando ambiente virtual..."
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependências
echo "Instalando dependências..."
pip install -r requirements.txt

# 4. Criar diretórios
mkdir -p logs config

# 5. Configurar YARN (se necessário)
echo "Verificando configuração YARN..."
if [ -f /etc/hadoop/conf/yarn-site.xml ]; then
    echo "✓ YARN configurado"
else
    echo "⚠ YARN não encontrado. Configure manualmente."
fi

# 6. Criar serviço systemd
sudo tee /etc/systemd/system/ai-orchestrator.service > /dev/null <<EOF
[Unit]
Description=AI Training Orchestrator
After=network.target redis.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$(pwd)/venv/bin/python app.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
