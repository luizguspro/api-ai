#!/bin/bash
# Script para configurar GPU scheduling no YARN

set -e

CLOUDERA_MANAGER_HOST="${CM_HOST:-localhost}"
CLOUDERA_API_VERSION="v19"
CLUSTER_NAME="${CLUSTER_NAME:-cluster}"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}[INFO] Iniciando configuração de GPU no YARN...${NC}"

# 1. Habilitar GPU scheduling no YARN
configure_yarn_gpu() {
    echo -e "${YELLOW}[STEP 1] Configurando YARN para GPU scheduling${NC}"
    
    cat > /tmp/yarn-gpu-config.xml <<EOF
<configuration>
    <property>
        <name>yarn.resource-types</name>
        <value>yarn.io/gpu</value>
    </property>
    
    <property>
        <name>yarn.nodemanager.resource-plugins</name>
        <value>yarn.io/gpu</value>
    </property>
    
    <property>
        <name>yarn.nodemanager.resource-plugins.gpu.allowed-gpu-devices</name>
        <value>auto</value>
    </property>
    
    <property>
        <name>yarn.nodemanager.resource-plugins.gpu.path-to-discovery-executables</name>
        <value>/usr/bin</value>
    </property>
</configuration>
EOF
    
    # Aplicar configuração
    hdfs dfsadmin -refreshNodes
}

# 2. Configurar Node Labels e Queue GPU
configure_node_labels() {
    echo -e "${YELLOW}[STEP 2] Criando node labels e queue GPU${NC}"
    
    # Criar label gpu-node
    yarn rmadmin -addToClusterNodeLabels "gpu(exclusive=false)"
    
    # Criar queue gpu.q
    cat > /tmp/capacity-scheduler.xml <<EOF
<configuration>
    <property>
        <name>yarn.scheduler.capacity.root.queues</name>
        <value>default,gpu</value>
    </property>
    <property>
        <name>yarn.scheduler.capacity.root.gpu.capacity</name>
        <value>30</value>
    </property>
    <property>
        <name>yarn.scheduler.capacity.root.gpu.accessible-node-labels</name>
        <value>gpu</value>
    </property>
    <property>
        <name>yarn.scheduler.capacity.root.gpu.default-node-label-expression</name>
        <value>gpu</value>
    </property>
</configuration>
EOF
    
    # Aplicar labels aos nodes com GPU
    GPU_NODES=$(nvidia-smi -L 2>/dev/null | grep "GPU" | wc -l)
    if [ "$GPU_NODES" -gt 0 ]; then
        HOSTNAME=$(hostname -f)
        echo -e "${GREEN}GPU detectada: $HOSTNAME${NC}"
        yarn rmadmin -replaceLabelsOnNode "$HOSTNAME=gpu"
    fi
}

# 3. Criar script de descoberta GPU
create_discovery_script() {
    echo -e "${YELLOW}[STEP 3] Criando script de descoberta GPU${NC}"
    
    cat > /opt/find_gpu.sh <<'SCRIPT'
#!/bin/bash
# GPU Discovery Script
GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
if [ -n "$GPUS" ]; then
    echo "{"name":"gpu","addresses":["$GPUS"]}"
else
    echo "{"name":"gpu","addresses":[]}"
fi
SCRIPT
    
    chmod +x /opt/find_gpu.sh
}

# 4. Configurar Spark para GPU
configure_spark_gpu() {
    echo -e "${YELLOW}[STEP 4] Configurando Spark para GPU${NC}"
    
    cat >> /etc/spark/conf/spark-defaults.conf <<EOF

# GPU Configuration
spark.executor.resource.gpu.amount=1
spark.executor.resource.gpu.discoveryScript=/opt/find_gpu.sh
spark.task.resource.gpu.amount=1
spark.rapids.memory.pinnedPool.size=2G
spark.executor.extraJavaOptions=-Djava.library.path=/usr/local/cuda/lib64
spark.executorEnv.LD_LIBRARY_PATH=/usr/local/cuda/lib64
spark.yarn.executor.nodeLabelExpression=gpu
EOF
}

# 5. Instalar drivers NVIDIA e CUDA
install_cuda_drivers() {
    echo -e "${YELLOW}[STEP 5] Verificando CUDA e drivers${NC}"
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}nvidia-smi não encontrado. Instalando...${NC}"
        sudo yum install -y nvidia-driver-latest-dkms cuda-toolkit-11-8
        
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /etc/profile.d/cuda.sh
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /etc/profile.d/cuda.sh
    else
        echo -e "${GREEN}CUDA instalado: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)${NC}"
    fi
}

# 6. Criar e distribuir ambiente Conda
distribute_conda_env() {
    echo -e "${YELLOW}[STEP 6] Criando ambiente Conda GPU${NC}"
    
    # Criar ambiente
    conda env create -f ../../environment.yml
    
    # Empacotar
    conda pack -n aicore-gpu -o /tmp/aicore-gpu.tar.gz
    
    # Upload para HDFS
    hdfs dfs -mkdir -p /environments
    hdfs dfs -put -f /tmp/aicore-gpu.tar.gz /environments/
    
    echo -e "${GREEN}Ambiente Conda distribuído${NC}"
}

# Executar todas as funções
main() {
    configure_yarn_gpu
    configure_node_labels
    create_discovery_script
    configure_spark_gpu
    install_cuda_drivers
    distribute_conda_env
    
    echo -e "${GREEN}[SUCCESS] Configuração GPU concluída!${NC}"
}

main "$@"
