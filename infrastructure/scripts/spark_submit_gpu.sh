#!/bin/bash
# Script para submissão de jobs Spark com GPU

set -e

# Parâmetros
PHASE="${1:-pipeline}"
INPUT_PATH="${2:-hdfs:///data/aicore/input/producao/novos_dados.parquet}"
OUTPUT_PATH="${3:-hdfs:///data/aicore/output}"
CONFIG_FILE="${4:-hdfs:///data/aicore/configs/gpu_config.yaml}"

# Cores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}[INFO] Iniciando job ${PHASE} com GPU${NC}"

# Configurações Spark GPU
SPARK_GPU_CONF="
    --master yarn
    --deploy-mode cluster
    --queue gpu.q
    --name AICore-${PHASE}-GPU
    --num-executors 2
    --executor-cores 4
    --executor-memory 12G
    --driver-memory 8G
    --conf spark.executor.memoryOverhead=4G
    --conf spark.executor.instances=2
    --conf spark.executor.resource.gpu.amount=1
    --conf spark.task.resource.gpu.amount=1
    --conf spark.executor.resource.gpu.discoveryScript=/opt/find_gpu.sh
    --conf spark.yarn.executor.nodeLabelExpression=gpu
    --conf spark.dynamicAllocation.enabled=false
    --archives hdfs:///environments/aicore-gpu.tar.gz#env
    --conf spark.pyspark.python=env/bin/python
    --conf spark.executorEnv.TF_CPP_MIN_LOG_LEVEL=2
    --conf spark.sql.adaptive.enabled=true
    --conf spark.sql.adaptive.coalescePartitions.enabled=true
"

# Função para Fase 1 - Preparação
run_preparacao() {
    echo -e "${YELLOW}[PHASE 1] Preparação de dados${NC}"
    
    spark-submit ${SPARK_GPU_CONF} \
        --py-files hdfs:///apps/aicore/libs.zip \
        ../../jobs/fase1_preparacao_job.py \
        --input ${INPUT_PATH} \
        --output ${OUTPUT_PATH}/preparacao \
        --config ${CONFIG_FILE}
}

# Função para Fase 2 - Treinamento
run_treino() {
    echo -e "${YELLOW}[PHASE 2] Treinamento com GPU${NC}"
    
    spark-submit ${SPARK_GPU_CONF} \
        --conf spark.executor.instances=1 \
        --conf spark.task.maxFailures=1 \
        --py-files hdfs:///apps/aicore/libs.zip \
        ../../jobs/fase2_treino_job.py \
        --input ${OUTPUT_PATH}/preparacao \
        --output ${OUTPUT_PATH}/modelos \
        --config ${CONFIG_FILE}
}

# Função para Fase 3 - Predição
run_predicao() {
    echo -e "${YELLOW}[PHASE 3] Predição/Ensemble${NC}"
    
    spark-submit ${SPARK_GPU_CONF} \
        --py-files hdfs:///apps/aicore/libs.zip \
        ../../jobs/fase3_predicao_job.py \
        --input ${INPUT_PATH} \
        --models ${OUTPUT_PATH}/modelos \
        --output ${OUTPUT_PATH}/predicoes \
        --config ${CONFIG_FILE} \
        --strategy maioria
}

# Pipeline completo
run_pipeline() {
    echo -e "${GREEN}[PIPELINE] Executando pipeline completo${NC}"
    
    spark-submit ${SPARK_GPU_CONF} \
        --py-files hdfs:///apps/aicore/libs.zip \
        ../../jobs/pipeline_completo_job.py \
        --input ${INPUT_PATH} \
        --output ${OUTPUT_PATH} \
        --config ${CONFIG_FILE}
}

# Executar fase
case ${PHASE} in
    preparacao) run_preparacao ;;
    treino) run_treino ;;
    predicao) run_predicao ;;
    pipeline) run_pipeline ;;
    *) echo "Uso: $0 {preparacao|treino|predicao|pipeline} [input] [output] [config]"; exit 1 ;;
esac

echo -e "${GREEN}[COMPLETE] Job ${PHASE} finalizado${NC}"
