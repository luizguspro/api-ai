#!/bin/bash
# Deploy do código para HDFS

set -e

echo "Deploying AI Core to HDFS..."

# Criar estrutura no HDFS
hdfs dfs -mkdir -p /apps/aicore
hdfs dfs -mkdir -p /data/aicore/{input,output,models,artifacts,configs}
hdfs dfs -mkdir -p /environments

# Empacotar dependências
cd ../../src
zip -r ../libs.zip core/ utils/ -x "*.pyc" -x "__pycache__/*"

# Upload código
hdfs dfs -put -f ../libs.zip /apps/aicore/
hdfs dfs -put -f ../jobs/*.py /apps/aicore/

# Upload configs
hdfs dfs -put -f ../infrastructure/configs/*.yaml /data/aicore/configs/

# Permissões
hdfs dfs -chmod -R 755 /apps/aicore
hdfs dfs -chmod -R 775 /data/aicore

echo "Deploy completo!"
