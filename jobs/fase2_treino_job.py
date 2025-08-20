"""
Job Spark para Fase 2 - Treinamento com GPU
"""
import argparse
import sys
import tensorflow as tf
from pyspark.sql import SparkSession

sys.path.append('/apps/aicore')

def train_on_gpu():
    """Função executada em cada executor com GPU"""
    from src.core import setup_gpu, HDFSAdapter
    from ai_core_treino import ModeloLogITS, ModeloAugmented, ModeloFonetica
    
    # Setup GPU
    gpu_ok = setup_gpu(memory_limit=None)  # Memory growth
    print(f"GPU {'disponível' if gpu_ok else 'não disponível'}")
    
    # Carregar dados do HDFS
    hdfs = HDFSAdapter()
    
    X_train = hdfs.read_parquet("hdfs:///data/aicore/output/preparacao/X_prepared")
    y_train = hdfs.read_parquet("hdfs:///data/aicore/output/preparacao/y_encoded")
    
    # Converter para numpy
    X_train = X_train.values
    y_train = y_train.values
    
    # Treinar modelos
    models = {
        'logits': ModeloLogITS,
        'augmented': ModeloAugmented,
        'fonetica': ModeloFonetica
    }
    
    for name, ModelClass in models.items():
        print(f"Treinando {name}...")
        
        model = ModelClass()
        model.build()
        
        history = model.fit(
            X_train, y_train,
            batch_size=64 if gpu_ok else 32,
            epochs=50,
            validation_split=0.2
        )
        
        # Salvar modelo
        hdfs.save_keras_model(
            model.model,
            f"hdfs:///data/aicore/output/modelos/{name}.keras"
        )
        
        print(f"Modelo {name} salvo")
    
    return "Treinamento concluído"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    
    spark = SparkSession.builder \
        .appName("AICore-Treino-GPU") \
        .getOrCreate()
    
    # Executar treinamento no executor com GPU
    # Como é single-node training, usar spark apenas para orquestração
    sc = spark.sparkContext
    
    # Executar em 1 executor com GPU
    result = sc.parallelize([1], 1).map(lambda x: train_on_gpu()).collect()
    
    print(result[0])
    spark.stop()

if __name__ == "__main__":
    main()
