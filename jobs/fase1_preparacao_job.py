"""
Job Spark para Fase 1 - Preparação de Dados
"""
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import sys
sys.path.append('/apps/aicore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    
    spark = SparkSession.builder \
        .appName("AICore-Preparacao") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    # Importar módulo de preparação original
    from ai_core_preparacao import PreparadorDados
    
    # Ler dados
    df = spark.read.parquet(args.input)
    
    # Converter para pandas e processar
    df_pandas = df.toPandas()
    
    preparador = PreparadorDados(config_path=args.config)
    X_prepared, y_encoded = preparador.preparar(df_pandas)
    
    # Salvar como Parquet
    spark.createDataFrame(X_prepared).write \
        .mode('overwrite') \
        .parquet(f"{args.output}/X_prepared")
    
    spark.createDataFrame(y_encoded).write \
        .mode('overwrite') \
        .parquet(f"{args.output}/y_encoded")
    
    # Salvar artefatos em /artifacts
    from src.core import HDFSAdapter
    hdfs = HDFSAdapter()
    
    hdfs.write_pickle(preparador.tokenizer, 
                     f"{args.output}/../artifacts/tokenizer.pkl")
    hdfs.write_pickle(preparador.label_encoders, 
                     f"{args.output}/../artifacts/label_encoders.pkl")
    
    print(f"Preparação concluída: {args.output}")
    spark.stop()

if __name__ == "__main__":
    main()
