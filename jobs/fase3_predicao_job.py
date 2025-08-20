"""
Job Spark para Fase 3 - Predição com GPU
"""
import argparse
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col
from typing import Iterator
import sys

sys.path.append('/apps/aicore')

# Variável global para modelo (carregado 1x por executor)
_models = None
_tokenizer = None

def load_models_once():
    """Carrega modelos uma vez por executor"""
    global _models, _tokenizer
    
    if _models is None:
        from src.core import HDFSAdapter, setup_gpu
        
        # Setup GPU
        setup_gpu()
        
        hdfs = HDFSAdapter()
        
        # Carregar modelos
        _models = {
            'logits': hdfs.load_keras_model("hdfs:///data/aicore/output/modelos/logits.keras"),
            'augmented': hdfs.load_keras_model("hdfs:///data/aicore/output/modelos/augmented.keras"),
            'fonetica': hdfs.load_keras_model("hdfs:///data/aicore/output/modelos/fonetica.keras")
        }
        
        # Carregar tokenizer
        _tokenizer = hdfs.read_pickle("hdfs:///data/aicore/artifacts/tokenizer.pkl")
    
    return _models, _tokenizer

@pandas_udf(returnType="struct<codigo_ncm:string,confidence:float,strategy:string>")
def predict_batch(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """
    UDF para predição em batch usando mapInPandas
    Carrega modelo 1x por partição
    """
    # Carregar modelos uma vez
    models, tokenizer = load_models_once()
    
    for batch_df in iterator:
        # Tokenizar batch
        texts = batch_df['descricao'].tolist()
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors='tf')
        
        # Predições de cada modelo
        predictions = {}
        for name, model in models.items():
            pred = model.predict(tokens['input_ids'], batch_size=256)
            predictions[name] = pred
        
        # Ensemble - estratégia de maioria
        import numpy as np
        final_predictions = []
        
        for i in range(len(texts)):
            votes = []
            for name in predictions:
                votes.append(np.argmax(predictions[name][i]))
            
            # Maioria simples
            from collections import Counter
            most_common = Counter(votes).most_common(1)[0]
            
            final_predictions.append({
                'codigo_ncm': str(most_common[0]),
                'confidence': most_common[1] / len(votes),
                'strategy': 'maioria'
            })
        
        yield pd.DataFrame(final_predictions)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--models', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--strategy', default='maioria')
    args = parser.parse_args()
    
    spark = SparkSession.builder \
        .appName("AICore-Predicao-GPU") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    # Ler dados novos
    df = spark.read.parquet(args.input)
    
    # Aplicar predições usando mapInPandas (carrega modelo 1x por executor)
    df_predictions = df.mapInPandas(
        predict_batch,
        schema="descricao string, codigo_ncm string, confidence float, strategy string"
    )
    
    # Salvar resultados
    df_predictions.write \
        .mode('overwrite') \
        .parquet(f"{args.output}/predicoes.parquet")
    
    # Exportar CSV opcional
    df_predictions.coalesce(1).write \
        .mode('overwrite') \
        .option('header', 'true') \
        .csv(f"{args.output}/predicoes.csv")
    
    print(f"Predições salvas em: {args.output}")
    spark.stop()

if __name__ == "__main__":
    main()
