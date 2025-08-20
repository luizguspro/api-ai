"""
Adapter para operações com HDFS usando fsspec
"""
import os
import yaml
import pickle
import pandas as pd
import pyarrow.parquet as pq
import fsspec
from typing import Dict, Any, Optional
from pathlib import Path

class HDFSAdapter:
    """Adapter para operações com HDFS usando fsspec"""
    
    def __init__(self, host: str = "default", port: int = 9000):
        """
        Inicializa adapter HDFS
        
        Args:
            host: Namenode host
            port: Namenode port
        """
        self.fs = fsspec.filesystem('hdfs', host=host, port=port)
        
    def read_parquet(self, hdfs_path: str) -> pd.DataFrame:
        """Lê arquivo Parquet do HDFS"""
        with self.fs.open(hdfs_path, 'rb') as f:
            return pd.read_parquet(f)
    
    def write_parquet(self, df: pd.DataFrame, hdfs_path: str):
        """Escreve DataFrame como Parquet no HDFS"""
        with self.fs.open(hdfs_path, 'wb') as f:
            df.to_parquet(f, engine='pyarrow', compression='snappy')
    
    def read_pickle(self, hdfs_path: str) -> Any:
        """Lê arquivo pickle do HDFS"""
        with self.fs.open(hdfs_path, 'rb') as f:
            return pickle.load(f)
    
    def write_pickle(self, obj: Any, hdfs_path: str):
        """Escreve objeto pickle no HDFS"""
        with self.fs.open(hdfs_path, 'wb') as f:
            pickle.dump(obj, f)
    
    def read_yaml(self, hdfs_path: str) -> Dict:
        """Lê arquivo YAML do HDFS"""
        with self.fs.open(hdfs_path, 'r') as f:
            return yaml.safe_load(f)
    
    def write_yaml(self, data: Dict, hdfs_path: str):
        """Escreve YAML no HDFS"""
        with self.fs.open(hdfs_path, 'w') as f:
            yaml.dump(data, f)
    
    def save_keras_model(self, model, hdfs_path: str):
        """Salva modelo Keras no HDFS"""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp:
            model.save(tmp.name)
            with open(tmp.name, 'rb') as local_f:
                with self.fs.open(hdfs_path, 'wb') as hdfs_f:
                    hdfs_f.write(local_f.read())
            os.unlink(tmp.name)
    
    def load_keras_model(self, hdfs_path: str):
        """Carrega modelo Keras do HDFS"""
        import tensorflow as tf
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp:
            with self.fs.open(hdfs_path, 'rb') as hdfs_f:
                tmp.write(hdfs_f.read())
            tmp.flush()
            model = tf.keras.models.load_model(tmp.name)
            os.unlink(tmp.name)
            return model
    
    def exists(self, hdfs_path: str) -> bool:
        """Verifica se arquivo existe no HDFS"""
        return self.fs.exists(hdfs_path)
    
    def mkdir(self, hdfs_path: str):
        """Cria diretório no HDFS"""
        self.fs.makedirs(hdfs_path, exist_ok=True)
    
    def list_files(self, hdfs_path: str):
        """Lista arquivos em um diretório"""
        return self.fs.ls(hdfs_path)
