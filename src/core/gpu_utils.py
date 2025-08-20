"""
Utilitários para configuração e gerenciamento de GPU
"""
import os
import tensorflow as tf
from typing import Optional, List

def setup_gpu(gpu_id: Optional[int] = None, memory_limit: Optional[int] = None) -> bool:
    """
    Configura GPU para TensorFlow
    
    Args:
        gpu_id: ID da GPU a usar (None = todas)
        memory_limit: Limite de memória em MB (None = growth)
    
    Returns:
        True se GPU configurada com sucesso
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if not gpus:
        print("Nenhuma GPU disponível")
        return False
    
    try:
        if gpu_id is not None:
            # Usar GPU específica
            tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
            gpus = [gpus[gpu_id]]
        
        for gpu in gpus:
            if memory_limit:
                # Limite fixo de memória
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=memory_limit
                    )]
                )
            else:
                # Memory growth
                tf.config.experimental.set_memory_growth(gpu, True)
        
        print(f"GPU(s) configurada(s): {len(gpus)}")
        return True
        
    except RuntimeError as e:
        print(f"Erro ao configurar GPU: {e}")
        return False

def get_optimal_batch_size(model, input_shape: tuple, max_batch: int = 512) -> int:
    """
    Determina batch size ótimo baseado na VRAM disponível
    
    Args:
        model: Modelo Keras
        input_shape: Shape do input (sem batch dimension)
        max_batch: Batch size máximo a testar
    
    Returns:
        Batch size ótimo
    """
    batch_size = 1
    
    while batch_size <= max_batch:
        try:
            # Testar forward pass
            test_input = tf.random.normal([batch_size] + list(input_shape))
            _ = model(test_input, training=False)
            
            # Se passou, dobrar batch size
            batch_size *= 2
            
        except tf.errors.ResourceExhaustedError:
            # VRAM estourou, voltar ao último válido
            return batch_size // 2
    
    return min(batch_size // 2, max_batch)

def enable_mixed_precision():
    """Habilita mixed precision training"""
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print('Mixed precision habilitado')

def enable_xla_optimization():
    """Habilita otimização XLA"""
    tf.config.optimizer.set_jit(True)
    print('XLA optimization habilitado')

def monitor_gpu_memory():
    """Monitora uso de memória GPU"""
    if tf.config.list_physical_devices('GPU'):
        tf.config.experimental.reset_memory_stats('GPU:0')
        stats = tf.config.experimental.get_memory_info('GPU:0')
        return {
            'current': stats['current'] / 1e9,  # GB
            'peak': stats['peak'] / 1e9  # GB
        }
    return None
