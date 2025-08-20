# GPU Setup Guide

## Prerequisites

- NVIDIA GPU with CUDA capability >= 7.0
- CUDA 11.8+
- cuDNN 8.6+
- Cloudera CDP 7.x

## Installation Steps

### 1. Configure YARN for GPU

```bash
cd infrastructure/scripts/
./iac_config_gpu.sh
```

### 2. Verify GPU Detection

```bash
yarn node -list | grep gpu
nvidia-smi
```

### 3. Deploy Conda Environment

```bash
conda env create -f environment.yml
conda pack -n aicore-gpu -o aicore-gpu.tar.gz
hdfs dfs -put aicore-gpu.tar.gz /environments/
```

### 4. Submit Test Job

```bash
./spark_submit_gpu.sh test
```

## Troubleshooting

### GPU Not Detected

1. Check NVIDIA drivers:
```bash
nvidia-smi
```

2. Verify CUDA installation:
```bash
nvcc --version
```

3. Check YARN configuration:
```bash
yarn node -status $(hostname) | grep gpu
```

### Out of Memory

Adjust batch size in `gpu_config.yaml`:
```yaml
pipeline:
  phases:
    treino:
      batch_size: 32  # Reduce if OOM
```
