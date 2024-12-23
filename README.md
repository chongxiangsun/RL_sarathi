
# RL_sarathi



### Setup CUDA

Sarathi-Serve has been tested with CUDA 12.3 on H100 and A100 GPUs.

### Clone repository

```sh
git 
```

### Create mamba environment

Setup mamba if you don't already have it,

```sh
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
bash Mambaforge-Linux-x86_64.sh # follow the instructions from there
```

Create a Python 3.10 environment,

```sh
mamba create -p ./env python=3.10  
```

### Install Sarathi-Serve

```sh
pip install -e . --extra-index-url https://flashinfer.ai/whl/cu121/torch2.3/
```

## Reproducing Results

Refer to readmes in individual folders corresponding to each figure in `osdi-experiments`.

## Citation

If you use our work, please consider citing our paper:

```
@article{
}
```

## Acknowledgment

This repository originally started as a fork of the [vLLM project](https://vllm-project.github.io/). Sarathi-Serve is a research prototype and does not have complete feature parity with open-source vLLM. We have only retained the most critical features and adopted the codebase for faster research iterations.

