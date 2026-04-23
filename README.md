# scPPC

we developed scPPC, a reconstruction framework for single-cell proteomics data. scPPC builds a heterogeneous graph that captures the hierarchical relationships among cells, proteins, and peptides/precursors, and leverages a heterogeneous graph Transformer autoencoder to reconstruct protein expression profiles.

<p align="center">
  <img src="./images/Figure1.jpg" width="500">
</p>

## Development Environment

* CUDA Version: 12.0
* python: 3.8.18
* pytorch: 1.12.0

### Installation

1. Create Environment

```bash
	conda create -n scPPC python=3.8.18
```

2. Activate Environment

```bash
	conda activate scPPC
```

3. Install dependencies

```bash
	pip install https://download.pytorch.org/whl/cu113/torch-1.12.0%2Bcu113-cp38-cp38-linux_x86_64.whl
	pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_scatter-2.1.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
	pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_sparse-0.6.16%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
	pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_cluster-1.6.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
```
