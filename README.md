# Learning Cooperative Trajectory Representations for Motion Forecasting

[![NeurIPS 2024](https://img.shields.io/badge/NeurIPS-2024-blue)](https://proceedings.neurips.cc/paper_files/paper/2024/file/1812042b83f20707a898ff6f8af7db84-Paper-Conference.pdf)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

This repository contains the official PyTorch implementation of our NeurIPS 2024 paper: "[Learning Cooperative Trajectory Representations for Motion Forecasting](https://proceedings.neurips.cc/paper_files/paper/2024/file/1812042b83f20707a898ff6f8af7db84-Paper-Conference.pdf)".

## Table of Contents
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation) 
- [Quick Start](#quick-start)
- [Benchmark Results](#benchmark-results)
- [Citation](#citation)
- [License](#license)
- [Related Resource](#related-resource)

## Installation

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/AIR-THU/V2X-Graph
cd V2X-Graph

# Create and activate conda environment
conda create -n v2x_graph python=3.8
conda activate v2x_graph

# Install dependencies
pip install -r requirements.txt
```

### Install Required APIs
```bash
# Clone and install DAIR-V2X-Seq API:
git clone https://github.com/AIR-THU/DAIR-V2X-Seq
export PYTHONPATH="${PYTHONPATH}:/path/to/DAIR-V2X-Seq/projects/dataset"

# Install Argoverse 1 API:
pip install git+https://github.com/argoverse/argoverse-api.git
```

## Dataset Preparation

### Download the Required Datasets
- [DAIR-V2X-Seq Dataset](https://drive.google.com/drive/folders/1yDnlrPCKImpVfI1OPBYyzLFWkhZP5v-7?usp=sharing)
- [V2X-Traj Dataset](https://drive.google.com/file/d/1-5cPcZfGx1L58aiNiHUGoJOkoTSojFtF/view?usp=sharing)

### Generate Trajectory Matching Labels
```bash
# For V2X-Seq-TFD
python generate_label.py \
    --data_root /path/to/V2X-Seq-TFD/cooperative-vehicle-infrastructure \
    --dataset V2X-Seq-TFD

# For V2X-Traj
python generate_label.py \
    --data_root /path/to/v2x-traj \
    --dataset V2X-Traj
```

### Preprocess the Datasets
```bash
# For V2X-Seq-TFD
python preprocess.py \
    --root /path/to/V2X-Seq-TFD/cooperative-vehicle-infrastructure \
    --dataset V2X-Seq-TFD

# For V2X-Traj
python preprocess.py \
    --root /path/to/v2x-traj \
    --dataset V2X-Traj
```

## Quick Start

Train or eval the model with different cooperation settings:
- V2X-Seq-TFD supports: ego/v2i
- V2X-Traj supports: ego/v2v/v2i/v2x

### Training
```bash
# For V2X-Seq-TFD
python train.py \
    --root /path/to/V2X-Seq-TFD/cooperative-vehicle-infrastructure \
    --dataset V2X-Seq-TFD \
    --cooperation v2i

# For V2X-Traj
python train.py \
    --root /path/to/v2x-traj \
    --dataset V2X-Traj \
    --cooperation v2x
```

### Evaluation
```bash
# For V2X-Seq-TFD
python eval.py \
    --root /path/to/V2X-Seq-TFD/cooperative-vehicle-infrastructure \
    --dataset V2X-Seq-TFD \
    --cooperation v2i \
    --ckpt_path /path/to/V2X-Graph/checkpoints/v2x-seq-tfd/v2i.ckpt

# For V2X-Traj
python eval.py \
    --root /path/to/v2x-traj \
    --dataset V2X-Traj \
    --cooperation v2x \
    --ckpt_path /path/to/V2X-Graph/checkpoints/v2x-traj/v2x.ckpt
```

## Benchmark Results

### V2X-Seq-TFD Validation Set Results

| Metric | Ego | V2I |
|:-------|:---:|:---:|
| minADE | 1.16 | 1.05 |
| minFDE | 2.02 | 1.79 |
| MR     | 0.30 | 0.25 |

### V2X-Traj Validation Set Results

| Metric | Ego | V2V | V2I | V2X |
|:-------|:---:|:---:|:---:|:---:|
| minADE | 0.90 | 0.77 | 0.80 | 0.72 |
| minFDE | 1.56 | 1.26 | 1.32 | 0.13 |
| MR     | 0.17 | 0.12 | 0.13 | 0.11 |

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{ruan2024v2xgraph,
  title={Learning Cooperative Trajectory Representations for Motion Forecasting},
  author={Hongzhi Ruan and Haibao Yu and Wenxian Yang and Siqi Fan and Zaiqing Nie},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Related Resources

- [DAIR-V2X-Seq](https://github.com/AIR-THU/DAIR-V2X-Seq)
- [Argoverse](https://github.com/argoverse/argoverse-api)
- [HiVT](https://github.com/ZikangZhou/HiVT)