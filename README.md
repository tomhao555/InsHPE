
# InsHaPE: Hand Pose Estimation Project

## Introduction

InsHaPE is a deep learning-based hand pose estimation project that supports multiple datasets and model architectures. This project implements the functionality of estimating 3D hand joint positions from depth images, making it suitable for applications such as hand tracking and gesture recognition.

## Requirements

### System Requirements

  - Python 3.7+
  - CUDA 10.0+ (Recommended)
  - Linux/Windows/macOS

### Dependencies

```bash
torch>=1.8.0
torchvision>=0.9.0
numpy
opencv-python
tqdm
tensorboardX
thop  # For FLOPs calculation
```

### Installation

```bash
pip install torch torchvision numpy opencv-python tqdm tensorboardX thop
```

## Dataset Preparation

### 1\. NYU Dataset

Place the NYU dataset in the `your_path/datasets/nyu/` directory. Ensure it follows the structure below:

```
nyu/
├── train/
│   ├── depth_1_0000001.png
│   ├── depth_1_0000002.png
│   └── ...
├── test/
│   ├── depth_1_0000001.png
│   └── ...
└── joints_data.mat
```

### 2\. IncNYU Dataset

Place the IncNYU dataset in the `your_path/datasets/IncNYU/` directory.


## Usage

### 1\. Training

#### Train on NYU Dataset

```bash
bash train_nyu.sh
```

#### Train on IncNYU Dataset

```bash
bash train_IncNYU.sh
```

#### Custom Training Parameters

```bash
python3 train.py \
    --gpu 0 \
    --dataset nyu \
    --dataset_path your_path/datasets/nyu \
    --stacks 3 \
    --model_name tgdnet \
    --step_size 40 \
```

### 2\. Evaluation

#### Evaluate on NYU Dataset

```bash
bash test_nyu.sh
```

#### Evaluate on IncNYU Dataset

```bash
bash test_IncNYU.sh
```

#### Custom Evaluation Parameters

```bash
python3 eval.py \
    --gpu 0 \
    --dataset nyu \
    --test_path your_path/datasets/nyu \
    --stacks 3 \
    --save_root_dir ./results \
    --model_checkpoint_dir your_path/checkpoints
```