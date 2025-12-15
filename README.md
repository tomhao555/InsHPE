# InsHPE: 手部姿态估计项目

## 项目简介

InsHPE是一个基于深度学习的手部姿态估计项目，支持多种数据集和模型架构。该项目实现了从深度图像中估计手部关节3D位置的功能，适用于手部跟踪、手势识别等应用场景。


```

## 环境要求

### 系统要求
- Python 3.7+
- CUDA 10.0+ (推荐)
- Linux/Windows/macOS

### 依赖包
```bash
torch>=1.8.0
torchvision>=0.9.0
numpy
opencv-python
tqdm
tensorboardX
thop  # 用于FLOPs计算
```

### 安装依赖
```bash
pip install torch torchvision numpy opencv-python tqdm tensorboardX thop
```

## 数据集准备

### 1. NYU数据集
将NYU数据集放置在 `your_path/datasets/nyu/` 目录下，确保包含以下结构：
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

### 2. IncNYU数据集
将IncNYU数据集放置在 `your_path/datasets/IncNYU/` 目录下。

### 3. DexYCB数据集
将DexYCB数据集放置在 `your_path/datasets/dexycb/` 目录下。

## 使用方法

### 1. 训练模型

#### 训练NYU数据集
```bash
bash train_nyu.sh
```

#### 训练IncNYU数据集
```bash
bash train_IncNYU.sh
```

#### 自定义训练参数
```bash
python3 train.py \
    --gpu 0 \
    --dataset nyu \
    --dataset_path your_path/datasets/nyu \
    --stacks 3 \
    --model_name tgdnet \
    --step_size 40 \
```

### 2. 评估模型

#### 评估NYU数据集
```bash
bash test_nyu.sh
```

#### 评估IncNYU数据集
```bash
bash test_IncNYU.sh
```

#### 自定义评估参数
```bash
python3 eval.py \
    --gpu 0 \
    --dataset nyu \
    --test_path your_path/datasets/nyu \
    --stacks 3 \
    --save_root_dir ./results \
    --model_checkpoint_dir your_path/checkpoints
```
