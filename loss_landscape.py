#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HandDAGT Loss Landscape Visualization Tool
基于ASAL_CVQA-main的loss_landscape.py，适用于HandDAGT训练过程

主要功能：
1. 在训练过程中计算loss landscape
2. 可视化模型在参数空间中的损失变化
3. 帮助分析模型的收敛性和稳定性
"""

import os
import time
import math
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import logging


def normalize_direction(direction, weights, norm='filter'):
    """
    重新缩放方向，使其在不同层级具有相似的范数
    
    Args:
        direction: 一层随机方向的变量
        weights: 一层原始模型的变量
        norm: 归一化方法, 'filter' | 'layer'
    """
    if norm == 'filter':
        # 重新缩放过滤器，使每个过滤器具有与weights中对应过滤器相同的范数
        for d, w in zip(direction, weights):
            d.mul_(w.norm()/(d.norm() + 1e-10))
    else:
        # 重新缩放层变量，使每层具有与weights中对应层相同的范数
        direction.mul_(weights.norm()/direction.norm())


def create_random_direction(weights, ignore='biasbn', norm='filter', device='cuda'):
    """
    设置与权重具有相同维度的随机（归一化）方向
    
    Args:
        weights: 给定的训练模型
        ignore: 'biasbn', 忽略偏置和BN参数
        norm: 归一化方法
        device: 计算设备
    
    Returns:
        direction: 与权重具有相同维度的随机方向
    """
    direction = []
    for w in weights:
        d = torch.randn(w.size())
        d = d.to(device)
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0)  # 忽略1维权重的方向
            else:
                # 保留每个节点只有1个的权重/偏置的方向
                d.copy_(w)
        else:
            normalize_direction(d, w, norm)
        direction.append(d)
    return direction


def set_weights(net, weights, directions=None, step=None):
    """
    用指定的张量列表覆盖网络的权重，或沿方向以步长改变权重
    
    Args:
        net: 网络
        weights: 权重列表
        directions: 方向列表
        step: 步长
    
    Returns:
        net: 修改后的网络
    """
    if directions is None:
        # 不能在没有方向的情况下指定步长
        for (p, w) in zip(net.parameters(), weights):
            p.data.copy_(w.type(type(p.data)))
    else:
        assert step is not None, '如果指定了方向，则必须指定步长'
        for (p, w, d) in zip(net.parameters(), weights, directions):
            p.data = w + d * step
    return net


def eval_loss_handdagt(model, dataloader, dataset, device='cuda'):
    """
    评估HandDAGT模型在给定数据上的损失
    
    Args:
        model: HandDAGT模型
        dataloader: 数据加载器
        dataset: 数据集对象（包含img2pcl_index等方法）
        device: 计算设备
    
    Returns:
        total_loss: 总损失
        avg_loss: 平均损失
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc="评估损失", leave=False):
            try:
                # 根据数据集类型处理数据
                if len(data) >= 9:  # IncNYU数据集
                    img, points, gt_xyz, uvd_gt, center, M, cube, cam_para, volume_length, \
                        teacher_data, teacher_pcl, teacher_center, teacher_M, teacher_cube = data
                    
                    # 移动到GPU
                    img = img.to(device)
                    points = points.to(device)
                    gt_xyz = gt_xyz.to(device)
                    center = center.to(device)
                    M = M.to(device)
                    cube = cube.to(device)
                    cam_para = cam_para.to(device)
                    teacher_data = teacher_data.to(device)
                    teacher_pcl = teacher_pcl.to(device)
                    teacher_center = teacher_center.to(device)
                    teacher_M = teacher_M.to(device)
                    teacher_cube = teacher_cube.to(device)
                    
                    # 计算损失
                    loss = model.get_loss(
                        points.transpose(1, 2), points.transpose(1, 2), 
                        img, dataset, center, M, cube, cam_para, 
                        gt_xyz.transpose(1, 2), teacher_pcl.transpose(1, 2), 
                        teacher_data, teacher_center, teacher_M, teacher_cube
                    )
                    
                elif len(data) >= 8:  # NYU数据集
                    img, points, gt_xyz, uvd_gt, center, M, cube, cam_para, volume_length = data
                    
                    # 移动到GPU
                    img = img.to(device)
                    points = points.to(device)
                    gt_xyz = gt_xyz.to(device)
                    center = center.to(device)
                    M = M.to(device)
                    cube = cube.to(device)
                    cam_para = cam_para.to(device)
                    
                    # 计算损失
                    loss = model.get_loss(
                        points.transpose(1, 2), points.transpose(1, 2), 
                        img, dataset, center, M, cube, cam_para, 
                        gt_xyz.transpose(1, 2), None
                    )
                    
                else:  # 其他数据集
                    img, points, gt_xyz, uvd_gt, center, M, cube, cam_para = data
                    
                    # 移动到GPU
                    img = img.to(device)
                    points = points.to(device)
                    gt_xyz = gt_xyz.to(device)
                    center = center.to(device)
                    M = M.to(device)
                    cube = cube.to(device)
                    cam_para = cam_para.to(device)
                    
                    # 计算损失
                    loss = model.get_loss(
                        points.transpose(1, 2), points.transpose(1, 2), 
                        img, dataset, center, M, cube, cam_para, 
                        gt_xyz.transpose(1, 2), None
                    )
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"处理批次时出错: {e}")
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return total_loss, avg_loss


def calculate_loss_landscape(model, dataloader, dataset, steps, dir_num=10, output_dir=None, device='cuda'):
    """
    计算HandDAGT模型的损失景观
    
    Args:
        model: HandDAGT模型
        dataloader: 数据加载器
        dataset: 数据集对象（包含img2pcl_index等方法）
        steps: 步长列表
        dir_num: 方向数量
        output_dir: 输出目录
        device: 计算设备
    
    Returns:
        train_losses: 训练损失数组
    """
    print("开始计算损失景观...")
    
    # 初始化损失数组
    train_losses = np.zeros((dir_num, len(steps)), dtype=np.float32)
    
    with torch.no_grad():
        # 保存训练好的权重
        trained_weights = copy.deepcopy(list(model.parameters()))
        
        for di in range(dir_num):
            print(f"计算方向 {di+1}/{dir_num}")
            torch.manual_seed(1024 + di)
            
            # 创建随机方向
            direction = create_random_direction(model.parameters(), device=device)
            
            for s, step in enumerate(steps):
                print(f"  步长 {s+1}/{len(steps)}: {step}")
                
                # 设置权重
                model = set_weights(model, trained_weights, direction, step)
                
                # 评估损失
                _, avg_loss = eval_loss_handdagt(model, dataloader, dataset, device)
                train_losses[di, s] = avg_loss
                
                print(f"    损失: {avg_loss:.6f}")
        
        print("恢复原始权重...")
        # 恢复原始权重
        model = set_weights(model, trained_weights)
    
    # 保存训练损失
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        npz_filename = os.path.join(output_dir, f'loss_landscape_{time.strftime("%Y-%m-%d-%H-%M-%S")}.npz')
        np.savez(npz_filename, train_loss=train_losses, steps=steps)
        print(f"损失景观已保存到: {npz_filename}")
    
    return train_losses


def plot_loss_landscape(losses, steps, output_dir, file_name=None, show=False):
    """
    绘制1D损失景观图
    
    Args:
        losses: 损失数组 [dir_num, steps]
        steps: 步长列表
        output_dir: 输出目录
        file_name: 文件名
        show: 是否显示图像
    """
    if file_name is None:
        file_name = time.strftime("%Y-%m-%d-%H-%M-%S")
    
    losses = np.array(losses)
    print("训练损失:")
    print(losses)
    
    # 保存损失数据
    save_losses = np.ones((losses.reshape((-1, 1)).shape[0], 3))
    r = 0
    
    # 创建损失图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制每个方向的损失曲线
    for di in range(losses.shape[0]):
        # 使用不同的颜色和更明显的线条
        color = plt.cm.viridis(di / max(1, losses.shape[0] - 1))  # 使用viridis颜色映射
        ax.plot(steps, losses[di, :], color=color, linewidth=1.5, alpha=0.8, label=f'Direction {di+1}')
        
        # 保存数据
        for s_idx, s in enumerate(steps):
            save_losses[r, 0] = di
            save_losses[r, 1] = s
            save_losses[r, 2] = losses[di, s_idx]
            r += 1
    
    # 计算平均损失
    mean_losses = np.mean(losses, axis=0)
    ax.plot(steps, mean_losses, 'r-', linewidth=2, label='Mean Loss')
    
    # 设置图表属性
    ax.set_xlabel('Disturbance Step')
    ax.set_ylabel('Loss')
    ax.set_title('HandDAGT Loss Landscape')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 自动调整布局
    plt.tight_layout()
    
    # 保存图像
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, f'loss_landscape_{file_name}.pdf')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"损失景观图已保存到: {fig_path}")
        
        # 保存CSV数据
        csv_path = os.path.join(output_dir, f'loss_landscape_{file_name}.csv')
        np.savetxt(csv_path, save_losses, delimiter=',', 
                   header='Direction,Step,Loss', comments='')
        print(f"损失数据已保存到: {csv_path}")
    
    if show:
        plt.show()
    
    plt.close()


def integrate_with_training(model, dataloader, epoch, output_dir, device='cuda'):
    """
    在训练过程中集成损失景观计算
    
    Args:
        model: HandDAGT模型
        dataloader: 数据加载器
        epoch: 当前训练轮次
        output_dir: 输出目录
        device: 计算设备
    """
    # 每10个epoch计算一次损失景观
    if epoch % 10 == 0:
        print(f"\n=== Epoch {epoch}: 计算损失景观 ===")
        
        # 定义步长范围
        steps = np.linspace(-0.5, 0.5, 21)  # 21个点，从-0.5到0.5
        
        # 计算损失景观
        losses = calculate_loss_landscape(
            model, dataloader, dataloader.dataset, steps, 
            dir_num=5,  # 减少方向数量以加快计算
            output_dir=os.path.join(output_dir, 'loss_landscape'),
            device=device
        )
        
        # 绘制损失景观
        plot_loss_landscape(
            losses, steps,
            output_dir=os.path.join(output_dir, 'loss_landscape'),
            file_name=f'epoch_{epoch:03d}',
            show=False
        )
        
        print(f"Epoch {epoch} 的损失景观计算完成\n")


def main():
    """
    主函数 - 用于独立运行损失景观分析
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='HandDAGT Loss Landscape Analysis')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--data_path', type=str, required=True, help='数据路径')
    parser.add_argument('--output_dir', type=str, default='./loss_landscape_results', help='输出目录')
    parser.add_argument('--steps', type=int, default=21, help='步长数量')
    parser.add_argument('--directions', type=int, default=10, help='方向数量')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    
    args = parser.parse_args()
    
    # 这里需要根据具体的模型和数据加载器进行适配
    print("请根据具体的HandDAGT模型和数据加载器修改此脚本")
    print("主要需要实现:")
    print("1. 模型加载")
    print("2. 数据加载器创建")
    print("3. 损失函数调用")
    
    # 示例用法
    steps = np.linspace(-0.5, 0.5, args.steps)
    print(f"将计算 {args.directions} 个方向，每个方向 {args.steps} 个步长的损失景观")
    print(f"步长范围: {steps[0]:.2f} 到 {steps[-1]:.2f}")


if __name__ == "__main__":
    main()
