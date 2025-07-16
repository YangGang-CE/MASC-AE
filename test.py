import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import VonMisesDataset
from model import SimplifiedVonMisesAutoEncoder
from utils.loss_utils import  single_task_loss_function
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
import pandas as pd
import json

def load_model(model_path, latent_dim, device, dropout_p=0, model_type='simplified'):
    """
    加载保存的AutoEncoder模型
    参数:
        model_path: 模型权重文件路径
        latent_dim: 潜空间维度
        device: 计算设备
        dropout_p: Dropout概率
        model_type: 模型类型 ('simplified')
    返回:
        model: 加载的模型
    """
    if model_type == 'simplified':
        model = SimplifiedVonMisesAutoEncoder(latent_dim=latent_dim, dropout_p=dropout_p).to(device)

    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"模型已从 {model_path} 加载")
    return model

def test_simplified_model(model, test_loader, device, output_dir, num_samples=5):
    """
    测试简化AutoEncoder模型并评估性能（只有应力预测）
    参数:
        model: 要测试的简化模型
        test_loader: 测试数据加载器
        device: 计算设备
        output_dir: 结果输出目录
        num_samples: 可视化样本数量
    返回:
        stress_metrics: 应力预测评估指标字典
    """
    model.eval()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化损失和指标累积器
    total_loss = 0.0
    total_mse_loss = 0.0
    total_ssim_loss = 0.0
    total_gradient_loss = 0.0
    
    # 用于计算详细指标的列表
    all_stress_true = []
    all_stress_pred = []
    
    # 收集可视化样本
    visualization_samples = []
    
    # 存储每个样本的详细指标
    sample_metrics = []
    sample_index = 0
    
    with torch.no_grad():
        for batch_idx, (data, stress_target) in enumerate(test_loader):
            data = data.to(device)
            stress_target = stress_target.to(device)
            
            # 前向传播
            stress_pred = model(data)
            
            # 计算损失
            loss_tuple = single_task_loss_function(
                stress_pred, stress_target,
                mse_weight=1.0, ssim_weight=1.0, gradient_weight=1.0
            )
            
            total_loss_batch, mse_loss, ssim_loss, gradient_loss = loss_tuple
            
            # 累积损失
            total_loss += total_loss_batch.item()
            total_mse_loss += mse_loss.item()
            total_ssim_loss += ssim_loss.item()
            total_gradient_loss += gradient_loss.item()
            
            # 收集数据用于详细指标计算
            stress_pred_np = stress_pred.cpu().numpy()
            stress_target_np = stress_target.cpu().numpy()
            
            all_stress_true.extend(stress_target_np.flatten())
            all_stress_pred.extend(stress_pred_np.flatten())
            
            # 为每个样本计算单独的指标
            for i in range(data.size(0)):
                sample_stress_true = stress_target_np[i].flatten()
                sample_stress_pred = stress_pred_np[i].flatten()
                
                # 计算单样本指标
                sample_mse = mean_squared_error(sample_stress_true, sample_stress_pred)
                sample_rmse = np.sqrt(sample_mse)
                sample_mae = mean_absolute_error(sample_stress_true, sample_stress_pred)
                sample_r2 = r2_score(sample_stress_true, sample_stress_pred)
                
                # 计算MAPE（平均绝对百分比误差）
                # 避免除零错误，添加小的epsilon值
                epsilon = 1e-8
                sample_mape = np.mean(np.abs((sample_stress_true - sample_stress_pred) / (sample_stress_true + epsilon))) * 100
                
                # 计算SSIM和PSNR
                stress_true_img = stress_target_np[i].squeeze()
                stress_pred_img = stress_pred_np[i].squeeze()
                
                # 确保值在[0,1]范围内
                stress_true_img = np.clip(stress_true_img, 0, 1)
                stress_pred_img = np.clip(stress_pred_img, 0, 1)
                
                sample_ssim = ssim_metric(stress_true_img, stress_pred_img, data_range=1.0)
                sample_psnr = psnr_metric(stress_true_img, stress_pred_img, data_range=1.0)
                
                # 存储样本指标
                sample_metric = {
                    'sample_index': sample_index,
                    'batch_index': batch_idx,
                    'batch_sample_index': i,
                    'mse': float(sample_mse),
                    'rmse': float(sample_rmse),
                    'mae': float(sample_mae),
                    'mape': float(sample_mape),
                    'r2': float(sample_r2),
                    'ssim': float(sample_ssim),
                    'psnr': float(sample_psnr)
                }
                
                sample_metrics.append(sample_metric)
                
                # 打印每个样本的指标
                print(f"样本 {sample_index:4d}: MSE={sample_mse:.6f}, RMSE={sample_rmse:.6f}, MAE={sample_mae:.6f}, MAPE={sample_mape:.2f}%, R²={sample_r2:.6f}, SSIM={sample_ssim:.6f}, PSNR={sample_psnr:.6f}")
                
                sample_index += 1
            
            # 收集可视化样本
            if len(visualization_samples) < num_samples:
                for i in range(min(num_samples - len(visualization_samples), data.size(0))):
                    visualization_samples.append({
                        'input': data[i].cpu().numpy(),
                        'stress_true': stress_target[i].cpu().numpy(),
                        'stress_pred': stress_pred[i].cpu().numpy()
                    })
    
    # 计算平均损失
    num_batches = len(test_loader)
    avg_total_loss = total_loss / num_batches
    avg_mse_loss = total_mse_loss / num_batches
    avg_ssim_loss = total_ssim_loss / num_batches
    avg_gradient_loss = total_gradient_loss / num_batches
    
    # 计算详细的应力预测指标
    all_stress_true = np.array(all_stress_true)
    all_stress_pred = np.array(all_stress_pred)
    
    stress_mse = mean_squared_error(all_stress_true, all_stress_pred)
    stress_rmse = np.sqrt(stress_mse)
    stress_mae = mean_absolute_error(all_stress_true, all_stress_pred)
    stress_r2 = r2_score(all_stress_true, all_stress_pred)
    
    # 计算整体MAPE
    epsilon = 1e-8
    stress_mape = np.mean(np.abs((all_stress_true - all_stress_pred) / (all_stress_true + epsilon))) * 100
    
    # 计算SSIM和PSNR（基于样本）
    ssim_scores = []
    psnr_scores = []
    
    for sample in visualization_samples:
        stress_true = sample['stress_true'].squeeze()
        stress_pred = sample['stress_pred'].squeeze()
        
        # 确保值在[0,1]范围内
        stress_true = np.clip(stress_true, 0, 1)
        stress_pred = np.clip(stress_pred, 0, 1)
        
        ssim_score = ssim_metric(stress_true, stress_pred, data_range=1.0)
        psnr_score = psnr_metric(stress_true, stress_pred, data_range=1.0)
        
        ssim_scores.append(ssim_score)
        psnr_scores.append(psnr_score)
    
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
    avg_psnr = np.mean(psnr_scores) if psnr_scores else 0.0
    
    # 组织应力指标
    stress_metrics = {
        'mse': float(stress_mse),
        'rmse': float(stress_rmse),
        'mae': float(stress_mae),
        'mape': float(stress_mape),
        'r2': float(stress_r2),
        'ssim': float(avg_ssim),
        'psnr': float(avg_psnr)
    }
    
    # 计算样本指标的统计信息
    sample_stats = {
        'mse': {
            'mean': float(np.mean([s['mse'] for s in sample_metrics])),
            'std': float(np.std([s['mse'] for s in sample_metrics])),
            'min': float(np.min([s['mse'] for s in sample_metrics])),
            'max': float(np.max([s['mse'] for s in sample_metrics]))
        },
        'rmse': {
            'mean': float(np.mean([s['rmse'] for s in sample_metrics])),
            'std': float(np.std([s['rmse'] for s in sample_metrics])),
            'min': float(np.min([s['rmse'] for s in sample_metrics])),
            'max': float(np.max([s['rmse'] for s in sample_metrics]))
        },
        'mae': {
            'mean': float(np.mean([s['mae'] for s in sample_metrics])),
            'std': float(np.std([s['mae'] for s in sample_metrics])),
            'min': float(np.min([s['mae'] for s in sample_metrics])),
            'max': float(np.max([s['mae'] for s in sample_metrics]))
        },
        'mape': {
            'mean': float(np.mean([s['mape'] for s in sample_metrics])),
            'std': float(np.std([s['mape'] for s in sample_metrics])),
            'min': float(np.min([s['mape'] for s in sample_metrics])),
            'max': float(np.max([s['mape'] for s in sample_metrics]))
        },
        'r2': {
            'mean': float(np.mean([s['r2'] for s in sample_metrics])),
            'std': float(np.std([s['r2'] for s in sample_metrics])),
            'min': float(np.min([s['r2'] for s in sample_metrics])),
            'max': float(np.max([s['r2'] for s in sample_metrics]))
        },
        'ssim': {
            'mean': float(np.mean([s['ssim'] for s in sample_metrics])),
            'std': float(np.std([s['ssim'] for s in sample_metrics])),
            'min': float(np.min([s['ssim'] for s in sample_metrics])),
            'max': float(np.max([s['ssim'] for s in sample_metrics]))
        },
        'psnr': {
            'mean': float(np.mean([s['psnr'] for s in sample_metrics])),
            'std': float(np.std([s['psnr'] for s in sample_metrics])),
            'min': float(np.min([s['psnr'] for s in sample_metrics])),
            'max': float(np.max([s['psnr'] for s in sample_metrics]))
        }
    }
    
    # 打印结果
    print(f"\n=== 简化模型测试结果 ===")
    print(f"平均总损失: {avg_total_loss:.6f}")
    print(f"平均MSE损失: {avg_mse_loss:.6f}")
    print(f"平均SSIM损失: {avg_ssim_loss:.6f}")
    print(f"平均梯度损失: {avg_gradient_loss:.6f}")
    print(f"\n=== 整体应力预测指标 ===")
    print(f"MSE: {stress_metrics['mse']:.6f}")
    print(f"RMSE: {stress_metrics['rmse']:.6f}")
    print(f"MAE: {stress_metrics['mae']:.6f}")
    print(f"R²: {stress_metrics['r2']:.6f}")
    print(f"SSIM: {stress_metrics['ssim']:.6f}")
    print(f"PSNR: {stress_metrics['psnr']:.6f}")
    
    print(f"\n=== 样本指标统计 ===")
    print(f"总样本数: {len(sample_metrics)}")
    for metric_name in ['mse', 'rmse', 'mae', 'r2', 'ssim', 'psnr']:
        stats = sample_stats[metric_name]
        print(f"{metric_name.upper()}: 均值={stats['mean']:.6f}, 标准差={stats['std']:.6f}, 最小值={stats['min']:.6f}, 最大值={stats['max']:.6f}")
    
    # 保存指标到Excel文件
    # 1. 保存整体指标
    overall_metrics_df = pd.DataFrame({
        '指标类型': ['总损失', 'MSE损失', 'SSIM损失', '梯度损失', 'MSE', 'RMSE', 'MAE', 'MAPE(%)', 'R²', 'SSIM', 'PSNR'],
        '数值': [
            float(avg_total_loss),
            float(avg_mse_loss), 
            float(avg_ssim_loss),
            float(avg_gradient_loss),
            float(stress_mse),
            float(stress_rmse),
            float(stress_mae),
            float(stress_mape),
            float(stress_r2),
            float(avg_ssim),
            float(avg_psnr)
        ]
    })
    
    # 2. 保存样本统计信息
    stats_data = []
    for metric_name in ['mse', 'rmse', 'mae', 'mape', 'r2', 'ssim', 'psnr']:
        stats = sample_stats[metric_name]
        stats_data.append({
            '指标': metric_name.upper(),
            '均值': stats['mean'],
            '标准差': stats['std'],
            '最小值': stats['min'],
            '最大值': stats['max']
        })
    
    sample_stats_df = pd.DataFrame(stats_data)
    
    # 3. 保存每个样本的详细指标
    sample_metrics_df = pd.DataFrame(sample_metrics)
    
    # 保存到Excel文件（多个工作表）
    excel_file_path = os.path.join(output_dir, 'simplified_test_metrics.xlsx')
    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        overall_metrics_df.to_excel(writer, sheet_name='整体指标', index=False)
        sample_stats_df.to_excel(writer, sheet_name='样本统计', index=False)
        sample_metrics_df.to_excel(writer, sheet_name='单样本指标', index=False)
    
    # 单独保存每个样本的详细指标到单独的Excel文件
    sample_metrics_df.to_excel(os.path.join(output_dir, 'individual_sample_metrics.xlsx'), index=False)
    
    # 保存样本统计信息到单独的Excel文件
    sample_stats_df.to_excel(os.path.join(output_dir, 'sample_statistics.xlsx'), index=False)
    
    print(f"\n详细指标已保存到:")
    print(f"- 完整指标: {excel_file_path}")
    print(f"- 单样本指标: {os.path.join(output_dir, 'individual_sample_metrics.xlsx')}")
    print(f"- 统计信息: {os.path.join(output_dir, 'sample_statistics.xlsx')}")
    
    # 可视化样本
    visualize_simplified_samples(visualization_samples, output_dir)
    
    return stress_metrics

def visualize_simplified_samples(samples, output_dir):
    """
    可视化简化模型的测试样本（应力预测）
    """
    num_samples = len(samples)
    if num_samples == 0:
        return
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(samples):
        input_img = sample['input'].squeeze()
        stress_true = sample['stress_true'].squeeze()
        stress_pred = sample['stress_pred'].squeeze()
        
        # 输入图像
        axes[i, 0].imshow(input_img, cmap='gray')
        axes[i, 0].set_title(f'Sample {i+1}: Input')
        axes[i, 0].axis('off')
        
        # 真实应力
        im1 = axes[i, 1].imshow(stress_true, cmap='jet', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Sample {i+1}: True Stress')
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # 预测应力
        im2 = axes[i, 2].imshow(stress_pred, cmap='jet', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Sample {i+1}: Predicted Stress')
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'simplified_test_samples.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化结果已保存到: {os.path.join(output_dir, 'simplified_test_samples.png')}")

def main():
    parser = argparse.ArgumentParser(description='Test AutoEncoder for Von Mises Stress Prediction')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth', help='path to the trained model')
    parser.add_argument('--latent_dim', type=int, default=256, help='dimension of latent space')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for testing')
    parser.add_argument('--output_dir', type=str, default='test_results', help='directory to save test results')
    parser.add_argument('--cuda', action='store_true', default=True, help='enable CUDA')
    parser.add_argument('--num_samples', type=int, default=5, help='number of samples to visualize')
    parser.add_argument('--dropout_p', type=float, default=0.2, help='Dropout probability for model layers')
    parser.add_argument('--model_type', type=str, default='simplified', choices=['simplified'])
    parser.add_argument('--crack_dir', type=str, default='../data/test/crack_arrays_256', help='directory containing crack images')
    parser.add_argument('--stress_dir', type=str, default='../data/test/stress', help='directory containing stress data')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建测试数据集和数据加载器（使用全部数据）
    test_dataset = VonMisesDataset(
        crack_dir=args.crack_dir,
        stress_dir=args.stress_dir,
        transform=None,
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f'测试数据集大小: {len(test_dataset)}')
    
    # 加载模型
    model = load_model(args.model_path, args.latent_dim, device, args.dropout_p, args.model_type)
    
    # 测试模型
    if args.model_type == 'simplified':
        stress_metrics = test_simplified_model(
            model, test_loader, device, args.output_dir, args.num_samples
        )
    
    print("\n测试完成！")

if __name__ == '__main__':
    main()