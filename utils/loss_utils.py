import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ssim


def gradient_loss(pred, target, reduction='mean'):
    """
    计算梯度损失，用于提升应力预测的锐度
    使用Sobel算子计算图像梯度，然后比较预测和真实图像的梯度差异
    
    参数:
        pred: 预测的应力图，形状为 (B, 1, H, W)
        target: 真实的应力图，形状为 (B, 1, H, W)
        reduction: 损失缩减方式 ('mean', 'sum', 'none')
    
    返回:
        gradient_loss: 梯度损失
    """
    # Sobel算子
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    
    # 扩展维度以匹配卷积要求 (out_channels, in_channels, H, W)
    sobel_x = sobel_x.view(1, 1, 3, 3).to(pred.device)
    sobel_y = sobel_y.view(1, 1, 3, 3).to(pred.device)
    
    # 计算预测图像的梯度
    pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
    pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
    pred_grad_magnitude = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
    
    # 计算真实图像的梯度
    target_grad_x = F.conv2d(target, sobel_x, padding=1)
    target_grad_y = F.conv2d(target, sobel_y, padding=1)
    target_grad_magnitude = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
    
    # 计算梯度损失
    grad_loss = F.mse_loss(pred_grad_magnitude, target_grad_magnitude, reduction=reduction)
    
    return grad_loss

def single_task_loss_function(stress_pred, stress_target, mse_weight=1.0, ssim_weight=1.0, gradient_weight=1.0):
    """
    单任务应力预测损失函数
    
    参数:
        stress_pred: 预测的应力图 [B, 1, H, W]
        stress_target: 真实的应力图 [B, 1, H, W]
        mse_weight: MSE损失权重
        ssim_weight: SSIM损失权重
        gradient_weight: 梯度损失权重
    
    返回:
        total_loss: 总损失
        mse_loss: MSE损失
        ssim_loss: SSIM损失
        gradient_loss: 梯度损失
    """
    # MSE损失
    mse_loss = F.mse_loss(stress_pred, stress_target)
    
    # SSIM损失
    ssim_loss = 1 - ssim(stress_pred, stress_target, size_average=True)
    
    # 梯度损失
    gradient_loss_value = gradient_loss(stress_pred, stress_target)
    
    # 总损失
    total_loss = (mse_weight * mse_loss + 
                  ssim_weight * ssim_loss + 
                  gradient_weight * gradient_loss_value)
    
    return total_loss, mse_loss, ssim_loss, gradient_loss_value