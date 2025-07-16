import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ssim  # 添加这行导入ssim函数


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

def mse_loss(pred, target, reduction='mean'):
    """
    计算应力图的均方误差损失
    
    参数:
        pred: 预测的应力图，形状为 (B, 1, H, W)
        target: 真实的应力图，形状为 (B, 1, H, W)
        reduction: 损失缩减方式 ('mean', 'sum', 'none')
    
    返回:
        mse_loss: 均方误差损失
    """
    if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("pred and target must be torch.Tensor")
    
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    
    return F.mse_loss(pred, target, reduction=reduction)

def multi_task_loss_function(outputs, targets, mask_targets, stress_weight=1.0, mask_weight=1.0,mse_weight=1.0, ssim_weight=1.0, gradient_weight=1.0,focal_weight=0.5, dice_weight=0.5):
    """
        增强的多任务损失函数:    - 应力损失 = w_mse*MSE + w_ssim*SSIM + w_grad*Gradient    - Mask损失 = L1 Loss
    """
    stress_pred = outputs["stress_pred"]
    mask_pred = outputs["failure_mask_pred"]

    # --- 1. 应力重构损失 (独立加权和) ---
    stress_mse_loss = F.mse_loss(stress_pred, targets, reduction='mean')
    stress_ssim_loss = 1 - ssim(stress_pred, targets) # 损失 = 1 - 相似度
    stress_gradient_loss = gradient_loss(stress_pred, targets, reduction='mean')

    stress_loss = (mse_weight * stress_mse_loss +
                   ssim_weight * stress_ssim_loss +
                   gradient_weight * stress_gradient_loss)

    # --- 2. Mask回归损失 (L1) ---
    # 使用L1 Loss (Mean Absolute Error) 进行灰度Mask回归
    # 之前的Focal和Dice Loss不适用于回归任务
    mask_combined_loss = F.l1_loss(mask_pred, mask_targets, reduction='mean')

    # --- 3. 总损失 ---
    total_loss = stress_weight * stress_loss + mask_weight * mask_combined_loss

    # 返回所有损失分量用于日志记录 (注意返回的是原始的mse和ssim损失值，而不是加权后的)
    return total_loss, stress_loss, mask_combined_loss, stress_mse_loss, (1 - stress_ssim_loss)


def combined_loss(stress_pred, stress_gt, mask_pred, mask_gt, 
                 alpha=1.0, beta=1.0, 
                 stress_reduction='mean', mask_reduction='mean',
                 use_dice=False, use_focal=False,
                 pos_weight=None, focal_alpha=1.0, focal_gamma=2.0):
    """
    计算联合损失函数：应力MSE损失 + mask分割损失
    
    参数:
        stress_pred: 预测的应力图，形状为 (B, 1, H, W)
        stress_gt: 真实的应力图，形状为 (B, 1, H, W)
        mask_pred: 预测的mask概率图，形状为 (B, 1, H, W)
        mask_gt: 真实的mask标签，形状为 (B, 1, H, W)
        alpha: 应力损失权重
        beta: mask损失权重
        stress_reduction: 应力损失缩减方式
        mask_reduction: mask损失缩减方式
        use_dice: 是否使用Dice损失替代BCE
        use_focal: 是否使用Focal损失替代BCE
        pos_weight: BCE损失的正样本权重
        focal_alpha: Focal损失的平衡因子
        focal_gamma: Focal损失的聚焦参数
    
    返回:
        total_loss: 总损失
        stress_loss: 应力损失
        mask_loss: mask损失
    """
    # 计算应力MSE损失
    stress_loss = mse_loss(stress_pred, stress_gt, reduction=stress_reduction)
    
    # 计算mask损失
    if use_dice:
        mask_loss = dice_loss(mask_pred, mask_gt)
    elif use_focal:
        mask_loss = focal_loss(mask_pred, mask_gt, alpha=focal_alpha, gamma=focal_gamma, reduction=mask_reduction)
    else:
        mask_loss = bce_loss(mask_pred, mask_gt, reduction=mask_reduction, pos_weight=pos_weight)
    
    # 计算总损失
    total_loss = alpha * stress_loss + beta * mask_loss
    
    return total_loss, stress_loss, mask_loss

def adaptive_combined_loss(stress_pred, stress_gt, mask_pred, mask_gt,
                          base_alpha=1.0, base_beta=1.0,
                          adaptive_weight=True):
    """
    自适应权重的联合损失函数
    
    参数:
        stress_pred: 预测的应力图
        stress_gt: 真实的应力图
        mask_pred: 预测的mask概率图
        mask_gt: 真实的mask标签
        base_alpha: 基础应力损失权重
        base_beta: 基础mask损失权重
        adaptive_weight: 是否使用自适应权重
    
    返回:
        total_loss: 总损失
        stress_loss: 应力损失
        mask_loss: mask损失
        alpha: 实际使用的应力权重
        beta: 实际使用的mask权重
    """
    # 计算基础损失
    stress_loss = mse_loss(stress_pred, stress_gt)
    mask_loss = bce_loss(mask_pred, mask_gt)
    
    if adaptive_weight:
        # 自适应权重：根据损失大小动态调整
        stress_magnitude = stress_loss.detach()
        mask_magnitude = mask_loss.detach()
        
        # 归一化权重
        total_magnitude = stress_magnitude + mask_magnitude + 1e-8
        alpha = base_alpha * (mask_magnitude / total_magnitude + 0.5)
        beta = base_beta * (stress_magnitude / total_magnitude + 0.5)
    else:
        alpha = base_alpha
        beta = base_beta
    
    total_loss = alpha * stress_loss + beta * mask_loss
    
    return total_loss, stress_loss, mask_loss, alpha, beta

class CombinedLossModule(nn.Module):
    """
    联合损失函数模块，可以作为nn.Module使用
    """
    def __init__(self, alpha=1.0, beta=1.0, use_dice=False, use_focal=False,
                 pos_weight=None, focal_alpha=1.0, focal_gamma=2.0):
        super(CombinedLossModule, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.use_dice = use_dice
        self.use_focal = use_focal
        self.pos_weight = pos_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def forward(self, stress_pred, stress_gt, mask_pred, mask_gt):
        return combined_loss(
            stress_pred, stress_gt, mask_pred, mask_gt,
            alpha=self.alpha, beta=self.beta,
            use_dice=self.use_dice, use_focal=self.use_focal,
            pos_weight=self.pos_weight,
            focal_alpha=self.focal_alpha, focal_gamma=self.focal_gamma
        )

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