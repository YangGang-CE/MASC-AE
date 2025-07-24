import argparse
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import gc

from dataset import create_dataloaders
from utils.loss_utils import single_task_loss_function


def train_simplified_model(model, train_loader, val_loader, optimizer, device, 
                          num_epochs=100, save_dir='checkpoints', log_interval=10, 
                          patience=10, delta=1e-4, lr_scheduler=None, 
                          mse_weight=1.0, ssim_weight=1.0, gradient_weight=1.0):
    """
    简化单任务训练模型 - 只训练应力预测任务
    
    参数:
        model: SimplifiedVonMisesAutoEncoder模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        device: 计算设备
        num_epochs: 训练轮数
        save_dir: 保存目录
        log_interval: 日志记录间隔
        patience: 早停耐心值
        delta: 早停最小改善值
        lr_scheduler: 学习率调度器
        mse_weight: MSE损失权重
        ssim_weight: SSIM损失权重
        gradient_weight: 梯度损失权重
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard_logs'))
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_stress_loss': [],
        'val_stress_loss': []
    }
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    
    print(f"开始简化单任务训练，共 {num_epochs} 轮")
    print(f"模型保存路径: {save_dir}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_stress_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (data, targets) in enumerate(train_pbar):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            stress_pred = model(data)
            
            total_loss, mse_loss, ssim_loss, gradient_loss = single_task_loss_function(
                stress_pred, targets,
                mse_weight=mse_weight,
                ssim_weight=ssim_weight,
                gradient_weight=gradient_weight
            )
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            train_stress_loss += total_loss.item()  # 在简化模型中，总损失就是应力损失
            
            train_pbar.set_postfix({
                'Loss': f'{total_loss.item():.6f}'
            })
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_stress_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for data, targets in val_pbar:
                data, targets = data.to(device), targets.to(device)
                
                stress_pred = model(data)
                
                from utils.loss_utils import single_task_loss_function
                total_loss, mse_loss, ssim_loss, gradient_loss = single_task_loss_function(
                    stress_pred, targets,
                    mse_weight=mse_weight,
                    ssim_weight=ssim_weight,
                    gradient_weight=gradient_weight
                )
                
                val_loss += total_loss.item()
                val_stress_loss += total_loss.item()
                
                val_pbar.set_postfix({
                    'Val Loss': f'{total_loss.item():.6f}'
                })
        
        # 计算平均损失
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_stress_loss = train_stress_loss / len(train_loader)
        avg_val_stress_loss = val_stress_loss / len(val_loader)
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_stress_loss'].append(avg_train_stress_loss)
        history['val_stress_loss'].append(avg_val_stress_loss)
        
        # TensorBoard记录
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Stress_Loss/Train', avg_train_stress_loss, epoch)
        writer.add_scalar('Stress_Loss/Validation', avg_val_stress_loss, epoch)
        
        # 学习率调度
        if lr_scheduler:
            lr_scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # 早停检查
        if avg_val_loss < best_val_loss - delta:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'history': history
            }, best_model_path)
            print(f"\n保存最佳模型 (Epoch {epoch+1}): Val Loss = {avg_val_loss:.6f}")
        else:
            epochs_without_improvement += 1
        
        # 打印训练信息
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {avg_val_loss:.6f}")
        print(f"  Train Stress Loss: {avg_train_stress_loss:.6f}")
        print(f"  Val Stress Loss: {avg_val_stress_loss:.6f}")
        
        if lr_scheduler:
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 定期保存模型和可视化
        if (epoch + 1) % log_interval == 0:
            checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'history': history
            }, checkpoint_path)
            
            # 可视化结果（简化版，只显示应力预测）
            visualize_simplified_results(model, val_loader, device, save_dir, epoch)
        
        # 早停
        if epochs_without_improvement >= patience:
            print(f"\n早停触发！已连续 {patience} 轮没有改善")
            break
    
    # 保存最终模型
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_loss,
        'history': history
    }, final_model_path)
    
    writer.close()
    print(f"\n简化单任务训练完成！")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"模型保存在: {save_dir}")
    
    # 清理内存
    torch.cuda.empty_cache()
    gc.collect()
    
    return history


def visualize_simplified_results(model, data_loader, device, save_dir, epoch, num_samples=3):
    """可视化简化模型结果，只显示应力预测"""
    img_dir = os.path.join(save_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    data_iter = iter(data_loader)
    data, targets = next(data_iter)
    data, targets = data.to(device), targets.to(device)
    
    with torch.no_grad():
        stress_pred = model(data)
    
    # 转换为numpy数组
    data_np = data.cpu().numpy()
    targets_np = targets.cpu().numpy()
    stress_pred_np = stress_pred.cpu().numpy()
    
    # 可视化前几个样本
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(num_samples, data.size(0))):
        # 输入裂缝图
        axes[i, 0].imshow(data_np[i, 0], cmap='gray')
        axes[i, 0].set_title(f'Input Crack {i+1}')
        axes[i, 0].axis('off')
        
        # 真实应力图
        axes[i, 1].imshow(targets_np[i, 0], cmap='viridis')
        axes[i, 1].set_title(f'GT Stress {i+1}')
        axes[i, 1].axis('off')
        
        # 预测应力图
        axes[i, 2].imshow(stress_pred_np[i, 0], cmap='viridis')
        axes[i, 2].set_title(f'Pred Stress {i+1}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f'results_epoch_{epoch+1}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='MASC-AE: Multi-Attention Von Mises Stress Correlation AutoEncoder')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=256, help='dimension of latent space')
    parser.add_argument('--cuda', action='store_true', default=True, help='enable CUDA')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--log_interval', type=int, default=10, help='interval for model saving and logging')
    parser.add_argument('--save_dir', type=str, default='checkpoints/', help='directory to save models')
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    parser.add_argument('--delta', type=float, default=0.0001, help='minimum change to be considered as improvement')

    parser.add_argument('--dropout_p', type=float, default=0.3, help='Dropout probability for model layers')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 regularization) for optimizer')
    
    # 损失函数权重
    parser.add_argument('--stress_weight', type=float, default=1.0, help='Weight for the total stress loss')
    parser.add_argument('--mse_weight', type=float, default=1.0, help='Weight for MSE loss in stress loss')
    parser.add_argument('--ssim_weight', type=float, default=1.0, help='Weight for SSIM loss in stress loss')
    parser.add_argument('--gradient_weight', type=float, default=2, help='Weight for Gradient loss in stress loss')

    # 学习率调度器相关参数
    parser.add_argument('--use_lr_scheduler', action='store_true', default=True, help='使用学习率调度器')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='学习率衰减因子')
    parser.add_argument('--lr_patience', type=int, default=10, help='学习率调度器耐心值')
    parser.add_argument('--lr_threshold', type=float, default=1e-4, help='学习率调度器阈值')
    parser.add_argument('--lr_min', type=float, default=1e-7, help='最小学习率')
    

    # 训练策略选择参数
    parser.add_argument('--training_strategy', type=str, default='simplified', 
                   choices=['simplified'])
    
    # 解析命令行参数 - 这一行是关键！
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    

    # 设置数据路径
    crack_dir = '../data/test/crack_arrays_256'
    stress_dir = '../data/test/stress'
    
    train_loader, val_loader, test_loader = create_dataloaders(
        crack_dir=crack_dir,
        stress_dir=stress_dir,
        batch_size=args.batch_size,
        train_ratio=0.8,
        val_ratio=0.1,
        num_workers=4,
        shuffle=True,
        transform=None
    )
    
    # 在训练策略选择部分修改
    if args.training_strategy == 'simplified':
        print(f"\n使用简化单任务训练策略: {args.epochs} epochs (应力预测)")
        

        # args.save_dir = 'checkpoints/simplified'
        
        # 创建简化模型
        from model import SimplifiedVonMisesAutoEncoder
        model = SimplifiedVonMisesAutoEncoder(latent_dim=args.latent_dim, dropout_p=args.dropout_p).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # 学习率调度器
        lr_scheduler = None
        if args.use_lr_scheduler:
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience,
                threshold=args.lr_threshold, min_lr=args.lr_min, verbose=True
            )
    
        
        
        history = train_simplified_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=args.epochs,
            save_dir=args.save_dir,
            log_interval=args.log_interval,
            patience=args.patience,
            delta=args.delta,
            lr_scheduler=lr_scheduler,
            mse_weight=args.mse_weight,
            ssim_weight=args.ssim_weight,
            gradient_weight=args.gradient_weight
        )
    
    # 绘制训练历史
    plot_history(history, args.save_dir)
    print(f"\n训练完成！模型和日志保存在: {args.save_dir}")


if __name__ == '__main__':
    main()