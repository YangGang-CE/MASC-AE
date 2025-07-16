import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
import re
import random


class VonMisesDataset(Dataset):
    def __init__(self, crack_dir, stress_dir, transform=None, augment=False):
        """
        Von Mises应力数据集
        
        参数:
            crack_dir: 裂缝图像目录
            stress_dir: 应力数据目录
            transform: 数据转换
            augment: 是否进行数据增强
        """
        self.crack_dir = crack_dir
        self.stress_dir = stress_dir
        self.transform = transform
        self.augment = augment

        # 获取所有裂缝文件
        self.crack_files = sorted(glob.glob(os.path.join(crack_dir, "*crack_*.npy")))
        
        # 构建配对映射
        self.pairs = self._build_pairs()

    def _build_pairs(self):
        """构建输入与标签的配对"""
        pairs = []
        
        for crack_file in self.crack_files:
            # 从文件名提取A和B
            match = re.search(r'crack_(\d+)_(\d+)\.npy', os.path.basename(crack_file))
            if match:
                a, b = match.groups()
                
                # 构建对应的应力文件名
                stress_file = os.path.join(self.stress_dir, f"{a}_var_Stress_{b}_Z.npy")

                # 检查应力文件是否存在
                if os.path.exists(stress_file):
                    pairs.append((crack_file, stress_file))
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        crack_file, stress_file = self.pairs[idx]
        
        # 加载数据
        crack_data = np.load(crack_file).astype(np.float32)
        stress_data = np.load(stress_file).astype(np.float32)
        
        # 添加通道维度并转换为Tensor
        crack_tensor = torch.from_numpy(crack_data).unsqueeze(0)  # (1, 512, 512)
        stress_tensor = torch.from_numpy(stress_data).unsqueeze(0)  # (1, 512, 512)
        
        # 修复：确保裂缝图像和应力图像应用相同的空间变换
        if self.augment:
            # 设置随机种子确保相同的变换
            seed = random.randint(0, 2**32 - 1)
            
            # 对裂缝图像应用空间变换
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            
            # 定义空间变换（不包括噪声）
            spatial_transforms = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=20, fill=0),
            ])
            
            crack_tensor = spatial_transforms(crack_tensor)
            
            # 对应力图像应用相同的空间变换
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            
            stress_tensor = spatial_transforms(stress_tensor)
            
            # 分别添加噪声（噪声可以不同）
            crack_tensor = crack_tensor + 0.01 * torch.randn_like(crack_tensor)
            stress_tensor = stress_tensor + 0.01 * torch.randn_like(stress_tensor)

        # 应用其他转换（如归一化等）
        if self.transform:
            crack_tensor = self.transform(crack_tensor)
            stress_tensor = self.transform(stress_tensor)
            
        # 数据验证
        assert not torch.isnan(crack_tensor).any(), f"crack_tensor NaN: {crack_file}"
        assert not torch.isnan(stress_tensor).any(), f"stress_tensor NaN: {stress_file}"
        assert not torch.isinf(crack_tensor).any(), f"crack_tensor Inf: {crack_file}"
        assert not torch.isinf(stress_tensor).any(), f"stress_tensor Inf: {stress_file}"

        return crack_tensor, stress_tensor


def create_dataloaders(crack_dir, stress_dir, batch_size=8, train_ratio=0.8, val_ratio=0.1,
                      num_workers=4, shuffle=True, transform=None):
    """
    创建训练、验证和测试数据加载器
    
    参数:
        crack_dir: 裂缝图像目录
        stress_dir: 应力数据目录
        batch_size: 批次大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        num_workers: 数据加载线程数
        shuffle: 是否打乱数据
        transform: 数据转换
    
    返回:
        train_loader, val_loader, test_loader
    """
    # 创建基础数据集（不进行数据增强）
    dataset = VonMisesDataset(crack_dir, stress_dir, transform, augment=False)
    
    # 计算分割索引
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # 分割数据集
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 为训练集启用数据增强，验证/测试集不使用
    train_dataset.dataset.augment = True
    val_dataset.dataset.augment = False
    test_dataset.dataset.augment = False

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader