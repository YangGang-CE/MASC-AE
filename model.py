import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 添加SSIM计算类
class SSIM(nn.Module):
    """
    结构相似性指数 (SSIM) 损失计算
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            math.exp(-(x - window_size//2)**2/float(2*sigma**2))
            for x in range(window_size)
        ])
        return gauss/gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


# 添加简化版SSIM计算函数
def ssim(img1, img2, window_size=11, size_average=True):
    """
    计算两个图像之间的SSIM
    """
    ssim_module = SSIM(window_size, size_average)
    # 确保SSIM模块与输入在同一设备上
    device = img1.device
    ssim_module = ssim_module.to(device)
    return ssim_module(img1, img2)


# 添加自注意力模块
class SelfAttention(nn.Module):
    """自注意力模块，用于捕获空间依赖关系"""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的缩放因子

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # 生成查询、键、值
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B x (H*W) x C'
        proj_key = self.key(x).view(batch_size, -1, height * width)  # B x C' x (H*W)
        attention = torch.bmm(proj_query, proj_key)  # B x (H*W) x (H*W)
        attention = F.softmax(attention, dim=-1)  # 注意力图
        
        proj_value = self.value(x).view(batch_size, -1, height * width)  # B x C x (H*W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x (H*W)
        out = out.view(batch_size, channels, height, width)  # B x C x H x W
        
        # 残差连接
        out = self.gamma * out + x
        return out


# 添加通道注意力模块
class ChannelAttention(nn.Module):
    """通道注意力模块，突出重要特征通道"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


# 添加空间注意力模块
class SpatialAttention(nn.Module):
    """空间注意力模块，关注重要空间位置"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 生成空间特征描述符
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        
        return self.sigmoid(out) * x


# 添加CBAM注意力模块（结合通道和空间注意力）
class CBAM(nn.Module):
    """CBAM注意力模块，结合通道和空间注意力"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# 添加FPN特征金字塔网络模块
class FeaturePyramidNetwork(nn.Module):
    """特征金字塔网络，用于多尺度特征融合"""
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.inner_layers = nn.ModuleList()
        self.outer_layers = nn.ModuleList()
        
        for in_channels in in_channels_list:
            inner_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            outer_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            
            self.inner_layers.append(inner_conv)
            self.outer_layers.append(outer_conv)
    
    def forward(self, features):
        # 从最深层开始处理
        last_inner = self.inner_layers[-1](features[-1])
        results = [self.outer_layers[-1](last_inner)]
        
        # 自顶向下处理
        for i in range(len(features) - 2, -1, -1):
            # 上采样
            upsample = F.interpolate(last_inner, size=features[i].shape[-2:], mode='nearest')
            # 横向连接
            inner = self.inner_layers[i](features[i]) + upsample
            # 保存当前层结果
            last_inner = inner
            results.insert(0, self.outer_layers[i](inner))
        
        return results


# 改进的编码器模块，包含注意力机制
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_attention=True, dropout_p=0.2):
        super(EncoderBlock, self).__init__()
        self.dropout_p = dropout_p
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(self.dropout_p),
        )
        
        # 残差连接：当输入输出通道数不同或步长不为1时，需要调整维度
        self.use_residual = True
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAM(out_channels)
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        
        # 残差连接 - 修复尺寸不匹配问题
        if self.use_residual:
            # 确保residual和out的尺寸完全匹配
            if residual.shape != out.shape:
                # 使用插值调整residual的尺寸以匹配out
                residual = F.interpolate(residual, size=out.shape[2:], mode='bilinear', align_corners=False)
            out = out + residual
        
        if self.use_attention:
            out = self.attention(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_attention=True, dropout_p=0.2):
        super(DecoderBlock, self).__init__()
        self.dropout_p = dropout_p
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(self.dropout_p),
        )
        
        # 残差连接：对于转置卷积，需要特殊处理
        self.use_residual = True
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAM(out_channels)
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.deconv(x)
        
        # 残差连接 - 修复尺寸不匹配问题
        if self.use_residual:
            # 确保residual和out的尺寸完全匹配
            if residual.shape != out.shape:
                # 使用插值调整residual的尺寸以匹配out
                residual = F.interpolate(residual, size=out.shape[2:], mode='bilinear', align_corners=False)
            out = out + residual
        
        if self.use_attention:
            out = self.attention(out)
        return out


class SimplifiedVonMisesAutoEncoder(nn.Module):
    def __init__(self, latent_dim=512, dropout_p=0.2):
        super(SimplifiedVonMisesAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.dropout_p = dropout_p

        # 编码器
        self.enc1 = EncoderBlock(1, 16, kernel_size=3, stride=1, padding=1, dropout_p=dropout_p)
        self.enc2 = EncoderBlock(16, 32, dropout_p=dropout_p)
        self.enc3 = EncoderBlock(32, 64, dropout_p=dropout_p)
        self.enc4 = EncoderBlock(64, 128, dropout_p=dropout_p)
        self.enc5 = EncoderBlock(128, 256, dropout_p=dropout_p)
        self.global_attention = SelfAttention(256)
        self.fpn = FeaturePyramidNetwork([16, 32, 64, 128, 256], 128)
        
        # 潜空间
        self.flat_size = 256 * 16 * 16
        self.fc_enc = nn.Linear(self.flat_size, latent_dim)
        
        # 应力预测解码器（只保留这一个解码器）
        self.fc_dec = nn.Linear(latent_dim, self.flat_size)
        self.stress_dec1 = DecoderBlock(256, 128, dropout_p=dropout_p)
        self.stress_dec2 = DecoderBlock(128, 64, dropout_p=dropout_p)
        self.stress_dec3 = DecoderBlock(64, 32, dropout_p=dropout_p)
        self.stress_dec4 = DecoderBlock(32, 16, dropout_p=dropout_p)
        self.stress_output_layer = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        # FPN适配层
        self.fpn_adapt1 = nn.Conv2d(128, 16, kernel_size=1)
        self.fpn_adapt2 = nn.Conv2d(128, 32, kernel_size=1)
        self.fpn_adapt3 = nn.Conv2d(128, 64, kernel_size=1)
        self.fpn_adapt4 = nn.Conv2d(128, 128, kernel_size=1)

    def encode(self, x):
        features = []
        e1 = self.enc1(x)
        features.append(e1)
        e2 = self.enc2(e1)
        features.append(e2)
        e3 = self.enc3(e2)
        features.append(e3)
        e4 = self.enc4(e3)
        features.append(e4)
        e5 = self.enc5(e4)
        features.append(e5)
        e5 = self.global_attention(e5)
        self.feature_maps = features
        self.fpn_features = self.fpn(features)
        x = e5.view(e5.size(0), -1)
        latent = self.fc_enc(x)
        return latent

    def decode(self, latent):
        """应力预测解码器"""
        x = self.fc_dec(latent)
        x = x.view(-1, 256, 16, 16)
        
        if not hasattr(self, 'fpn_features'):
            x = self.stress_dec1(x)
            x = self.stress_dec2(x)
            x = self.stress_dec3(x)
            x = self.stress_dec4(x)
            return self.stress_output_layer(x)
        
        # 使用FPN特征进行多尺度融合
        x1 = self.stress_dec1(x)
        fpn4_resized = F.interpolate(self.fpn_adapt4(self.fpn_features[4]), size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = x1 + fpn4_resized
        
        x2 = self.stress_dec2(x)
        fpn3_resized = F.interpolate(self.fpn_adapt3(self.fpn_features[3]), size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = x2 + fpn3_resized
        
        x3 = self.stress_dec3(x)
        fpn2_resized = F.interpolate(self.fpn_adapt2(self.fpn_features[2]), size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = x3 + fpn2_resized
        
        x4 = self.stress_dec4(x)
        fpn1_resized = F.interpolate(self.fpn_adapt1(self.fpn_features[1]), size=x4.shape[2:], mode='bilinear', align_corners=False)
        x = x4 + fpn1_resized
        
        return self.stress_output_layer(x)

    def forward(self, x):
        """前向传播，只返回应力预测"""
        latent = self.encode(x)
        stress_pred = self.decode(latent)
        return stress_pred