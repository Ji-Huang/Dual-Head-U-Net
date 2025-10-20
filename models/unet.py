# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.checkpoint import checkpoint


# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2 with optional improvements"""

#     def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.1):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
        
#         layers = [
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         ]
        
#         self.double_conv = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.double_conv(x)


# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels, dropout=0.1):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels, dropout=dropout)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)


# class Up(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True, dropout=0.1):
#         super().__init__()

#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout=dropout)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
        
#         # 处理尺寸不匹配问题
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         # 对称填充以确保尺寸匹配
#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
        
#         # 确保填充后尺寸完全匹配
#         assert x1.size()[2:] == x2.size()[2:], \
#             f"上采样特征尺寸 {x1.size()[2:]} 与跳跃连接特征尺寸 {x2.size()[2:]} 不匹配"
        
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)


# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)


# class AttentionBlock(nn.Module):
#     """Attention gate for better feature fusion"""
#     def __init__(self, F_g, F_l, F_int):
#         super(AttentionBlock, self).__init__()
#         self.W_g = nn.Sequential(
#             nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )
        
#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )
        
#         self.psi = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
        
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, g, x):
#         # 确保输入特征图尺寸匹配
#         if g.size()[2:] != x.size()[2:]:
#             # 使用与Up模块相同的填充策略
#             diffY = x.size()[2] - g.size()[2]
#             diffX = x.size()[3] - g.size()[3]
#             g = F.pad(g, [diffX // 2, diffX - diffX // 2,
#                          diffY // 2, diffY - diffY // 2])
        
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)

#         # 现在g1和x1应该有相同的尺寸，不需要额外插值
#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)
        
#         return x * psi


# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=False, dropout=0.1, use_attention=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#         self.use_attention = use_attention
#         self.dropout_rate = dropout

#         self.inc = DoubleConv(n_channels, 64, dropout=dropout)
#         self.down1 = Down(64, 128, dropout=dropout)
#         self.down2 = Down(128, 256, dropout=dropout)
#         self.down3 = Down(256, 512, dropout=dropout)
        
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor, dropout=dropout)
        
#         # 注意力门（如果启用）
#         if use_attention:
#             # 修正注意力块的参数，确保通道数匹配
#             self.att1 = AttentionBlock(F_g=1024//factor, F_l=512, F_int=256)
#             self.att2 = AttentionBlock(F_g=512//factor, F_l=256, F_int=128)
#             self.att3 = AttentionBlock(F_g=256//factor, F_l=128, F_int=64)
#             self.att4 = AttentionBlock(F_g=128//factor, F_l=64, F_int=32)
        
#         # 修正上采样模块的输入通道数
#         self.up1 = Up(1024, 512 // factor, bilinear, dropout=dropout)
#         self.up2 = Up(512, 256 // factor, bilinear, dropout=dropout)
#         self.up3 = Up(256, 128 // factor, bilinear, dropout=dropout)
#         self.up4 = Up(128, 64, bilinear, dropout=dropout)
        
#         self.outc = OutConv(64, n_classes)
        
#         # 初始化权重
#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         # 编码器
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
        
#         # 解码器（带可选注意力机制）
#         if self.use_attention:
#             # 使用注意力机制加权跳跃连接
#             x4_att = self.att1(x5, x4)
#             x_up1 = self.up1(x5, x4_att)
            
#             x3_att = self.att2(x_up1, x3)
#             x_up2 = self.up2(x_up1, x3_att)
            
#             x2_att = self.att3(x_up2, x2)
#             x_up3 = self.up3(x_up2, x2_att)
            
#             x1_att = self.att4(x_up3, x1)
#             x = self.up4(x_up3, x1_att)
#         else:
#             # 非注意力分支
#             x = self.up1(x5, x4)
#             x = self.up2(x, x3)
#             x = self.up3(x, x2)
#             x = self.up4(x, x1)
        
#         logits = self.outc(x)
#         return logits

#     def use_checkpointing(self):
#         """启用梯度检查点以节省内存"""
#         # 定义检查点包装器类
#         class CheckpointWrapper(nn.Module):
#             def __init__(self, module):
#                 super().__init__()
#                 self.module = module
                
#             def forward(self, *args, **kwargs):
#                 # 明确指定use_reentrant参数以避免警告
#                 return checkpoint(self.module, *args, use_reentrant=False, **kwargs)
        
#         # 处理单输入模块
#         self.inc = CheckpointWrapper(self.inc)
#         self.down1 = CheckpointWrapper(self.down1)
#         self.down2 = CheckpointWrapper(self.down2)
#         self.down3 = CheckpointWrapper(self.down3)
#         self.down4 = CheckpointWrapper(self.down4)
#         self.outc = CheckpointWrapper(self.outc)
        
#         # 处理需要两个输入的上采样模块
#         class CheckpointWrapper2Inputs(nn.Module):
#             def __init__(self, module):
#                 super().__init__()
#                 self.module = module
                
#             def forward(self, x, skip_x):
#                 return checkpoint(self.module, x, skip_x, use_reentrant=False)
        
#         self.up1 = CheckpointWrapper2Inputs(self.up1)
#         self.up2 = CheckpointWrapper2Inputs(self.up2)
#         self.up3 = CheckpointWrapper2Inputs(self.up3)
#         self.up4 = CheckpointWrapper2Inputs(self.up4)
        
#         # 如果有注意力模块也需要处理
#         if self.use_attention:
#             class CheckpointWrapperAttention(nn.Module):
#                 def __init__(self, module):
#                     super().__init__()
#                     self.module = module
                    
#                 def forward(self, x, skip_x):
#                     return checkpoint(self.module, x, skip_x, use_reentrant=False)
            
#             self.att1 = CheckpointWrapperAttention(self.att1)
#             self.att2 = CheckpointWrapperAttention(self.att2)
#             self.att3 = CheckpointWrapperAttention(self.att3)
#             self.att4 = CheckpointWrapperAttention(self.att4)


#     def get_intermediate_features(self, x, layers=None):
#         """提取指定层的中间特征"""
#         if layers is None:
#             layers = ['x1', 'x2', 'x3', 'x4', 'x5', 'up1', 'up2', 'up3', 'up4']
#             if self.use_attention:
#                 layers.extend(['x1_att', 'x2_att', 'x3_att', 'x4_att'])

#         features = {}
        
#         # 编码器
#         x1 = self.inc(x)
#         if 'x1' in layers:
#             features['x1'] = x1

#         x2 = self.down1(x1)
#         if 'x2' in layers:
#             features['x2'] = x2

#         x3 = self.down2(x2)
#         if 'x3' in layers:
#             features['x3'] = x3

#         x4 = self.down3(x3)
#         if 'x4' in layers:
#             features['x4'] = x4

#         x5 = self.down4(x4)
#         if 'x5' in layers:
#             features['x5'] = x5

#         # 解码器
#         if self.use_attention:
#             x4_att = self.att1(x5, x4)
#             if 'x4_att' in layers:
#                 features['x4_att'] = x4_att
#             x_up1 = self.up1(x5, x4_att)
            
#             x3_att = self.att2(x_up1, x3)
#             if 'x3_att' in layers:
#                 features['x3_att'] = x3_att
#             x_up2 = self.up2(x_up1, x3_att)
            
#             x2_att = self.att3(x_up2, x2)
#             if 'x2_att' in layers:
#                 features['x2_att'] = x2_att
#             x_up3 = self.up3(x_up2, x2_att)
            
#             x1_att = self.att4(x_up3, x1)
#             if 'x1_att' in layers:
#                 features['x1_att'] = x1_att
#             x_up4 = self.up4(x_up3, x1_att)
#         else:
#             x_up1 = self.up1(x5, x4)
#             x_up2 = self.up2(x_up1, x3)
#             x_up3 = self.up3(x_up2, x2)
#             x_up4 = self.up4(x_up3, x1)
        
#         if 'up1' in layers:
#             features['up1'] = x_up1
#         if 'up2' in layers:
#             features['up2'] = x_up2
#         if 'up3' in layers:
#             features['up3'] = x_up3
#         if 'up4' in layers:
#             features['up4'] = x_up4

#         return features






import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, dropout=0.1):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 处理尺寸不匹配问题
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # 对称填充以确保尺寸匹配
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class OffsetHead(nn.Module):
    """专门用于回归偏移值的头部"""
    def __init__(self, in_channels, hidden_channels=256, dropout=0.3):
        super(OffsetHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        self.conv3 = nn.Conv2d(hidden_channels // 2, 2, kernel_size=1)  # 输出2个通道：δx, δy

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)  # 无激活函数，直接回归偏移值
        return x


class BoundaryAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 边界检测卷积核（使用Laplacian边缘检测核）
        self.boundary_conv = nn.Conv2d(
            channels, 1, kernel_size=3, padding=1, bias=False
        )
        # 初始化卷积核为边缘检测核（中心8，周围-1）
        nn.init.constant_(self.boundary_conv.weight, -1.0)
        self.boundary_conv.weight.data[:, :, 1, 1] = 8.0  # 中心像素权重
        
        # 注意力门控（增强边界区域权重）
        self.attention = nn.Sequential(
            nn.Sigmoid(),  # 将边界响应归一化到[0,1]
            nn.Conv2d(1, 1, kernel_size=1)  # 压缩为单通道注意力图
        )
    
    def forward(self, x):
        # 1. 提取边界特征（x为DoubleConv输出的特征图）
        boundary_features = self.boundary_conv(x)  # (B,1,H,W)
        
        # 2. 生成边界注意力图（边界区域权重高）
        boundary_attention = self.attention(boundary_features)  # (B,1,H,W)，值在[0,1]
        
        # 3. 特征加权：边界区域特征 = 原始特征 * (1 + 注意力权重)
        #    非边界区域特征基本不变，边界区域特征被放大
        attended_features = x * (1 + boundary_attention)
        
        return attended_features


class SemanticHead(nn.Module):
    def __init__(self, in_channels, n_classes, dropout=0.0):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, 32, dropout=dropout)
        self.boundary_attn = BoundaryAttention(channels=32)  # 新增边界注意力
        self.out_conv = OutConv(32, n_classes)
    
    def forward(self, x):
        # 原有DoubleConv特征提取
        x = self.double_conv(x)
        # 边界注意力增强
        x = self.boundary_attn(x)
        # 输出分割结果
        x = self.out_conv(x)
        return x


class EarlyFusionUNet(nn.Module):
    """
    早期融合U-Net：将RGB和深度图在输入层融合
    """
    def __init__(self, n_classes, bilinear=False, dropout=0.1):
        super(EarlyFusionUNet, self).__init__()
        self.bilinear = bilinear
        self.dropout_rate = dropout
        self.n_classes = n_classes

        # 输入通道：3 (RGB) + 1 (Depth) = 4
        self.inc = DoubleConv(4, 64, dropout=dropout)
        self.down1 = Down(64, 128, dropout=dropout)
        self.down2 = Down(128, 256, dropout=dropout)
        self.down3 = Down(256, 512, dropout=dropout)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, dropout=dropout)
        
        # 上采样模块
        self.up1 = Up(1024, 512 // factor, bilinear, dropout=dropout)
        self.up2 = Up(512, 256 // factor, bilinear, dropout=dropout)
        self.up3 = Up(256, 128 // factor, bilinear, dropout=dropout)
        self.up4 = Up(128, 64, bilinear, dropout=dropout)
        
        # 双输出头
        self.semantic_head = SemanticHead(in_channels=64, n_classes=self.n_classes, dropout=0.2)
        
        self.offset_head = OffsetHead(64, hidden_channels=128, dropout=0.3)
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, rgb, depth):
        # 早期融合：在输入层连接RGB和深度图
        x = torch.cat([rgb, depth], dim=1)  # 沿着通道维度连接
        
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 解码器
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 双头输出
        semantic_output = self.semantic_head(x)
        offset_output = self.offset_head(x)
        
        return semantic_output, offset_output


class LateFusionUNet(nn.Module):
    """
    晚期融合U-Net：分别处理RGB和深度图，在解码器阶段融合
    """
    def __init__(self, n_classes, bilinear=False, dropout=0.1):
        super(LateFusionUNet, self).__init__()
        self.bilinear = bilinear
        self.dropout_rate = dropout

        # RGB分支编码器
        self.rgb_inc = DoubleConv(3, 32, dropout=dropout)
        self.rgb_down1 = Down(32, 64, dropout=dropout)
        self.rgb_down2 = Down(64, 128, dropout=dropout)
        self.rgb_down3 = Down(128, 256, dropout=dropout)
        
        # 深度分支编码器
        self.depth_inc = DoubleConv(1, 32, dropout=dropout)
        self.depth_down1 = Down(32, 64, dropout=dropout)
        self.depth_down2 = Down(64, 128, dropout=dropout)
        self.depth_down3 = Down(128, 256, dropout=dropout)
        
        factor = 2 if bilinear else 1
        # 融合后的编码器
        self.fusion_down4 = Down(512, 512 // factor, dropout=dropout)
        
        # 上采样模块
        self.up1 = Up(512, 256 // factor, bilinear, dropout=dropout)
        self.up2 = Up(256, 128 // factor, bilinear, dropout=dropout)
        self.up3 = Up(128, 64 // factor, bilinear, dropout=dropout)
        self.up4 = Up(64, 32, bilinear, dropout=dropout)
        
        # 双输出头
        self.semantic_head = nn.Sequential(
            DoubleConv(32, 16, dropout=dropout),
            OutConv(16, n_classes)
        )
        
        self.offset_head = OffsetHead(32, hidden_channels=64, dropout=dropout)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, rgb, depth):
        # 分别处理RGB和深度图
        rgb1 = self.rgb_inc(rgb)
        rgb2 = self.rgb_down1(rgb1)
        rgb3 = self.rgb_down2(rgb2)
        rgb4 = self.rgb_down3(rgb3)
        
        depth1 = self.depth_inc(depth)
        depth2 = self.depth_down1(depth1)
        depth3 = self.depth_down2(depth2)
        depth4 = self.depth_down3(depth3)
        
        # 在瓶颈层融合
        x5 = torch.cat([rgb4, depth4], dim=1)
        x5 = self.fusion_down4(x5)
        
        # 解码器
        x = self.up1(x5, rgb4)  # 主要使用RGB特征进行上采样
        x = self.up2(x, rgb3)
        x = self.up3(x, rgb2)
        x = self.up4(x, rgb1)
        
        semantic_output = self.semantic_head(x)
        offset_output = self.offset_head(x)
        
        return semantic_output, offset_output


# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""
#     def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.1):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
        
#         layers = [
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         ]
        
#         self.double_conv = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.double_conv(x)


# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""
#     def __init__(self, in_channels, out_channels, dropout=0.1):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels, dropout=dropout)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)


# class Up(nn.Module):
#     """Upscaling then double conv"""
#     def __init__(self, in_channels, out_channels, bilinear=True, dropout=0.1):
#         super().__init__()
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout=dropout)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # 处理尺寸不匹配问题
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]
#         # 对称填充以确保尺寸匹配
#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)


# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)


# class OffsetHead(nn.Module):
#     """专门用于回归偏移值的头部"""
#     def __init__(self, in_channels, hidden_channels=256, dropout=0.2):
#         super(OffsetHead, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(hidden_channels)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.dropout1 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
#         self.conv2 = nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(hidden_channels // 2)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.dropout2 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
#         self.conv3 = nn.Conv2d(hidden_channels // 2, 2, kernel_size=1)  # 输出2个通道：δx, δy

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         x = self.dropout1(x)
        
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#         x = self.dropout2(x)
        
#         x = self.conv3(x)  # 无激活函数，直接回归偏移值
#         return x


# class BoundaryAttention(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         # 边界检测卷积核（使用Laplacian边缘检测核）
#         self.boundary_conv = nn.Conv2d(
#             channels, 1, kernel_size=3, padding=1, bias=False
#         )
#         # 初始化卷积核为边缘检测核（中心8，周围-1）
#         nn.init.constant_(self.boundary_conv.weight, -1.0)
#         self.boundary_conv.weight.data[:, :, 1, 1] = 8.0  # 中心像素权重
        
#         # 注意力门控（增强边界区域权重）
#         self.attention = nn.Sequential(
#             nn.Sigmoid(),  # 将边界响应归一化到[0,1]
#             nn.Conv2d(1, 1, kernel_size=1)  # 压缩为单通道注意力图
#         )
    
#     def forward(self, x):
#         # 1. 提取边界特征
#         boundary_features = self.boundary_conv(x)  # (B,1,H,W)
#         # 2. 生成边界注意力图
#         boundary_attention = self.attention(boundary_features)  # (B,1,H,W)
#         # 3. 特征加权：边界区域特征放大
#         attended_features = x * (1 + boundary_attention)
#         return attended_features


# class SemanticHead(nn.Module):
#     def __init__(self, in_channels, n_classes, dropout=0.0):
#         super().__init__()
#         self.double_conv = DoubleConv(in_channels, 32, dropout=dropout)
#         self.boundary_attn = BoundaryAttention(channels=32)  # 边界注意力增强
#         self.out_conv = OutConv(32, n_classes)
    
#     def forward(self, x):
#         x = self.double_conv(x)
#         x = self.boundary_attn(x)
#         x = self.out_conv(x)
#         return x


# # --------------------------- 核心改动：添加任务专用特征分支 ---------------------------
# class EarlyFusionUNet(nn.Module):
#     """
#     早期融合U-Net：共享骨干网络 + 分割/偏移任务独立特征分支
#     改动点：
#     1. 在解码器输出（up4之后）添加两个独立分支
#     2. 分割分支：侧重局部边界特征
#     3. 偏移分支：侧重全局几何特征（大卷积核+多尺度）
#     """
#     def __init__(self, n_classes, bilinear=False, dropout=0.1):
#         super(EarlyFusionUNet, self).__init__()
#         self.bilinear = bilinear
#         self.dropout_rate = dropout
#         self.n_classes = n_classes

#         # 输入通道：3 (RGB) + 1 (Depth) = 4
#         self.inc = DoubleConv(4, 64, dropout=dropout)
#         self.down1 = Down(64, 128, dropout=dropout)
#         self.down2 = Down(128, 256, dropout=dropout)
#         self.down3 = Down(256, 512, dropout=dropout)
        
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor, dropout=dropout)
        
#         # 共享解码器
#         self.up1 = Up(1024, 512 // factor, bilinear, dropout=dropout)
#         self.up2 = Up(512, 256 // factor, bilinear, dropout=dropout)
#         self.up3 = Up(256, 128 // factor, bilinear, dropout=dropout)
#         self.up4 = Up(128, 64, bilinear, dropout=dropout)  # 共享解码器输出：(B,64,H,W)
        
#         # --------------------------- 任务专用特征分支（核心改动） ---------------------------
#         # 1. 分割任务专用分支：强化局部边界特征
#         self.seg_branch = nn.Sequential(
#             DoubleConv(64, 64, dropout=dropout),  # 细化分割特征
#             BoundaryAttention(64),  # 早期边界增强（比原SemanticHead更早介入）
#             DoubleConv(64, 32, dropout=dropout)   # 适配原SemanticHead输入
#         )
        
#         # 2. 偏移任务专用分支：强化全局几何特征
#         self.offset_branch = nn.Sequential(
#             DoubleConv(64, 64, dropout=dropout),  # 基础特征提取
#             # 大卷积核捕捉全局上下文（实例尺度/位置关系）
#             nn.Conv2d(64, 64, kernel_size=7, padding=3, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             DoubleConv(64, 64, dropout=dropout),  # 巩固全局特征
#             # 多尺度融合（1x1捕捉细节 + 3x3捕捉上下文）
#             nn.Conv2d(64, 128, kernel_size=1, padding=0),  # 细节分支
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 融合分支
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True)
#         )
#         # -----------------------------------------------------------------------------------
        
#         # 双输出头（复用原头部，输入适配分支输出）
#         self.semantic_head = SemanticHead(in_channels=32, n_classes=self.n_classes, dropout=0.2)
#         self.offset_head = OffsetHead(in_channels=64, hidden_channels=128, dropout=0.2)
        
#         # 初始化权重
#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, rgb, depth):
#         # 早期融合：输入层连接RGB和深度图
#         x = torch.cat([rgb, depth], dim=1)  # (B,4,H,W)
        
#         # 共享骨干（编码器+解码器）
#         x1 = self.inc(x)       # (B,64,H,W)
#         x2 = self.down1(x1)    # (B,128,H/2,W/2)
#         x3 = self.down2(x2)    # (B,256,H/4,W/4)
#         x4 = self.down3(x3)    # (B,512,H/8,W/8)
#         x5 = self.down4(x4)    # (B,512/1024,H/16,W/16)
        
#         x = self.up1(x5, x4)   # (B,256,H/8,W/8)
#         x = self.up2(x, x3)    # (B,128,H/4,W/4)
#         x = self.up3(x, x2)    # (B,64,H/2,W/2)
#         shared_decoder_out = self.up4(x, x1)  # (B,64,H,W) —— 共享解码器输出
        
#         # --------------------------- 任务专用分支前向（核心改动） ---------------------------
#         # 分割分支：共享特征 → 分割专用特征 → 分割头
#         seg_feat = self.seg_branch(shared_decoder_out)  # (B,32,H,W)
#         semantic_output = self.semantic_head(seg_feat)
        
#         # 偏移分支：共享特征 → 偏移专用特征 → 偏移头
#         offset_feat = self.offset_branch(shared_decoder_out)  # (B,64,H,W)
#         offset_output = self.offset_head(offset_feat)
#         # -----------------------------------------------------------------------------------
        
#         return semantic_output, offset_output


# # # 示例用法
# # if __name__ == "__main__":
# #     # 创建模型
# #     early_fusion_model = EarlyFusionUNet(n_classes=1, bilinear=True, dropout=0.1)
# #     late_fusion_model = LateFusionUNet(n_classes=1, bilinear=True, dropout=0.1)
    
# #     # 测试输入
# #     batch_size, height, width = 2, 480, 640
# #     rgb_input = torch.randn(batch_size, 3, height, width)
# #     depth_input = torch.randn(batch_size, 1, height, width)
    
# #     # 测试早期融合模型
# #     semantic_pred1, offset_pred1 = early_fusion_model(rgb_input, depth_input)
# #     print(f"早期融合模型:")
# #     print(f"  语义分割输出尺寸: {semantic_pred1.shape}")  # [2, 1, 480, 640]
# #     print(f"  偏移回归输出尺寸: {offset_pred1.shape}")    # [2, 2, 480, 640]
    
# #     # 测试晚期融合模型
# #     semantic_pred2, offset_pred2 = late_fusion_model(rgb_input, depth_input)
# #     print(f"晚期融合模型:")
# #     print(f"  语义分割输出尺寸: {semantic_pred2.shape}")  # [2, 1, 480, 640]
# #     print(f"  偏移回归输出尺寸: {offset_pred2.shape}")    # [2, 2, 480, 640]
    
# #     # 参数量统计
# #     early_params = sum(p.numel() for p in early_fusion_model.parameters())
# #     late_params = sum(p.numel() for p in late_fusion_model.parameters())
# #     print(f"早期融合模型参数量: {early_params:,}")
# #     print(f"晚期融合模型参数量: {late_params:,}")