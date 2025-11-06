import os
import yaml
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datetime import datetime
import cv2
# from sklearn.model_selection import ParameterGrid  # 用于超参数搜索

from utils.data_loading import DualHeadUNetDataset
from models.unet import EarlyFusionUNet
from utils.metrics import compute_iou, compute_mse, compute_masked_mse


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_dir, experiment_name):
    """设置日志记录：同时输出到文件和控制台"""
    log_path = os.path.join(log_dir, f"{experiment_name}.log")
    os.makedirs(log_dir, exist_ok=True)

    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def save_checkpoint(model, optimizer, epoch, metrics, config, save_path):
    """保存训练 checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,  # 保存当前最佳指标
        'config': config     # 保存当前实验配置
    }
    torch.save(checkpoint, save_path)
    logging.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """加载 checkpoint 恢复训练"""
    if not os.path.exists(checkpoint_path):
        logging.warning(f"Checkpoint {checkpoint_path} not found, starting from scratch")
        return model, optimizer, 0, {}

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
    best_metrics = checkpoint['metrics']
    logging.info(f"Loaded checkpoint from {checkpoint_path}, starting at epoch {start_epoch}")
    return model, optimizer, start_epoch, best_metrics


def background_zero_loss(pred_offset, sem_gt):
    """约束背景区域的偏移值必须为0"""
    background_mask = (sem_gt == 0).float().unsqueeze(1)  # (B,1,H,W)，背景为1
    # 计算背景偏移与0的L1损失（对异常值更敏感）
    return (background_mask * torch.abs(pred_offset)).mean() * 5


def dice_loss(pred, gt):
    """Dice Loss（用于分割，抑制类别不平衡）"""
    pred_sigmoid = torch.sigmoid(pred)
    intersection = (pred_sigmoid * gt).sum()
    union = pred_sigmoid.sum() + gt.sum()
    return 1 - (2 * intersection + 1e-6) / (union + 1e-6)  # 加1e-6避免除零


def compute_depth_gradient(depth_map, grad_scale=100.0):
    """
    计算深度图的x/y方向梯度，突出结构边缘
    Args:
        depth_map: (B,1,H,W) 原始深度图（值已归一化到[0,1]）
        grad_scale: 梯度值放大系数（让梯度特征更明显）
    Returns:
        depth_grad: (B,2,H,W) x/y方向梯度图（归一化到[0,1]）
    """
    # 1. 计算x方向梯度（左右相邻像素深度差）
    grad_x = depth_map[:, :, :, 1:] - depth_map[:, :, :, :-1]  # (B,1,H,W-1)
    grad_x = F.pad(grad_x, (0, 1, 0, 0, 0, 0), mode='replicate')  # 补全为(B,1,H,W)
    
    # 2. 计算y方向梯度（上下相邻像素深度差）
    grad_y = depth_map[:, :, 1:, :] - depth_map[:, :, :-1, :]  # (B,1,H-1,W)
    grad_y = F.pad(grad_y, (0, 0, 0, 1, 0, 0), mode='replicate')  # 补全为(B,1,H,W)
    
    # 3. 梯度值归一化（消除深度尺度影响）
    grad_x = torch.abs(grad_x) * grad_scale  # 取绝对值并放大
    grad_y = torch.abs(grad_y) * grad_scale
    grad_x = torch.clamp(grad_x, 0.0, 1.0)  # 限制在[0,1]，避免异常值
    grad_y = torch.clamp(grad_y, 0.0, 1.0)
    
    # 4. 拼接x/y梯度，形成2通道梯度图
    depth_grad = torch.cat([grad_x, grad_y], dim=1)  # (B,2,H,W)

    return depth_grad


def false_positive_penalty_loss(sem_pred, depth_grad, sem_gt, pos_grad_threshold=500): #768.20
    # 新增：裁剪极端高梯度（如>5000，对应统计中的q98=5132，排除2%极端值）
    depth_grad = torch.clamp(depth_grad, 0.0, 5000.0)  # 避免噪声影响
    
    # 原有逻辑不变
    pred_foreground = (torch.sigmoid(sem_pred) > 0.5).float()
    real_background = (sem_gt == 0).float().unsqueeze(1)
    low_gradient = (torch.max(depth_grad, dim=1, keepdim=True)[0] < pos_grad_threshold).float()
    false_positive_mask = pred_foreground * real_background * low_gradient
    pred_confidence = torch.sigmoid(sem_pred)
    fp_loss = (false_positive_mask * pred_confidence).mean()

    return fp_loss * 5


class InstanceAwareOffsetLoss(nn.Module):
    def __init__(self, boundary_margin=3.0):
        super().__init__()
        self.boundary_margin = boundary_margin

    def get_instance_boundary_mask(self, semantic_mask, erosion_radius=1):
        """
        获取实例边界区域（批量处理，支持二值语义掩码）
        Args:
            semantic_mask: (B, 1, H, W) 或 (B, H, W) 的语义掩码，前景为1，背景为0
            erosion_radius: 形态学操作的半径，控制边界粗细
        Returns:
            boundary_mask: (B, H, W) 边界掩码，边界区域为True
        """
        # 统一形状为 (B, 1, H, W)
        if semantic_mask.dim() == 3:
            semantic_mask = semantic_mask.unsqueeze(1)  # (B, 1, H, W)
        B, _, H, W = semantic_mask.shape

        # 二值化掩码（前景为1，背景为0）
        binary_mask = (semantic_mask > 0.5).float()

        # 定义圆形结构元素（用于形态学操作）
        kernel_size = 2 * erosion_radius + 1
        kernel = torch.ones(kernel_size, kernel_size, device=semantic_mask.device)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, kernel_size, kernel_size)

        # 膨胀操作（扩大前景区域）
        dilated = F.max_pool2d(
            binary_mask,
            kernel_size=kernel_size,
            stride=1,
            padding=erosion_radius
        )

        # 腐蚀操作（缩小前景区域）
        eroded = -F.max_pool2d(
            -binary_mask,
            kernel_size=kernel_size,
            stride=1,
            padding=erosion_radius
        )

        # 边界 = 膨胀区域 - 腐蚀区域（非重叠部分为边界）
        boundary_mask = (dilated - eroded) > 1e-5  # (B, 1, H, W)

        return boundary_mask.squeeze(1)  # (B, H, W)

    def boundary_aware_offset_loss(self, offset_pred, semantic_mask):
        """
        边界感知偏移损失：在边界处强制更大的偏移幅度
        """
        B, _, H, W = offset_pred.shape
        semantic_mask = semantic_mask.squeeze(1) if semantic_mask.dim() == 4 else semantic_mask
        
        # 获取边界和内部掩码
        boundary_mask = self.get_instance_boundary_mask(semantic_mask)  # (B, H, W)
        
        # 计算偏移向量的模长
        offset_magnitude = torch.norm(offset_pred, dim=1)  # (B, H, W)

        # 惩罚边界处偏移太小的情况
        boundary_offset_penalty = F.relu(1.0 - offset_magnitude)  # 边界处偏移应大于1
        
        boundary_pixels = torch.clamp(boundary_mask.sum() + 1e-8, min=1e-4)       
        loss = (boundary_offset_penalty * boundary_mask).sum() / boundary_pixels
        
        return loss

    def directional_divergence_loss(self, offset_pred, semantic_mask):
        """
        方向散度损失：促进偏移场从边缘向中心收敛（内部区域散度为负）
        """
        B, _, H, W = offset_pred.shape
        semantic_mask = semantic_mask.squeeze(1) if semantic_mask.dim() == 4 else semantic_mask  # (B, H, W)
        
        # 获取边界掩码和内部掩码
        boundary_mask = self.get_instance_boundary_mask(semantic_mask)  # (B, H, W)
        foreground_mask = (semantic_mask > 0.5).float()  # (B, H, W)
        interior_mask = foreground_mask * (~boundary_mask).float()  # (B, H, W)

        # 计算指向中心的向量（-offset_pred）
        to_center_vectors = -offset_pred  # (B, 2, H, W)
        dy = to_center_vectors[:, 0]  # y方向分量 (B, H, W)
        dx = to_center_vectors[:, 1]  # x方向分量 (B, H, W)

        # 计算散度
        dy_dy = torch.gradient(dy, dim=1)[0]  # ∂dy/∂y（沿H维度）
        dx_dx = torch.gradient(dx, dim=2)[0]  # ∂dx/∂x（沿W维度）
        divergence = dy_dy + dx_dx  # (B, H, W)

        # 惩罚正散度（内部区域应收敛，散度为负）
        positive_divergence = F.relu(divergence)  # (B, H, W)
        positive_divergence = torch.clamp(positive_divergence, max=5.0)  # 限制散度惩罚上限
        total_interior_pixels = torch.clamp(interior_mask.sum() + 1e-8, min=1e-4)  # 强制最小为0.0001
        divergence_loss = (positive_divergence * interior_mask).sum() / total_interior_pixels

        return divergence_loss

    def center_compactness_loss(self, offset_pred, semantic_mask):
        """
        中心紧凑性损失：约束同一实例的预测中心在局部邻域内聚集
        """
        B, _, H, W = offset_pred.shape
        semantic_mask = semantic_mask.squeeze(1) if semantic_mask.dim() == 4 else semantic_mask  # (B, H, W)
        
        # 前景掩码（批量处理）
        foreground_mask = (semantic_mask > 0.5).float().unsqueeze(1)  # (B, 1, H, W)
        
        # 创建坐标网格（y, x）
        y_coords = torch.arange(H, device=offset_pred.device, dtype=torch.float32)
        x_coords = torch.arange(W, device=offset_pred.device, dtype=torch.float32)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')  # (H, W)
        coords = torch.stack([y_grid, x_grid], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 2, H, W)
        
        # 预测中心 = 像素坐标 + 偏移量
        pred_centers = coords + offset_pred  # (B, 2, H, W)

        # 局部平均中心（3x3邻域卷积）
        kernel = torch.ones(1, 1, 3, 3, device=offset_pred.device) / 25.0  # 3x3平均滤波
        # 计算y/x方向的局部平均中心
        local_mean_y = F.conv2d(pred_centers[:, 0:1] * foreground_mask, kernel, padding=1)  # (B, 1, H, W)
        local_mean_x = F.conv2d(pred_centers[:, 1:2] * foreground_mask, kernel, padding=1)  # (B, 1, H, W)
        # 局部有效像素数（用于归一化）
        local_count = F.conv2d(foreground_mask, kernel, padding=1) + 1e-8  # (B, 1, H, W)
        local_mean_y = local_mean_y / local_count
        local_mean_x = local_mean_x / local_count

        # 拼接局部平均中心（B, 2, H, W）
        local_mean_centers = torch.cat([local_mean_y, local_mean_x], dim=1)

        # 计算每个像素的预测中心与局部平均中心的距离
        distances = torch.norm(pred_centers - local_mean_centers, dim=1)  # (B, H, W)

        # 仅计算内部区域的距离损失
        boundary_mask = self.get_instance_boundary_mask(semantic_mask)  # (B, H, W)
        interior_mask = foreground_mask.squeeze(1) * (~boundary_mask).float()  # (B, H, W)
        total_interior_pixels = torch.clamp(interior_mask.sum() + 1e-8, min=1e-4)  # 强制最小为0.0001
        compactness_loss = (distances * interior_mask).sum() / total_interior_pixels

        return compactness_loss

    def forward(self, offset_pred, semantic_mask):
        """
        前向传播：组合三个辅助损失
        Args:
            offset_pred: (B, 2, H, W) 预测的偏移场（y, x方向）
            semantic_mask: (B, 1, H, W) 或 (B, H, W) 语义掩码（前景为1，背景为0）
        Returns:
            total_loss: 组合损失值
        """
        boundary_aware_loss = self.boundary_aware_offset_loss(offset_pred, semantic_mask)
        divergence_loss = self.directional_divergence_loss(offset_pred, semantic_mask)
        compactness_loss = self.center_compactness_loss(offset_pred, semantic_mask)

        # 保持原权重组合（未归一化，按原设计）
        total_loss = (1.0 * boundary_aware_loss +
                     1.0 * divergence_loss +
                     1.5 * compactness_loss)

        return total_loss


def train_one_epoch(model, train_loader, seg_criterion, offset_criterion, aux_offset_loss_func,
                   optimizer, device, epoch, writer, grad_clip=None):
    """训练单个epoch"""
    model.train()
    total_loss = 0.0
    total_seg_loss = 0.0
    total_offset_loss = 0.0
    total_bce_loss = 0.0
    total_seg_dice_loss = 0.0
    total_mse_loss = 0.0 
    total_bg_zero_loss = 0.0
    total_fp_penalty_loss = 0.0
    total_aux_offset_loss = 0.0
    all_iou = []
    all_mse = []
    all_bg_mse = []
    all_fg_mse = []

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # 数据移至设备
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device)
        sem_gt = batch['semantic_gt'].to(device)
        offset_gt_norm = batch['offset_gt_norm'].to(device)
        offset_gt_raw = batch['offset_gt_raw'].to(device)  # 用于计算原始偏移MAE

        # 前向传播
        sem_pred, offset_pred_norm = model(rgb, depth)

        # 计算损失
        aux_offset_loss = aux_offset_loss_func(offset_pred_norm, sem_gt)  # 增加辅助分割的偏移损失
        seg_loss = seg_criterion(sem_pred, sem_gt.unsqueeze(1).float())  # 语义标签增加通道维
        # offset_loss = offset_criterion(offset_pred_norm, offset_gt_norm)
        foreground_mask = (sem_gt == 1).float().unsqueeze(1)
        offset_loss = offset_criterion(offset_pred_norm * foreground_mask, offset_gt_norm * foreground_mask) * 1.5 + \
        offset_criterion(offset_pred_norm * (1-foreground_mask), offset_gt_norm * (1-foreground_mask)) * 0.5
        
        bg_zero_loss = background_zero_loss(offset_pred_norm, sem_gt)
        seg_dice_loss = dice_loss(sem_pred, sem_gt.unsqueeze(1).float())
        depth_grad = compute_depth_gradient(depth)
        fp_penalty_loss = false_positive_penalty_loss(sem_pred, depth_grad, sem_gt)

        aux_weight = min(0.02 + epoch / 100, 0.2)  # 前18 epoch从0.02增至0.2，之后保持
        seg_weight = max(1 - epoch / 100, 0.6)  # 前40 epoch从1降至0.6，之后保持

        loss = seg_loss * seg_weight + seg_dice_loss * 5 * seg_weight + fp_penalty_loss * 0.2 \
            + offset_loss * 0.3 + bg_zero_loss * 0.2 + aux_offset_loss* aux_weight

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=grad_clip) 
        optimizer.step()

        # 计算指标
        sem_pred_argmax = torch.sigmoid(sem_pred) > 0.5  # 二分类阈值
        iou = compute_iou(sem_pred_argmax.squeeze(1).cpu().numpy(), 
                            sem_gt.cpu().numpy())
        all_iou.append(iou)

        train_dataset = train_loader.dataset
        offset_pred_raw = train_dataset.denormalize_offset(offset_pred_norm)

        # denorm_offset_gt_norm = train_dataset.denormalize_offset(offset_gt_norm)
        # print(f"offset_gt_raw mean: {offset_gt_raw.mean().item():.4f}")
        # print(f"denorm_offset_gt_norm mean: {denorm_offset_gt_norm.mean().item():.4f}")

        mse = compute_mse(offset_pred_raw.cpu().numpy(), 
                            offset_gt_raw.cpu().numpy())
        all_mse.append(mse)

        background_mask = (sem_gt == 0).float().unsqueeze(1)
        bg_mse = compute_masked_mse(offset_pred_raw.cpu().numpy(), 
                            offset_gt_raw.cpu().numpy(), 
                            background_mask.cpu().numpy())
        all_bg_mse.append(bg_mse) 

        foreground_mask = (sem_gt == 1).float().unsqueeze(1)
        fg_mse = compute_masked_mse(offset_pred_raw.cpu().numpy(), 
                            offset_gt_raw.cpu().numpy(), 
                            foreground_mask.cpu().numpy())
        all_fg_mse.append(fg_mse)

        # 累计损失
        total_loss += loss.item() * rgb.size(0)
        total_seg_loss += (seg_loss.item() * seg_weight + seg_dice_loss.item() * 5 * seg_weight + fp_penalty_loss * 0.2) * rgb.size(0)
        total_offset_loss += (offset_loss.item() * 0.3 + bg_zero_loss.item() * 0.2 + aux_offset_loss * aux_weight) * rgb.size(0)
        total_bce_loss += seg_loss.item() * seg_weight * rgb.size(0)
        total_seg_dice_loss += seg_dice_loss.item() * 5 * seg_weight * rgb.size(0)
        total_mse_loss += offset_loss.item() * 0.3 * rgb.size(0)
        total_bg_zero_loss += bg_zero_loss.item() * 0.2 * rgb.size(0)
        total_fp_penalty_loss += fp_penalty_loss.item() * 0.2 * rgb.size(0)
        total_aux_offset_loss += aux_offset_loss.item() * aux_weight * rgb.size(0)  # 增加辅助分割的偏移损失

        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'seg_loss': f"{(seg_loss.item() * seg_weight) + (seg_dice_loss.item() * 5 * seg_weight) + (fp_penalty_loss.item() * 0.2):.4f}",
            'offset_loss': f"{(offset_loss.item() * 0.3 + bg_zero_loss.item() * 0.2 + aux_offset_loss.item() * aux_weight):.4f}"
            # '细分-seg_bce': f"{(seg_loss.item() * 0.3):.4f}",
            # '细分-seg_dice': f"{(seg_dice_loss.item() * 5 * 0.3):.4f}",
            # '细分-fp_penalty': f"{(fp_penalty_loss.item() * 0.1):.4f}",
            # '细分-offset': f"{(offset_loss.item() * 0.4):.4f}",
            # '细分-bg_zero': f"{(bg_zero_loss.item() * 0.1):.4f}"
        })

    # 计算平均指标
    avg_loss = total_loss / len(train_loader.dataset)
    avg_seg_loss = total_seg_loss / len(train_loader.dataset)
    avg_offset_loss = total_offset_loss / len(train_loader.dataset)
    avg_iou = np.mean(all_iou) if all_iou else 0.0
    avg_mse = np.mean(all_mse) if all_mse else 0.0
    avg_bg_mse = np.mean(all_bg_mse) if all_bg_mse else 0.0
    avg_fg_mse = np.mean(all_fg_mse) if all_fg_mse else 0.0

    avg_bce_loss = total_bce_loss / len(train_loader.dataset)
    avg_seg_dice_loss = total_seg_dice_loss / len(train_loader.dataset)
    avg_mse_loss = total_mse_loss / len(train_loader.dataset)
    avg_bg_zero_loss = total_bg_zero_loss / len(train_loader.dataset)
    avg_fp_penalty_loss = total_fp_penalty_loss / len(train_loader.dataset)
    avg_aux_offset_loss = total_aux_offset_loss / len(train_loader.dataset)  # 增加辅助分割的偏移损失

    # 记录到TensorBoard
    # writer.add_scalar('train/loss', avg_loss, epoch)
    # writer.add_scalar('train/seg_loss', avg_seg_loss, epoch)
    # writer.add_scalar('train/offset_loss', avg_offset_loss, epoch)
    writer.add_scalars(
    main_tag='train/loss_com',  # 图表标题（对应TensorBoard中的标签）
    tag_scalar_dict={                 # 曲线名称→指标值的映射
        'Total Loss': avg_loss,
        'Seg Loss': avg_seg_loss,
        'Offset Loss': avg_offset_loss
    },
    global_step=epoch)
    writer.add_scalar('train/iou', avg_iou, epoch)
    writer.add_scalar('train/mse', avg_mse, epoch)
    # writer.add_scalar('train/bg_mse', avg_bg_mse, epoch)
    # writer.add_scalar('train/fg_mse', avg_fg_mse, epoch)
    writer.add_scalars(
    main_tag='train/mse_com',
    tag_scalar_dict={
        'MSE': avg_mse,
        'BG MSE': avg_bg_mse,
        'FG MSE': avg_fg_mse
    },
    global_step=epoch)
    writer.add_scalars(
    main_tag='train/seg_loss_com',
    tag_scalar_dict={
        'Total Seg Loss': avg_seg_loss,
        'BCE Loss': avg_bce_loss,
        'Dice Loss': avg_seg_dice_loss,
        'FP Penalty Loss': avg_fp_penalty_loss
    },
    global_step=epoch)
    writer.add_scalars(
    main_tag='train/offset_loss_com',
    tag_scalar_dict={
        'Total Offset Loss': avg_offset_loss,
        'MSE Loss': avg_mse_loss,
        'BG Zero Loss': avg_bg_zero_loss,
        'AUX Loss': avg_aux_offset_loss  # 增加辅助分割的偏移损失
    },
    global_step=epoch)

    logging.info(
        f"Train Epoch {epoch} - Loss: {avg_loss:.4f}, "
        f"Seg Loss: {avg_seg_loss:.4f}, Offset Loss: {avg_offset_loss:.4f}, "
        f"IoU: {avg_iou:.4f}, MSE: {avg_mse:.4f}, bg_MSE: {avg_bg_mse:.4f}, fg_MSE: {avg_fg_mse:.4f}, "
        f"BCE Loss: {avg_bce_loss:.4f}, Dice Loss: {avg_seg_dice_loss:.4f}, FP Penalty Loss: {avg_fp_penalty_loss:.4f}, "
        f"MSE Loss: {avg_mse_loss:.4f}, BG Zero Loss: {avg_bg_zero_loss:.4f}, AUX Loss: {avg_aux_offset_loss:.4f}"
    )

    return {
        'loss': avg_loss,
        'seg_loss': avg_seg_loss,
        'offset_loss': avg_offset_loss,
        'iou': avg_iou,
        'mse': avg_mse
    }


def validate(model, val_loader, seg_criterion, offset_criterion, aux_offset_loss_func, device, epoch, writer, dataset):
    """验证模型性能"""
    model.eval()
    total_loss = 0.0
    total_seg_loss = 0.0
    total_offset_loss = 0.0
    total_bce_loss = 0.0
    total_seg_dice_loss = 0.0
    total_mse_loss = 0.0 
    total_bg_zero_loss = 0.0
    total_fp_penalty_loss = 0.0
    total_aux_offset_loss = 0.0
    all_iou = []
    all_mse = []
    all_bg_mse = []
    all_fg_mse = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Val Epoch {epoch}")
        for batch in pbar:
            # 数据移至设备
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            sem_gt = batch['semantic_gt'].to(device)
            offset_gt_norm = batch['offset_gt_norm'].to(device)
            offset_gt_raw = batch['offset_gt_raw'].to(device)

            # 前向传播
            sem_pred, offset_pred_norm = model(rgb, depth)
            
            # 计算损失
            aux_offset_loss = aux_offset_loss_func(offset_pred_norm, sem_gt)  # 增加辅助分割的偏移损失
            seg_loss = seg_criterion(sem_pred, sem_gt.unsqueeze(1).float())
            offset_loss = offset_criterion(offset_pred_norm, offset_gt_norm)
            bg_zero_loss = background_zero_loss(offset_pred_norm, sem_gt)
            seg_dice_loss = dice_loss(sem_pred, sem_gt.unsqueeze(1).float())
            depth_grad = compute_depth_gradient(depth)
            fp_penalty_loss = false_positive_penalty_loss(sem_pred, depth_grad, sem_gt)

            aux_weight = min(0.02 + epoch / 100, 0.2)  # 前18 epoch从0.02增至0.2，之后保持
            seg_weight = max(1 - epoch / 100, 0.6)  # 前40 epoch从1降至0.6，之后保持

            loss = seg_loss * seg_weight + seg_dice_loss * 5 * seg_weight + fp_penalty_loss * 0.2 \
                + offset_loss * 0.3 + bg_zero_loss * 0.2 + aux_offset_loss* aux_weight
            
            # 计算指标
            sem_pred_argmax = torch.sigmoid(sem_pred) > 0.5
            iou = compute_iou(sem_pred_argmax.squeeze(1).cpu().numpy(), 
                             sem_gt.cpu().numpy())
            all_iou.append(iou)

            offset_pred_raw = dataset.denormalize_offset(offset_pred_norm)
            mse = compute_mse(offset_pred_raw.cpu().numpy(), 
                             offset_gt_raw.cpu().numpy())
            all_mse.append(mse)

            background_mask = (sem_gt == 0).float().unsqueeze(1)
            bg_mse = compute_masked_mse(offset_pred_raw.cpu().numpy(), 
                                offset_gt_raw.cpu().numpy(), 
                                background_mask.cpu().numpy())
            all_bg_mse.append(bg_mse) 

            foreground_mask = (sem_gt == 1).float().unsqueeze(1)
            fg_mse = compute_masked_mse(offset_pred_raw.cpu().numpy(), 
                                offset_gt_raw.cpu().numpy(), 
                                foreground_mask.cpu().numpy())
            all_fg_mse.append(fg_mse)
            
            # 累计损失
            total_loss += loss.item() * rgb.size(0)
            total_seg_loss += (seg_loss.item() * seg_weight + seg_dice_loss.item() * 5 * seg_weight + fp_penalty_loss * 0.2) * rgb.size(0)
            total_offset_loss += (offset_loss.item() * 0.3 + bg_zero_loss.item() * 0.2 + aux_offset_loss.item() * aux_weight) * rgb.size(0)
            total_bce_loss += seg_loss.item() * seg_weight * rgb.size(0)
            total_seg_dice_loss += seg_dice_loss.item() * 5 * seg_weight * rgb.size(0)
            total_mse_loss += offset_loss.item() * 0.3 * rgb.size(0)
            total_bg_zero_loss += bg_zero_loss.item() * 0.2 * rgb.size(0)
            total_fp_penalty_loss += fp_penalty_loss.item() * 0.2 * rgb.size(0)
            total_aux_offset_loss += aux_offset_loss.item() * aux_weight * rgb.size(0)  # 增加辅助分割的偏移损失

            # 更新进度条
            pbar.set_postfix({
            'val_loss': f"{loss.item():.4f}",
            'val_seg_loss': f"{(seg_loss.item() * seg_weight) + (seg_dice_loss.item() * 5 * seg_weight) + (fp_penalty_loss.item() * 0.2):.4f}",
            'val_offset_loss': f"{(offset_loss.item() * 0.3 + bg_zero_loss.item() * 0.2 + aux_offset_loss.item() * aux_weight):.4f}"
            })

    # 计算平均指标
    avg_loss = total_loss / len(val_loader.dataset)
    avg_seg_loss = total_seg_loss / len(val_loader.dataset)
    avg_offset_loss = total_offset_loss / len(val_loader.dataset)
    avg_iou = np.mean(all_iou) if all_iou else 0.0
    avg_mse = np.mean(all_mse) if all_mse else 0.0
    avg_bg_mse = np.mean(all_bg_mse) if all_bg_mse else 0.0
    avg_fg_mse = np.mean(all_fg_mse) if all_fg_mse else 0.0

    avg_bce_loss = total_bce_loss / len(val_loader.dataset)
    avg_seg_dice_loss = total_seg_dice_loss / len(val_loader.dataset)
    avg_mse_loss = total_mse_loss / len(val_loader.dataset)
    avg_bg_zero_loss = total_bg_zero_loss / len(val_loader.dataset)
    avg_fp_penalty_loss = total_fp_penalty_loss / len(val_loader.dataset)
    avg_aux_offset_loss = total_aux_offset_loss / len(val_loader.dataset)  # 增加辅助分割的偏移损失

    # 记录到TensorBoard
    # writer.add_scalar('val/loss', avg_loss, epoch)
    # writer.add_scalar('val/seg_loss', avg_seg_loss, epoch)
    # writer.add_scalar('val/offset_loss', avg_offset_loss, epoch)
    writer.add_scalars(
    main_tag='val/loss_com',  # 图表标题（对应TensorBoard中的标签）
    tag_scalar_dict={                 # 曲线名称→指标值的映射
        'Total Loss': avg_loss,
        'Seg Loss': avg_seg_loss,
        'Offset Loss': avg_offset_loss
    },
    global_step=epoch)
    writer.add_scalar('val/iou', avg_iou, epoch)
    writer.add_scalar('val/mse', avg_mse, epoch)
    writer.add_scalars(
    main_tag='val/mse_com',
    tag_scalar_dict={
        'MSE': avg_mse,
        'BG MSE': avg_bg_mse,
        'FG MSE': avg_fg_mse
    },
    global_step=epoch)
    writer.add_scalars(
    main_tag='val/seg_loss_com',
    tag_scalar_dict={
        'Total Seg Loss': avg_seg_loss,
        'BCE Loss': avg_bce_loss,
        'Dice Loss': avg_seg_dice_loss,
        'FP Penalty Loss': avg_fp_penalty_loss
    },
    global_step=epoch)
    writer.add_scalars(
    main_tag='val/offset_loss_com',
    tag_scalar_dict={
        'Total Offset Loss': avg_offset_loss,
        'MSE Loss': avg_mse_loss,
        'BG Zero Loss': avg_bg_zero_loss,
        'AUX Loss': avg_aux_offset_loss
    },
    global_step=epoch)

    logging.info(
        f"Val Epoch {epoch} - Loss: {avg_loss:.4f}, "
        f"Seg Loss: {avg_seg_loss:.4f}, Offset Loss: {avg_offset_loss:.4f}, "
        f"IoU: {avg_iou:.4f}, MSE: {avg_mse:.4f}, bg_MSE: {avg_bg_mse:.4f}, fg_MSE: {avg_fg_mse:.4f}, "
        f"BCE Loss: {avg_bce_loss:.4f}, Dice Loss: {avg_seg_dice_loss:.4f}, FP Penalty Loss: {avg_fp_penalty_loss:.4f}, "
        f"MSE Loss: {avg_mse_loss:.4f}, BG Zero Loss: {avg_bg_zero_loss:.4f}, AUX Loss: {avg_aux_offset_loss:.4f}"
    )

    return {
        'loss': avg_loss,
        'seg_loss': avg_seg_loss,
        'offset_loss': avg_offset_loss,
        'iou': avg_iou,
        'mse': avg_mse,
        'bg_mse': avg_bg_mse,
        'fg_mse': avg_fg_mse,
        'fp_penalty_loss': avg_fp_penalty_loss,
        'aux_offset_loss': avg_aux_offset_loss
    }


def run_training(config, experiment_name):
    """执行训练流程"""
    # 设备配置
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # 创建数据集和数据加载器
    train_dataset = DualHeadUNetDataset(
        data_root=config['data_root'],
        split='train',
        # resize=config['resize'],
        rgb_mean=config.get('rgb_mean'),
        rgb_std=config.get('rgb_std'),
        depth_mean=config.get('depth_mean'),
        depth_std=config.get('depth_std'),
        offset_mean=config.get('offset_mean'),
        offset_std=config.get('offset_std')
    )

    val_dataset = DualHeadUNetDataset(
        data_root=config['data_root'],
        split='val',
        # resize=config['resize'],
        rgb_mean=config.get('rgb_mean'),
        rgb_std=config.get('rgb_std'),
        depth_mean=config.get('depth_mean'),
        depth_std=config.get('depth_std'),
        offset_mean=config.get('offset_mean'),
        offset_std=config.get('offset_std')
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # 初始化模型
    model = EarlyFusionUNet(
        n_classes=config['n_classes'],
        bilinear=config['bilinear'],
        dropout=config['dropout']
    ).to(device)

    # 损失函数和优化器
    seg_criterion = nn.BCEWithLogitsLoss() # if config['n_classes'] == 1 else nn.CrossEntropyLoss()
    offset_criterion = nn.MSELoss()

    # offset_criterion = nn.L1Loss()
    aux_offset_loss_func = InstanceAwareOffsetLoss().to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config['lr']),
        weight_decay=float(config['weight_decay'])
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['scheduler_factor'],
        patience=config['scheduler_patience'],
        verbose=True
    )

    # 梯度裁剪
    grad_clip = config.get('grad_clip', 1.0)  # 从配置读取，默认1.0

    # 初始化日志和TensorBoard
    log_dir = os.path.join(config['log_root'], experiment_name)
    writer = SummaryWriter(log_dir=log_dir)
    
    # 断点续训
    checkpoint_dir = os.path.join(config['checkpoint_root'], experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'latest.pth')
    model, optimizer, start_epoch, best_metrics = load_checkpoint(model, optimizer, checkpoint_path)

    # 初始化最佳指标
    best_iou = best_metrics.get('iou', 0.0)
    best_mse = best_metrics.get('mse', float('inf'))
    best_score = -1

    # 开始训练
    logging.info(f"Starting training from epoch {start_epoch}, total epochs: {config['epochs']}")
    for epoch in range(start_epoch, config['epochs']):
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, seg_criterion, offset_criterion, aux_offset_loss_func,
            optimizer, device, epoch, writer, grad_clip
        )

        # 验证
        val_metrics = validate(
            model, val_loader, seg_criterion, offset_criterion, aux_offset_loss_func,
            device, epoch, writer, val_dataset
        )

        # 学习率调度
        scheduler.step(val_metrics['loss'])
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, epoch)
        logging.info(f"Epoch {epoch} - Current Learning Rate: {current_lr:.6f}")

        # 保存最新checkpoint
        save_checkpoint(model, optimizer, epoch, val_metrics, config, checkpoint_path)

        # MAE归一化，当MAE > max_mae时，normalized_mae强制为0（惩罚严重偏离）
        max_mse = 500
        normalized_mse = 1 - (val_metrics['mse'] / max_mse) # if val_metrics['mse'] <= max_mse else 0.0
        normalized_mse = max(normalized_mse, 0.0)  # 确保不小于0

        # 改进2：增加"背景偏移MAE"的惩罚项（若有此指标）
        # 假设val_metrics新增了background_mae（背景区域的MAE）
        if 'bg_mse' in val_metrics:
            # 背景MAE越大，惩罚越重（降低总分）
            bg_mse_penalty = val_metrics['bg_mse'] / max_mse # 背景MAE权重更高
            bg_mse_penalty = min(bg_mse_penalty, 0.5)  # 惩罚上限0.5
        else:
            bg_mse_penalty = 0.0

        # 最终评分，减去背景偏移惩罚
        score = 0.6 * val_metrics['iou'] + 0.4 * (normalized_mse - bg_mse_penalty)
        logging.info(f"Score: {score:.4f}, from_IoU: {(0.6 * val_metrics['iou']):.4f}, from_MSE: {(0.4 * normalized_mse):.4f}, from_BGP: {(-0.4 * bg_mse_penalty):.4f}")

        #保存最佳模型
        # if val_metrics['iou'] > best_iou:  # 根据IoU判断
        if score > best_score:  # 根据综合评分判断
            best_score = score
            best_iou = val_metrics['iou']
            best_mse = val_metrics['mse']
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best.pth')
            save_checkpoint(model, optimizer, epoch, val_metrics, config, best_checkpoint_path)
            logging.info(f"New best model! Score: {best_score:.4f}, IoU: {best_iou:.4f}, MSE: {best_mse:.4f}")

    # 训练结束
    writer.close()
    logging.info(f"Training completed. Best Score: {best_score:.4f}, IoU: {best_iou:.4f}, MSE: {best_mse:.4f}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train Dual-Head UNet')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--experiment', type=str, default=None, help='Experiment name (optional)')
    args = parser.parse_args()

    # 加载基础配置
    config = load_config(args.config)

    # 生成实验名称（如果未指定）
    if args.experiment is None:
        timestamp = datetime.now().strftime("%Y%m%d%H")
        experiment_name = f"unet_train_{timestamp}"
    else:
        experiment_name = args.experiment

    # 设置日志
    log_root = os.path.join(config['log_root'], experiment_name)
    logger = setup_logging(log_root, experiment_name)
    logger.info(f"Experiment name: {experiment_name}")
    logger.info(f"Using config: {config}")

    run_training(config, experiment_name)


if __name__ == "__main__":
    main()
