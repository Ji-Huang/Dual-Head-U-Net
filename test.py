import os
import yaml
import torch
import numpy as np
import argparse
import logging
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# 导入自定义模块
from utils.data_loading import DualHeadUNetDataset
from models.unet import EarlyFusionUNet
from utils.metrics import compute_iou, compute_mae


def setup_logging(log_dir):
    """设置测试日志"""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "test.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_model(config, checkpoint_path, device):
    """加载模型和权重"""
    model = EarlyFusionUNet(
        n_classes=config['n_classes'],
        bilinear=config['bilinear'],
        dropout=config['dropout']
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


# 辅助函数：将数组归一化到0-255的uint8格式
def normalize_to_uint8(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max - arr_min < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    normalized = (arr - arr_min) / (arr_max - arr_min) * 255
    return normalized.astype(np.uint8)

# 辅助函数：PIL实现颜色映射（模拟OpenCV的COLORMAP_HOT）
def apply_colormap_pil(data, cmap_name="hot"):
    """
    对数据应用颜色映射，返回PIL图像
    cmap_name: "hot" 模拟OpenCV的COLORMAP_HOT
    """
    # 归一化数据到0-255
    normalized = normalize_to_uint8(data)
    
    # 创建颜色映射（热图：黑→红→黄→白）
    def hot_colormap(value):
        if value < 64:
            # 黑→红
            return (int(value * 4), 0, 0)
        elif value < 128:
            # 红→橙
            return (255, int((value - 64) * 4), 0)
        elif value < 192:
            # 橙→黄
            return (255, 255, int((value - 128) * 4))
        else:
            # 黄→白
            return (255, 255, 127 + int((value - 192) * 1.28))
    
    # 应用颜色映射
    h, w = normalized.shape
    rgb_array = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            rgb_array[i, j] = hot_colormap(normalized[i, j])
    
    return Image.fromarray(rgb_array)

# 辅助函数：绘制矢量场
def draw_vector_field_pil(offset_np, stride=20):
    h, w = offset_np.shape[:2]
    
    # 生成背景图：矢量幅度灰度图转RGB
    magnitude = np.sqrt(offset_np[..., 0] **2 + offset_np[..., 1]** 2)
    background_gray = normalize_to_uint8(magnitude)
    background_pil = Image.fromarray(background_gray, mode="L").convert("RGB")
    
    # 稀疏采样矢量坐标
    y_coords = np.arange(0, h, stride)
    x_coords = np.arange(0, w, stride)
    u_sampled = offset_np[y_coords[:, None], x_coords, 0]
    v_sampled = offset_np[y_coords[:, None], x_coords, 1]
    
    # 绘制箭头
    draw = ImageDraw.Draw(background_pil)
    arrow_color = (255, 0, 0)  # 红色
    arrow_width = 1
    tip_length = 0.3
    scale_factor = 5
    
    for i in range(len(y_coords)):
        for j in range(len(x_coords)):
            y = y_coords[i]
            x = x_coords[j]
            dx = u_sampled[i, j]
            dy = v_sampled[i, j]
            
            if abs(dx) <= 1 and abs(dy) <= 1:
                continue
            
            scaled_dx = dx / scale_factor
            scaled_dy = dy / scale_factor
            end_x = np.clip(int(x + scaled_dx), 0, w - 1)
            end_y = np.clip(int(y + scaled_dy), 0, h - 1)
            
            # 绘制箭头主线
            draw.line(xy=[(x, y), (end_x, end_y)], fill=arrow_color, width=arrow_width)
            
            # 绘制箭头尖端
            line_length = np.sqrt(scaled_dx **2 + scaled_dy** 2)
            if line_length < 1e-6:
                continue
            actual_tip_len = line_length * tip_length
            angle = np.arctan2(scaled_dy, scaled_dx)
            tip1_x = end_x - actual_tip_len * np.cos(angle + np.pi*5/6)
            tip1_y = end_y - actual_tip_len * np.sin(angle + np.pi*5/6)
            tip2_x = end_x - actual_tip_len * np.cos(angle - np.pi*5/6)
            tip2_y = end_y - actual_tip_len * np.sin(angle - np.pi*5/6)
            draw.line(xy=[(end_x, end_y), (int(tip1_x), int(tip1_y))], fill=arrow_color, width=arrow_width)
            draw.line(xy=[(end_x, end_y), (int(tip2_x), int(tip2_y))], fill=arrow_color, width=arrow_width)
    
    return background_pil

# 辅助函数：添加文字标题
def add_text_pil(image_pil, text, text_color=(255, 255, 255)):
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        # Windows: font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
        # macOS: font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default(size=16)
    
    text_bbox = draw.textbbox((0,0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    draw.rectangle([(10,10), (10+text_w+4, 10+text_h+4)], fill=(0,0,0,180))
    draw.text((12, 12), text, fill=text_color, font=font)
    return image_pil

# 主可视化函数（包含热力图）
def visualize_results(rgb, sem_gt, sem_pred, offset_gt, offset_pred, save_path, idx, vector_stride=20):
    """
    完整可视化函数：包含RGB图、语义掩码、矢量场和热力图
    布局：2行4列
    """
    # -------------------------- 1. 基础数据转换 --------------------------
    # RGB图处理
    rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.180, 0.246, 0.285])
    std = np.array([0.207, 0.202, 0.230])
    rgb_np = np.clip((rgb_np * std + mean) * 255, 0, 255).astype(np.uint8)
    rgb_pil = Image.fromarray(rgb_np)
    
    # 语义掩码处理
    def sem_to_pil(sem_tensor):
        sem_np = sem_tensor.cpu().numpy().astype(np.uint8)
        sem_gray = Image.fromarray(sem_np * 255, mode="L")
        return sem_gray.convert("RGB")
    sem_gt_pil = sem_to_pil(sem_gt)
    sem_pred_pil = sem_to_pil(sem_pred)
    
    # -------------------------- 2. 矢量场可视化 --------------------------
    offset_gt_np = offset_gt.permute(1, 2, 0).cpu().numpy()
    offset_pred_np = offset_pred.permute(1, 2, 0).cpu().numpy()
    offset_gt_pil = draw_vector_field_pil(offset_gt_np, stride=vector_stride)
    offset_pred_pil = draw_vector_field_pil(offset_pred_np, stride=vector_stride)
    
    # -------------------------- 3. 热力图可视化（新增） --------------------------
    # 计算偏移幅度（用于热力图）
    magnitude_gt = np.sqrt(offset_gt_np[..., 0]** 2 + offset_gt_np[..., 1] **2)
    magnitude_pred = np.sqrt(offset_pred_np[..., 0]** 2 + offset_pred_np[..., 1] ** 2)
    
    # 生成热力图（使用PIL实现的热图颜色映射）
    heatmap_gt_pil = apply_colormap_pil(magnitude_gt, cmap_name="hot")
    heatmap_pred_pil = apply_colormap_pil(magnitude_pred, cmap_name="hot")
    
    # -------------------------- 4. 添加标题 --------------------------
    rgb_pil = add_text_pil(rgb_pil, "RGB Image")
    sem_gt_pil = add_text_pil(sem_gt_pil, "Semantic GT")
    sem_pred_pil = add_text_pil(sem_pred_pil, "Semantic Prediction")
    offset_gt_pil = add_text_pil(offset_gt_pil, f"Offset GT (Stride={vector_stride})")
    offset_pred_pil = add_text_pil(offset_pred_pil, f"Offset Prediction")
    heatmap_gt_pil = add_text_pil(heatmap_gt_pil, "GT Magnitude Heatmap")
    heatmap_pred_pil = add_text_pil(heatmap_pred_pil, "Prediction Magnitude Heatmap")
    
    # -------------------------- 5. 拼接图像（2行4列新布局） --------------------------
    img_w, img_h = rgb_pil.size
    black_pil = Image.new("RGB", (img_w, img_h), color=(0, 0, 0))
    
    # 第一行：RGB + 语义GT + 语义预测 + 预留位
    row1 = Image.new("RGB", (img_w * 4, img_h))
    row1.paste(rgb_pil, (0, 0))
    row1.paste(sem_gt_pil, (img_w, 0))
    row1.paste(sem_pred_pil, (img_w * 2, 0))
    row1.paste(black_pil, (img_w * 3, 0))  # 占位
    
    # 第二行：偏移GT矢量场 + 偏移预测矢量场 + GT热力图 + 预测热力图
    row2 = Image.new("RGB", (img_w * 4, img_h))
    row2.paste(offset_gt_pil, (0, 0))
    row2.paste(offset_pred_pil, (img_w, 0))
    row2.paste(heatmap_gt_pil, (img_w * 2, 0))
    row2.paste(heatmap_pred_pil, (img_w * 3, 0))
    
    # 垂直拼接
    final_pil = Image.new("RGB", (img_w * 4, img_h * 2))
    final_pil.paste(row1, (0, 0))
    final_pil.paste(row2, (0, img_h))
    
    # -------------------------- 6. 保存结果 --------------------------
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"result_{idx}.png")
    final_pil.save(save_file, quality=95)
    print(f"已保存样本 {idx} 的可视化结果：{save_file}")


def test_model(config, checkpoint_path):
    """测试模型主函数"""
    # 配置设备
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # 设置日志和结果保存目录
    result_dir = os.path.join(config['test_result_dir'], os.path.basename(os.path.dirname(checkpoint_path)))
    os.makedirs(result_dir, exist_ok=True)
    logger = setup_logging(result_dir)
    
    # 加载模型
    logger.info(f"Loading model from {checkpoint_path}")
    model = load_model(config, checkpoint_path, device)
    
    # 创建测试数据集和数据加载器
    test_dataset = DualHeadUNetDataset(
        data_root=config['data_root'],
        split='test',
        # resize=config['resize'],
        rgb_mean=config.get('rgb_mean'),
        rgb_std=config.get('rgb_std'),
        depth_mean=config.get('depth_mean'),
        depth_std=config.get('depth_std'),
        offset_mean=config.get('offset_mean'),
        offset_std=config.get('offset_std')
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # 评估指标初始化
    total_iou = []
    total_mae = []
    
    # 开始测试
    logger.info(f"Starting test on {len(test_dataset)} samples")
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch_idx, batch in enumerate(pbar):
            # 加载数据
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            sem_gt = batch['semantic_gt'].to(device)
            offset_gt_raw = batch['offset_gt_raw'].to(device)
            
            # 模型推理
            sem_pred, offset_pred_norm = model(rgb, depth)
            
            # 处理语义预测结果
            sem_pred_sigmoid = torch.sigmoid(sem_pred)
            sem_pred_mask = (sem_pred_sigmoid > 0.5).squeeze(1).float()  # 二值化
            
            # 反归一化偏移预测结果
            offset_pred_raw = test_dataset.denormalize_offset(offset_pred_norm)
            
            # 计算指标
            iou = compute_iou(sem_pred_mask.cpu().numpy(), sem_gt.cpu().numpy())
            mae = compute_mae(offset_pred_raw.cpu().numpy(), offset_gt_raw.cpu().numpy())
            
            total_iou.append(iou)
            total_mae.append(mae)
            pbar.set_postfix({"IoU": f"{iou:.4f}", "MAE": f"{mae:.4f}"})
            
            # # 可视化部分结果（每10个batch保存一次）
            # if batch_idx % 10 == 0:
            for i in range(min(8, rgb.size(0))):  # 每个batch最多可视化8个样本
                visualize_results(
                    rgb[i], 
                    sem_gt[i], 
                    sem_pred_mask[i], 
                    offset_gt_raw[i], 
                    offset_pred_raw[i],
                    result_dir,
                    idx=batch_idx * config['batch_size'] + i
                )
    
    # 计算平均指标
    avg_iou = np.mean(total_iou)
    avg_mae = np.mean(total_mae)
    
    # 输出最终结果
    logger.info(f"Test completed! Average IoU: {avg_iou:.4f}, Average MAE: {avg_mae:.4f}")
    with open(os.path.join(result_dir, "metrics.txt"), "w") as f:
        f.write(f"Average IoU: {avg_iou:.4f}\n")
        f.write(f"Average MAE: {avg_mae:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description='Test Dual-Head UNet')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 运行测试
    test_model(config, args.checkpoint)


if __name__ == "__main__":
    main()
