import os
import yaml
import torch
import numpy as np
import argparse
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from tqdm import tqdm

# 导入自定义模块
from models.unet import EarlyFusionUNet
from utils.data_loading import DualHeadUNetDataset
from utils.post_process import (
    post_process_mask,
    mask_offset_values,
    extract_and_cluster_centroids
)


def load_model(config, checkpoint_path, device):
    """加载模型"""
    model = EarlyFusionUNet(
        n_classes=config['n_classes'],
        bilinear=config['bilinear'],
        dropout=0.0  # 推理时关闭dropout
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def preprocess_input(rgb_path, depth_path, config, device):
    """预处理输入图像（RGB和深度图），同时返回原始RGB用于可视化"""
    # 加载图像（保留原始RGB用于可视化）
    rgb_img_raw = Image.open(rgb_path).convert("RGB")  # 原始RGB图（未标准化）
    depth_img = Image.open(depth_path)  # 深度图
    
    # 调整大小（与模型输入一致）
    resize = T.Resize(config['resize'])
    rgb_img = resize(rgb_img_raw.copy())  # 用于模型输入的缩放图
    rgb_img_raw = resize(rgb_img_raw)     # 原始RGB同步缩放（便于可视化对齐）
    depth_img = resize(depth_img)
    
    # 转换为numpy数组并标准化（模型输入用）
    rgb_np = np.array(rgb_img, dtype=np.float32) / 255.0
    depth_np = np.array(depth_img, dtype=np.float32)
    
    # 标准化RGB图
    rgb_mean = config.get('rgb_mean', [0.485, 0.456, 0.406])
    rgb_std = config.get('rgb_std', [0.229, 0.224, 0.225])
    for c in range(3):
        rgb_np[..., c] = (rgb_np[..., c] - rgb_mean[c]) / rgb_std[c]
    
    # 标准化深度图
    depth_mean = config.get('depth_mean', 2997.516)
    depth_std = config.get('depth_std', 1324.9409)
    depth_np = (depth_np - depth_mean) / depth_std
    
    # 调整维度并转换为Tensor
    rgb_tensor = torch.from_numpy(rgb_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
    depth_tensor = torch.from_numpy(depth_np[np.newaxis, ...]).unsqueeze(0).to(device)
    
    return rgb_tensor, depth_tensor, rgb_img_raw  # 返回原始缩放RGB图用于可视化


def normalize_offset_for_vis(offset_map):
    """将偏移值归一化到0-255，用于可视化"""
    offset_x = offset_map[..., 0]
    offset_y = offset_map[..., 1]
    
    # 分别归一化x和y通道（避免相互干扰）
    def norm_channel(channel):
        ch_min, ch_max = channel.min(), channel.max()
        if ch_max - ch_min < 1e-6:
            return np.zeros_like(channel, dtype=np.uint8)
        return ((channel - ch_min) / (ch_max - ch_min) * 255).astype(np.uint8)
    
    offset_x_vis = norm_channel(offset_x)
    offset_y_vis = norm_channel(offset_y)
    
    # 合并为RGB图（x=红，y=绿，无偏移=蓝）
    offset_vis = np.stack([offset_x_vis, offset_y_vis, np.zeros_like(offset_x_vis)], axis=-1)
    return Image.fromarray(offset_vis)


def draw_visualization(rgb_raw, sem_pred_mask, offset_map, 
                       processed_mask=None, clusters=None):
    """
    基于PIL绘制可视化对比图
    布局：2行3列 → [原始RGB, 原始掩膜, 偏移可视化], [后处理掩膜, 聚类结果, 空白]
    """
    H, W = rgb_raw.size[1], rgb_raw.size[0]
    # 创建画布（2行3列，每个子图大小H×W，间距10像素）
    canvas_w = 3 * W + 20
    canvas_h = 2 * H + 20
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # 加载字体（适配不同环境，无字体则用默认）
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except IOError:
        font = ImageFont.load_default()
    
    # 子图1：原始RGB图
    canvas.paste(rgb_raw, (10, 10))
    draw.text((10 + W//2 - 30, 10 + H + 5), "Original RGB", fill=(0, 0, 0), font=font)
    
    # 子图2：原始语义掩膜（白色=前景，黑色=背景，半透明叠加灰色底）
    sem_raw_vis = Image.new("RGB", (W, H), color=(128, 128, 128))
    sem_mask_pil = Image.fromarray((sem_pred_mask * 255).astype(np.uint8)).convert("L")
    sem_raw_vis.paste(Image.new("RGB", (W, H), color=(255, 255, 255)), 
                      mask=sem_mask_pil)  # 掩码叠加
    canvas.paste(sem_raw_vis, (W + 20, 10))
    draw.text((W + 20 + W//2 - 35, 10 + H + 5), "Raw Mask", fill=(0, 0, 0), font=font)
    
    # 子图3：偏移值可视化
    offset_vis = normalize_offset_for_vis(offset_map)
    canvas.paste(offset_vis, (2*W + 30, 10))
    draw.text((2*W + 30 + W//2 - 40, 10 + H + 5), "Offset (X=R,Y=G)", fill=(0, 0, 0), font=font)
    
    # 子图4：后处理掩膜（若无则显示“未启用”）
    if processed_mask is not None:
        sem_proc_vis = Image.new("RGB", (W, H), color=(128, 128, 128))
        sem_proc_pil = Image.fromarray((processed_mask * 255).astype(np.uint8)).convert("L")
        sem_proc_vis.paste(Image.new("RGB", (W, H), color=(0, 255, 0)), mask=sem_proc_pil)
        canvas.paste(sem_proc_vis, (10, H + 20))
        draw.text((10 + W//2 - 45, H + 20 + H + 5), "Processed Mask", fill=(0, 0, 0), font=font)
    else:
        no_proc = Image.new("RGB", (W, H), color=(220, 220, 220))
        draw_no = ImageDraw.Draw(no_proc)
        draw_no.text((W//2 - 50, H//2 - 10), "Postprocess\nDisabled", 
                    fill=(0, 0, 0), font=font, align="center")
        canvas.paste(no_proc, (10, H + 20))
    
    # 子图5：聚类结果（原始RGB + 聚类中心标记）
    cluster_vis = rgb_raw.copy()
    cluster_draw = ImageDraw.Draw(cluster_vis)
    if clusters is not None and len(clusters) > 0:
        for i, (x, y) in enumerate(clusters):
            # 绘制红色叉号（10×10像素）
            cluster_draw.line((x-5, y-5, x+5, y+5), fill=(255, 0, 0), width=3)
            cluster_draw.line((x+5, y-5, x-5, y+5), fill=(255, 0, 0), width=3)
            # 标注聚类序号
            cluster_draw.text((x+8, y-15), str(i+1), fill=(255, 0, 0), font=font, anchor="lt")
        draw.text((W + 20 + W//2 - 30, H + 20 + H + 5), f"Clusters ({len(clusters)})", 
                  fill=(0, 0, 0), font=font)
    else:
        cluster_draw.text((W//2 - 40, H//2 - 10), "No Clusters", 
                        fill=(0, 0, 0), font=font, align="center")
        draw.text((W + 20 + W//2 - 30, H + 20 + H + 5), "Clusters", fill=(0, 0, 0), font=font)
    canvas.paste(cluster_vis, (W + 20, H + 20))
    
    # 子图6：空白（占位，保持布局对称）
    blank = Image.new("RGB", (W, H), color=(240, 240, 240))
    canvas.paste(blank, (2*W + 30, H + 20))
    
    return canvas


def save_inference_results(sem_pred, offset_pred, save_dir, filename, 
                           post_processed=None, clusters=None, rgb_raw=None, save_vis=False):
    """保存推理结果（含可视化选项）"""
    # 创建基础保存目录
    sem_dir = os.path.join(save_dir, "semantic_masks")
    offset_dir = os.path.join(save_dir, "offset_maps")
    vis_dir = os.path.join(save_dir, "visualizations")  # 可视化目录
    os.makedirs(sem_dir, exist_ok=True)
    os.makedirs(offset_dir, exist_ok=True)
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)
    
    # 保存原始语义分割掩码
    sem_mask = (sem_pred * 255).astype(np.uint8)
    sem_save_path = os.path.join(sem_dir, f"{filename}_sem.png")
    Image.fromarray(sem_mask).save(sem_save_path)
    
    # 保存原始偏移向量
    offset_save_path = os.path.join(offset_dir, f"{filename}_offset.npy")
    np.save(offset_save_path, offset_pred)
    
    save_paths = {
        "semantic_mask": sem_save_path,
        "offset_map": offset_save_path
    }
    
    # 保存后处理结果（如果启用）
    if post_processed is not None:
        proc_sem_dir = os.path.join(save_dir, "processed_semantic_masks")
        proc_offset_dir = os.path.join(save_dir, "processed_offset_maps")
        cluster_dir = os.path.join(save_dir, "clusters")
        os.makedirs(proc_sem_dir, exist_ok=True)
        os.makedirs(proc_offset_dir, exist_ok=True)
        os.makedirs(cluster_dir, exist_ok=True)
        
        # 保存处理后的掩膜
        proc_mask = (post_processed["processed_mask"] * 255).astype(np.uint8)
        proc_sem_path = os.path.join(proc_sem_dir, f"{filename}_proc_sem.png")
        Image.fromarray(proc_mask).save(proc_sem_path)
        
        # 保存处理后的偏移值
        proc_offset_path = os.path.join(proc_offset_dir, f"{filename}_proc_offset.npy")
        np.save(proc_offset_path, post_processed["masked_offset"])
        
        # 保存聚类结果
        if clusters is not None and len(clusters) > 0:
            cluster_path = os.path.join(cluster_dir, f"{filename}_clusters.txt")
            with open(cluster_path, "w") as f:
                for i, (x, y) in enumerate(clusters):
                    f.write(f"cluster_{i}: {x:.2f}, {y:.2f}\n")
            save_paths["clusters"] = cluster_path
        
        save_paths.update({
            "processed_semantic_mask": proc_sem_path,
            "processed_offset_map": proc_offset_path
        })
    
    # 保存可视化结果（如果启用）
    if save_vis and rgb_raw is not None:
        vis_img = draw_visualization(
            rgb_raw=rgb_raw,
            sem_pred_mask=sem_pred,
            offset_map=offset_pred,
            processed_mask=post_processed["processed_mask"] if post_processed else None,
            clusters=clusters
        )
        vis_save_path = os.path.join(vis_dir, f"{filename}_vis.png")
        vis_img.save(vis_save_path, quality=95)  # 高质量保存
        save_paths["visualization"] = vis_save_path
    
    return save_paths


def inference(config, checkpoint_path, input_dir, output_dir, 
              enable_postprocess=False, save_visualization=False):
    """
    批量推理主函数
    功能：加载模型→预处理输入→模型推理→结果后处理（可选）→结果保存（含可视化可选）
    Args:
        config: 配置字典（含模型参数、预处理参数等）
        checkpoint_path: 模型权重文件路径
        input_dir: 输入目录（需包含images和depth子目录）
        output_dir: 结果保存目录
        enable_postprocess: 是否启用后处理（噪声过滤+聚类）
        save_visualization: 是否保存PIL生成的可视化对比图
    """
    # -------------------------- 1. 初始化设备 --------------------------
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}（若需使用GPU，请确保CUDA已配置）")

    # -------------------------- 2. 加载模型 --------------------------
    model = load_model(config, checkpoint_path, device)

    # -------------------------- 3. 创建数据集实例（复用反归一化方法） --------------------------
    # 仅用于加载偏移值的mean/std，不实际加载训练数据
    dataset = DualHeadUNetDataset(
        data_root=config['data_root'],
        split='train',  # 任意split均可（仅需统计量）
        offset_mean=config.get('offset_mean'),
        offset_std=config.get('offset_std')
    )

    # -------------------------- 4. 验证输入目录结构 --------------------------
    input_rgb_dir = os.path.join(input_dir, "images")  # RGB图像目录
    input_depth_dir = os.path.join(input_dir, "depth")  # 深度图目录
    # 检查目录是否存在
    if not os.path.exists(input_rgb_dir):
        raise FileNotFoundError(f"输入RGB目录不存在：{input_rgb_dir}")
    if not os.path.exists(input_depth_dir):
        raise FileNotFoundError(f"输入深度图目录不存在：{input_depth_dir}")
    
    # 获取RGB文件列表（支持png/jpg/jpeg格式）
    rgb_suffixes = (".png", ".jpg", ".jpeg")
    rgb_files = [f for f in os.listdir(input_rgb_dir) if f.lower().endswith(rgb_suffixes)]
    if len(rgb_files) == 0:
        raise ValueError(f"在{input_rgb_dir}中未找到任何RGB图像（支持格式：{rgb_suffixes}）")
    print(f"找到 {len(rgb_files)} 个推理样本，开始批量处理...")

    result_dir = os.path.join(output_dir, os.path.basename(os.path.dirname(checkpoint_path)))
    os.makedirs(result_dir, exist_ok=True)

    # -------------------------- 5. 批量推理与结果处理 --------------------------
    # 进度条（显示处理进度）
    with tqdm(total=len(rgb_files), desc="批量推理进度") as pbar:
        for idx, rgb_filename in enumerate(rgb_files):
            # 5.1 构建文件路径（假设RGB和深度图文件名前缀一致，深度图后缀为"_depth.png"）
            file_prefix = os.path.splitext(rgb_filename)[0]  # 文件名前缀（不含后缀）
            rgb_path = os.path.join(input_rgb_dir, rgb_filename)

            depth_filename = rgb_filename.replace("_color", "_depth")
            depth_filename = depth_filename.replace(".jpg", ".png")
            depth_path = os.path.join(input_depth_dir, depth_filename)  # 深度图路径
            
            # 跳过不存在的深度图
            if not os.path.exists(depth_path):
                print(f"\n警告：深度图 {depth_path} 不存在，已跳过该样本")
                pbar.update(1)
                continue

            # 5.2 输入预处理（返回模型输入Tensor + 原始缩放RGB图）
            print("输入预处理")
            rgb_tensor, depth_tensor, rgb_raw_scaled = preprocess_input(
                rgb_path=rgb_path,
                depth_path=depth_path,
                config=config,
                device=device
            )

            # 5.3 模型推理（关闭梯度计算，加速并减少内存占用）
            print("模型推理")
            with torch.no_grad():
                sem_pred_norm, offset_pred_norm = model(rgb_tensor, depth_tensor)  # 模型输出（归一化后）

            # 5.4 处理推理结果（反归一化+格式转换）
            # 语义分割结果：sigmoid激活→二值化（阈值0.5）→(H,W) numpy数组
            sem_pred_sigmoid = torch.sigmoid(sem_pred_norm).squeeze().cpu().numpy()  # (H,W)，值范围[0,1]
            sem_pred_mask = (sem_pred_sigmoid > 0.5).astype(np.uint8)  # 二值掩膜（0=背景，1=前景）
            
            # 偏移值结果：反归一化→调整维度为(H,W,2)（便于后处理和可视化）
            offset_pred_raw = dataset.denormalize_offset(offset_pred_norm)  # 反归一化（恢复原始数值范围）
            offset_pred_raw = offset_pred_raw.squeeze().cpu().numpy()  # 原始输出为(2,H,W)，挤压后仍为(2,H,W)
            offset_pred_raw = offset_pred_raw.transpose(1, 2, 0)  # 转换为(H,W,2)（通道0=dx，通道1=dy）

            # 5.5 后处理（可选：噪声过滤+偏移值掩膜+聚类）
            print("后处理")
            post_processed = None  # 存储后处理结果（掩膜+偏移值）
            clusters = None        # 存储聚类中心（列表，每个元素为(x,y)）
            if enable_postprocess:
                # 步骤1：处理语义掩膜（去除小噪声+填充空洞+过滤小区域）
                processed_mask = post_process_mask(
                    mask=sem_pred_mask,
                    min_area=config.get('postprocess_min_area', 500),  # 最小保留区域面积（默认500像素）
                    kernel_size=config.get('postprocess_kernel_size', 3)  # 形态学核大小（默认3×3）
                )
                print("步骤1")
                # 步骤2：掩膜外偏移值置0（仅保留前景区域的偏移值）
                masked_offset = mask_offset_values(
                    offset_map=offset_pred_raw,
                    mask=processed_mask
                )
                print("步骤2")
                # 步骤3：提取候选中心点并DBSCAN聚类
                candidates, clusters = extract_and_cluster_centroids(
                    processed_mask=processed_mask,
                    masked_offset=masked_offset,
                    eps=config.get('dbscan_eps', 50),  # 聚类半径（默认50像素）
                    min_samples=config.get('dbscan_min_samples', 10)  # 形成聚类的最小样本数（默认10）
                )
                print("步骤3")
                # 存储后处理结果
                post_processed = {
                    "processed_mask": processed_mask,
                    "masked_offset": masked_offset
                }

            # 5.6 保存结果（原始结果+后处理结果+可视化结果）
            print("保存结果")
            save_inference_results(
                sem_pred=sem_pred_mask,
                offset_pred=offset_pred_raw,
                save_dir=result_dir,
                filename=file_prefix,  # 用文件名前缀作为保存标识
                post_processed=post_processed,
                clusters=clusters,
                rgb_raw=rgb_raw_scaled,  # 用于可视化的原始缩放RGB图
                save_vis=save_visualization
            )

            # 5.7 更新进度条
            pbar.update(1)
            # 进度条附加信息（显示当前处理的文件名）
            pbar.set_postfix({"当前样本": rgb_filename, "已处理": f"{idx+1}/{len(rgb_files)}"})

    # -------------------------- 6. 推理完成提示 --------------------------
    print(f"\n批量推理全部完成！结果已保存至：{output_dir}")
    # 打印结果目录结构（方便用户查找）
    print("\n结果目录结构说明：")
    print(f"1. 原始分割掩膜：{os.path.join(output_dir, 'semantic_masks/raw')}")
    print(f"2. 原始偏移值图：{os.path.join(output_dir, 'offset_maps/raw')}")
    if enable_postprocess:
        print(f"3. 后处理分割掩膜：{os.path.join(output_dir, 'semantic_masks/processed')}")
        print(f"4. 后处理偏移值图：{os.path.join(output_dir, 'offset_maps/processed')}")
        print(f"5. 聚类结果（TXT）：{os.path.join(output_dir, 'clusters')}")
    if save_visualization:
        print(f"6. 可视化对比图：{os.path.join(output_dir, 'visualizations')}")


def main():
    """
    命令行入口函数
    功能：解析命令行参数→加载配置文件→调用inference函数
    """
    # -------------------------- 1. 解析命令行参数 --------------------------
    parser = argparse.ArgumentParser(description="双端UNet模型批量推理脚本（支持后处理与PIL可视化）")
    # 必选参数
    parser.add_argument('--config', type=str, default='config.yaml', help="配置文件路径（如config.yaml）")
    parser.add_argument('--checkpoint', type=str, required=True, help="模型权重文件路径（如model_best.pth）")
    parser.add_argument('--input', type=str, required=True, help="输入目录路径（需包含images和depth子目录）")
    # 可选参数
    parser.add_argument('--output', type=str, default="inference_results", help="结果保存目录（默认：inference_results）")
    parser.add_argument('--postprocess', action='store_true', help="启用后处理（噪声过滤+DBSCAN聚类）")
    parser.add_argument('--save-vis', action='store_true', help="保存PIL生成的可视化对比图")
    
    args = parser.parse_args()

    # -------------------------- 2. 加载配置文件 --------------------------
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"配置文件加载成功：{args.config}")
    except Exception as e:
        raise RuntimeError(f"加载配置文件失败：{str(e)}") from e

    # -------------------------- 3. 调用推理函数 --------------------------
    inference(
        config=config,
        checkpoint_path=args.checkpoint,
        input_dir=args.input,
        output_dir=args.output,
        enable_postprocess=args.postprocess,
        save_visualization=args.save_vis
    )


if __name__ == "__main__":
    main()