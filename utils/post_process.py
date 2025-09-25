import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from scipy.ndimage import binary_opening, binary_closing, binary_dilation
from PIL import Image, ImageDraw, ImageFont
import os
from typing import Tuple, Dict, List

# --------------------------- 工具函数：统一数据格式转换 ---------------------------
def numpy_to_pil(np_img: np.ndarray, mode: str = "L") -> Image.Image:
    """
    将numpy数组转为PIL图像（适配二值掩膜、RGB图等）
    Args:
        np_img: 输入numpy数组（H,W）或（H,W,3）
        mode: PIL图像模式（L=灰度/二值，RGB=彩色）
    Returns:
        pil_img: PIL图像对象
    """
    # 二值图（0/1）转为0-255范围
    if mode == "L" and np_img.dtype in [np.uint8, np.bool_]:
        np_img = (np_img * 255).astype(np.uint8)
    # RGB图确保通道顺序正确（numpy默认HWC，PIL默认RGB）
    if mode == "RGB" and np_img.shape[-1] == 3:
        np_img = np.clip(np_img, 0, 255).astype(np.uint8)
    return Image.fromarray(np_img, mode=mode)

def pil_to_numpy(pil_img: Image.Image) -> np.ndarray:
    """PIL图像转为numpy数组（适配后续计算）"""
    return np.array(pil_img)

# --------------------------- 核心1：分割掩膜后处理 ---------------------------
def post_process_mask(
    mask: np.ndarray, 
    min_area: int = 500, 
    kernel_size: int = 3
) -> np.ndarray:
    """
    去除掩膜噪声与小块区域，输出处理后的二值掩膜
    Args:
        mask: 原始二值掩膜（H,W），值为0或1
        min_area: 最小保留区域面积（像素），拆垛场景建议300-1000
        kernel_size: 形态学操作核大小（3或5，噪声多则增大）
    Returns:
        processed_mask: 处理后二值掩膜（H,W），值为0或1
    """
    # 1. 形态学开运算：先腐蚀再膨胀，去除小噪声（如孤立像素）
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_open = binary_opening(mask, structure=kernel).astype(np.uint8)
    
    # 2. 形态学闭运算：先膨胀再腐蚀，填充掩膜内部小空洞
    mask_close = binary_closing(mask_open, structure=kernel).astype(np.uint8)
    print(mask_close)
    if not isinstance(mask_close, np.ndarray):
        raise ValueError("输入的掩码为不是numpy")

    if mask_close is None or mask_close.size == 0:
        raise ValueError("输入的掩码为空")

    # 如果是多通道图像，转换为单通道
    if len(mask_close.shape) > 2 and mask_close.shape[2] > 1:
        mask_close = cv2.cvtColor(mask_close, cv2.COLOR_BGR2GRAY)

    # 3. 连通区域分析：过滤面积小于min_area的区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_close, connectivity=8  # 8连通（更贴合料箱连通区域）
    )
    large_area_mask = np.zeros_like(mask_close)
    for label in range(1, num_labels):  # 跳过背景（label=0）
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            large_area_mask[labels == label] = 1
    
    # 4. 轻微膨胀：恢复开运算导致的边缘收缩（避免掩膜边缘损失）
    processed_mask = binary_dilation(large_area_mask, structure=kernel).astype(np.uint8)
    
    return processed_mask

# --------------------------- 核心2：偏移值图掩膜过滤 ---------------------------
def mask_offset_values(
    offset_map: np.ndarray, 
    processed_mask: np.ndarray
) -> np.ndarray:
    """
    将掩膜外的偏移值置0（仅保留有效区域的偏移信息）
    Args:
        offset_map: 模型输出偏移值图（H,W,2），通道0=dx，通道1=dy
        processed_mask: 后处理后的二值掩膜（H,W）
    Returns:
        masked_offset: 过滤后的偏移值图（H,W,2）
    """
    # 扩展掩膜维度至3D（H,W）→（H,W,1），与偏移图通道匹配
    mask_3d = np.expand_dims(processed_mask, axis=-1)
    # 掩膜外区域偏移值置0
    masked_offset = offset_map * mask_3d
    return masked_offset

# --------------------------- 核心3：候选点提取与DBSCAN聚类 ---------------------------
def extract_and_cluster_centroids(
    processed_mask: np.ndarray, 
    masked_offset: np.ndarray, 
    eps: float = 50.0, 
    min_samples: int = 10
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    从有效区域提取候选中心点，用DBSCAN聚类得到最终目标中心
    Args:
        processed_mask: 后处理掩膜（H,W）
        masked_offset: 过滤后的偏移值图（H,W,2）
        eps: DBSCAN聚类半径（像素），根据料箱间距调整（40-80）
        min_samples: 形成聚类的最小样本数（8-15，避免噪声聚类）
    Returns:
        candidates: 所有候选中心点（N,2），格式(x,y)
        cluster_centers: 最终聚类中心列表，每个元素为(x,y)，保留2位小数
    """
    # 1. 提取掩膜内所有像素坐标（y为行号，x为列号）
    y_coords, x_coords = np.where(processed_mask == 1)
    if len(y_coords) == 0:
        return np.array([]), []  # 无有效区域，返回空
    
    # 2. 计算每个有效像素对应的目标中心点（x+dx, y+dy）
    candidates = []
    for x, y in zip(x_coords, y_coords):
        dx, dy = masked_offset[y, x]  # 偏移值：dx=x方向偏移，dy=y方向偏移
        centroid_x = x + dx
        centroid_y = y + dy
        candidates.append([centroid_x, centroid_y])
    candidates = np.array(candidates)
    
    # 3. DBSCAN聚类（过滤离散候选点，聚合同一料箱的中心点）
    if len(candidates) < min_samples:
        return candidates, []  # 候选点不足，无法形成聚类
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(candidates)
    
    # 4. 计算每个聚类的平均中心（排除噪声点label=-1）
    cluster_centers = []
    unique_labels = set(labels) - {-1}
    for label in unique_labels:
        cluster_points = candidates[labels == label]
        avg_x = round(np.mean(cluster_points[:, 0]), 2)
        avg_y = round(np.mean(cluster_points[:, 1]), 2)
        cluster_centers.append((avg_x, avg_y))
    
    return candidates, cluster_centers

# --------------------------- 核心4：结果可视化与保存（PIL实现） ---------------------------
def draw_result_visualization(
    original_mask: np.ndarray,
    processed_mask: np.ndarray,
    masked_offset: np.ndarray,
    candidates: np.ndarray,
    cluster_centers: List[Tuple[float, float]],
    img_h: int,
    img_w: int
) -> Image.Image:
    """
    绘制后处理结果对比图（4个子图：原始掩膜、处理后掩膜、偏移值幅度、聚类结果）
    Args:
        各输入参数同前；img_h/img_w：图像高度/宽度（统一子图尺寸）
    Returns:
        result_img: 拼接后的结果可视化图（PIL RGB图像）
    """
    # 1. 准备单个子图尺寸（2行2列，总尺寸为2*img_w × 2*img_h）
    subplot_size = (img_w, img_h)
    total_size = (2 * img_w, 2 * img_h)
    result_img = Image.new("RGB", total_size, color=(255, 255, 255))  # 白色背景
    draw = ImageDraw.Draw(result_img)
    
    # 2. 加载字体（适配不同环境，无字体则跳过文字标注）
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    
    # 3. 子图1：原始掩膜（灰色）
    original_mask_pil = numpy_to_pil(original_mask, mode="L")
    original_mask_pil = original_mask_pil.resize(subplot_size)
    result_img.paste(original_mask_pil, (0, 0))
    draw.text((10, 10), "Original Mask", fill=(255, 0, 0), font=font)
    
    # 4. 子图2：处理后掩膜（灰色）
    processed_mask_pil = numpy_to_pil(processed_mask, mode="L")
    processed_mask_pil = processed_mask_pil.resize(subplot_size)
    result_img.paste(processed_mask_pil, (img_w, 0))
    draw.text((img_w + 10, 10), "Processed Mask", fill=(255, 0, 0), font=font)
    
    # 5. 子图3：偏移值幅度图（彩色，值越大越亮）
    offset_magnitude = np.linalg.norm(masked_offset, axis=2)  # 计算偏移幅度（dx²+dy²）^0.5
    # 幅度值归一化到0-255（便于可视化）
    if offset_magnitude.max() > 0:
        offset_magnitude = (offset_magnitude / offset_magnitude.max()) * 255
    offset_magnitude_pil = numpy_to_pil(offset_magnitude.astype(np.uint8), mode="L")
    offset_magnitude_pil = offset_magnitude_pil.resize(subplot_size)
    # 转为伪彩色（用PIL的调色板）
    offset_magnitude_pil = offset_magnitude_pil.convert("P")
    offset_magnitude_pil.putpalette([
        0, 0, 0,    # 0=黑色（无偏移）
        0, 0, 255,  # 低幅度=蓝色
        0, 255, 255,# 中幅度=青色
        255, 255, 0,# 高幅度=黄色
        255, 0, 0   # 最高幅度=红色
    ])
    offset_magnitude_pil = offset_magnitude_pil.convert("RGB")
    result_img.paste(offset_magnitude_pil, (0, img_h))
    draw.text((10, img_h + 10), "Offset Magnitude", fill=(255, 0, 0), font=font)
    
    # 6. 子图4：聚类结果（处理后掩膜+候选点+聚类中心）
    cluster_bg_pil = processed_mask_pil.convert("RGB")  # 灰色背景转为RGB
    cluster_draw = ImageDraw.Draw(cluster_bg_pil)
    # 绘制候选点（蓝色小点，透明度通过颜色叠加实现）
    if len(candidates) > 0:
        # 候选点坐标缩放到子图尺寸
        scale_x = img_w / original_mask.shape[1]
        scale_y = img_h / original_mask.shape[0]
        scaled_candidates = candidates * [scale_x, scale_y]
        for (x, y) in scaled_candidates:
            cluster_draw.ellipse(
                (x-1, y-1, x+1, y+1),  # 2x2像素小圆
                fill=(0, 0, 255, 100),  # 蓝色，半透明
                outline=None
            )
    # 绘制聚类中心（红色叉号，尺寸10x10）
    for i, (x, y) in enumerate(cluster_centers):
        scaled_x = x * scale_x
        scaled_y = y * scale_y
        # 叉号：左上→右下，右上→左下
        cluster_draw.line(
            (scaled_x-5, scaled_y-5, scaled_x+5, scaled_y+5),
            fill=(255, 0, 0), width=2
        )
        cluster_draw.line(
            (scaled_x+5, scaled_y-5, scaled_x-5, scaled_y+5),
            fill=(255, 0, 0), width=2
        )
        # 标注聚类序号
        cluster_draw.text(
            (scaled_x+8, scaled_y-8), str(i+1), fill=(255, 0, 0), font=font
        )
    result_img.paste(cluster_bg_pil, (img_w, img_h))
    draw.text((img_w + 10, img_h + 10), f"Clusters (Total: {len(cluster_centers)})", fill=(255, 0, 0), font=font)
    
    return result_img

# --------------------------- 主函数：一键调用后处理流程 ---------------------------
def model_output_post_process(
    model_output: Dict[str, np.ndarray],
    save_dir: str = "./post_process_results",
    min_area: int = 500,
    kernel_size: int = 3,
    eps: float = 50.0,
    min_samples: int = 10,
    save_visualization: bool = True,
    save_masks: bool = True
) -> Dict[str, any]:
    """
    衔接模型输出的完整后处理流程：输入模型输出→后处理→保存结果→返回关键信息
    Args:
        model_output: 模型输出字典，需包含：
            - "mask": 原始分割掩膜（H,W），值0/1
            - "offset_map": 偏移值图（H,W,2），通道0=dx，通道1=dy
        save_dir: 结果保存目录（自动创建）
        min_area/kernel_size/eps/min_samples: 后处理参数（同前）
        save_visualization: 是否保存结果可视化图
        save_masks: 是否保存处理后的掩膜
    Returns:
        post_process_result: 后处理结果字典，包含：
            - "processed_mask": 处理后掩膜（np.ndarray）
            - "masked_offset": 过滤后偏移值图（np.ndarray）
            - "cluster_centers": 聚类中心列表（List[(x,y)]）
            - "candidates": 候选中心点（np.ndarray）
            - "save_paths": 保存文件路径（可视化图、处理后掩膜）
    """
    # 1. 解析模型输出（确保数据格式正确）
    original_mask = model_output["mask"].squeeze()  # 确保为2D（H,W）
    offset_map = model_output["offset_map"].squeeze()  # 确保为3D（H,W,2）
    img_h, img_w = original_mask.shape[0], original_mask.shape[1]
    
    # 2. 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    save_paths = {}
    
    # 3. 执行后处理核心步骤
    processed_mask = post_process_mask(original_mask, min_area, kernel_size)
    masked_offset = mask_offset_values(offset_map, processed_mask)
    candidates, cluster_centers = extract_and_cluster_centroids(
        processed_mask, masked_offset, eps, min_samples
    )
    
    # 4. 保存处理后的掩膜（若开启）
    if save_masks:
        processed_mask_pil = numpy_to_pil(processed_mask, mode="L")
        mask_save_path = os.path.join(save_dir, "processed_mask.png")
        processed_mask_pil.save(mask_save_path)
        save_paths["processed_mask"] = mask_save_path
    
    # 5. 生成并保存可视化结果（若开启）
    if save_visualization:
        result_visual_img = draw_result_visualization(
            original_mask, processed_mask, masked_offset,
            candidates, cluster_centers, img_h, img_w
        )
        vis_save_path = os.path.join(save_dir, "post_process_visualization.png")
        result_visual_img.save(vis_save_path, quality=95)
        save_paths["visualization"] = vis_save_path
    
    # 6. 整理返回结果
    post_process_result = {
        "processed_mask": processed_mask,
        "masked_offset": masked_offset,
        "cluster_centers": cluster_centers,
        "candidates": candidates,
        "save_paths": save_paths,
        "num_clusters": len(cluster_centers)  # 方便后续统计
    }
    
    # 打印关键信息（便于日志跟踪）
    print(f"后处理完成：检测到{len(cluster_centers)}个料箱聚类中心")