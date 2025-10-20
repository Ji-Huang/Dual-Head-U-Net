# import numpy as np
# import cv2
# from sklearn.cluster import DBSCAN
# from scipy.ndimage import binary_opening, binary_closing, binary_dilation
# from PIL import Image, ImageDraw, ImageFont
# import os
# from typing import Tuple, Dict, List

# # --------------------------- 工具函数：统一数据格式转换 ---------------------------
# def numpy_to_pil(np_img: np.ndarray, mode: str = "L") -> Image.Image:
#     """
#     将numpy数组转为PIL图像（适配二值掩膜、RGB图等）
#     Args:
#         np_img: 输入numpy数组（H,W）或（H,W,3）
#         mode: PIL图像模式（L=灰度/二值，RGB=彩色）
#     Returns:
#         pil_img: PIL图像对象
#     """
#     # 二值图（0/1）转为0-255范围
#     if mode == "L" and np_img.dtype in [np.uint8, np.bool_]:
#         np_img = (np_img * 255).astype(np.uint8)
#     # RGB图确保通道顺序正确（numpy默认HWC，PIL默认RGB）
#     if mode == "RGB" and np_img.shape[-1] == 3:
#         np_img = np.clip(np_img, 0, 255).astype(np.uint8)
#     return Image.fromarray(np_img, mode=mode)

# def pil_to_numpy(pil_img: Image.Image) -> np.ndarray:
#     """PIL图像转为numpy数组（适配后续计算）"""
#     return np.array(pil_img)

# # --------------------------- 核心1：分割掩膜后处理 ---------------------------
# def post_process_mask(
#     mask: np.ndarray, 
#     min_area: int = 500, 
#     kernel_size: int = 3
# ) -> np.ndarray:
#     """
#     去除掩膜噪声与小块区域，输出处理后的二值掩膜
#     Args:
#         mask: 原始二值掩膜（H,W），值为0或1
#         min_area: 最小保留区域面积（像素），拆垛场景建议300-1000
#         kernel_size: 形态学操作核大小（3或5，噪声多则增大）
#     Returns:
#         processed_mask: 处理后二值掩膜（H,W），值为0或1
#     """
#     # 1. 形态学开运算：先腐蚀再膨胀，去除小噪声（如孤立像素）
#     kernel = np.ones((kernel_size, kernel_size), np.uint8)
#     mask_open = binary_opening(mask, structure=kernel).astype(np.uint8)
    
#     # 2. 形态学闭运算：先膨胀再腐蚀，填充掩膜内部小空洞
#     mask_close = binary_closing(mask_open, structure=kernel).astype(np.uint8)
#     print(mask_close)
#     if not isinstance(mask_close, np.ndarray):
#         raise ValueError("输入的掩码为不是numpy")

#     if mask_close is None or mask_close.size == 0:
#         raise ValueError("输入的掩码为空")

#     # 如果是多通道图像，转换为单通道
#     if len(mask_close.shape) > 2 and mask_close.shape[2] > 1:
#         mask_close = cv2.cvtColor(mask_close, cv2.COLOR_BGR2GRAY)

#     # 3. 连通区域分析：过滤面积小于min_area的区域
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
#         mask_close, connectivity=8  # 8连通（更贴合料箱连通区域）
#     )
#     large_area_mask = np.zeros_like(mask_close)
#     for label in range(1, num_labels):  # 跳过背景（label=0）
#         if stats[label, cv2.CC_STAT_AREA] >= min_area:
#             large_area_mask[labels == label] = 1
    
#     # 4. 轻微膨胀：恢复开运算导致的边缘收缩（避免掩膜边缘损失）
#     processed_mask = binary_dilation(large_area_mask, structure=kernel).astype(np.uint8)
    
#     return processed_mask

# # --------------------------- 核心2：偏移值图掩膜过滤 ---------------------------
# def mask_offset_values(
#     offset_map: np.ndarray, 
#     processed_mask: np.ndarray
# ) -> np.ndarray:
#     """
#     将掩膜外的偏移值置0（仅保留有效区域的偏移信息）
#     Args:
#         offset_map: 模型输出偏移值图（H,W,2），通道0=dx，通道1=dy
#         processed_mask: 后处理后的二值掩膜（H,W）
#     Returns:
#         masked_offset: 过滤后的偏移值图（H,W,2）
#     """
#     # 扩展掩膜维度至3D（H,W）→（H,W,1），与偏移图通道匹配
#     mask_3d = np.expand_dims(processed_mask, axis=-1)
#     # 掩膜外区域偏移值置0
#     masked_offset = offset_map * mask_3d
#     return masked_offset

# # --------------------------- 核心3：候选点提取与DBSCAN聚类 ---------------------------
# def extract_and_cluster_centroids(
#     processed_mask: np.ndarray, 
#     masked_offset: np.ndarray, 
#     eps: float = 50.0, 
#     min_samples: int = 10
# ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
#     """
#     从有效区域提取候选中心点，用DBSCAN聚类得到最终目标中心
#     Args:
#         processed_mask: 后处理掩膜（H,W）
#         masked_offset: 过滤后的偏移值图（H,W,2）
#         eps: DBSCAN聚类半径（像素），根据料箱间距调整（40-80）
#         min_samples: 形成聚类的最小样本数（8-15，避免噪声聚类）
#     Returns:
#         candidates: 所有候选中心点（N,2），格式(x,y)
#         cluster_centers: 最终聚类中心列表，每个元素为(x,y)，保留2位小数
#     """
#     # 1. 提取掩膜内所有像素坐标（y为行号，x为列号）
#     y_coords, x_coords = np.where(processed_mask == 1)
#     if len(y_coords) == 0:
#         return np.array([]), []  # 无有效区域，返回空
    
#     # 2. 计算每个有效像素对应的目标中心点（x+dx, y+dy）
#     candidates = []
#     for x, y in zip(x_coords, y_coords):
#         dx, dy = masked_offset[y, x]  # 偏移值：dx=x方向偏移，dy=y方向偏移
#         centroid_x = x + dx
#         centroid_y = y + dy
#         candidates.append([centroid_x, centroid_y])
#     candidates = np.array(candidates)
    
#     # 3. DBSCAN聚类（过滤离散候选点，聚合同一料箱的中心点）
#     if len(candidates) < min_samples:
#         return candidates, []  # 候选点不足，无法形成聚类
    
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     labels = dbscan.fit_predict(candidates)
    
#     # 4. 计算每个聚类的平均中心（排除噪声点label=-1）
#     cluster_centers = []
#     unique_labels = set(labels) - {-1}
#     for label in unique_labels:
#         cluster_points = candidates[labels == label]
#         avg_x = round(np.mean(cluster_points[:, 0]), 2)
#         avg_y = round(np.mean(cluster_points[:, 1]), 2)
#         cluster_centers.append((avg_x, avg_y))
    
#     return candidates, cluster_centers

# # --------------------------- 核心4：结果可视化与保存（PIL实现） ---------------------------
# def draw_result_visualization(
#     original_mask: np.ndarray,
#     processed_mask: np.ndarray,
#     masked_offset: np.ndarray,
#     candidates: np.ndarray,
#     cluster_centers: List[Tuple[float, float]],
#     img_h: int,
#     img_w: int
# ) -> Image.Image:
#     """
#     绘制后处理结果对比图（4个子图：原始掩膜、处理后掩膜、偏移值幅度、聚类结果）
#     Args:
#         各输入参数同前；img_h/img_w：图像高度/宽度（统一子图尺寸）
#     Returns:
#         result_img: 拼接后的结果可视化图（PIL RGB图像）
#     """
#     # 1. 准备单个子图尺寸（2行2列，总尺寸为2*img_w × 2*img_h）
#     subplot_size = (img_w, img_h)
#     total_size = (2 * img_w, 2 * img_h)
#     result_img = Image.new("RGB", total_size, color=(255, 255, 255))  # 白色背景
#     draw = ImageDraw.Draw(result_img)
    
#     # 2. 加载字体（适配不同环境，无字体则跳过文字标注）
#     try:
#         font = ImageFont.truetype("arial.ttf", 16)
#     except IOError:
#         font = ImageFont.load_default()
    
#     # 3. 子图1：原始掩膜（灰色）
#     original_mask_pil = numpy_to_pil(original_mask, mode="L")
#     original_mask_pil = original_mask_pil.resize(subplot_size)
#     result_img.paste(original_mask_pil, (0, 0))
#     draw.text((10, 10), "Original Mask", fill=(255, 0, 0), font=font)
    
#     # 4. 子图2：处理后掩膜（灰色）
#     processed_mask_pil = numpy_to_pil(processed_mask, mode="L")
#     processed_mask_pil = processed_mask_pil.resize(subplot_size)
#     result_img.paste(processed_mask_pil, (img_w, 0))
#     draw.text((img_w + 10, 10), "Processed Mask", fill=(255, 0, 0), font=font)
    
#     # 5. 子图3：偏移值幅度图（彩色，值越大越亮）
#     offset_magnitude = np.linalg.norm(masked_offset, axis=2)  # 计算偏移幅度（dx²+dy²）^0.5
#     # 幅度值归一化到0-255（便于可视化）
#     if offset_magnitude.max() > 0:
#         offset_magnitude = (offset_magnitude / offset_magnitude.max()) * 255
#     offset_magnitude_pil = numpy_to_pil(offset_magnitude.astype(np.uint8), mode="L")
#     offset_magnitude_pil = offset_magnitude_pil.resize(subplot_size)
#     # 转为伪彩色（用PIL的调色板）
#     offset_magnitude_pil = offset_magnitude_pil.convert("P")
#     offset_magnitude_pil.putpalette([
#         0, 0, 0,    # 0=黑色（无偏移）
#         0, 0, 255,  # 低幅度=蓝色
#         0, 255, 255,# 中幅度=青色
#         255, 255, 0,# 高幅度=黄色
#         255, 0, 0   # 最高幅度=红色
#     ])
#     offset_magnitude_pil = offset_magnitude_pil.convert("RGB")
#     result_img.paste(offset_magnitude_pil, (0, img_h))
#     draw.text((10, img_h + 10), "Offset Magnitude", fill=(255, 0, 0), font=font)
    
#     # 6. 子图4：聚类结果（处理后掩膜+候选点+聚类中心）
#     cluster_bg_pil = processed_mask_pil.convert("RGB")  # 灰色背景转为RGB
#     cluster_draw = ImageDraw.Draw(cluster_bg_pil)
#     # 绘制候选点（蓝色小点，透明度通过颜色叠加实现）
#     if len(candidates) > 0:
#         # 候选点坐标缩放到子图尺寸
#         scale_x = img_w / original_mask.shape[1]
#         scale_y = img_h / original_mask.shape[0]
#         scaled_candidates = candidates * [scale_x, scale_y]
#         for (x, y) in scaled_candidates:
#             cluster_draw.ellipse(
#                 (x-1, y-1, x+1, y+1),  # 2x2像素小圆
#                 fill=(0, 0, 255, 100),  # 蓝色，半透明
#                 outline=None
#             )
#     # 绘制聚类中心（红色叉号，尺寸10x10）
#     for i, (x, y) in enumerate(cluster_centers):
#         scaled_x = x * scale_x
#         scaled_y = y * scale_y
#         # 叉号：左上→右下，右上→左下
#         cluster_draw.line(
#             (scaled_x-5, scaled_y-5, scaled_x+5, scaled_y+5),
#             fill=(255, 0, 0), width=2
#         )
#         cluster_draw.line(
#             (scaled_x+5, scaled_y-5, scaled_x-5, scaled_y+5),
#             fill=(255, 0, 0), width=2
#         )
#         # 标注聚类序号
#         cluster_draw.text(
#             (scaled_x+8, scaled_y-8), str(i+1), fill=(255, 0, 0), font=font
#         )
#     result_img.paste(cluster_bg_pil, (img_w, img_h))
#     draw.text((img_w + 10, img_h + 10), f"Clusters (Total: {len(cluster_centers)})", fill=(255, 0, 0), font=font)
    
#     return result_img

# # --------------------------- 主函数：一键调用后处理流程 ---------------------------
# def model_output_post_process(
#     model_output: Dict[str, np.ndarray],
#     save_dir: str = "./post_process_results",
#     min_area: int = 500,
#     kernel_size: int = 3,
#     eps: float = 50.0,
#     min_samples: int = 10,
#     save_visualization: bool = True,
#     save_masks: bool = True
# ) -> Dict[str, any]:
#     """
#     衔接模型输出的完整后处理流程：输入模型输出→后处理→保存结果→返回关键信息
#     Args:
#         model_output: 模型输出字典，需包含：
#             - "mask": 原始分割掩膜（H,W），值0/1
#             - "offset_map": 偏移值图（H,W,2），通道0=dx，通道1=dy
#         save_dir: 结果保存目录（自动创建）
#         min_area/kernel_size/eps/min_samples: 后处理参数（同前）
#         save_visualization: 是否保存结果可视化图
#         save_masks: 是否保存处理后的掩膜
#     Returns:
#         post_process_result: 后处理结果字典，包含：
#             - "processed_mask": 处理后掩膜（np.ndarray）
#             - "masked_offset": 过滤后偏移值图（np.ndarray）
#             - "cluster_centers": 聚类中心列表（List[(x,y)]）
#             - "candidates": 候选中心点（np.ndarray）
#             - "save_paths": 保存文件路径（可视化图、处理后掩膜）
#     """
#     # 1. 解析模型输出（确保数据格式正确）
#     original_mask = model_output["mask"].squeeze()  # 确保为2D（H,W）
#     offset_map = model_output["offset_map"].squeeze()  # 确保为3D（H,W,2）
#     img_h, img_w = original_mask.shape[0], original_mask.shape[1]
    
#     # 2. 创建保存目录
#     os.makedirs(save_dir, exist_ok=True)
#     save_paths = {}
    
#     # 3. 执行后处理核心步骤
#     processed_mask = post_process_mask(original_mask, min_area, kernel_size)
#     masked_offset = mask_offset_values(offset_map, processed_mask)
#     candidates, cluster_centers = extract_and_cluster_centroids(
#         processed_mask, masked_offset, eps, min_samples
#     )
    
#     # 4. 保存处理后的掩膜（若开启）
#     if save_masks:
#         processed_mask_pil = numpy_to_pil(processed_mask, mode="L")
#         mask_save_path = os.path.join(save_dir, "processed_mask.png")
#         processed_mask_pil.save(mask_save_path)
#         save_paths["processed_mask"] = mask_save_path
    
#     # 5. 生成并保存可视化结果（若开启）
#     if save_visualization:
#         result_visual_img = draw_result_visualization(
#             original_mask, processed_mask, masked_offset,
#             candidates, cluster_centers, img_h, img_w
#         )
#         vis_save_path = os.path.join(save_dir, "post_process_visualization.png")
#         result_visual_img.save(vis_save_path, quality=95)
#         save_paths["visualization"] = vis_save_path
    
#     # 6. 整理返回结果
#     post_process_result = {
#         "processed_mask": processed_mask,
#         "masked_offset": masked_offset,
#         "cluster_centers": cluster_centers,
#         "candidates": candidates,
#         "save_paths": save_paths,
#         "num_clusters": len(cluster_centers)  # 方便后续统计
#     }
    
#     # 打印关键信息（便于日志跟踪）
#     print(f"后处理完成：检测到{len(cluster_centers)}个料箱聚类中心")

# import numpy as np
# from sklearn.cluster import DBSCAN
# from scipy.ndimage import binary_opening, binary_closing, binary_dilation
# from scipy import ndimage
# from skimage.measure import label, regionprops
# from PIL import Image, ImageDraw, ImageFont
# import os
# from typing import Tuple, Dict, List

# # --------------------------- 工具函数：统一数据格式转换 ---------------------------
# def numpy_to_pil(np_img: np.ndarray, mode: str = "L") -> Image.Image:
#     """
#     将numpy数组转为PIL图像（适配二值掩膜、RGB图等）
#     Args:
#         np_img: 输入numpy数组（H,W）或（H,W,3）
#         mode: PIL图像模式（L=灰度/二值，RGB=彩色）
#     Returns:
#         pil_img: PIL图像对象
#     """
#     # 二值图（0/1）转为0-255范围
#     if mode == "L" and np_img.dtype in [np.uint8, np.bool_]:
#         np_img = (np_img * 255).astype(np.uint8)
#     # RGB图确保通道顺序正确（numpy默认HWC，PIL默认RGB）
#     if mode == "RGB" and np_img.shape[-1] == 3:
#         np_img = np.clip(np_img, 0, 255).astype(np.uint8)
#     return Image.fromarray(np_img, mode=mode)

# def pil_to_numpy(pil_img: Image.Image) -> np.ndarray:
#     """PIL图像转为numpy数组（适配后续计算）"""
#     return np.array(pil_img)

# # --------------------------- 核心1：分割掩膜后处理（无cv2版本） ---------------------------
# def post_process_mask(
#     mask: np.ndarray, 
#     min_area: int = 500, 
#     kernel_size: int = 3
# ) -> np.ndarray:
#     """
#     去除掩膜噪声与小块区域，输出处理后的二值掩膜
#     Args:
#         mask: 原始二值掩膜（H,W），值为0或1
#         min_area: 最小保留区域面积（像素），拆垛场景建议300-1000
#         kernel_size: 形态学操作核大小（3或5，噪声多则增大）
#     Returns:
#         processed_mask: 处理后二值掩膜（H,W），值为0或1
#     """
#     # 1. 形态学开运算：先腐蚀再膨胀，去除小噪声（如孤立像素）
#     kernel = np.ones((kernel_size, kernel_size), np.uint8)
#     mask_open = binary_opening(mask, structure=kernel).astype(np.uint8)
    
#     # 2. 形态学闭运算：先膨胀再腐蚀，填充掩膜内部小空洞
#     mask_close = binary_closing(mask_open, structure=kernel).astype(np.uint8)
    
#     # 输入验证
#     if not isinstance(mask_close, np.ndarray):
#         raise ValueError("输入的掩码不是numpy数组")

#     if mask_close is None or mask_close.size == 0:
#         raise ValueError("输入的掩码为空")

#     # 如果是多通道图像，转换为单通道
#     if len(mask_close.shape) > 2 and mask_close.shape[2] > 1:
#         mask_close = np.mean(mask_close, axis=2).astype(np.uint8)

#     # 3. 连通区域分析：使用skimage替代cv2.connectedComponentsWithStats
#     labeled_mask = label(mask_close, connectivity=2)  # 2表示8连通（skimage中2=8连通）
#     regions = regionprops(labeled_mask)
    
#     large_area_mask = np.zeros_like(mask_close)
#     for region in regions:
#         if region.area >= min_area:
#             # 获取该区域的坐标并设置为1
#             coords = region.coords
#             large_area_mask[coords[:, 0], coords[:, 1]] = 1
    
#     # 4. 轻微膨胀：恢复开运算导致的边缘收缩（避免掩膜边缘损失）
#     processed_mask = binary_dilation(large_area_mask, structure=kernel).astype(np.uint8)
    
#     return processed_mask

# # --------------------------- 核心2：偏移值图掩膜过滤 ---------------------------
# def mask_offset_values(
#     offset_map: np.ndarray, 
#     processed_mask: np.ndarray
# ) -> np.ndarray:
#     """
#     将掩膜外的偏移值置0（仅保留有效区域的偏移信息）
#     Args:
#         offset_map: 模型输出偏移值图（H,W,2），通道0=dx，通道1=dy
#         processed_mask: 后处理后的二值掩膜（H,W）
#     Returns:
#         masked_offset: 过滤后的偏移值图（H,W,2）
#     """
#     # 扩展掩膜维度至3D（H,W）→（H,W,1），与偏移图通道匹配
#     mask_3d = np.expand_dims(processed_mask, axis=-1)
#     # 掩膜外区域偏移值置0
#     masked_offset = offset_map * mask_3d
#     return masked_offset

# # --------------------------- 核心3：候选点提取与DBSCAN聚类 ---------------------------
# def extract_and_cluster_centroids(
#     processed_mask: np.ndarray, 
#     masked_offset: np.ndarray, 
#     eps: float = 50.0, 
#     min_samples: int = 10
# ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
#     """
#     从有效区域提取候选中心点，用DBSCAN聚类得到最终目标中心
#     Args:
#         processed_mask: 后处理掩膜（H,W）
#         masked_offset: 过滤后的偏移值图（H,W,2）
#         eps: DBSCAN聚类半径（像素），根据料箱间距调整（40-80）
#         min_samples: 形成聚类的最小样本数（8-15，避免噪声聚类）
#     Returns:
#         candidates: 所有候选中心点（N,2），格式(x,y)
#         cluster_centers: 最终聚类中心列表，每个元素为(x,y)，保留2位小数
#     """
#     # 1. 提取掩膜内所有像素坐标（y为行号，x为列号）
#     y_coords, x_coords = np.where(processed_mask == 1)
#     if len(y_coords) == 0:
#         return np.array([]), []  # 无有效区域，返回空
    
#     # 2. 计算每个有效像素对应的目标中心点（x+dx, y+dy）
#     candidates = []
#     for x, y in zip(x_coords, y_coords):
#         dx, dy = masked_offset[y, x]  # 偏移值：dx=x方向偏移，dy=y方向偏移
#         centroid_x = x + dx
#         centroid_y = y + dy
#         candidates.append([centroid_x, centroid_y])
#     candidates = np.array(candidates)
    
#     # 3. DBSCAN聚类（过滤离散候选点，聚合同一料箱的中心点）
#     if len(candidates) < min_samples:
#         return candidates, []  # 候选点不足，无法形成聚类
    
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     labels = dbscan.fit_predict(candidates)
    
#     # 4. 计算每个聚类的平均中心（排除噪声点label=-1）
#     cluster_centers = []
#     unique_labels = set(labels) - {-1}
#     for label in unique_labels:
#         cluster_points = candidates[labels == label]
#         avg_x = round(np.mean(cluster_points[:, 0]), 2)
#         avg_y = round(np.mean(cluster_points[:, 1]), 2)
#         cluster_centers.append((avg_x, avg_y))
    
#     return candidates, cluster_centers

# # --------------------------- 核心4：结果可视化与保存（PIL实现） ---------------------------
# def draw_result_visualization(
#     original_mask: np.ndarray,
#     processed_mask: np.ndarray,
#     masked_offset: np.ndarray,
#     candidates: np.ndarray,
#     cluster_centers: List[Tuple[float, float]],
#     img_h: int,
#     img_w: int
# ) -> Image.Image:
#     """
#     绘制后处理结果对比图（4个子图：原始掩膜、处理后掩膜、偏移值幅度、聚类结果）
#     Args:
#         各输入参数同前；img_h/img_w：图像高度/宽度（统一子图尺寸）
#     Returns:
#         result_img: 拼接后的结果可视化图（PIL RGB图像）
#     """
#     # 1. 准备单个子图尺寸（2行2列，总尺寸为2*img_w × 2*img_h）
#     subplot_size = (img_w, img_h)
#     total_size = (2 * img_w, 2 * img_h)
#     result_img = Image.new("RGB", total_size, color=(255, 255, 255))  # 白色背景
#     draw = ImageDraw.Draw(result_img)
    
#     # 2. 加载字体（适配不同环境，无字体则跳过文字标注）
#     try:
#         font = ImageFont.truetype("arial.ttf", 16)
#     except IOError:
#         font = ImageFont.load_default()
    
#     # 3. 子图1：原始掩膜（灰色）
#     original_mask_pil = numpy_to_pil(original_mask, mode="L")
#     original_mask_pil = original_mask_pil.resize(subplot_size)
#     result_img.paste(original_mask_pil, (0, 0))
#     draw.text((10, 10), "Original Mask", fill=(255, 0, 0), font=font)
    
#     # 4. 子图2：处理后掩膜（灰色）
#     processed_mask_pil = numpy_to_pil(processed_mask, mode="L")
#     processed_mask_pil = processed_mask_pil.resize(subplot_size)
#     result_img.paste(processed_mask_pil, (img_w, 0))
#     draw.text((img_w + 10, 10), "Processed Mask", fill=(255, 0, 0), font=font)
    
#     # 5. 子图3：偏移值幅度图（彩色，值越大越亮）
#     offset_magnitude = np.linalg.norm(masked_offset, axis=2)  # 计算偏移幅度（dx²+dy²）^0.5
#     # 幅度值归一化到0-255（便于可视化）
#     if offset_magnitude.max() > 0:
#         offset_magnitude = (offset_magnitude / offset_magnitude.max()) * 255
#     offset_magnitude_pil = numpy_to_pil(offset_magnitude.astype(np.uint8), mode="L")
#     offset_magnitude_pil = offset_magnitude_pil.resize(subplot_size)
#     # 转为伪彩色（用PIL的调色板）
#     offset_magnitude_pil = offset_magnitude_pil.convert("P")
#     offset_magnitude_pil.putpalette([
#         0, 0, 0,    # 0=黑色（无偏移）
#         0, 0, 255,  # 低幅度=蓝色
#         0, 255, 255,# 中幅度=青色
#         255, 255, 0,# 高幅度=黄色
#         255, 0, 0   # 最高幅度=红色
#     ])
#     offset_magnitude_pil = offset_magnitude_pil.convert("RGB")
#     result_img.paste(offset_magnitude_pil, (0, img_h))
#     draw.text((10, img_h + 10), "Offset Magnitude", fill=(255, 0, 0), font=font)
    
#     # 6. 子图4：聚类结果（处理后掩膜+候选点+聚类中心）
#     cluster_bg_pil = processed_mask_pil.convert("RGB")  # 灰色背景转为RGB
#     cluster_draw = ImageDraw.Draw(cluster_bg_pil)
#     # 绘制候选点（蓝色小点，透明度通过颜色叠加实现）
#     if len(candidates) > 0:
#         # 候选点坐标缩放到子图尺寸
#         scale_x = img_w / original_mask.shape[1]
#         scale_y = img_h / original_mask.shape[0]
#         scaled_candidates = candidates * [scale_x, scale_y]
#         for (x, y) in scaled_candidates:
#             cluster_draw.ellipse(
#                 (x-1, y-1, x+1, y+1),  # 2x2像素小圆
#                 fill=(0, 0, 255),  # 蓝色
#                 outline=None
#             )
#     # 绘制聚类中心（红色叉号，尺寸10x10）
#     for i, (x, y) in enumerate(cluster_centers):
#         scaled_x = x * scale_x
#         scaled_y = y * scale_y
#         # 叉号：左上→右下，右上→左下
#         cluster_draw.line(
#             (scaled_x-5, scaled_y-5, scaled_x+5, scaled_y+5),
#             fill=(255, 0, 0), width=2
#         )
#         cluster_draw.line(
#             (scaled_x+5, scaled_y-5, scaled_x-5, scaled_y+5),
#             fill=(255, 0, 0), width=2
#         )
#         # 标注聚类序号
#         cluster_draw.text(
#             (scaled_x+8, scaled_y-8), str(i+1), fill=(255, 0, 0), font=font
#         )
#     result_img.paste(cluster_bg_pil, (img_w, img_h))
#     draw.text((img_w + 10, img_h + 10), f"Clusters (Total: {len(cluster_centers)})", fill=(255, 0, 0), font=font)
    
#     return result_img

# # --------------------------- 主函数：一键调用后处理流程 ---------------------------
# def model_output_post_process(
#     model_output: Dict[str, np.ndarray],
#     save_dir: str = "./post_process_results",
#     min_area: int = 500,
#     kernel_size: int = 3,
#     eps: float = 50.0,
#     min_samples: int = 10,
#     save_visualization: bool = True,
#     save_masks: bool = True
# ) -> Dict[str, any]:
#     """
#     衔接模型输出的完整后处理流程：输入模型输出→后处理→保存结果→返回关键信息
#     Args:
#         model_output: 模型输出字典，需包含：
#             - "mask": 原始分割掩膜（H,W），值0/1
#             - "offset_map": 偏移值图（H,W,2），通道0=dx，通道1=dy
#         save_dir: 结果保存目录（自动创建）
#         min_area/kernel_size/eps/min_samples: 后处理参数（同前）
#         save_visualization: 是否保存结果可视化图
#         save_masks: 是否保存处理后的掩膜
#     Returns:
#         post_process_result: 后处理结果字典，包含：
#             - "processed_mask": 处理后掩膜（np.ndarray）
#             - "masked_offset": 过滤后偏移值图（np.ndarray）
#             - "cluster_centers": 聚类中心列表（List[(x,y)]）
#             - "candidates": 候选中心点（np.ndarray）
#             - "save_paths": 保存文件路径（可视化图、处理后掩膜）
#     """
#     # 1. 解析模型输出（确保数据格式正确）
#     original_mask = model_output["mask"].squeeze()  # 确保为2D（H,W）
#     offset_map = model_output["offset_map"].squeeze()  # 确保为3D（H,W,2）
#     img_h, img_w = original_mask.shape[0], original_mask.shape[1]
    
#     # 2. 创建保存目录
#     os.makedirs(save_dir, exist_ok=True)
#     save_paths = {}
    
#     # 3. 执行后处理核心步骤
#     processed_mask = post_process_mask(original_mask, min_area, kernel_size)
#     masked_offset = mask_offset_values(offset_map, processed_mask)
#     candidates, cluster_centers = extract_and_cluster_centroids(
#         processed_mask, masked_offset, eps, min_samples
#     )
    
#     # 4. 保存处理后的掩膜（若开启）
#     if save_masks:
#         processed_mask_pil = numpy_to_pil(processed_mask, mode="L")
#         mask_save_path = os.path.join(save_dir, "processed_mask.png")
#         processed_mask_pil.save(mask_save_path)
#         save_paths["processed_mask"] = mask_save_path
    
#     # 5. 生成并保存可视化结果（若开启）
#     if save_visualization:
#         result_visual_img = draw_result_visualization(
#             original_mask, processed_mask, masked_offset,
#             candidates, cluster_centers, img_h, img_w
#         )
#         vis_save_path = os.path.join(save_dir, "post_process_visualization.png")
#         result_visual_img.save(vis_save_path, quality=95)
#         save_paths["visualization"] = vis_save_path
    
#     # 6. 整理返回结果
#     post_process_result = {
#         "processed_mask": processed_mask,
#         "masked_offset": masked_offset,
#         "cluster_centers": cluster_centers,
#         "candidates": candidates,
#         "save_paths": save_paths,
#         "num_clusters": len(cluster_centers)  # 方便后续统计
#     }
    
#     # 打印关键信息（便于日志跟踪）
#     print(f"后处理完成：检测到{len(cluster_centers)}个料箱聚类中心")
#     return post_process_result


# import numpy as np
# from PIL import Image, ImageDraw, ImageFont
# import os
# from typing import Tuple, Dict, List

# # --------------------------- 工具函数：统一数据格式转换 ---------------------------
# def numpy_to_pil(np_img: np.ndarray, mode: str = "L") -> Image.Image:
#     """
#     将numpy数组转为PIL图像（适配二值掩膜、RGB图等）
#     """
#     if mode == "L" and np_img.dtype in [np.uint8, bool]:
#         np_img = (np_img * 255).astype(np.uint8)
#     if mode == "RGB" and np_img.shape[-1] == 3:
#         np_img = np.clip(np_img, 0, 255).astype(np.uint8)
#     return Image.fromarray(np_img, mode=mode)

# def pil_to_numpy(pil_img: Image.Image) -> np.ndarray:
#     return np.array(pil_img)

# # --------------------------- 形态学操作（纯numpy实现） ---------------------------
# def binary_erosion(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
#     """二值图像腐蚀操作"""
#     H, W = mask.shape
#     k = kernel_size // 2
#     eroded = np.zeros_like(mask)
    
#     for i in range(H):
#         for j in range(W):
#             if mask[i, j] == 1:
#                 # 检查kernel范围内的所有像素是否都为1
#                 min_i, max_i = max(0, i-k), min(H, i+k+1)
#                 min_j, max_j = max(0, j-k), min(W, j+k+1)
                
#                 # 如果邻域内所有像素都是1，则保留该点
#                 if np.all(mask[min_i:max_i, min_j:max_j] == 1):
#                     eroded[i, j] = 1
                    
#     return eroded

# def binary_dilation(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
#     """二值图像膨胀操作"""
#     H, W = mask.shape
#     k = kernel_size // 2
#     dilated = np.zeros_like(mask)
    
#     for i in range(H):
#         for j in range(W):
#             if mask[i, j] == 1:
#                 # 将kernel范围内的所有像素设为1
#                 min_i, max_i = max(0, i-k), min(H, i+k+1)
#                 min_j, max_j = max(0, j-k), min(W, j+k+1)
#                 dilated[min_i:max_i, min_j:max_j] = 1
                
#     return dilated

# def binary_opening(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
#     """开运算：先腐蚀后膨胀"""
#     eroded = binary_erosion(mask, kernel_size)
#     opened = binary_dilation(eroded, kernel_size)
#     return opened

# def binary_closing(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
#     """闭运算：先膨胀后腐蚀"""
#     dilated = binary_dilation(mask, kernel_size)
#     closed = binary_erosion(dilated, kernel_size)
#     return closed

# # --------------------------- 连通区域分析（纯numpy实现） ---------------------------
# def connected_components_analysis(mask: np.ndarray, min_area: int = 500) -> np.ndarray:
#     """
#     使用两遍扫描算法实现连通区域分析（4连通）
#     """
#     H, W = mask.shape
#     labeled = np.zeros((H, W), dtype=int)
#     current_label = 1
#     equivalences = {}
    
#     # 第一遍扫描：标记连通区域
#     for i in range(H):
#         for j in range(W):
#             if mask[i, j] == 1:
#                 neighbors = []
#                 # 检查上方邻居
#                 if i > 0 and labeled[i-1, j] > 0:
#                     neighbors.append(labeled[i-1, j])
#                 # 检查左方邻居
#                 if j > 0 and labeled[i, j-1] > 0:
#                     neighbors.append(labeled[i, j-1])
                
#                 if not neighbors:
#                     # 没有邻居，创建新标签
#                     labeled[i, j] = current_label
#                     current_label += 1
#                 else:
#                     # 使用最小的邻居标签
#                     min_label = min(neighbors)
#                     labeled[i, j] = min_label
#                     # 记录等价关系
#                     for neighbor in neighbors:
#                         if neighbor != min_label:
#                             equivalences[neighbor] = min_label
    
#     # 解析等价关系，统一标签
#     for label_id in range(1, current_label):
#         current = label_id
#         while current in equivalences:
#             current = equivalences[current]
#         equivalences[label_id] = current
    
#     # 第二遍扫描：统一标签并计算区域面积
#     final_labels = np.zeros_like(labeled)
#     area_dict = {}
    
#     for i in range(H):
#         for j in range(W):
#             if labeled[i, j] > 0:
#                 final_label = equivalences[labeled[i, j]]
#                 final_labels[i, j] = final_label
#                 area_dict[final_label] = area_dict.get(final_label, 0) + 1
    
#     # 过滤小面积区域
#     filtered_mask = np.zeros_like(mask)
#     for label_id, area in area_dict.items():
#         if area >= min_area:
#             filtered_mask[final_labels == label_id] = 1
    
#     return filtered_mask

# # --------------------------- 核心1：分割掩膜后处理 ---------------------------
# def post_process_mask(
#     mask: np.ndarray, 
#     min_area: int = 500, 
#     kernel_size: int = 3
# ) -> np.ndarray:
#     """
#     去除掩膜噪声与小块区域，输出处理后的二值掩膜
#     """
#     # 输入验证和处理
#     if not isinstance(mask, np.ndarray):
#         raise ValueError("输入的掩码不是numpy数组")
#     if mask is None or mask.size == 0:
#         raise ValueError("输入的掩码为空")
    
#     # 多通道转单通道
#     if len(mask.shape) > 2 and mask.shape[2] > 1:
#         mask = np.mean(mask, axis=2)
    
#     # 确保是二值图像
#     mask = (mask > 0.5).astype(np.uint8)
    
#     # 1. 形态学开运算：先腐蚀再膨胀，去除小噪声
#     print("形态学开运算")
#     mask_open = binary_opening(mask, kernel_size)
    
#     # 2. 形态学闭运算：先膨胀后腐蚀，填充内部小空洞
#     mask_close = binary_closing(mask_open, kernel_size)
#     print("形态学闭运算")
#     # 3. 连通区域分析：过滤面积小于min_area的区域
#     processed_mask = connected_components_analysis(mask_close, min_area)
#     print("连通区域分析")
#     # 4. 轻微膨胀：恢复开运算导致的边缘收缩
#     processed_mask = binary_dilation(processed_mask, kernel_size)
#     print("轻微膨胀")
#     return processed_mask

# # --------------------------- 核心2：偏移值图掩膜过滤 ---------------------------
# def mask_offset_values(offset_map: np.ndarray, processed_mask: np.ndarray) -> np.ndarray:
#     """将掩膜外的偏移值置0"""
#     mask_3d = np.expand_dims(processed_mask, axis=-1)
#     masked_offset = offset_map * mask_3d
#     return masked_offset

# # --------------------------- 核心3：候选点提取与聚类 ---------------------------
# def extract_and_cluster_centroids(
#     processed_mask: np.ndarray, 
#     masked_offset: np.ndarray, 
#     eps: float = 50.0, 
#     min_samples: int = 10
# ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
#     """
#     从有效区域提取候选中心点，用简单聚类得到最终目标中心
#     """
#     # 1. 提取掩膜内所有像素坐标
#     y_coords, x_coords = np.where(processed_mask == 1)
#     if len(y_coords) == 0:
#         return np.array([]), []
    
#     # 2. 计算每个有效像素对应的目标中心点
#     candidates = []
#     for x, y in zip(x_coords, y_coords):
#         dx, dy = masked_offset[y, x]
#         centroid_x = x + dx
#         centroid_y = y + dy
#         candidates.append([centroid_x, centroid_y])
#     candidates = np.array(candidates)
    
#     # 3. 简单聚类算法
#     cluster_centers = simple_clustering(candidates, eps, min_samples)
    
#     return candidates, cluster_centers

# def simple_clustering(points: np.ndarray, cluster_radius: float, min_points: int) -> List[Tuple[float, float]]:
#     """简单的基于距离的聚类算法"""
#     if len(points) == 0:
#         return []
    
#     clusters = []
#     visited = set()
    
#     for i, point in enumerate(points):
#         if i in visited:
#             continue
            
#         # 找到半径内的所有点
#         distances = np.linalg.norm(points - point, axis=1)
#         neighbor_indices = np.where(distances < cluster_radius)[0]
        
#         if len(neighbor_indices) >= min_points:
#             # 形成一个聚类
#             cluster_points = points[neighbor_indices]
#             avg_x = round(np.mean(cluster_points[:, 0]), 2)
#             avg_y = round(np.mean(cluster_points[:, 1]), 2)
#             clusters.append((avg_x, avg_y))
            
#             # 标记为已访问
#             visited.update(neighbor_indices)
    
#     return clusters

# # --------------------------- 核心4：结果可视化与保存 ---------------------------
# def draw_result_visualization(
#     original_mask: np.ndarray,
#     processed_mask: np.ndarray,
#     masked_offset: np.ndarray,
#     candidates: np.ndarray,
#     cluster_centers: List[Tuple[float, float]],
#     img_h: int,
#     img_w: int
# ) -> Image.Image:
#     """绘制后处理结果对比图"""
#     subplot_size = (img_w, img_h)
#     total_size = (2 * img_w, 2 * img_h)
#     result_img = Image.new("RGB", total_size, color=(255, 255, 255))
#     draw = ImageDraw.Draw(result_img)
    
#     try:
#         font = ImageFont.truetype("arial.ttf", 16)
#     except IOError:
#         font = ImageFont.load_default()
    
#     # 子图1：原始掩膜
#     original_mask_pil = numpy_to_pil(original_mask, mode="L")
#     original_mask_pil = original_mask_pil.resize(subplot_size)
#     result_img.paste(original_mask_pil, (0, 0))
#     draw.text((10, 10), "Original Mask", fill=(255, 0, 0), font=font)
    
#     # 子图2：处理后掩膜
#     processed_mask_pil = numpy_to_pil(processed_mask, mode="L")
#     processed_mask_pil = processed_mask_pil.resize(subplot_size)
#     result_img.paste(processed_mask_pil, (img_w, 0))
#     draw.text((img_w + 10, 10), "Processed Mask", fill=(255, 0, 0), font=font)
    
#     # 子图3：偏移值幅度图
#     offset_magnitude = np.linalg.norm(masked_offset, axis=2)
#     if offset_magnitude.max() > 0:
#         offset_magnitude = (offset_magnitude / offset_magnitude.max()) * 255
#     offset_magnitude_pil = numpy_to_pil(offset_magnitude.astype(np.uint8), mode="L")
#     offset_magnitude_pil = offset_magnitude_pil.resize(subplot_size)
#     offset_magnitude_pil = offset_magnitude_pil.convert("RGB")
#     result_img.paste(offset_magnitude_pil, (0, img_h))
#     draw.text((10, img_h + 10), "Offset Magnitude", fill=(255, 0, 0), font=font)
    
#     # 子图4：聚类结果
#     cluster_bg_pil = processed_mask_pil.convert("RGB")
#     cluster_draw = ImageDraw.Draw(cluster_bg_pil)
    
#     if len(candidates) > 0:
#         scale_x = img_w / original_mask.shape[1]
#         scale_y = img_h / original_mask.shape[0]
#         scaled_candidates = candidates * [scale_x, scale_y]
#         for (x, y) in scaled_candidates:
#             cluster_draw.ellipse((x-1, y-1, x+1, y+1), fill=(0, 0, 255), outline=None)
    
#     for i, (x, y) in enumerate(cluster_centers):
#         scaled_x = x * scale_x
#         scaled_y = y * scale_y
#         cluster_draw.line((scaled_x-5, scaled_y-5, scaled_x+5, scaled_y+5), fill=(255, 0, 0), width=2)
#         cluster_draw.line((scaled_x+5, scaled_y-5, scaled_x-5, scaled_y+5), fill=(255, 0, 0), width=2)
#         cluster_draw.text((scaled_x+8, scaled_y-8), str(i+1), fill=(255, 0, 0), font=font)
    
#     result_img.paste(cluster_bg_pil, (img_w, img_h))
#     draw.text((img_w + 10, img_h + 10), f"Clusters: {len(cluster_centers)}", fill=(255, 0, 0), font=font)
    
#     return result_img

# # --------------------------- 主函数 ---------------------------
# def model_output_post_process(
#     model_output: Dict[str, np.ndarray],
#     save_dir: str = "./post_process_results",
#     min_area: int = 500,
#     kernel_size: int = 3,
#     eps: float = 50.0,
#     min_samples: int = 10,
#     save_visualization: bool = True,
#     save_masks: bool = True
# ) -> Dict[str, any]:
#     """衔接模型输出的完整后处理流程"""
#     original_mask = model_output["mask"].squeeze()
#     offset_map = model_output["offset_map"].squeeze()
#     img_h, img_w = original_mask.shape[0], original_mask.shape[1]
    
#     os.makedirs(save_dir, exist_ok=True)
#     save_paths = {}
    
#     processed_mask = post_process_mask(original_mask, min_area, kernel_size)
#     masked_offset = mask_offset_values(offset_map, processed_mask)
#     candidates, cluster_centers = extract_and_cluster_centroids(
#         processed_mask, masked_offset, eps, min_samples
#     )
    
#     if save_masks:
#         processed_mask_pil = numpy_to_pil(processed_mask, mode="L")
#         mask_save_path = os.path.join(save_dir, "processed_mask.png")
#         processed_mask_pil.save(mask_save_path)
#         save_paths["processed_mask"] = mask_save_path
    
#     if save_visualization:
#         result_visual_img = draw_result_visualization(
#             original_mask, processed_mask, masked_offset,
#             candidates, cluster_centers, img_h, img_w
#         )
#         vis_save_path = os.path.join(save_dir, "post_process_visualization.png")
#         result_visual_img.save(vis_save_path, quality=95)
#         save_paths["visualization"] = vis_save_path
    
#     post_process_result = {
#         "processed_mask": processed_mask,
#         "masked_offset": masked_offset,
#         "cluster_centers": cluster_centers,
#         "candidates": candidates,
#         "save_paths": save_paths,
#         "num_clusters": len(cluster_centers)
#     }
    
#     print(f"后处理完成：检测到{len(cluster_centers)}个料箱聚类中心")
#     return post_process_result


import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from typing import Tuple, Dict, List

# --------------------------- 工具函数：统一数据格式转换 ---------------------------
def numpy_to_pil(np_img: np.ndarray, mode: str = "L") -> Image.Image:
    if mode == "L" and np_img.dtype in [np.uint8, bool]:
        np_img = (np_img * 255).astype(np.uint8)
    if mode == "RGB" and np_img.shape[-1] == 3:
        np_img = np.clip(np_img, 0, 255).astype(np.uint8)
    return Image.fromarray(np_img, mode=mode)

def pil_to_numpy(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img)

# --------------------------- 优化的形态学操作 ---------------------------
def binary_erosion_fast(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """使用卷积加速的腐蚀操作"""
    H, W = mask.shape
    k = kernel_size // 2
    pad_mask = np.pad(mask, k, mode='constant', constant_values=0)
    
    # 创建滑动窗口视图
    strides = pad_mask.strides * 2
    shape = (H, W, kernel_size, kernel_size)
    windows = np.lib.stride_tricks.as_strided(pad_mask, shape, strides)
    
    # 检查每个窗口是否全为1
    eroded = np.all(windows == 1, axis=(2, 3)).astype(np.uint8)
    return eroded

def binary_dilation_fast(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """使用卷积加速的膨胀操作"""
    H, W = mask.shape
    k = kernel_size // 2
    pad_mask = np.pad(mask, k, mode='constant', constant_values=0)
    
    # 创建滑动窗口视图
    strides = pad_mask.strides * 2
    shape = (H, W, kernel_size, kernel_size)
    windows = np.lib.stride_tricks.as_strided(pad_mask, shape, strides)
    
    # 检查每个窗口是否有至少一个1
    dilated = np.any(windows == 1, axis=(2, 3)).astype(np.uint8)
    return dilated

def binary_opening_fast(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """优化的开运算"""
    eroded = binary_erosion_fast(mask, kernel_size)
    opened = binary_dilation_fast(eroded, kernel_size)
    return opened

def binary_closing_fast(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """优化的闭运算"""
    dilated = binary_dilation_fast(mask, kernel_size)
    closed = binary_erosion_fast(dilated, kernel_size)
    return closed

# --------------------------- 优化的连通区域分析 ---------------------------
def connected_components_fast(mask: np.ndarray, min_area: int = 500) -> np.ndarray:
    """
    使用并查集(Union-Find)算法优化连通区域分析
    """
    H, W = mask.shape
    labeled = np.zeros((H, W), dtype=int)
    parent = {}  # 并查集父节点字典
    area = {}    # 区域面积统计
    
    def find(x):
        """查找根节点（路径压缩）"""
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        """合并两个集合"""
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            # 按秩合并（总是将小树合并到大树）
            if area[root_x] < area[root_y]:
                parent[root_x] = root_y
                area[root_y] += area[root_x]
            else:
                parent[root_y] = root_x
                area[root_x] += area[root_y]
    
    current_label = 1
    
    # 第一遍扫描
    for i in range(H):
        for j in range(W):
            if mask[i, j] == 1:
                neighbors = []
                # 检查4连通邻居
                if i > 0 and labeled[i-1, j] > 0:
                    neighbors.append(labeled[i-1, j])
                if j > 0 and labeled[i, j-1] > 0:
                    neighbors.append(labeled[i, j-1])
                
                if not neighbors:
                    # 新区域
                    labeled[i, j] = current_label
                    parent[current_label] = current_label
                    area[current_label] = 1
                    current_label += 1
                else:
                    # 连接到现有区域
                    min_label = min(neighbors)
                    labeled[i, j] = min_label
                    area[min_label] += 1
                    
                    # 合并所有相连的区域
                    for neighbor in neighbors:
                        if neighbor != min_label:
                            union(min_label, neighbor)
    
    # 第二遍扫描：统一标签
    final_labels = np.zeros_like(labeled)
    final_areas = {}
    
    for i in range(H):
        for j in range(W):
            if labeled[i, j] > 0:
                root = find(labeled[i, j])
                final_labels[i, j] = root
                final_areas[root] = final_areas.get(root, 0) + 1
    
    # 过滤小面积区域
    filtered_mask = np.zeros_like(mask)
    for label_id, label_area in final_areas.items():
        if label_area >= min_area:
            filtered_mask[final_labels == label_id] = 1
    
    return filtered_mask

# --------------------------- 更快的连通区域分析（使用BFS） ---------------------------
def connected_components_bfs(mask: np.ndarray, min_area: int = 500) -> np.ndarray:
    """
    使用BFS的连通区域分析，适合稀疏图像
    """
    H, W = mask.shape
    visited = np.zeros((H, W), dtype=bool)
    result = np.zeros_like(mask)
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4连通
    
    for i in range(H):
        for j in range(W):
            if mask[i, j] == 1 and not visited[i, j]:
                # BFS搜索连通区域
                queue = [(i, j)]
                region = []
                visited[i, j] = True
                
                while queue:
                    x, y = queue.pop(0)
                    region.append((x, y))
                    
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < H and 0 <= ny < W and 
                            not visited[nx, ny] and mask[nx, ny] == 1):
                            visited[nx, ny] = True
                            queue.append((nx, ny))
                
                # 如果区域面积足够大，保留
                if len(region) >= min_area:
                    for x, y in region:
                        result[x, y] = 1
    
    return result

# --------------------------- 核心1：分割掩膜后处理 ---------------------------
def post_process_mask(
    mask: np.ndarray, 
    min_area: int = 500, 
    kernel_size: int = 3,
    use_fast_method: bool = True
) -> np.ndarray:
    """
    去除掩膜噪声与小块区域，输出处理后的二值掩膜
    """
    # 输入验证和处理
    if not isinstance(mask, np.ndarray):
        raise ValueError("输入的掩码不是numpy数组")
    if mask is None or mask.size == 0:
        raise ValueError("输入的掩码为空")
    
    # 多通道转单通道
    if len(mask.shape) > 2 and mask.shape[2] > 1:
        mask = np.mean(mask, axis=2)
    
    # 确保是二值图像
    mask = (mask > 0.5).astype(np.uint8)
    
    if use_fast_method:
        # 使用优化版本
        mask_open = binary_opening_fast(mask, kernel_size)
        mask_close = binary_closing_fast(mask_open, kernel_size)
        
        # 根据图像稀疏性选择算法
        density = np.sum(mask_close) / mask_close.size
        if density < 0.3:  # 稀疏图像使用BFS
            processed_mask = connected_components_bfs(mask_close, min_area)
        else:  # 密集图像使用并查集
            processed_mask = connected_components_fast(mask_close, min_area)
            
        processed_mask = binary_dilation_fast(processed_mask, kernel_size)
    else:
        # 备用简单版本（用于调试）
        processed_mask = _simple_post_process(mask, min_area, kernel_size)
    
    return processed_mask

def _simple_post_process(mask: np.ndarray, min_area: int, kernel_size: int) -> np.ndarray:
    """简单的后处理实现（用于调试）"""
    # 简单的形态学操作
    def simple_dilation(mask, k):
        H, W = mask.shape
        result = mask.copy()
        for i in range(H):
            for j in range(W):
                if mask[i, j] == 1:
                    min_i = max(0, i-1); max_i = min(H, i+2)
                    min_j = max(0, j-1); max_j = min(W, j+2)
                    result[min_i:max_i, min_j:max_j] = 1
        return result
    
    def simple_erosion(mask, k):
        H, W = mask.shape
        result = np.zeros_like(mask)
        for i in range(H):
            for j in range(W):
                if mask[i, j] == 1:
                    min_i = max(0, i-1); max_i = min(H, i+2)
                    min_j = max(0, j-1); max_j = min(W, j+2)
                    if np.all(mask[min_i:max_i, min_j:max_j] == 1):
                        result[i, j] = 1
        return result
    
    # 简化处理流程
    mask_processed = simple_erosion(mask, kernel_size)
    mask_processed = simple_dilation(mask_processed, kernel_size)
    
    # 简单的面积过滤
    H, W = mask_processed.shape
    visited = np.zeros((H, W), dtype=bool)
    
    for i in range(H):
        for j in range(W):
            if mask_processed[i, j] == 1 and not visited[i, j]:
                # 简单区域生长
                stack = [(i, j)]
                region = []
                visited[i, j] = True
                
                while stack:
                    x, y = stack.pop()
                    region.append((x, y))
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nx, ny = x+dx, y+dy
                        if 0<=nx<H and 0<=ny<W and not visited[nx,ny] and mask_processed[nx,ny]==1:
                            visited[nx,ny] = True
                            stack.append((nx, ny))
                
                # 面积过滤
                if len(region) < min_area:
                    for x, y in region:
                        mask_processed[x, y] = 0
    
    return mask_processed

# --------------------------- 其他函数保持不变 ---------------------------
def mask_offset_values(offset_map: np.ndarray, processed_mask: np.ndarray) -> np.ndarray:
    mask_3d = np.expand_dims(processed_mask, axis=-1)
    masked_offset = offset_map * mask_3d
    return masked_offset

def extract_and_cluster_centroids(
    processed_mask: np.ndarray, 
    masked_offset: np.ndarray, 
    eps: float = 50.0, 
    min_samples: int = 10
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    y_coords, x_coords = np.where(processed_mask == 1)
    if len(y_coords) == 0:
        return np.array([]), []
    
    candidates = []
    for x, y in zip(x_coords, y_coords):
        dx, dy = masked_offset[y, x]
        centroid_x = x + dx
        centroid_y = y + dy
        candidates.append([centroid_x, centroid_y])
    candidates = np.array(candidates)
    
    cluster_centers = simple_clustering(candidates, eps, min_samples)
    return candidates, cluster_centers

def simple_clustering(points: np.ndarray, cluster_radius: float, min_points: int) -> List[Tuple[float, float]]:
    if len(points) == 0:
        return []
    
    clusters = []
    visited = set()
    
    for i, point in enumerate(points):
        if i in visited:
            continue
            
        distances = np.linalg.norm(points - point, axis=1)
        neighbor_indices = np.where(distances < cluster_radius)[0]
        
        if len(neighbor_indices) >= min_points:
            cluster_points = points[neighbor_indices]
            avg_x = round(np.mean(cluster_points[:, 0]), 2)
            avg_y = round(np.mean(cluster_points[:, 1]), 2)
            clusters.append((avg_x, avg_y))
            visited.update(neighbor_indices)
    
    return clusters

def draw_result_visualization(
    original_mask: np.ndarray,
    processed_mask: np.ndarray,
    masked_offset: np.ndarray,
    candidates: np.ndarray,
    cluster_centers: List[Tuple[float, float]],
    img_h: int,
    img_w: int
) -> Image.Image:
    subplot_size = (img_w, img_h)
    total_size = (2 * img_w, 2 * img_h)
    result_img = Image.new("RGB", total_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(result_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    
    original_mask_pil = numpy_to_pil(original_mask, mode="L").resize(subplot_size)
    result_img.paste(original_mask_pil, (0, 0))
    draw.text((10, 10), "Original Mask", fill=(255, 0, 0), font=font)
    
    processed_mask_pil = numpy_to_pil(processed_mask, mode="L").resize(subplot_size)
    result_img.paste(processed_mask_pil, (img_w, 0))
    draw.text((img_w + 10, 10), "Processed Mask", fill=(255, 0, 0), font=font)
    
    offset_magnitude = np.linalg.norm(masked_offset, axis=2)
    if offset_magnitude.max() > 0:
        offset_magnitude = (offset_magnitude / offset_magnitude.max()) * 255
    offset_magnitude_pil = numpy_to_pil(offset_magnitude.astype(np.uint8), mode="L").resize(subplot_size)
    result_img.paste(offset_magnitude_pil, (0, img_h))
    draw.text((10, img_h + 10), "Offset Magnitude", fill=(255, 0, 0), font=font)
    
    cluster_bg_pil = processed_mask_pil.convert("RGB")
    cluster_draw = ImageDraw.Draw(cluster_bg_pil)
    
    if len(candidates) > 0:
        scale_x = img_w / original_mask.shape[1]
        scale_y = img_h / original_mask.shape[0]
        scaled_candidates = candidates * [scale_x, scale_y]
        for (x, y) in scaled_candidates:
            cluster_draw.ellipse((x-1, y-1, x+1, y+1), fill=(0, 0, 255))
    
    for i, (x, y) in enumerate(cluster_centers):
        scaled_x = x * scale_x
        scaled_y = y * scale_y
        cluster_draw.line((scaled_x-5, scaled_y-5, scaled_x+5, scaled_y+5), fill=(255, 0, 0), width=2)
        cluster_draw.line((scaled_x+5, scaled_y-5, scaled_x-5, scaled_y+5), fill=(255, 0, 0), width=2)
        cluster_draw.text((scaled_x+8, scaled_y-8), str(i+1), fill=(255, 0, 0), font=font)
    
    result_img.paste(cluster_bg_pil, (img_w, img_h))
    draw.text((img_w + 10, img_h + 10), f"Clusters: {len(cluster_centers)}", fill=(255, 0, 0), font=font)
    
    return result_img

def model_output_post_process(
    model_output: Dict[str, np.ndarray],
    save_dir: str = "./post_process_results",
    min_area: int = 500,
    kernel_size: int = 3,
    eps: float = 50.0,
    min_samples: int = 10,
    save_visualization: bool = True,
    save_masks: bool = True,
    use_fast_method: bool = True  # 新增参数，控制是否使用快速方法
) -> Dict[str, any]:
    original_mask = model_output["mask"].squeeze()
    offset_map = model_output["offset_map"].squeeze()
    img_h, img_w = original_mask.shape[0], original_mask.shape[1]
    
    os.makedirs(save_dir, exist_ok=True)
    save_paths = {}
    
    processed_mask = post_process_mask(original_mask, min_area, kernel_size, use_fast_method)
    masked_offset = mask_offset_values(offset_map, processed_mask)
    candidates, cluster_centers = extract_and_cluster_centroids(
        processed_mask, masked_offset, eps, min_samples
    )
    
    if save_masks:
        processed_mask_pil = numpy_to_pil(processed_mask, mode="L")
        mask_save_path = os.path.join(save_dir, "processed_mask.png")
        processed_mask_pil.save(mask_save_path)
        save_paths["processed_mask"] = mask_save_path
    
    if save_visualization:
        result_visual_img = draw_result_visualization(
            original_mask, processed_mask, masked_offset,
            candidates, cluster_centers, img_h, img_w
        )
        vis_save_path = os.path.join(save_dir, "post_process_visualization.png")
        result_visual_img.save(vis_save_path, quality=95)
        save_paths["visualization"] = vis_save_path
    
    post_process_result = {
        "processed_mask": processed_mask,
        "masked_offset": masked_offset,
        "cluster_centers": cluster_centers,
        "candidates": candidates,
        "save_paths": save_paths,
        "num_clusters": len(cluster_centers)
    }
    
    print(f"后处理完成：检测到{len(cluster_centers)}个料箱聚类中心")
    return post_process_result