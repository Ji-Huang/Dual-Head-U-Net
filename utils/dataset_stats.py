import os
import numpy as np
from PIL import Image

import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# 统计训练集RGB图像的均值和标准差（仅需运行一次）
def calculate_rgb_stats(data_root, split="train"):
    rgb_dir = os.path.join(data_root, split, "images")
    filenames = [f for f in os.listdir(rgb_dir) if f.endswith((".png", ".jpg"))]
    
    mean = np.zeros(3, dtype=np.float32)
    std = np.zeros(3, dtype=np.float32)
    total_pixels = 0
    
    for fn in filenames:
        img = Image.open(os.path.join(rgb_dir, fn)).convert("RGB")
        img_np = np.array(img, dtype=np.float32) / 255.0  # 归一化到[0,1]
        h, w, c = img_np.shape
        total_pixels += h * w
        mean += img_np.mean(axis=(0,1)) * h * w  # 按像素数加权求和
        std += img_np.std(axis=(0,1)) * h * w
    
    mean /= total_pixels
    std /= total_pixels
    print(f"RGB Mean: {mean.round(4)}")
    print(f"RGB Std: {std.round(4)}")
    return mean, std


# calculate_rgb_stats(data_root="/root/unet/data/")
# RGB Mean: [0.1796 0.2458 0.2846]
# RGB Std: [0.2067 0.2024 0.229 ]


# 统计深度图均值/标准差（类似RGB统计，仅需运行一次）
def calculate_depth_stats(data_root, split="train"):
    depth_dir = os.path.join(data_root, split, "depth")
    filenames = [f for f in os.listdir(depth_dir) if f.endswith((".png", ".jpg"))]
    
    mean = 0.0
    std = 0.0
    total_pixels = 0
    
    for fn in filenames:
        img = Image.open(os.path.join(depth_dir, fn))
        # 处理16位/8位深度图
        if img.mode == "I" or img.mode == "I;16":
            depth_np = np.array(img, dtype=np.float32)
        else:
            depth_np = np.array(img.convert("L"), dtype=np.float32)
        
        # 异常值过滤（如剔除0、65535等无效值）
        depth_np = depth_np[(depth_np > 500) & (depth_np < 65000)]
        if depth_np.size == 0:
            continue
        
        p = depth_np.shape
        total_pixels += depth_np.size
        mean += depth_np.mean() * depth_np.size
        std += depth_np.std() * depth_np.size
    
    mean /= total_pixels
    std /= total_pixels
    print(f"Depth Mean: {mean.round(4)}")
    print(f"Depth Std: {std.round(4)}")
    return mean, std


# calculate_depth_stats(data_root="/root/unet/data/")
# Depth Mean: 2997.516
# Depth Std: 1324.9409


# 统计偏移标签的min/max（用于min-max归一化）或mean/std（用于标准化）
def calculate_offset_stats(data_root, split="train"):
    offset_dir = os.path.join(data_root, split, "gt_offset")
    filenames = [f for f in os.listdir(offset_dir) if f.endswith(".npy")]
    
    all_offsets = []
    for fn in filenames:
        offset_np = np.load(os.path.join(offset_dir, fn))  # (H,W,2)
        all_offsets.append(offset_np.reshape(-1, 2))  # 展平为(N,2)
    
    all_offsets = np.concatenate(all_offsets, axis=0)
    offset_min = all_offsets.min(axis=0)  # 如 [0, 0]
    offset_max = all_offsets.max(axis=0)  # 如 [80, 60]
    offset_mean = all_offsets.mean(axis=0)  # 如 [25.3, 18.7]
    offset_std = all_offsets.std(axis=0)    # 如 [12.5, 10.2]
    
    print(f"Offset Min: {offset_min}")
    print(f"Offset Max: {offset_max}")
    print(f"Offset Mean: {offset_mean.round(4)}")
    print(f"Offset Std: {offset_std.round(4)}")
    return offset_min, offset_max, offset_mean, offset_std


# calculate_offset_stats(data_root="/root/unet/data/")
# Offset Min: [-215. -145.]
# Offset Max: [207. 140.]
# Offset Mean: [-0.0681 -0.0754]
# Offset Std: [17.0844 10.4708]


import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


# --------------------------- 1. 深度梯度计算函数（不变） ---------------------------
def compute_local_depth_gradient(depth_map, kernel_size=3):
    depth_tensor = torch.tensor(depth_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    grad_y = F.conv2d(depth_tensor, sobel_y, padding=kernel_size//2)
    grad_map = torch.abs(grad_y).squeeze(0).squeeze(0).numpy()
    return grad_map

# --------------------------- 2. 梯度分布统计函数（不变） ---------------------------
def stats_foreground_gradient(dataset, sample_num=200):
    foreground_grads = []
    total_foreground_pixels = 0
    sample_indices = np.random.choice(len(dataset), size=min(sample_num, len(dataset)), replace=False)
    for idx in tqdm(sample_indices, desc="统计真实前景梯度分布"):
        data = dataset[idx]
        depth_map = data["depth"]
        sem_gt = data["sem_gt"]
        grad_map = compute_local_depth_gradient(depth_map, kernel_size=3)
        foreground_mask = (sem_gt == 1)
        foreground_grad = grad_map[foreground_mask]
        valid_foreground_grad = foreground_grad[foreground_grad > 0]
        if len(valid_foreground_grad) > 0:
            foreground_grads.extend(valid_foreground_grad.tolist())
            total_foreground_pixels += len(valid_foreground_grad)
    foreground_grads = np.array(foreground_grads)
    grad_stats = {
        "mean": np.mean(foreground_grads),
        "median": np.median(foreground_grads),
        "q95": np.percentile(foreground_grads, 95),
        "q98": np.percentile(foreground_grads, 98),
        "max": np.max(foreground_grads),
        "total_pixels": total_foreground_pixels
    }
    return grad_stats, foreground_grads

# --------------------------- 3. 阈值生成与可视化函数（不变） ---------------------------
def generate_pos_grad_threshold(grad_stats, safety_factor=1.1):
    pos_grad_threshold = grad_stats["q95"] * safety_factor
    if pos_grad_threshold > grad_stats["max"] * 0.8:
        pos_grad_threshold = grad_stats["q98"] * safety_factor
    return round(pos_grad_threshold, 2)


# --------------------------- 4. 数据集类（核心修改：适配_augID命名） ---------------------------
class StackedBoxDataset(Dataset):
    def __init__(self, data_root, split="train"):
        """
        适配含_augID的文件名：
        - 深度图：1757314912916_640x480_depth_augID.png
        - 语义图：1757314912916_640x480_color_augID.png
        - 提取核心前缀：1757314912916_640x480（剔除_depth_augID/_color_augID）
        """
        self.split = split
        self.base_dir = os.path.join(data_root, split) if split in ["train", "val", "test"] else data_root
        self.depth_dir = os.path.join(self.base_dir, "depth")
        self.sem_gt_dir = os.path.join(self.base_dir, "gt_semantic")
        
        # 核心修改：提取含_augID文件名的核心前缀
        self.sample_info = self._get_sample_info()  # 存储 (核心前缀, 深度图路径, 语义图路径)
        if len(self.sample_info) == 0:
            raise ValueError(f"在{self.depth_dir}中未找到符合格式的深度图文件（需包含_depth_augID）")

    def _get_sample_info(self):
        """提取所有样本的核心前缀和对应文件路径"""
        sample_info = []
        # 遍历深度图目录，匹配含_depth_augID的文件名
        for depth_filename in os.listdir(self.depth_dir):
            # 条件1：深度图文件名包含_depth_（如_depth_aug1, _depth_aug10）
            if "_depth_" not in depth_filename:
                continue
            # 条件2：深度图后缀为.png
            if not depth_filename.endswith(".png"):
                continue
            
            # 步骤1：提取深度图的核心前缀（剔除_depth_augID.png）
            # 示例：1757314912916_640x480_depth_aug1.png → 分割为["1757314912916_640x480", "depth", "aug1.png"]
            core_prefix = depth_filename.split("_depth_")[0]
            depth_path = os.path.join(self.depth_dir, depth_filename)
            
            # 步骤2：匹配对应的语义图（含_color_augID.png）
            # 语义图命名规则：核心前缀 + _color_augID.png（augID与深度图一致）
            # 示例：核心前缀=1757314912916_640x480 → 语义图=1757314912916_640x480_color_aug1.png
            sem_aug_suffix = depth_filename.split("_depth_")[1]  # 提取augID.png（如aug1.png）
            sem_filename = f"{core_prefix}_color_{sem_aug_suffix}"
            sem_path = os.path.join(self.sem_gt_dir, sem_filename)
            
            # 容错：若语义图后缀是.jpg（如_color_augID.jpg），补充匹配
            if not os.path.exists(sem_path):
                sem_filename_jpg = f"{core_prefix}_color_{sem_aug_suffix.replace('.png', '.jpg')}"
                sem_path = os.path.join(self.sem_gt_dir, sem_filename_jpg)
            
            # 验证语义图是否存在
            if not os.path.exists(sem_path):
                print(f"警告：语义图{sem_filename}不存在，跳过该样本")
                continue
            
            # 保存样本信息
            sample_info.append({
                "core_prefix": core_prefix,
                "depth_path": depth_path,
                "sem_path": sem_path,
                "full_name": depth_filename  # 保留完整文件名，便于调试
            })
        
        return sample_info

    def __len__(self):
        return len(self.sample_info)
    
    def __getitem__(self, idx):
        sample = self.sample_info[idx]
        core_prefix = sample["core_prefix"]
        depth_path = sample["depth_path"]
        sem_path = sample["sem_path"]
        
        # 1. 加载深度图（16位，单位mm）
        depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth_map is None:
            raise ValueError(f"无法读取深度图: {depth_path}")
        depth_map = depth_map.astype(np.float32)
        
        # 2. 加载语义图（灰度图，前景255→1，背景0）
        # 处理语义图可能的通道问题（若为RGB图，转灰度）
        sem_gt = cv2.imread(sem_path)
        if sem_gt is None:
            raise ValueError(f"无法读取语义图: {sem_path}")
        # 若为RGB图，转灰度（适配_color_augID.jpg可能是彩色标注的情况）
        if len(sem_gt.shape) == 3:
            sem_gt = cv2.cvtColor(sem_gt, cv2.COLOR_BGR2GRAY)
        # 二值化：前景（顶面）255→1，背景（侧面）0
        sem_gt = (sem_gt == 255).astype(np.uint8)
        
        # 3. 深度图预处理（过滤无效值）
        depth_map[depth_map <= 0] = 0  # 过滤0值无效深度
        depth_map[depth_map > 5000] = 0  # 过滤过远深度（根据实际场景调整，单位mm）
        
        return {
            "depth": depth_map,
            "sem_gt": sem_gt,
            "core_prefix": core_prefix,
            "full_name": sample["full_name"]  # 用于调试时定位样本
        }

# --------------------------- 5. 主函数（执行统计与阈值生成） ---------------------------
def main():
    # 配置参数（适配你的目录结构）
    DATA_ROOT = "./data"        # 数据集根目录
    SPLIT = "train"             # 用训练集统计（数据量多，分布全）
    SAMPLE_NUM = 600            # 抽样数量（建议100-200，平衡效率与准确性）
    SAFETY_FACTOR = 1.15        # 安全系数（1.1~1.2，避免误判真实前景）
    
    # 加载数据集（自动适配_augID命名）
    print(f"正在加载{SPLIT}集数据，适配含_augID的文件名...")
    dataset = StackedBoxDataset(data_root=DATA_ROOT, split=SPLIT)
    print(f"数据集加载完成，共找到{len(dataset)}个有效样本（深度图+语义图匹配）")
    
    # 统计真实前景的梯度分布
    print("\n开始统计真实前景（料箱顶面）的梯度分布...")
    grad_stats, foreground_grads = stats_foreground_gradient(dataset, sample_num=SAMPLE_NUM)
    
    # 生成pos_grad_threshold
    pos_grad_threshold = generate_pos_grad_threshold(grad_stats, safety_factor=SAFETY_FACTOR)

    
    # 输出结果并保存阈值
    print("\n" + "="*60)
    print("真实前景（料箱顶面）梯度统计结果（含数据增强样本）：")
    print(f"参与统计的样本数: {min(SAMPLE_NUM, len(dataset))}")
    print(f"参与统计的前景像素数: {grad_stats['total_pixels']:,}")
    print(f"梯度均值: {grad_stats['mean']:.2f}")
    print(f"梯度中位数: {grad_stats['median']:.2f}")
    print(f"梯度95%分位数: {grad_stats['q95']:.2f}")
    print(f"梯度98%分位数: {grad_stats['q98']:.2f}")
    print(f"梯度最大值: {grad_stats['max']:.2f}")
    print("="*60)
    print(f"最终生成的pos_grad_threshold: {pos_grad_threshold:.2f}")
    print("="*60)

if __name__ == "__main__":
    main()