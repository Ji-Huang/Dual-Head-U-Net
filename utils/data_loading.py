import os
import torch
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
from PIL import Image
import numpy as np
import cv2


class DualHeadUNetDataset(Dataset):
    """
    适配双头U-Net的数据集类: 加载RGB图、深度图、语义分割标签、偏移回归标签
    """
    def __init__(self, data_root, split="train", resize=None, 
                 # rgb_mean=(0.485, 0.456, 0.406), rgb_std=(0.229, 0.224, 0.225),  # ImageNet
                 rgb_mean=(0.180, 0.246, 0.285), rgb_std=(0.207, 0.202, 0.230),  # Ours
                 depth_mean=2997.516, depth_std=1324.941,  # Ours
                 offset_mean = np.array([-0.0681, -0.0754]),  # Ours
                 offset_std = np.array([17.0844, 10.4708])):  # Ours
        """
        Args:
            data_root: 数据集根路径(如 "data/")
            split: 数据集子集()"train"/"val"/"test")
            resize: 可选，图像 resize 尺寸(如 (480, 640)，格式 (height, width))
            rgb_mean: RGB图像归一化的均值
            rgb_std: RGB图像归一化的标准差
            depth_mean: 深度图归一化的均值
            depth_std: 深度图归一化的标准差
            offset_mean: 偏移值图归一化的均值
            offset_std: 偏移值图归一化的标准差
        """
        self.data_root = data_root
        self.split = split
        self.resize = resize

        self.rgb_mean = np.array(rgb_mean, dtype=np.float32)
        self.rgb_std = np.array(rgb_std, dtype=np.float32)
        self.depth_mean = depth_mean
        self.depth_std = depth_std
        self.offset_mean = np.array(offset_mean, dtype=np.float32)  # x/y 均值
        self.offset_std = np.array(offset_std, dtype=np.float32)   # x/y 标准差
        
        # 1. 定义数据路径
        self.rgb_dir = os.path.join(data_root, split, "images")
        self.depth_dir = os.path.join(data_root, split, "depth")
        self.semantic_dir = os.path.join(data_root, split, "gt_semantic")
        self.offset_dir = os.path.join(data_root, split, "gt_offset")
        
        # 2. 验证路径存在性
        assert os.path.exists(self.rgb_dir), f"RGB路径不存在: {self.rgb_dir}"
        assert os.path.exists(self.depth_dir), f"深度图路径不存在：{self.depth_dir}"
        assert os.path.exists(self.semantic_dir), f"语义标签路径不存在：{self.semantic_dir}"
        assert os.path.exists(self.offset_dir), f"偏移标签路径不存在：{self.offset_dir}"
        
        # 3. 获取图像文件名（以RGB图为基准，确保其他文件一一对应）
        self.rgb_filenames = [f for f in os.listdir(self.rgb_dir) if f.endswith((".png", ".jpg"))]
        # 验证文件名匹配（避免数据缺失）
        for rgb_fn in self.rgb_filenames:
            depth_fn = rgb_fn.replace("_color", "_depth")
            depth_fn = depth_fn.replace(".jpg", ".png")
            semantic_fn = rgb_fn.replace(".jpg", ".png")
            offset_fn = rgb_fn.replace(".jpg", ".npy")
            
            assert os.path.exists(os.path.join(self.depth_dir, depth_fn)), f"深度图缺失：{depth_fn}"
            assert os.path.exists(os.path.join(self.semantic_dir, semantic_fn)), f"语义标签缺失：{semantic_fn}"
            assert os.path.exists(os.path.join(self.offset_dir, offset_fn)), f"偏移标签缺失：{offset_fn}"

    # def _numpy_resize(self, img, target_size, interpolation="bicubic"):
    #     """Resize numpy array using PIL"""
    #     print(img.shape)
    #     single_channel_flag = False
    #     try:
    #         # Convert numpy array to PIL Image
    #         if len(img.shape) == 3 and img.shape[0] == 1:  # Single channel with channel dimension
    #             img = img[0]  # Remove channel dimension for PIL
    #             mode = 'L'
    #             single_channel_flag = True
    #         elif len(img.shape) == 3:  # Multi-channel (RGB, etc.)
    #             mode = 'RGB'
    #         else:  # Grayscale
    #             mode = 'L'
    #         print(img.shape)
    #         pil_img = Image.fromarray(img, mode)
            
    #         # Map interpolation string to PIL method
    #         interpolation_map = {
    #             "nearest": Image.NEAREST,
    #             "bilinear": Image.BILINEAR,
    #             "bicubic": Image.BICUBIC,
    #             "lanczos": Image.LANCZOS
    #         }
            
    #         pil_interpolation = interpolation_map.get(interpolation.lower(), Image.BICUBIC)
            
    #         # Resize
    #         resized_pil = pil_img.resize((target_size[1], target_size[0]), pil_interpolation)
            
    #         # Convert back to numpy array
    #         resized = np.array(resized_pil)
    #         print(resized.shape)
    #         # Add channel dimension back if it was single channel with channel dimension
    #         if single_channel_flag:
    #             resized = resized[np.newaxis, :, :]
    #         print(resized.shape)
    #         return resized
            
    #     except Exception as e:
    #         raise RuntimeError(f"Failed to resize image (shape={img.shape}, target={target_size}): {str(e)}")

    def _numpy_resize(self, img, target_size, interpolation="bicubic"):
        """Resize numpy array using PIL with proper channel handling"""
        interpolation_map = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC
        }
        single_channel_flag = False
        try:
            original_shape = img.shape
            # print(f"Debug: Original shape: {original_shape}")  # 调试信息
            
            # 处理2通道特殊情况
            if len(img.shape) == 3 and img.shape[0] == 2:
                # 2通道图像 - 分别处理每个通道
                resized_channels = []
                for i in range(img.shape[0]):
                    channel = img[i]
                    pil_img = Image.fromarray(channel, 'L')  # 单通道
                    
                    pil_interpolation = interpolation_map.get(interpolation.lower(), Image.BICUBIC)                  
                    resized_pil = pil_img.resize((target_size[1], target_size[0]), pil_interpolation)
                    resized_channel = np.array(resized_pil)
                    resized_channels.append(resized_channel)
                
                # 重新组合通道
                resized = np.stack(resized_channels, axis=0)
                # print(resized.shape)
                return resized
                
            # 正常处理其他情况
            elif len(img.shape) == 3 and img.shape[0] in [1, 3]:
                # 标准1或3通道图像
                if img.shape[0] == 1:
                    img = img[0]
                    mode = 'L'
                    single_channel_flag = True
                else:
                    img = img.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                    mode = 'RGB'
                
                pil_img = Image.fromarray(img, mode)              
                pil_interpolation = interpolation_map.get(interpolation.lower(), Image.BICUBIC)
                
                # Resize
                resized_pil = pil_img.resize((target_size[1], target_size[0]), pil_interpolation)
                
                # Convert back to numpy array
                resized = np.array(resized_pil)
                # print(resized.shape)
                # Add channel dimension back if it was single channel with channel dimension
                if single_channel_flag:
                    resized = resized[np.newaxis, :, :]
                else:
                    resized = resized.transpose(2, 0, 1)  #  (H, W, C) -> (C, H, W)
                # print(resized.shape)
                return resized
                
        except Exception as e:
            raise RuntimeError(f"Failed to resize image: {str(e)}")
        


    def __len__(self):
        """返回数据集样本总数"""
        return len(self.rgb_filenames)

    def __getitem__(self, idx):
        """按索引加载单个样本: RGB图、深度图、语义标签、偏移标签"""
        # 1. 获取当前样本的文件名
        rgb_fn = self.rgb_filenames[idx]
        depth_fn = rgb_fn.replace("_color", "_depth")
        depth_fn = depth_fn.replace(".jpg", ".png")
        semantic_fn = rgb_fn.replace(".jpg", ".png")
        offset_fn = rgb_fn.replace(".jpg", ".npy")
        
        # 2. 加载RGB图
        rgb_path = os.path.join(self.rgb_dir, rgb_fn)
        rgb_img = Image.open(rgb_path).convert("RGB")  # 确保为3通道
        rgb_np = np.array(rgb_img, dtype=np.float32)  # (H, W, 3)，值范围0~255
        rgb_np = rgb_np.transpose(2, 0, 1) / 255.0    # 转置为(3, H, W)，归一化到0~1
        # 应用RGB归一化
        for c in range(3):
            rgb_np[c] = (rgb_np[c] - self.rgb_mean[c]) / self.rgb_std[c]
        # 可选resize
        if self.resize is not None:
            h, w = self.resize
            rgb_np = self._numpy_resize(rgb_np, (h, w))
        rgb_tensor = torch.from_numpy(rgb_np).float()  # 转为Tensor

        
        # 3. 加载深度图
        depth_path = os.path.join(self.depth_dir, depth_fn)
        # 若深度图是16位（如存储实际深度值，范围0~65535），需先转为8位或归一化；若已为8位，直接转L模式
        depth_img = Image.open(depth_path)
        if depth_img.mode == "I" or depth_img.mode == "I;16":
            depth_np = np.array(depth_img, dtype=np.float32)
            depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        else:
            depth_np = np.array(depth_img.convert("L"), dtype=np.float32) / 255.0  # 归一化到0~1
        depth_np = depth_np[np.newaxis, :, :]  # 增加通道维，变为(1, H, W)
        # 应用深度图归一化
        depth_np = (depth_np - self.depth_mean) / self.depth_std
        # 可选resize
        if self.resize is not None:
            depth_np = self._numpy_resize(depth_np, (h, w))
        depth_tensor = torch.from_numpy(depth_np).float()
        
        # 4. 加载语义分割标签（单通道，像素值为类别ID）
        semantic_path = os.path.join(self.semantic_dir, semantic_fn)
        semantic_img = Image.open(semantic_path).convert("L")  # 确保单通道
        semantic_np = np.array(semantic_img, dtype=np.int64)  # 类别ID用int64
        # 语义分割标签值归一化
        semantic_np = semantic_np / 255
        # 可选resize（用最近邻插值，避免类别混淆）
        if self.resize is not None:
            semantic_np = self._numpy_resize(semantic_np[np.newaxis, :, :], (h, w), interpolation="nearest")[0]
        semantic_tensor = torch.from_numpy(semantic_np).long()
        
        # 5. 加载偏移回归标签（两通道图像：δx、δy）
        offset_path = os.path.join(self.offset_dir, offset_fn)
        offset_gt_raw = np.load(offset_path).transpose(2, 0, 1)  # 维度转置（H×W×C → C×H×W）

        # Z-Score 标准化（支持批次数据，按 x/y 方向分别处理）
        offset_gt_norm = offset_gt_raw.copy()  # 避免修改原始数据
        # 假设形状为(2, H, W)（单样本）或(B, 2, H, W)（批次）
        channel_dim = 0 if offset_gt_norm.ndim == 3 else 1  # 3维: (2,H,W)；4维: (B,2,H,W)
        for c in range(2):  # c=0:x方向, c=1:y方向
            mean = self.offset_mean[c]
            std = self.offset_std[c] + 1e-8  # 避免除零          
            # 根据通道维度索引，适配不同形状
            if channel_dim == 0:
                offset_gt_norm[c, :, :] = (offset_gt_raw[c, :, :] - mean) / std
            else:
                offset_gt_norm[:, c, :, :] = (offset_gt_raw[:, c, :, :] - mean) / std
        # 可选resize
        if self.resize is not None:
            offset_gt_raw = self._numpy_resize(offset_gt_raw, (h, w))
            offset_gt_norm = self._numpy_resize(offset_gt_norm, (h, w))
        offset_tensor_raw = torch.from_numpy(offset_gt_raw).float()
        offset_tensor_norm = torch.from_numpy(offset_gt_norm).float()

        assert offset_tensor_raw.shape[0] == 2, f"偏移标签需为2通道 (δx、δy), 当前为{offset_tensor_raw.shape[0]}通道"
        
        # 返回：输入（RGB+深度）、标签（语义+偏移）
        return {
            "rgb": rgb_tensor,          # (3, H, W)
            "depth": depth_tensor,      # (1, H, W)
            "semantic_gt": semantic_tensor,  # (H, W)，long类型
            "offset_gt_raw": offset_tensor_raw,   # (2, H, W)，float类型
            "offset_gt_norm": offset_tensor_norm   # (2, H, W)，float类型
        }
    
    def denormalize_offset(self, pred_offset):
        """
        反归一化模型输出的偏移值
        Args:
            pred_offset: 模型输出的归一化偏移, shape=(B, 2, H, W)（Tensor或numpy数组）
        Returns:
            反归一化后的真实偏移, shape=(B, 2, H, W)
        """
        # 处理 Tensor 类型（若输入是Tensor，先转为numpy数组，避免修改原Tensor）
        is_tensor = False
        if isinstance(pred_offset, torch.Tensor):
            is_tensor = True
            pred_offset = pred_offset.detach().cpu().numpy()
        
        # 按 x/y 方向分别反归一化（使用实例的 offset_mean 和 offset_std）
        pred_offset[:, 0, :, :] = pred_offset[:, 0, :, :] * self.offset_std[0] + self.offset_mean[0]  # x方向
        pred_offset[:, 1, :, :] = pred_offset[:, 1, :, :] * self.offset_std[1] + self.offset_mean[1]  # y方向
        
        # 若原输入是Tensor，转回Tensor并保持设备一致
        if is_tensor:
            pred_offset = torch.from_numpy(pred_offset).to(self.device if hasattr(self, 'device') else 'cpu')
        
        return pred_offset


def create_dual_head_dataloader(data_root, split="train", resize=None, 
                                batch_size=8, num_workers=4, shuffle=True):
    """
    创建双头U-Net的DataLoader
    
    Args:
        data_root: 数据集根路径
        split: 数据集子集（"train"/"val"/"test"）
        resize: 可选，图像resize尺寸（height, width）
        batch_size: 批次大小
        num_workers: 数据加载线程数（CPU核心数充足时可设为4~8）
        shuffle: 是否打乱数据（train设为True，val/test设为False）
    
    Returns:
        DataLoader: 适配双头U-Net的DataLoader
    """
    # 1. 初始化数据集
    dataset = DualHeadUNetDataset(
        data_root=data_root,
        split=split,
        resize=resize
    )
    
    # 2. 创建DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # 若使用GPU，设为True可加速数据传输（减少CPU→GPU延迟）
    )
    
    return dataloader


# # ------------------- 示例用法 -------------------
# if __name__ == "__main__":
#     # 数据集根路径（根据你的实际路径修改）
#     DATA_ROOT = "/root/unet/data/"
#     # 图像resize尺寸（可选，若不需要resize设为None）
#     RESIZE_SIZE = (480, 640)  # (height, width)，与你模型输入一致
    
#     # 1. 创建训练集DataLoader
#     train_loader = create_dual_head_dataloader(
#         data_root=DATA_ROOT,
#         split="train",
#         resize=None,
#         batch_size=2,  # 示例批次大小
#         num_workers=0,
#         shuffle=True
#     )
    
#     # 2. 创建验证集DataLoader
#     val_loader = create_dual_head_dataloader(
#         data_root=DATA_ROOT,
#         split="val",
#         resize=None,
#         batch_size=2,
#         num_workers=0,
#         shuffle=False  # 验证集不打乱
#     )
    
#     # 3. 测试数据加载（查看数据维度是否匹配模型输入）
#     for batch in train_loader:
#         rgb = batch["rgb"]          # (batch_size, 3, H, W)
#         depth = batch["depth"]      # (batch_size, 1, H, W)
#         semantic_gt = batch["semantic_gt"]  # (batch_size, H, W)
#         offset_gt = batch["offset_gt"]       # (batch_size, 2, H, W)
        
#         print(f"RGB tensor shape: {rgb.shape}")
#         print(f"Depth tensor shape: {depth.shape}")
#         print(f"Semantic GT shape: {semantic_gt.shape}")
#         print(f"Offset GT shape: {offset_gt.shape}")
#         print(f"Semantic GT dtype: {semantic_gt.dtype}")  # 需为long
#         print(f"Offset GT dtype: {offset_gt.dtype}")      # 需为float
        
#         # 若加载模型，可直接传入数据
#         # model = EarlyFusionUNet(n_classes=1)
#         # semantic_pred, offset_pred = model(rgb, depth)
        
#         break  # 仅测试一个批次