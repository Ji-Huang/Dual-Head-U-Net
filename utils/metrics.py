import numpy as np


def compute_iou(pred_mask, true_mask, eps=1e-8):
    """
    计算语义分割的交并比(IoU)
    Args:
        pred_mask: 预测的掩码数组，shape=(N, H, W)，值为0或1（二分类）
        true_mask: 真实的掩码数组，shape=(N, H, W)，值为0或1（二分类）
        eps: 防止除零的微小值
    Returns:
        平均IoU值（在批次和像素上平均）
    """
    # 确保输入是numpy数组
    if isinstance(pred_mask, np.ndarray) is False:
        pred_mask = np.array(pred_mask)
    if isinstance(true_mask, np.ndarray) is False:
        true_mask = np.array(true_mask)
    
    # 检查形状是否匹配
    assert pred_mask.shape == true_mask.shape, \
        f"预测掩码形状 {pred_mask.shape} 与真实掩码形状 {true_mask.shape} 不匹配"
    
    # 计算交并比（批次内每个样本的IoU，再取平均）
    batch_size = pred_mask.shape[0]
    iou_sum = 0.0
    
    for i in range(batch_size):
        # 计算交集和并集
        intersection = np.logical_and(pred_mask[i], true_mask[i]).sum()
        union = np.logical_or(pred_mask[i], true_mask[i]).sum()
        iou = (intersection + eps) / (union + eps)  # 加eps防止除零
        iou_sum += iou
    
    return iou_sum / batch_size


def compute_mae(pred_offset, true_offset):
    """
    计算偏移回归的平均绝对误差(MAE)
    Args:
        pred_offset: 预测的偏移数组，shape=(N, 2, H, W) 或 (N, H, W, 2)
                     其中2表示x和y方向的偏移
        true_offset: 真实的偏移数组，shape与pred_offset相同
    Returns:
        平均MAE值（在批次、通道和像素上平均）
    """
    # 确保输入是numpy数组
    if isinstance(pred_offset, np.ndarray) is False:
        pred_offset = np.array(pred_offset)
    if isinstance(true_offset, np.ndarray) is False:
        true_offset = np.array(true_offset)
    
    # 检查形状是否匹配
    assert pred_offset.shape == true_offset.shape, \
        f"预测偏移形状 {pred_offset.shape} 与真实偏移形状 {true_offset.shape} 不匹配"
    
    # 计算绝对误差并平均（支持不同通道维度顺序）
    absolute_error = np.abs(pred_offset - true_offset)
    return np.mean(absolute_error)  # 在所有维度上取平均


def compute_mse(pred_offset, true_offset):
    """
    计算偏移回归的均方误差(MSE)
    Args:
        pred_offset: 预测的偏移数组，shape=(N, 2, H, W) 或 (N, H, W, 2)
                     其中2表示x和y方向的偏移
        true_offset: 真实的偏移数组，shape与pred_offset相同
    Returns:
        平均MSE值（在批次、通道和像素上平均）
    """
    # 确保输入是numpy数组
    if isinstance(pred_offset, np.ndarray) is False:
        pred_offset = np.array(pred_offset)
    if isinstance(true_offset, np.ndarray) is False:
        true_offset = np.array(true_offset)
    
    # 检查形状是否匹配
    assert pred_offset.shape == true_offset.shape, \
        f"预测偏移形状 {pred_offset.shape} 与真实偏移形状 {true_offset.shape} 不匹配"
    
    # 计算平方误差并平均（支持不同通道维度顺序）
    squared_error = np.square(pred_offset - true_offset)  # 核心修改：绝对误差→平方误差
    return np.mean(squared_error)  # 在所有维度上取平均


def compute_masked_mse(pred, gt, mask):
    """
    计算掩码区域的MSE（仅目标区域，按目标像素数均值）
    输入为numpy数组版本
    Args:
        pred: 预测偏移 (B, 2, H, W) numpy数组
        gt: 真实偏移 (B, 2, H, W) numpy数组
        mask: 目标区域掩码 (B, 1, H, W) numpy数组（1=目标区域，0=非目标区域）
    Returns:
        masked_mse: 掩码区域的MSE（无目标像素时返回0）
    """
    # 1. 扩展掩码维度以匹配pred（(B,1,H,W)→(B,2,H,W)）
    # 沿通道维度复制（将1通道掩码扩展为2通道，与pred的x/y方向对应）
    mask = np.repeat(mask, 2, axis=1)  # numpy中用repeat扩展通道维度
    
    # 2. 仅计算掩码区域的误差（非目标区域误差置0）
    squared_error = (pred - gt) **2 * mask  # 仅目标区域保留误差
    
    # 3. 按“目标区域像素数”计算均值（避免分母为0）
    total_target_pixels = mask.sum() + 1e-8  # 目标区域总像素数（加1e-8防除零）
    masked_mse = squared_error.sum() / total_target_pixels
    
    return masked_mse
