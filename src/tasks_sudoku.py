"""
数独任务定义（基于 diffusion-vs-ar 风格）
==========================================

任务描述：
- 输入：162 维向量（81位题目 + 81位答案）
- 输出：81 个类别的预测（0-9）
- Loss：Cross-Entropy Loss（分类任务）
- 评估指标：Cell Accuracy 和 Board Accuracy
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict


def decode_sudoku_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    将 logits 解码为 81 个类别预测
    
    Args:
        logits: [B, n_respond, 81, 10] 或 [B, 81, 10] 的 logits 向量
    
    Returns:
        pred_classes: [B, n_respond, 81] 或 [B, 81] 预测的类别（0-9）
    """
    if logits.dim() == 4:
        # [B, n_respond, 81, 10] -> [B, n_respond, 81]
        return torch.argmax(logits, dim=-1)
    elif logits.dim() == 3:
        # [B, 81, 10] -> [B, 81]
        return torch.argmax(logits, dim=-1)
    else:
        raise ValueError(f"不支持的 logits 维度: {logits.shape}")


def decode_sudoku_onehot(ys: torch.Tensor) -> torch.Tensor:
    """
    从 solution 向量中提取类别标签
    
    Args:
        ys: [B, n_points, 81] 或 [B, 81] 的 solution 向量
    
    Returns:
        labels: [B, n_respond, 81] 或 [B, 81] 类别标签（0-9）
    """
    if ys.dim() == 3:
        # [B, n_points, 81] -> 提取最后一个点的 solution（假设 respond 是最后一个点）
        solution_part = ys[:, -1, :]  # [B, 81]
        return solution_part.long()
    elif ys.dim() == 2:
        # [B, 81] -> 直接返回
        return ys.long()
    else:
        raise ValueError(f"不支持的 ys 维度: {ys.shape}")


def sudoku_accuracy(pred_logits: torch.Tensor, true_ys: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    计算数独任务的准确率
    
    Args:
        pred_logits: [B, n_respond, 81, 10] 预测的 logits
        true_ys: [B, n_points, 81] 真实的 solution 向量
    
    Returns:
        dict: {
            'cell_accuracy': 格子级别的准确率（标量）
            'board_accuracy': 数独级别的准确率（标量）
        }
    """
    # 解码预测和标签
    pred_digits = decode_sudoku_logits(pred_logits)  # [B, n_respond, 81]
    true_digits = decode_sudoku_onehot(true_ys)  # [B, 81]
    
    # 如果 pred_digits 有 n_respond 维度，取最后一个（或平均）
    if pred_digits.dim() == 3:
        # 取最后一个 respond 的预测
        pred_digits = pred_digits[:, -1, :]  # [B, 81]
    
    # 扩展 true_digits 以匹配 pred_digits
    if true_digits.dim() == 2 and true_digits.shape[0] == pred_digits.shape[0]:
        # true_digits: [B, 81], pred_digits: [B, 81]
        pass
    else:
        # 如果维度不匹配，尝试广播
        if true_digits.shape[0] != pred_digits.shape[0]:
            # 取最后一个 batch 的 true_digits
            true_digits = true_digits[-1:].expand_as(pred_digits)
    
    # 计算格子级别的准确率
    correct = (pred_digits == true_digits).float()  # [B, 81]
    cell_accuracy = correct.mean()  # 标量
    
    # 计算数独级别的准确率（整盘 81 格全对）
    board_correct = (correct.mean(dim=-1) == 1.0).float()  # [B]
    board_accuracy = board_correct.mean()  # 标量
    
    return {
        'cell_accuracy': cell_accuracy,
        'board_accuracy': board_accuracy,
    }


class SudokuTask:
    """
    数独求解任务
    
    任务描述：
    - 输入：xs=[B, n_points, 81] (quiz), ys=[B, n_points, 81] (solution)
    - 输出：81 个类别的预测（0-9）
    - Loss：Cross-Entropy Loss（分类任务）
    """
    
    def __init__(
        self,
        n_dims: int,
        batch_size: int,
        data_sampler,
        w_type: str = "sudoku",
        pool_dict: Optional[Dict] = None,
        seeds: Optional[list] = None,
        **kwargs
    ):
        """
        Args:
            n_dims: 维度（数独任务固定为 81，quiz 部分）
            batch_size: batch大小
            data_sampler: 数独数据采样器实例
            w_type: 任务类型标识
            pool_dict: 保留兼容性
            seeds: 随机种子列表
        """
        self.n_dims = 81  # 数独任务：quiz 部分为 81 维
        self.b_size = batch_size
        self.data_sampler = data_sampler
        self.w_type = w_type
        self.pool_dict = pool_dict
        self.seeds = seeds
        
        # 存储当前batch的问题和答案
        self._current_xs = None
        self._current_ys = None
    
    def evaluate(self, xs: torch.Tensor) -> torch.Tensor:
        """
        根据问题（X）返回答案（Y）
        
        Args:
            xs: [B, n_points, 81] 数独问题（quiz 部分）
        
        Returns:
            ys: [B, n_points, 81] 数独答案（solution 部分）
        """
        # 从数据采样器获取对应的 solution
        self._current_xs = xs
        ys = self.data_sampler.get_solutions(xs)
        self._current_ys = ys
        return ys
    
    @staticmethod
    def get_metric():
        """返回评估指标（准确率）"""
        return sudoku_accuracy
    
    @staticmethod
    def get_training_metric():
        """返回训练指标（Cross-Entropy Loss）"""
        # 数独的 Loss 在模型内部通过 CrossEntropy 计算
        return None
    
    @staticmethod
    def generate_pool_dict(n_dims: int, num_tasks: int, w_type: str, **kwargs):
        """
        保留兼容性：数独任务不使用 pool_dict
        """
        return None


def get_sudoku_task_sampler(
    n_dims: int,
    batch_size: int,
    data_sampler,
    w_type: str = "sudoku",
    pool_dict: Optional[Dict] = None,
    num_tasks: Optional[int] = None,
    **kwargs
):
    """
    工厂函数：创建数独任务采样器
    
    Args:
        n_dims: 维度（数独任务固定为 162）
        batch_size: batch大小
        data_sampler: 数独数据采样器实例
        w_type: 任务类型
        pool_dict: 保留兼容性
        num_tasks: 保留兼容性
        **kwargs: 其他参数
    
    Returns:
        lambda函数，调用时返回 SudokuTask 实例
    """
    return lambda **args: SudokuTask(
        n_dims=n_dims,
        batch_size=batch_size,
        data_sampler=data_sampler,
        w_type=w_type,
        pool_dict=pool_dict,
        **args,
        **kwargs
    )
