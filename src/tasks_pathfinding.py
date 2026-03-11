"""
路径查找任务定义（基于 diffusion-vs-ar 风格）
==========================================

任务描述：
- 输入：边列表 + 查询（起点/终点）
- 输出：路径节点序列
- Loss：Cross-Entropy Loss（分类任务）
- 评估指标：Node Accuracy 和 Path Accuracy
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict


def decode_pathfinding_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    将 logits 解码为路径节点预测

    Args:
        logits: [B, n_respond, path_len, vocab_size] 或 [B, path_len, vocab_size] 的 logits 向量

    Returns:
        pred_nodes: [B, n_respond, path_len] 或 [B, path_len] 预测的节点 ID
    """
    if logits.dim() == 4:
        # [B, n_respond, path_len, vocab_size] -> [B, n_respond, path_len]
        return torch.argmax(logits, dim=-1)
    elif logits.dim() == 3:
        # [B, path_len, vocab_size] -> [B, path_len]
        return torch.argmax(logits, dim=-1)
    else:
        raise ValueError(f"不支持的 logits 维度: {logits.shape}")


def decode_pathfinding_target(ys: torch.Tensor) -> torch.Tensor:
    """
    从路径向量中提取节点标签

    Args:
        ys: [B, n_points, path_len] 或 [B, path_len] 的路径向量

    Returns:
        labels: [B, n_respond, path_len] 或 [B, path_len] 节点标签
    """
    if ys.dim() == 3:
        # [B, n_points, path_len] -> 提取最后一个点的路径（假设 respond 是最后一个点）
        path_part = ys[:, -1, :]  # [B, path_len]
        return path_part.long()
    elif ys.dim() == 2:
        # [B, path_len] -> 直接返回
        return ys.long()
    else:
        raise ValueError(f"不支持的 ys 维度: {ys.shape}")


def pathfinding_accuracy(pred_logits: torch.Tensor, true_ys: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    计算路径查找任务的准确率

    Args:
        pred_logits: [B, n_respond, path_len, vocab_size] 预测的 logits
        true_ys: [B, n_points, path_len] 真实的路径向量

    Returns:
        dict: {
            'node_accuracy': 节点级别的准确率（标量）
            'path_accuracy': 路径级别的准确率（标量）
        }
    """
    # 解码预测和标签
    pred_nodes = decode_pathfinding_logits(pred_logits)  # [B, n_respond, path_len]
    true_nodes = decode_pathfinding_target(true_ys)  # [B, path_len]

    # 如果 pred_nodes 有 n_respond 维度，取最后一个
    if pred_nodes.dim() == 3:
        # 取最后一个 respond 的预测
        pred_nodes = pred_nodes[:, -1, :]  # [B, path_len]

    # 扩展 true_nodes 以匹配 pred_nodes
    if true_nodes.dim() == 2 and true_nodes.shape[0] == pred_nodes.shape[0]:
        # true_nodes: [B, path_len], pred_nodes: [B, path_len]
        pass
    else:
        # 如果维度不匹配，尝试广播
        if true_nodes.shape[0] != pred_nodes.shape[0]:
            # 取最后一个 batch 的 true_nodes
            true_nodes = true_nodes[-1:].expand_as(pred_nodes)

    # 计算节点级别的准确率
    correct = (pred_nodes == true_nodes).float()  # [B, path_len]
    node_accuracy = correct.mean()  # 标量

    # 计算路径级别的准确率（整条路径全对）
    path_correct = (correct.mean(dim=-1) == 1.0).float()  # [B]
    path_accuracy = path_correct.mean()  # 标量

    return {
        'node_accuracy': node_accuracy,
        'path_accuracy': path_accuracy,
    }


class PathfindingTask:
    """
    路径查找任务

    任务描述：
    - 输入：xs=[B, n_points, edge_len] (边列表 + 查询), ys=[B, n_points, path_len] (路径)
    - 输出：路径节点序列
    - Loss：Cross-Entropy Loss（分类任务）
    """

    def __init__(
        self,
        n_dims: int,
        batch_size: int,
        data_sampler,
        w_type: str = "pathfinding",
        pool_dict: Optional[Dict] = None,
        seeds: Optional[list] = None,
        **kwargs
    ):
        """
        Args:
            n_dims: 维度（路径查找任务：edge_len = (path_len-1)*degree*2 + 2）
            batch_size: batch大小
            data_sampler: 路径查找数据采样器实例
            w_type: 任务类型标识
            pool_dict: 保留兼容性
            seeds: 随机种子列表
        """
        self.n_dims = n_dims
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
            xs: [B, n_points, edge_len] 路径查找问题（边列表 + 查询）

        Returns:
            ys: [B, n_points, path_len] 路径答案
        """
        # 从数据采样器获取对应的路径
        self._current_xs = xs
        ys = self.data_sampler.get_solutions(xs)
        self._current_ys = ys
        return ys

    @staticmethod
    def get_metric():
        """返回评估指标（准确率）"""
        return pathfinding_accuracy

    @staticmethod
    def get_training_metric():
        """返回训练指标（Cross-Entropy Loss）"""
        # 路径查找的 Loss 在模型内部通过 CrossEntropy 计算
        return None

    @staticmethod
    def generate_pool_dict(n_dims: int, num_tasks: int, w_type: str, **kwargs):
        """
        保留兼容性：路径查找任务不使用 pool_dict
        """
        return None


def get_pathfinding_task_sampler(
    n_dims: int,
    batch_size: int,
    data_sampler,
    w_type: str = "pathfinding",
    pool_dict: Optional[Dict] = None,
    num_tasks: Optional[int] = None,
    **kwargs
):
    """
    工厂函数：创建路径查找任务采样器

    Args:
        n_dims: 维度（路径查找任务：edge_len = (path_len-1)*degree*2 + 2）
        batch_size: batch大小
        data_sampler: 路径查找数据采样器实例
        w_type: 任务类型
        pool_dict: 保留兼容性
        num_tasks: 保留兼容性
        **kwargs: 其他参数

    Returns:
        lambda函数，调用时返回 PathfindingTask 实例
    """
    return lambda **args: PathfindingTask(
        n_dims=n_dims,
        batch_size=batch_size,
        data_sampler=data_sampler,
        w_type=w_type,
        pool_dict=pool_dict,
        **args,
        **kwargs
    )
