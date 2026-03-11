"""
路径查找数据采样器（基于 diffusion-vs-ar 风格）
===========================================

数据格式：
- 每个点（Point）包含：边列表 + 查询（起点/终点）+ 目标路径
- 格式：edge_list/start,goal=target_path
- 示例：32,3|16,12|3,19/32,12=32,34,6,16,12
"""

import torch
import numpy as np
import os
from typing import Optional, List


class PathfindingDataSampler:
    """
    路径查找数据采样器

    从文本文件读取路径查找数据，构造 ICL 序列。
    数据格式：edge_list/start,goal=target_path
    """

    def __init__(self, data_path, n_dims=None, num_nodes=50, degree=2, path_len=5, test_mode=False, **kwargs):
        """
        Args:
            data_path: 文本文件路径，每行一个路径查找问题
            n_dims: 保留兼容性，路径查找任务根据 num_nodes, degree, path_len 计算
            num_nodes: 图中节点数量
            degree: 每个节点的度数
            path_len: 路径长度
            test_mode: 是否使用测试集
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"路径查找数据文件不存在: {data_path}")

        self.num_nodes = num_nodes
        self.degree = degree
        self.path_len = path_len

        # 读取数据
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(line)

        # 验证数据格式
        self._validate_data()

        # 存储最后一次采样的索引（用于get_solutions）
        self._last_indices = None
        self._last_parsed_data = None

        print(f"[Pathfinding Sampler] Loaded {len(self.data)} problems from {data_path}")
        print(f"[Pathfinding Sampler] Graph: {num_nodes} nodes, degree {degree}, path length {path_len}")

    def _validate_data(self):
        """验证数据格式"""
        for i in range(min(5, len(self.data))):
            line = self.data[i]
            if '/' not in line or '=' not in line:
                raise ValueError(
                    f"路径查找数据格式错误（行{i}）: 期望格式 'edge_list/start,goal=target_path'"
                )

    def _parse_line(self, line):
        """
        解析一行数据

        Args:
            line: "edge_list/start,goal=target_path"

        Returns:
            edges: List of (node1, node2) tuples
            query: (start, goal) tuple
            path: List of node IDs
        """
        # 分割 edge_list 和 query/path
        edge_part, query_path_part = line.split('/')
        query_part, path_part = query_path_part.split('=')

        # 解析边列表
        edges = []
        for edge_str in edge_part.split('|'):
            node1, node2 = map(int, edge_str.split(','))
            edges.append((node1, node2))

        # 解析查询（起点/终点）
        start, goal = map(int, query_part.split(','))
        query = (start, goal)

        # 解析路径
        path = list(map(int, path_part.split(',')))

        return edges, query, path

    def sample_xs(self, n_points, bsz, n_dims=None, seeds=None):
        """
        采样路径查找数据，构造 ICL 序列

        Args:
            n_points: 每个batch需要的点数（n_prompt + n_respond）
            bsz: batch size
            n_dims: 保留兼容性
            seeds: 随机种子列表，用于确定性采样

        Returns:
            xs_b: [bsz, n_points, edge_len] 的张量，每个点包含边列表和查询
                  edge_len = (path_len - 1) * degree * 2 + 2 (edges + query)
        """
        edge_len = (self.path_len - 1) * self.degree * 2 + 2  # edges + query
        xs_b = torch.zeros(bsz, n_points, edge_len, dtype=torch.long)

        # 存储采样的索引，供 get_solutions() 使用
        indices_b = torch.zeros(bsz, n_points, dtype=torch.long)
        parsed_data_b = []  # 存储解析后的数据

        for b in range(bsz):
            if seeds is not None:
                np.random.seed(seeds[b])

            # 随机采样 n_points 个路径查找问题
            indices = np.random.choice(len(self.data), n_points, replace=False)
            indices_b[b] = torch.tensor(indices, dtype=torch.long)

            batch_parsed = []
            for j, idx in enumerate(indices):
                line = self.data[idx]
                edges, query, path = self._parse_line(line)
                batch_parsed.append((edges, query, path))

                # 构造 xs: [edges (2*d*l), query (2)]
                xs_data = []
                for edge in edges:
                    xs_data.extend(edge)
                xs_data.extend(query)

                xs_b[b, j] = torch.tensor(xs_data, dtype=torch.long)

            parsed_data_b.append(batch_parsed)

        # 存储索引和解析数据供 get_solutions() 使用
        self._last_indices = indices_b
        self._last_parsed_data = parsed_data_b

        return xs_b

    def get_solutions(self, xs):
        """
        根据问题（xs）返回对应的路径（ys）

        Args:
            xs: [B, n_points, edge_len] 问题部分（边列表 + 查询）

        Returns:
            ys: [B, n_points, path_len] 路径部分
        """
        B, n_points, _ = xs.shape
        ys = torch.zeros(B, n_points, self.path_len, dtype=torch.long)

        # 使用存储的解析数据直接获取路径
        if self._last_parsed_data is None:
            raise RuntimeError(
                "get_solutions() 必须在 sample_xs() 之后调用！"
                "请确保先调用 sample_xs() 生成问题数据。"
            )

        # 验证维度匹配
        if len(self._last_parsed_data) != B:
            raise RuntimeError(
                f"Batch size 不匹配！"
                f"期望: {B}, 实际: {len(self._last_parsed_data)}。"
            )

        # 使用解析数据直接填充路径
        for b in range(B):
            for j in range(n_points):
                _, _, path = self._last_parsed_data[b][j]
                ys[b, j] = torch.tensor(path, dtype=torch.long)

        return ys


def get_pathfinding_data_sampler(data_path, n_dims=None, num_nodes=50, degree=2, path_len=5, **kwargs):
    """
    工厂函数：创建路径查找数据采样器

    Args:
        data_path: 文本文件路径
        n_dims: 保留兼容性
        num_nodes: 图中节点数量
        degree: 每个节点的度数
        path_len: 路径长度
        **kwargs: 其他参数

    Returns:
        PathfindingDataSampler 实例
    """
    return PathfindingDataSampler(
        data_path=data_path,
        n_dims=n_dims,
        num_nodes=num_nodes,
        degree=degree,
        path_len=path_len,
        **kwargs
    )
