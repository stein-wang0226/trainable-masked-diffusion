"""
数独数据采样器（基于 diffusion-vs-ar 风格）
===========================================

数据格式：
- 每个点（Point）是 162 维向量：81位题目（quiz） + 81位答案（solution）
- ICL 序列：n_prompt 个完整的 [题目+答案] 对 + 1 个 [题目 + MASK]
- 使用 Embedding 处理 0-10（0-9 为数字，10 为 MASK）
"""

import torch
import numpy as np
import pandas as pd
import os
from typing import Optional, List


class SudokuDataSampler:
    """
    数独数据采样器
    
    从 CSV 文件读取数独数据，构造 ICL 序列。
    每个点（Point）是 162 维向量：81位题目 + 81位答案
    """
    
    def __init__(self, data_path, n_dims=None, test_mode=False, **kwargs):
        """
        Args:
            data_path: CSV 文件路径，包含 'quizzes' 和 'solutions' 列
            n_dims: 保留兼容性，数独任务固定为 162
            test_mode: 是否使用测试集
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数独数据文件不存在: {data_path}")

        # IMPORTANT: preserve leading zeros in quizzes/solutions
        self.df = pd.read_csv(data_path, dtype={"quizzes": str, "solutions": str})

        # 验证列名
        if 'quizzes' not in self.df.columns or 'solutions' not in self.df.columns:
            raise ValueError(f"CSV文件必须包含 'quizzes' 和 'solutions' 列")

        # 验证数据格式
        self._normalize_and_validate_data()

        # 🆕 存储最后一次采样的索引（用于get_solutions）
        self._last_indices = None

        print(f"[Sudoku Sampler] Loaded {len(self.df)} puzzles from {data_path}")
        print(f"[Sudoku Sampler] 数据格式: xs=[B, n_points, 81] (quiz), ys=[B, n_points, 81] (solution)")
    
    def _normalize_and_validate_data(self):
        """
        Normalize + validate:
        - force string type (already via dtype)
        - strip whitespace
        - zfill(81) to avoid leading-zero loss
        """
        self.df["quizzes"] = self.df["quizzes"].astype(str).str.strip().str.zfill(81)
        self.df["solutions"] = self.df["solutions"].astype(str).str.strip().str.zfill(81)

        for i in range(min(5, len(self.df))):
            quiz = self.df.iloc[i]["quizzes"]
            solution = self.df.iloc[i]["solutions"]
            
            if len(quiz) != 81 or len(solution) != 81:
                raise ValueError(
                    f"数独数据格式错误（索引{i}）: "
                    f"quiz长度={len(quiz)}, solution长度={len(solution)}, 期望81"
                )
            
            if not all(c.isdigit() for c in quiz):
                raise ValueError(f"quiz包含非数字字符（索引{i}）: {quiz}")
            if not all(c.isdigit() for c in solution):
                raise ValueError(f"solution包含非数字字符（索引{i}）: {solution}")
    
    def sample_xs(self, n_points, bsz, n_dims=None, seeds=None):
        """
        采样数独数据，构造 ICL 序列

        Args:
            n_points: 每个batch需要的点数（n_prompt + n_respond）
            bsz: batch size
            n_dims: 保留兼容性，数独任务固定为 81（quiz 部分）
            seeds: 随机种子列表，用于确定性采样

        Returns:
            xs_b: [bsz, n_points, 81] 的张量，每个点包含 quiz（题目部分）
        """
        xs_b = torch.zeros(bsz, n_points, 81, dtype=torch.long)
        # 🆕 存储采样的索引，供 get_solutions() 使用
        indices_b = torch.zeros(bsz, n_points, dtype=torch.long)

        for b in range(bsz):
            if seeds is not None:
                np.random.seed(seeds[b])

            # 随机采样 n_points 个数独
            indices = np.random.choice(len(self.df), n_points, replace=False)
            indices_b[b] = torch.tensor(indices, dtype=torch.long)

            for j, idx in enumerate(indices):
                q_str = self.df.iloc[idx]["quizzes"]

                # 转换为整数列表（quiz 部分）
                quiz = [int(d) for d in q_str]  # [81]
                xs_b[b, j] = torch.tensor(quiz, dtype=torch.long)

        # 🆕 存储索引供 get_solutions() 使用
        self._last_indices = indices_b

        return xs_b
    
    def get_solutions(self, xs):
        """
        根据 quiz（xs）返回对应的 solution（ys）

        Args:
            xs: [B, n_points, 81] quiz 部分

        Returns:
            ys: [B, n_points, 81] solution 部分
        """
        B, n_points, _ = xs.shape
        ys = torch.zeros_like(xs)

        # 🆕 使用存储的索引直接获取solution，避免字符串匹配问题
        if self._last_indices is None:
            raise RuntimeError(
                "get_solutions() 必须在 sample_xs() 之后调用！"
                "请确保先调用 sample_xs() 生成quiz数据。"
            )

        # 验证索引维度匹配
        if self._last_indices.shape != (B, n_points):
            raise RuntimeError(
                f"索引维度不匹配！"
                f"期望: [{B}, {n_points}], 实际: {self._last_indices.shape}。"
                f"可能是sample_xs()和get_solutions()的参数不一致。"
            )

        # 使用索引直接查找solution（100%正确，无string匹配问题）
        for b in range(B):
            for j in range(n_points):
                idx = self._last_indices[b, j].item()
                s_str = self.df.iloc[idx]["solutions"]
                sol = [int(d) for d in s_str]
                ys[b, j] = torch.tensor(sol, dtype=torch.long)

        return ys
    
    def get_solution(self, quiz_vector):
        """
        根据问题向量查找对应的答案（保留兼容性，数独任务中可能不需要）
        
        Args:
            quiz_vector: [162] 的问题向量（前81位是quiz）
        
        Returns:
            solution_vector: [162] 的答案向量
        """
        # 提取 quiz 部分（前81位）
        quiz_digits = quiz_vector[:81].tolist()
        quiz_string = ''.join(str(d) for d in quiz_digits)
        
        # 查找匹配的答案
        try:
            # quizzes/solutions are normalized to zfill(81) strings
            idx = self.df["quizzes"].tolist().index(quiz_string)
            s_str = self.df.iloc[idx]["solutions"]
            sol = [int(d) for d in s_str]
            # 返回完整的 162 维向量
            return torch.tensor(quiz_digits + sol, dtype=torch.long)
        except ValueError:
            # 如果找不到精确匹配，使用随机答案
            print(f"⚠️  警告: 未找到精确匹配的数独答案")
            random_idx = np.random.randint(0, len(self.df))
            s_str = self.df.iloc[random_idx]["solutions"]
            sol = [int(d) for d in s_str]
            return torch.tensor(quiz_digits + sol, dtype=torch.long)


def get_sudoku_data_sampler(data_path, n_dims=None, **kwargs):
    """
    工厂函数：创建数独数据采样器
    
    Args:
        data_path: CSV 文件路径
        n_dims: 保留兼容性，数独任务固定为 162
        **kwargs: 其他参数
    
    Returns:
        SudokuDataSampler 实例
    """
    return SudokuDataSampler(data_path=data_path, n_dims=n_dims, **kwargs)
