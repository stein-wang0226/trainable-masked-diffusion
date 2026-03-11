import math

import torch
import random
import numpy as np

class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims
        self.epoch_seed = 0  # 用于记录当前 epoch 的种子

    def sample_xs(self):
        raise NotImplementedError
        
    def reset_epoch(self):
            """每个epoch开始时调用，确保数据顺序相同"""
            self.epoch_seed += 1  # 增加epoch的种子，每个epoch数据顺序不变
            random.seed(self.epoch_seed)
            np.random.seed(self.epoch_seed)
            torch.manual_seed(self.epoch_seed)
            torch.cuda.manual_seed(self.epoch_seed)
            print(f"[DataSampler] Epoch seed reset: {self.epoch_seed}")

def rand_select_sampler(sampler1, sampler2): # todo 添加随机选择sample
    if random.random() < 0.5:
        return sampler1
    else:
        return sampler2


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "uniform":UniformSampler
        # add
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError



def sample_transformation(eigenvalues, normalize=False): # 根据给定的特征值（eigenvalues）生成一个线性变换矩阵，用于对数据进行线性变换
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t

class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None, standardize=False, unit_norm=False, unit_norm_eps=1e-12):# standardize表示是否标准化
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale
        self.standardize = standardize  # 是否标准化数据
        self.unit_norm = unit_norm  # 是否对每个点做 L2 单位化（消除模长捷径）
        self.unit_norm_eps = unit_norm_eps

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)

        # 标准化数据：零均值单位方差
        # 修复：使用 dim=1（序列维度）而非 dim=0（批次维度），避免批次间信息泄露
        if self.standardize:
            mean = xs_b.mean(dim=1, keepdim=True)  # [b_size, 1, n_dims]
            std = xs_b.std(dim=1, keepdim=True)    # [b_size, 1, n_dims]
            xs_b = (xs_b - mean) / (std + 1e-8)    # 每个样本独立标准化，防止除以零

        if self.scale is not None:
            xs_b = xs_b @ self.scale

        if self.bias is not None:
            xs_b += self.bias

        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0

        # 🆕 归一化：把每个点的向量模长归一为 1（在所有线性变换/截断之后进行）
        # 目的：消除 “按模长排序 (norm sorting)” 这类 AR 可能利用的捷径
        if self.unit_norm:
            norms = torch.linalg.norm(xs_b, dim=-1, keepdim=True)
            xs_b = xs_b / (norms + self.unit_norm_eps)

        return xs_b

# class GaussianSampler(DataSampler):
#     def __init__(self, n_dims, bias=None, scale=None):
#         super().__init__(n_dims)
#         self.bias = bias
#         self.scale = scale
#     def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
#         if seeds is None: #
#             xs_b = torch.randn(b_size, n_points, self.n_dims)
#         else: # 利用seeds
#             xs_b = torch.zeros(b_size, n_points, self.n_dims)
#             generator = torch.Generator()
#             assert len(seeds) == b_size
#             for i, seed in enumerate(seeds):
#                 generator.manual_seed(seed)
#                 xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
#                 # 缩放矩阵
#         if self.scale is not None:
#             xs_b = xs_b @ self.scale
#         if self.bias is not None:
#             xs_b += self.bias
#         if n_dims_truncated is not None: # 截断特征维度 将多余的维度置零
#             xs_b[:, :, n_dims_truncated:] = 0
#         return xs_b


class UniformSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None, unit_norm=False, unit_norm_eps=1e-12):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale
        self.unit_norm = unit_norm
        self.unit_norm_eps = unit_norm_eps

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            a = -1.96 #
            b=   1.96
            xs_b = a + (b - a) * torch.rand(b_size, n_points, self.n_dims)

            # xs_b = torch.rand(b_size, n_points, self.n_dims) # U [a,b]  [-+1.96]
        else: # 利用seed
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                np.random.seed(seed)
                xs_b[i] = torch.rand(n_points, self.n_dims, generator=generator)

        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None: # 截断特征维度 将多余的维度置零
            xs_b[:, :, n_dims_truncated:] = 0

        # 🆕 归一化：把每个点的向量模长归一为 1
        if self.unit_norm:
            norms = torch.linalg.norm(xs_b, dim=-1, keepdim=True)
            xs_b = xs_b / (norms + self.unit_norm_eps)
        return xs_b

