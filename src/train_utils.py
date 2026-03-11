"""
Training Utilities for Prompt-Respond Models
============================================

辅助函数和类：
- 随机种子设置
- 数据种子采样
- 日志记录和可视化
- 训练步骤函数
"""

import os
import json
import random
import tempfile
import shutil
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.profiler import profile, ProfilerActivity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Model Attribute Access Utilities (for DDP compatibility)
# ============================================================

def get_model_attr(model, attr_name, default=None):
    """
    安全地访问模型属性，兼容 DDP 包装。
    
    当模型被 DistributedDataParallel 或 Accelerator 包装时，
    实际模型在 model.module 中。
    
    Args:
        model: 可能是原始模型或 DDP 包装的模型
        attr_name: 要访问的属性名
        default: 如果属性不存在，返回的默认值
    
    Returns:
        属性值
    """
    # 尝试从包装的模型中获取（DDP/Accelerator）
    if hasattr(model, 'module'):
        return getattr(model.module, attr_name, default)
    # 否则直接从模型获取
    return getattr(model, attr_name, default)


# ============================================================
# Random Seed Utilities
# ============================================================

def set_seed(seed):
    """设置随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置PyTorch的确定性模式（可能影响性能，但保证可复现性）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sample_seeds(pool_size=None, bs=None, step=None):
    """从种子池采样，增加随机性"""
    if pool_size is None:
        return None
    seeds = set()
    while len(seeds) < bs:
        seeds.add(random.randint(0, pool_size - 1))
    return list(seeds)


# ============================================================
# Validation and Epoch Seed Generation
# ============================================================

def build_validation_seed_batches(batch_count, batch_size, seed):
    """为固定验证集预先生成 data/task seeds"""
    rng = random.Random(seed)
    batches = []
    for _ in range(batch_count):
        data_seeds = [rng.randint(0, 2**31 - 1) for _ in range(batch_size)]
        task_seeds = [s + 1 for s in data_seeds]
        batches.append({"data_seeds": data_seeds, "task_seeds": task_seeds})
    return batches


def build_epoch_seed_pool(num_epochs, steps_per_epoch, batch_size, base_seed=42, shuffle_between_epochs=True):
    """
    为多epoch训练预生成固定的种子池
    
    Args:
        num_epochs: epoch数量
        steps_per_epoch: 每个epoch的步数
        batch_size: batch大小
        base_seed: 基础随机种子
        shuffle_between_epochs: 是否在epoch之间shuffle数据顺序
    
    Returns:
        List of dicts, 每个dict包含 {"epoch": int, "step_in_epoch": int, "data_seeds": list, "task_seeds": list}
    """
    rng = random.Random(base_seed)
    
    # 首先生成一个epoch的所有种子
    base_epoch_seeds = []
    for step in range(steps_per_epoch):
        data_seeds = [rng.randint(0, 2**31 - 1) for _ in range(batch_size)]
        task_seeds = [s + 1 for s in data_seeds]
        base_epoch_seeds.append({
            "step_in_epoch": step,
            "data_seeds": data_seeds,
            "task_seeds": task_seeds
        })
    
    # 为每个epoch复制并可选地shuffle
    epoch_seed_pool = []
    for epoch in range(num_epochs):
        # 复制基础epoch的种子
        epoch_seeds = [dict(item, epoch=epoch) for item in base_epoch_seeds]
        
        # 如果启用shuffle且不是第一个epoch，则shuffle顺序
        if shuffle_between_epochs and epoch > 0:
            # 使用epoch编号作为shuffle种子，保证可复现
            epoch_rng = random.Random(base_seed + epoch * 1000)
            epoch_rng.shuffle(epoch_seeds)
            # 更新step_in_epoch以反映新顺序
            for new_step, item in enumerate(epoch_seeds):
                item["step_in_epoch"] = new_step
        
        epoch_seed_pool.extend(epoch_seeds)
    
    return epoch_seed_pool


# ============================================================
# Data Shuffling for Non-Sequential ICL
# ============================================================

def shuffle_prompt_respond_pairs(xs, ys, n_prompt, n_respond):
    """
    Shuffle prompt and respond pairs for Non-Sequential ICL mode.
    
    Args:
        xs: [B, n_prompt + n_respond, D]
        ys: [B, n_prompt + n_respond]
        n_prompt: number of prompt pairs
        n_respond: number of respond pairs
    
    Returns:
        xs_shuffled: [B, n_prompt + n_respond, D] - shuffled input
        ys_shuffled: [B, n_prompt + n_respond] - shuffled output
        respond_position_mask: [B, n_prompt + n_respond] - boolean mask marking respond positions
    """
    B, total_points, D = xs.shape
    assert total_points == n_prompt + n_respond, f"Total points {total_points} != {n_prompt} + {n_respond}"
    
    # 创建打乱的 xs, ys 和 respond_position_mask
    xs_shuffled = torch.zeros_like(xs)
    ys_shuffled = torch.zeros_like(ys)
    respond_position_mask = torch.zeros(B, total_points, dtype=torch.bool, device=xs.device)
    
    # 为每个 batch 独立打乱
    for b in range(B):
        # 创建索引：前 n_prompt 个是 prompt pairs，后 n_respond 个是 respond pairs
        indices = torch.arange(total_points, device=xs.device)
        
        # 打乱索引
        shuffled_indices = indices[torch.randperm(total_points, device=xs.device)]
        
        # 使用打乱后的索引重排 xs 和ys
        xs_shuffled[b] = xs[b, shuffled_indices]
        ys_shuffled[b] = ys[b, shuffled_indices]
        
        # 标记哪些位置是 respond pairs（原来在 n_prompt 之后的位置）
        # 找到打乱后的索引中，哪些来自原始的 respond 部分
        respond_position_mask[b] = shuffled_indices >= n_prompt
    
    return xs_shuffled, ys_shuffled, respond_position_mask


def generate_fixed_permutation(n_prompt, n_respond, seed=42):
    """
    Generate a fixed permutation for the prompt part only.
    
    重要：只为 prompt 部分生成排列，respond 部分保持原始顺序
    
    Args:
        n_prompt: number of prompt pairs
        n_respond: number of respond pairs (not used, kept for API compatibility)
        seed: random seed for reproducibility
    
    Returns:
        fixed_permutation: [n_prompt] tensor - the fixed permutation indices for prompt part only
    """
    # 只为 prompt 部分生成排列
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    indices = torch.arange(n_prompt)
    fixed_permutation = indices[torch.randperm(n_prompt, generator=generator)]
    
    return fixed_permutation


def apply_fixed_permutation(xs, ys, n_prompt, n_respond, fixed_permutation_x, fixed_permutation_y=None):
    """
    Apply fixed permutation to PROMPT part only, keep RESPOND part unchanged.
    
    重要：
    - 只打乱 Prompt 部分（前 n_prompt 个 pairs）
    - Respond 部分（后 n_respond 个 pairs）保持正确的 (x,y) 对应关系，不打乱
    
    Sequence structure:
    [Prompt: x₀,y₀, x₁,y₁, ..., x_{p-1},y_{p-1}] | [Respond: xₚ,?, x_{p+1},?, ..., x_{p+r-1},?]
     ↑ 这部分打乱                                    ↑ 这部分保持不变
    
    Two modes:
    1. fixed_permutation_y=None (只打乱 prompt 的 x): 
       - 打乱 prompt 部分的 x（必须）
       - prompt 部分的 y 保持不动
       - respond 部分完全不动
    2. fixed_permutation_y!=None (同时打乱 prompt 的 x 和 y):
       - 打乱 prompt 部分的 x（必须）
       - 打乱 prompt 部分的 y（可选）
       - respond 部分完全不动
    
    Args:
        xs: [B, n_prompt + n_respond, D]
        ys: [B, n_prompt + n_respond]
        n_prompt: number of prompt pairs
        n_respond: number of respond pairs
        fixed_permutation_x: [n_prompt] - fixed permutation for x in PROMPT part only
        fixed_permutation_y: [n_prompt] or None - fixed permutation for y in PROMPT part only
                            If None, y in prompt part is not permuted
    
    Returns:
        xs_permuted: [B, n_prompt + n_respond, D] - permuted input
        ys_permuted: [B, n_prompt + n_respond] - permuted output
        respond_position_mask: [B, n_prompt + n_respond] - boolean mask marking respond positions
    """
    B, total_points, D = xs.shape
    assert total_points == n_prompt + n_respond, f"Total points {total_points} != {n_prompt} + {n_respond}"
    assert len(fixed_permutation_x) == n_prompt, f"Permutation length {len(fixed_permutation_x)} != {n_prompt}"
    
    device = xs.device
    fixed_permutation_x = fixed_permutation_x.to(device)
    if fixed_permutation_y is not None:
        assert len(fixed_permutation_y) == n_prompt, f"Y Permutation length {len(fixed_permutation_y)} != {n_prompt}"
        fixed_permutation_y = fixed_permutation_y.to(device)
    
    # 创建打乱的 xs, ys 和 respond_position_mask
    xs_permuted = torch.zeros_like(xs)
    ys_permuted = torch.zeros_like(ys)
    respond_position_mask = torch.zeros(B, total_points, dtype=torch.bool, device=device)
    
    # 为所有 batch 应用相同的固定排列
    for b in range(B):
        # === Prompt 部分（前 n_prompt 个）：打乱 ===
        
        # 打乱 prompt 的 x（必须）
        xs_permuted[b, :n_prompt] = xs[b, fixed_permutation_x]
        
        # 打乱 prompt 的 y（可选）
        if fixed_permutation_y is not None:
            ys_permuted[b, :n_prompt] = ys[b, fixed_permutation_y]
        else:
            # prompt 的 y 不打乱，保持原顺序
            ys_permuted[b, :n_prompt] = ys[b, :n_prompt]
        
        # === Respond 部分（后 n_respond 个）：完全不动 ===
        xs_permuted[b, n_prompt:] = xs[b, n_prompt:]
        ys_permuted[b, n_prompt:] = ys[b, n_prompt:]
        
        # 标记 respond 位置（respond 部分从 n_prompt 开始）
        respond_position_mask[b, n_prompt:] = True
    
    return xs_permuted, ys_permuted, respond_position_mask


def apply_random_permutation(xs, ys, n_prompt, n_respond, permute_y=False):
    """
    Apply RANDOM permutation to PROMPT part only, keep RESPOND part unchanged.
    Each batch gets a DIFFERENT random permutation (harder task than fixed permutation).
    
    重要：
    - 只打乱 Prompt 部分（前 n_prompt 个 pairs）
    - Respond 部分（后 n_respond 个 pairs）保持正确的 (x,y) 对应关系，不打乱
    - 每个 batch 都使用不同的随机排列（比 fixed_permutation 更难）
    
    Sequence structure:
    [Prompt: x₀,y₀, x₁,y₁, ..., x_{p-1},y_{p-1}] | [Respond: xₚ,?, x_{p+1},?, ..., x_{p+r-1},?]
     ↑ 每个batch随机打乱                           ↑ 这部分保持不变
    
    Two modes:
    1. permute_y=False (只打乱 prompt 的 x): 
       - 打乱 prompt 部分的 x（每个batch不同排列）
       - prompt 部分的 y 保持不动
       - respond 部分完全不动
    2. permute_y=True (同时打乱 prompt 的 x 和 y):
       - 打乱 prompt 部分的 x（每个batch不同排列）
       - 打乱 prompt 部分的 y（每个batch不同排列）
       - respond 部分完全不动
    
    Args:
        xs: [B, n_prompt + n_respond, D]
        ys: [B, n_prompt + n_respond]
        n_prompt: number of prompt pairs
        n_respond: number of respond pairs
        permute_y: if True, also permute y in prompt part (harder mode)
    
    Returns:
        xs_permuted: [B, n_prompt + n_respond, D] - permuted input
        ys_permuted: [B, n_prompt + n_respond] - permuted output
        respond_position_mask: [B, n_prompt + n_respond] - boolean mask marking respond positions
    """
    B, total_points, D = xs.shape
    assert total_points == n_prompt + n_respond, f"Total points {total_points} != {n_prompt} + {n_respond}"
    
    device = xs.device
    
    # 创建打乱的 xs, ys 和 respond_position_mask
    xs_permuted = torch.zeros_like(xs)
    ys_permuted = torch.zeros_like(ys)
    respond_position_mask = torch.zeros(B, total_points, dtype=torch.bool, device=device)
    
    # 为每个 batch 生成不同的随机排列（这是与 fixed_permutation 的关键区别）
    for b in range(B):
        # === Prompt 部分（前 n_prompt 个）：随机打乱 ===
        
        # 生成 x 的随机排列（每个batch不同）
        perm_x = torch.randperm(n_prompt, device=device)
        xs_permuted[b, :n_prompt] = xs[b, perm_x]
        
        # 打乱 prompt 的 y（可选）
        if permute_y:
            # 生成 y 的随机排列（每个batch不同）
            perm_y = torch.randperm(n_prompt, device=device)
            ys_permuted[b, :n_prompt] = ys[b, perm_y]
        else:
            # prompt 的 y 不打乱，保持原顺序
            ys_permuted[b, :n_prompt] = ys[b, :n_prompt]
        
        # === Respond 部分（后 n_respond 个）：完全不动 ===
        xs_permuted[b, n_prompt:] = xs[b, n_prompt:]
        ys_permuted[b, n_prompt:] = ys[b, n_prompt:]
        
        # 标记 respond 位置（respond 部分从 n_prompt 开始）
        respond_position_mask[b, n_prompt:] = True
    
    return xs_permuted, ys_permuted, respond_position_mask


def generate_permutation_pool(n_prompt, pool_size=20, seed=42):
    """
    Generate a fixed pool of permutations for the prompt part only.
    
    生成固定大小的排列池，用于 pool_permutation 模式。
    每次从池中随机采样一个排列（训练和验证都从同一个池中采样）。
    
    Args:
        n_prompt: number of prompt pairs
        pool_size: size of the permutation pool (default: 20)
        seed: random seed for reproducibility
    
    Returns:
        permutation_pool: [pool_size, n_prompt] tensor - pool of fixed permutations
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    permutation_pool = []
    for i in range(pool_size):
        # 为每个排列使用不同的种子（seed + i），确保池中每个排列都不同
        perm_generator = torch.Generator()
        perm_generator.manual_seed(seed + i)
        indices = torch.arange(n_prompt)
        perm = indices[torch.randperm(n_prompt, generator=perm_generator)]
        permutation_pool.append(perm)
    
    return torch.stack(permutation_pool)  # [pool_size, n_prompt]


def apply_pool_permutation(xs, ys, n_prompt, n_respond, permutation_pool_x, permutation_pool_y=None):
    """
    Apply permutation from a fixed pool to PROMPT part only, keep RESPOND part unchanged.
    Each batch randomly samples a permutation from the pool.
    
    重要：
    - 只打乱 Prompt 部分（前 n_prompt 个 pairs）
    - Respond 部分（后 n_respond 个 pairs）保持正确的 (x,y) 对应关系，不打乱
    - 每个 batch 从固定的排列池中随机采样一个排列（介于 fixed 和 random 之间）
    
    Sequence structure:
    [Prompt: x₀,y₀, x₁,y₁, ..., x_{p-1},y_{p-1}] | [Respond: xₚ,?, x_{p+1},?, ..., x_{p+r-1},?]
     ↑ 从池中随机采样排列打乱                      ↑ 这部分保持不变
    
    Two modes:
    1. permutation_pool_y=None (只打乱 prompt 的 x): 
       - 打乱 prompt 部分的 x（从池中随机采样）
       - prompt 部分的 y 保持不动
       - respond 部分完全不动
    2. permutation_pool_y!=None (同时打乱 prompt 的 x 和 y):
       - 打乱 prompt 部分的 x（从池中随机采样）
       - 打乱 prompt 部分的 y（从池中随机采样）
       - respond 部分完全不动
    
    Args:
        xs: [B, n_prompt + n_respond, D]
        ys: [B, n_prompt + n_respond]
        n_prompt: number of prompt pairs
        n_respond: number of respond pairs
        permutation_pool_x: [pool_size, n_prompt] - pool of fixed permutations for x
        permutation_pool_y: [pool_size, n_prompt] or None - pool of fixed permutations for y
                           If None, y in prompt part is not permuted
    
    Returns:
        xs_permuted: [B, n_prompt + n_respond, D] - permuted input
        ys_permuted: [B, n_prompt + n_respond] - permuted output
        respond_position_mask: [B, n_prompt + n_respond] - boolean mask marking respond positions
    """
    B, total_points, D = xs.shape
    assert total_points == n_prompt + n_respond, f"Total points {total_points} != {n_prompt} + {n_respond}"
    assert len(permutation_pool_x) > 0, "Permutation pool must not be empty"
    assert len(permutation_pool_x[0]) == n_prompt, f"Permutation length {len(permutation_pool_x[0])} != {n_prompt}"
    
    device = xs.device
    pool_size = len(permutation_pool_x)
    permutation_pool_x = permutation_pool_x.to(device)
    if permutation_pool_y is not None:
        assert len(permutation_pool_y) == pool_size, f"Y pool size {len(permutation_pool_y)} != X pool size {pool_size}"
        assert len(permutation_pool_y[0]) == n_prompt, f"Y Permutation length {len(permutation_pool_y[0])} != {n_prompt}"
        permutation_pool_y = permutation_pool_y.to(device)
    
    # 创建打乱的 xs, ys 和 respond_position_mask
    xs_permuted = torch.zeros_like(xs)
    ys_permuted = torch.zeros_like(ys)
    respond_position_mask = torch.zeros(B, total_points, dtype=torch.bool, device=device)
    
    # 为每个 batch 从池中随机采样一个排列
    for b in range(B):
        # === Prompt 部分（前 n_prompt 个）：从池中采样排列打乱 ===
        
        # 从池中随机采样一个排列（每个batch独立采样）
        pool_idx_x = torch.randint(0, pool_size, (1,), device=device).item()
        fixed_permutation_x = permutation_pool_x[pool_idx_x]
        xs_permuted[b, :n_prompt] = xs[b, fixed_permutation_x]
        
        # 打乱 prompt 的 y（可选）
        if permutation_pool_y is not None:
            # 从 y 的池中随机采样一个排列（每个batch独立采样）
            pool_idx_y = torch.randint(0, pool_size, (1,), device=device).item()
            fixed_permutation_y = permutation_pool_y[pool_idx_y]
            ys_permuted[b, :n_prompt] = ys[b, fixed_permutation_y]
        else:
            # prompt 的 y 不打乱，保持原顺序
            ys_permuted[b, :n_prompt] = ys[b, :n_prompt]
        
        # === Respond 部分（后 n_respond 个）：完全不动 ===
        xs_permuted[b, n_prompt:] = xs[b, n_prompt:]
        ys_permuted[b, n_prompt:] = ys[b, n_prompt:]
        
        # 标记 respond 位置（respond 部分从 n_prompt 开始）
        respond_position_mask[b, n_prompt:] = True
    
    return xs_permuted, ys_permuted, respond_position_mask


# ============================================================
# Training Step Functions
# ============================================================

def train_step_ar(model, xs, ys, optimizer, loss_func, respond_position_mask=None, accelerator=None):
    """
    训练步骤 - AR Model
    
    Args:
        model: AR Transformer model
        xs: [B, n_prompt + n_respond, D]
        ys: [B, n_prompt + n_respond]
        optimizer: optimizer
        loss_func: loss function
        respond_position_mask: [B, n_prompt + n_respond] boolean mask marking respond positions
                               (for Non-Sequential ICL mode)
    
    Returns:
        loss: scalar
        output: predictions
    """
    optimizer.zero_grad()
    
    # Forward pass: AR模型只预测respond部分
    pred, loss_mask = model(xs, ys, respond_position_mask=respond_position_mask)
    
    # 🔧 从 pred 的形状推断 actual_n_respond（pred 的形状是 [B, actual_n_respond]）
    actual_n_respond = pred.shape[1]
    
    # 🔧 安全地获取 n_prompt（兼容 DDP 包装）
    n_prompt = get_model_attr(model, 'n_prompt')
    if n_prompt is None:
        raise AttributeError(f"Model does not have 'n_prompt' attribute. Model type: {type(model)}")
    
    # 🔧 使用辅助函数：获取respond部分的真实值
    respond_true = extract_respond_values(ys, n_prompt, respond_position_mask, actual_n_respond=actual_n_respond)
    
    # 计算loss（loss_mask应该全为1，因为只预测respond部分）
    loss = loss_func(pred * loss_mask, respond_true * loss_mask)
    
    # 🆕 分布式训练：使用 accelerator.backward()
    if accelerator is not None:
        accelerator.backward(loss)
    else:
        loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # AR 模型没有 masked_indices（对所有 respond 位置计算 loss）
    return loss.item(), pred.detach(), None


def train_step_mdm(model, xs, ys, optimizer, loss_func, respond_position_mask=None, accelerator=None):
    """
    训练步骤 - MDM Model (LLaDA, Dream)
    
    Args:
        model: Diffusion model
        xs: [B, n_prompt + n_respond, D]
        ys: [B, n_prompt + n_respond]
        optimizer: optimizer
        loss_func: loss function (not used, model returns loss)
        respond_position_mask: [B, n_prompt + n_respond] boolean mask marking respond positions
                               (for Non-Sequential ICL mode)
    
    Returns:
        loss: scalar
        output: predictions
        respond_masked_indices: [B, actual_n_respond] boolean tensor (only for MDM models, None for AR models)
    """
    optimizer.zero_grad()
    
    # Forward pass (returns loss, pred, t, respond_masked_indices for MDM models)
    forward_result = model(xs, ys, train_mode=True, respond_position_mask=respond_position_mask)
    if len(forward_result) == 4:
        # MDM model: returns (loss, pred, t, respond_masked_indices)
        loss, pred, t_scalar, respond_masked_indices = forward_result
    else:
        # AR model: returns (loss, pred) - no masked_indices
        loss, pred = forward_result
        respond_masked_indices = None
    
    # 🆕 分布式训练：使用 accelerator.backward()
    if accelerator is not None:
        accelerator.backward(loss)
    else:
        loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item(), pred.detach(), respond_masked_indices


# ============================================================
# FLOPs Profiling Functions
# ============================================================

def profile_model_flops(model, sample_batch, optimizer, loss_func, is_sudoku_task=False, 
                        is_pathfinding_task=False, respond_position_mask=None, accelerator=None, config=None):
    """
    使用 torch.profiler 测量模型单步训练的 FLOPs
    
    Args:
        model: 模型实例（可能是 DDP 包装的）
        sample_batch: 样本批次，包含 (xs, ys) 或类似结构
        optimizer: 优化器实例
        loss_func: 损失函数
        is_sudoku_task: 是否为数独任务
        is_pathfinding_task: 是否为路径查找任务
        respond_position_mask: respond位置mask（可选）
        accelerator: Accelerator实例（分布式训练，可选）
        config: 配置字典（用于获取推理步数等参数，可选）
    
    Returns:
        flops_dict: 包含以下字段的字典
            - train_step_flops: 单次训练步总 FLOPs（前向+反向）
            - forward_flops: 单次前向传播 FLOPs（约占总FLOPs的1/3）
            - single_inference_flops: 单次推理 FLOPs（根据模型类型计算）
            - model_family: 模型家族（AR 或 MDM/Diff）
    """
    model.train()
    
    # 提取样本数据
    if isinstance(sample_batch, tuple):
        xs, ys = sample_batch
    else:
        xs = sample_batch['xs']
        ys = sample_batch['ys']
    
    # 获取模型家族类型
    model_family = get_model_attr(model, 'family', 'unknown')
    
    # 如果未传入 is_pathfinding_task，尝试从 model_family 推断
    if not is_pathfinding_task:
        is_pathfinding_task = (model_family and model_family.startswith('pathfinding_'))
    
    # 判断是否为 AR 模型（排除 pathfinding 和 sudoku 任务，因为它们使用不同的接口）
    is_ar_model = (model_family in ['llama', 'llama2', 'llama3', 'gpt2', 'gptj', 'qwen', 'qwen2', 'qwen2.5']
                   or ('ar' in model_family.lower() and not is_pathfinding_task))
    
    # 获取 n_respond（用于AR模型的推理FLOPs计算）
    n_respond = get_model_attr(model, 'n_respond', None)
    if n_respond is None and config is not None:
        n_respond = config.get('model', {}).get('n_respond', None)
    
    # 获取采样步数（用于Diff模型的推理FLOPs计算）
    sampling_steps = 1  # 默认单步
    if config is not None:
        # 首先尝试从 inference 配置中获取
        inference_config = config.get('inference', {})
        sampling_steps = inference_config.get('steps', inference_config.get('inference_steps', 1))
        # 如果 inference 配置中没有，尝试从 model 配置中获取（数独/路径查找任务）
        if sampling_steps == 1:
            model_config = config.get('model', {})
            sampling_steps = model_config.get('inference_steps', 1)
        # 也尝试从模型中获取
        if hasattr(model, 'module') and hasattr(model.module, 'inference_steps'):
            sampling_steps = model.module.inference_steps
        elif hasattr(model, 'inference_steps'):
            sampling_steps = model.inference_steps
    
    # 🆕 Pathfinding 任务：如果使用多步推理，需要根据 use_multistep_inference 调整
    if is_pathfinding_task:
        # 检查是否使用多步推理
        use_multistep = False
        if hasattr(model, 'module') and hasattr(model.module, 'use_multistep_inference'):
            use_multistep = model.module.use_multistep_inference
        elif hasattr(model, 'use_multistep_inference'):
            use_multistep = model.use_multistep_inference
        
        if use_multistep:
            # 多步推理：使用 inference_steps
            if hasattr(model, 'module') and hasattr(model.module, 'inference_steps'):
                sampling_steps = model.module.inference_steps
            elif hasattr(model, 'inference_steps'):
                sampling_steps = model.inference_steps
        else:
            # 单步推理：只需要 1 次前向传播
            sampling_steps = 1
    
    # 使用 profiler 测量 FLOPs
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_flops=True,
        record_shapes=False
    ) as prof:
        # 执行一次完整的训练步骤
        optimizer.zero_grad(set_to_none=True)
        
        if is_sudoku_task or is_pathfinding_task:
            # 数独/路径查找任务：直接调用模型的forward
            forward_result = model(xs, ys, train_mode=True, respond_position_mask=respond_position_mask)
            if isinstance(forward_result, tuple):
                loss = forward_result[0]
            else:
                loss = forward_result
        else:
            # 标准任务：根据模型类型选择训练步骤
            if is_ar_model:
                pred, loss_mask = model(xs, ys, respond_position_mask=respond_position_mask)
                n_prompt = get_model_attr(model, 'n_prompt')
                respond_true = extract_respond_values(ys, n_prompt, respond_position_mask, actual_n_respond=pred.shape[1])
                loss = loss_func(pred * loss_mask, respond_true * loss_mask)
            else:
                # MDM模型
                forward_result = model(xs, ys, train_mode=True, respond_position_mask=respond_position_mask)
                if isinstance(forward_result, tuple):
                    loss = forward_result[0]
                else:
                    loss = forward_result
        
        # 反向传播
        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()
        
        # 只执行一次，用于测量（不实际更新参数）
        # 注意：这里我们不调用 optimizer.step()，因为只是测量
        
    # 提取 FLOPs 数据
    key_averages = prof.key_averages()
    
    # 计算总 FLOPs（前向+反向）
    total_flops = 0
    forward_flops = 0
    
    for event in key_averages:
        if event.key != 'cuda_time_total':
            # 累加所有操作的 FLOPs
            if hasattr(event, 'flops') and event.flops > 0:
                total_flops += event.flops
    
    # 估算前向传播 FLOPs（约为总FLOPs的1/3，这是经验值）
    # 反向传播大约是前向的2倍
    forward_flops = total_flops / 3
    
    # 计算单次推理 FLOPs
    if is_ar_model:
        # AR模型：一次推理需要 n_respond 次前向传播（自回归）
        single_inference_flops = forward_flops * (n_respond if n_respond else 1)
    else:
        # Diff/MDM模型：一次推理需要 sampling_steps 次前向传播
        single_inference_flops = forward_flops * max(sampling_steps, 1)
    
    # 清理梯度（因为我们没有调用 optimizer.step()）
    optimizer.zero_grad(set_to_none=True)
    
    return {
        'train_step_flops': total_flops,
        'forward_flops': forward_flops,
        'single_inference_flops': single_inference_flops,
        'model_family': model_family,
        'is_ar_model': is_ar_model,
        'n_respond': n_respond,
        'sampling_steps': sampling_steps,
    }


# ============================================================
# MSE Logger Class
# ============================================================

def extract_respond_values(ys, n_prompt, respond_position_mask=None, actual_n_respond=None):
    """
    从完整序列中提取 respond 部分的值
    
    Args:
        ys: [B, total_points] 完整序列
        n_prompt: prompt 点数
        respond_position_mask: [B, total_points] boolean tensor 或 None
        actual_n_respond: 实际 respond 点数（如果为 None，则从 total_points 推断）
    
    Returns:
        respond_values: [B, actual_n_respond] respond 部分的值
    """
    if respond_position_mask is not None:
        # Non-sequential: 使用 mask 提取真实的 respond 值
        respond_values = torch.stack([
            ys[i, respond_position_mask[i]]
            for i in range(ys.shape[0])
        ])
    else:
        # Sequential: 直接切片
        if actual_n_respond is None:
            # 如果没有提供 actual_n_respond，从 total_points 推断
            actual_n_respond = ys.shape[1] - n_prompt
        respond_start = n_prompt
        respond_values = ys[:, respond_start:respond_start+actual_n_respond]
    
    return respond_values


# ============================================================
# MSE Logger Class
# ============================================================

class LocalMSELogger:
    """本地 MSE 记录器 + 可视化器"""

    def __init__(self, train_log_path, validation_log_path, plot_dir, log_interval, model_type, is_main_process=True):
        """
        Args:
            train_log_path: 训练日志路径
            validation_log_path: 验证日志路径
            plot_dir: 绘图输出目录
            log_interval: 日志刷新间隔
            model_type: 模型类型
            is_main_process: 是否为主进程（分布式训练时，只有主进程写入日志）
        """
        self.train_log_path = train_log_path
        self.validation_log_path = validation_log_path
        self.plot_dir = plot_dir
        self.log_interval = max(1, log_interval)
        self.model_type = model_type  # 🆕 保存模型类型用于导出阈值
        self.is_main_process = is_main_process  # 🆕 主进程标志
        
        # 🆕 只有主进程才创建目录和初始化文件
        if self.is_main_process:
            os.makedirs(os.path.dirname(train_log_path), exist_ok=True)
            os.makedirs(os.path.dirname(validation_log_path), exist_ok=True)
            os.makedirs(plot_dir, exist_ok=True)

        self.train_buffer = []
        self.train_count = 0
        self.train_accumulated_avg = 0.0
        self.validation_count = 0
        self.validation_accumulated_avg = 0.0
        
        # 🆕 性能-算力对齐：阈值跟踪
        self.flops_to_threshold = None  # 达到阈值所需的 FLOPs
        self.threshold_reached_step = None  # 达到阈值的步数
        self.threshold_value = None  # 实际达到的阈值值
        
        # 🆕 断点续传：记录已存在的步数，避免重复记录
        self.existing_train_steps = set()
        self.existing_validation_steps = set()  # 单步验证的步数集合
        self.existing_validation_keys = set()  # 🆕 多步验证的 (step, inference_steps) 集合
        
        # 🆕 如果日志文件已存在，加载已记录的步数（用于断点续传时避免重复）
        if self.is_main_process:
            if os.path.exists(self.train_log_path):
                existing_train_records = self._load_json_lines(self.train_log_path)
                self.existing_train_steps = {r.get('step', -1) for r in existing_train_records if 'step' in r}
                if self.existing_train_steps:
                    max_step = max(self.existing_train_steps)
                    print(f"[Resume] 检测到已有训练日志，最大步数: {max_step}，已记录 {len(self.existing_train_steps)} 个步数")
            
            if os.path.exists(self.validation_log_path):
                existing_val_records = self._load_json_lines(self.validation_log_path)
                # 🆕 同时加载单步和多步验证的记录
                for r in existing_val_records:
                    step = r.get('step', -1)
                    inference_steps = r.get('inference_steps', None)
                    if step >= 0:
                        if inference_steps is not None:
                            # 多步验证：使用 (step, inference_steps) 作为唯一标识
                            self.existing_validation_keys.add((step, inference_steps))
                        else:
                            # 单步验证：使用 step 作为唯一标识
                            self.existing_validation_steps.add(step)
                if self.existing_validation_steps or self.existing_validation_keys:
                    max_step = max(
                        [s for s in self.existing_validation_steps if s >= 0] + 
                        [s for s, _ in self.existing_validation_keys if s >= 0],
                        default=-1
                    )
                    total_count = len(self.existing_validation_steps) + len(self.existing_validation_keys)
                    print(f"[Resume] 检测到已有验证日志，最大步数: {max_step}，已记录 {total_count} 个记录（单步: {len(self.existing_validation_steps)}, 多步: {len(self.existing_validation_keys)}）")
        self.threshold_metric = None  # 阈值指标类型（'cell_accuracy' 或 'mse'）

    def _convert_to_json_serializable(self, value):
        """
        将值转换为 JSON 可序列化的类型
        
        处理 PyTorch Tensor、numpy 数组等类型
        
        Args:
            value: 要转换的值
            
        Returns:
            JSON 可序列化的值
        """
        if isinstance(value, torch.Tensor):
            # 如果是标量 Tensor，转换为 Python 原生类型
            if value.numel() == 1:
                return value.item()
            # 如果是多维 Tensor，转换为列表
            return value.detach().cpu().tolist()
        elif isinstance(value, np.ndarray):
            # numpy 数组转换为列表
            return value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            # numpy 标量类型转换为 Python 原生类型
            return value.item()
        elif isinstance(value, (list, tuple)):
            # 递归处理列表和元组
            return [self._convert_to_json_serializable(item) for item in value]
        elif isinstance(value, dict):
            # 递归处理字典
            return {k: self._convert_to_json_serializable(v) for k, v in value.items()}
        else:
            # 其他类型直接返回（如 int, float, str, bool, None）
            return value

    def record_train(self, step, raw_mse, dims, respond_points, prompt_points, model_type, cell_accuracy=None, sudoku_accuracy=None, node_accuracy=None, path_accuracy=None, cumulative_flops=None, single_inference_flops=None, train_loss=None):
        """
        记录训练结果

        🆕 断点续传保护：如果该 step 已存在，跳过记录（避免重复）

        Args:
            step: 训练步数
            raw_mse: 原始 MSE
            dims: 特征维度
            respond_points: respond 点数
            prompt_points: prompt 点数
            model_type: 模型类型
            cell_accuracy: 格子准确率（数独任务，可选）
            sudoku_accuracy: 数独准确率（数独任务，可选）
            node_accuracy: 节点准确率（路径查找任务，可选）
            path_accuracy: 路径准确率（路径查找任务，可选）
            cumulative_flops: 累计算力（FLOPs，可选）
            single_inference_flops: 单次推理FLOPs（可选）
            train_loss: 训练 loss（可选，🆕 新增）
        """
        # 🆕 断点续传保护：检查该 step 是否已记录
        if step in self.existing_train_steps:
            return  # 已存在，跳过记录（避免重复）
        # 🆕 只有主进程才记录训练日志，避免多进程竞争导致数据覆盖
        if not self.is_main_process:
            return
        
        entry = {
            "step": step,
            "model_type": model_type,
            "dims": dims,
            "respond_points": respond_points,
            "prompt_points": prompt_points,
        }
        # 🆕 数独任务：只记录准确率（不记录 MSE）
        if cell_accuracy is not None:
            entry["cell_accuracy"] = self._convert_to_json_serializable(cell_accuracy)
        if sudoku_accuracy is not None:
            entry["sudoku_accuracy"] = self._convert_to_json_serializable(sudoku_accuracy)
        # 🆕 Pathfinding 任务：只记录准确率（不记录 MSE）
        if node_accuracy is not None:
            entry["node_accuracy"] = self._convert_to_json_serializable(node_accuracy)
        if path_accuracy is not None:
            entry["path_accuracy"] = self._convert_to_json_serializable(path_accuracy)
        # 非数独任务：记录 MSE
        if raw_mse is not None:
            # 🔧 确保 raw_mse 是 Python 原生类型
            raw_mse = self._convert_to_json_serializable(raw_mse)
            self.train_count += 1
            self.train_accumulated_avg += (raw_mse - self.train_accumulated_avg) / self.train_count
            entry["raw_mse"] = raw_mse
            entry["accumulated_avg_mse"] = self.train_accumulated_avg
        # 🆕 记录累计算力
        if cumulative_flops is not None:
            entry["cumulative_flops"] = self._convert_to_json_serializable(cumulative_flops)
        # 🆕 记录单次推理FLOPs
        if single_inference_flops is not None:
            entry["single_inference_flops"] = self._convert_to_json_serializable(single_inference_flops)
        # 🆕 记录训练 loss
        if train_loss is not None:
            entry["train_loss"] = self._convert_to_json_serializable(train_loss)
        
        self.train_buffer.append(entry)
        # 🆕 将当前 step 添加到已记录集合（在添加到缓冲区后）
        self.existing_train_steps.add(step)
        
        if len(self.train_buffer) >= self.log_interval:
            self.flush_train()

    def flush_train(self, force=False):
        # 🆕 只有主进程才写入日志文件，避免多进程竞争
        if not self.is_main_process:
            self.train_buffer.clear()  # 非主进程清空缓冲区但不写入
            return
        
        if not self.train_buffer and not force:
            return
        if not self.train_buffer:
            return
        
        # 🆕 确保目录存在（主进程）
        os.makedirs(os.path.dirname(self.train_log_path), exist_ok=True)
        
        # 🔧 优化：使用追加模式写入，但捕获可能的 OSS 写入错误
        # 如果路径在 OSS 上，写入可能会很慢，但我们已经确保只有主进程写入
        try:
            with open(self.train_log_path, "a") as f:
                for entry in self.train_buffer:
                    # 🔧 确保 entry 中的所有值都是 JSON 可序列化的（双重保护）
                    entry_serializable = self._convert_to_json_serializable(entry)
                    f.write(json.dumps(entry_serializable) + "\n")
                f.flush()  # 🔧 强制刷新，确保数据写入（虽然可能慢，但能保证数据不丢失）
        except (IOError, OSError) as e:
            # 🔧 如果写入失败（如 OSS 网络问题），打印警告但不中断训练
            print(f"⚠️  警告: 写入训练日志失败: {e}")
            print(f"   日志路径: {self.train_log_path}")
        finally:
            self.train_buffer.clear()

    def _check_and_record_threshold(self, step, raw_mse, cell_accuracy, dims, cumulative_flops, threshold_cell_accuracy=0.9):
        """
        🆕 性能-算力对齐：检测是否达到阈值并记录所需的 FLOPs
        
        Args:
            step: 训练步数
            raw_mse: 原始 MSE（线性回归任务）
            cell_accuracy: 格子准确率（数独任务）
            dims: 特征维度（用于线性回归任务阈值调整）
            cumulative_flops: 累计算力
            threshold_cell_accuracy: 数独任务阈值（默认0.9）
        
        Returns:
            bool: 是否首次达到阈值
        """
        # 如果已经达到阈值，不再重复记录
        if self.flops_to_threshold is not None:
            return False
        
        # 如果累计算力未测量，无法记录阈值
        if cumulative_flops is None or cumulative_flops <= 0:
            return False
        
        # 数独任务：检测 cell_accuracy >= threshold_cell_accuracy
        if cell_accuracy is not None:
            if cell_accuracy >= threshold_cell_accuracy:
                self.flops_to_threshold = cumulative_flops
                self.threshold_reached_step = step
                self.threshold_value = cell_accuracy
                self.threshold_metric = 'cell_accuracy'
                # 🆕 终端打印：达到阈值时立即输出
                print(f"\n🎯 [性能-算力对齐] 达到阈值！")
                print(f"   阈值指标: {self.threshold_metric} >= {threshold_cell_accuracy}")
                print(f"   实际值: {cell_accuracy:.4f}")
                print(f"   达到步数: step {step}")
                print(f"   所需 FLOPs: {cumulative_flops:.2e}")
                print(f"   log₁₀(FLOPs): {np.log10(float(cumulative_flops)):.2f}")
                return True
        
        # 线性回归任务：检测 validation/MSE < τ
        # τ = 0.1 if d <= 20, else 0.2
        if raw_mse is not None:
            tau = 0.1 if dims <= 20 else 0.2
            if raw_mse < tau:
                self.flops_to_threshold = cumulative_flops
                self.threshold_reached_step = step
                self.threshold_value = raw_mse
                self.threshold_metric = 'mse'
                # 🆕 终端打印：达到阈值时立即输出
                print(f"\n🎯 [性能-算力对齐] 达到阈值！")
                print(f"   阈值指标: {self.threshold_metric} < {tau} (dims={dims}, τ={'0.1' if dims <= 20 else '0.2'})")
                print(f"   实际值: {raw_mse:.6f}")
                print(f"   达到步数: step {step}")
                print(f"   所需 FLOPs: {cumulative_flops:.2e}")
                print(f"   log₁₀(FLOPs): {np.log10(float(cumulative_flops)):.2f}")
                return True
        
        return False

    def record_validation(self, step, raw_mse, dims, respond_points, batch_means, model_type, inference_steps=None, cell_accuracy=None, sudoku_accuracy=None, node_accuracy=None, path_accuracy=None, cumulative_flops=None, single_inference_flops=None):
        """
        记录验证结果

        Args:
            step: 训练步数
            raw_mse: 原始 MSE
            dims: 特征维度
            respond_points: respond 点数
            batch_means: batch 均值列表
            model_type: 模型类型
            inference_steps: 推理步数（可选，用于多步数评估模式）
            cell_accuracy: 格子准确率（数独任务，可选）
            sudoku_accuracy: 数独准确率（数独任务，可选）
            node_accuracy: 节点准确率（路径查找任务，可选）
            path_accuracy: 路径准确率（路径查找任务，可选）
            cumulative_flops: 累计算力（FLOPs，可选）
            single_inference_flops: 单次推理FLOPs（可选）

        🆕 断点续传保护：如果该 step 已存在，跳过记录（避免重复）
        """
        # 🆕 断点续传保护：检查该 step 是否已记录（考虑 inference_steps）
        # 对于多步验证，使用 (step, inference_steps) 作为唯一标识
        if inference_steps is not None:
            val_key = (step, inference_steps)
            # 🆕 优化：使用已加载的 existing_validation_keys 集合，避免每次重新读取文件
            if val_key in self.existing_validation_keys:
                return  # 已存在，跳过记录
        else:
            if step in self.existing_validation_steps:
                return  # 已存在，跳过记录
        entry = {
            "step": step,
            "model_type": model_type,
            "dims": dims,
            "respond_points": respond_points,
        }
        # 🆕 数独任务：只记录准确率（不记录 MSE）
        if cell_accuracy is not None:
            entry["cell_accuracy"] = self._convert_to_json_serializable(cell_accuracy)
        if sudoku_accuracy is not None:
            entry["sudoku_accuracy"] = self._convert_to_json_serializable(sudoku_accuracy)
        # 🆕 Pathfinding 任务：只记录准确率（不记录 MSE）
        if node_accuracy is not None:
            entry["node_accuracy"] = self._convert_to_json_serializable(node_accuracy)
        if path_accuracy is not None:
            entry["path_accuracy"] = self._convert_to_json_serializable(path_accuracy)
        # 非数独任务：记录 MSE
        if raw_mse is not None:
            # 🔧 确保 raw_mse 是 Python 原生类型
            raw_mse = self._convert_to_json_serializable(raw_mse)
            self.validation_count += 1
            self.validation_accumulated_avg += (raw_mse - self.validation_accumulated_avg) / self.validation_count
            entry["raw_mse"] = raw_mse
            entry["accumulated_avg_mse"] = self.validation_accumulated_avg
            entry["batch_means"] = self._convert_to_json_serializable(batch_means)
        # 🆕 多步推理：记录推理步数
        if inference_steps is not None:
            entry["inference_steps"] = inference_steps
        # 🆕 记录累计算力
        if cumulative_flops is not None:
            entry["cumulative_flops"] = self._convert_to_json_serializable(cumulative_flops)
        # 🆕 记录单次推理FLOPs
        if single_inference_flops is not None:
            entry["single_inference_flops"] = self._convert_to_json_serializable(single_inference_flops)
        
        # 🆕 性能-算力对齐：检测并记录达到阈值所需的 FLOPs
        # 注意：只检测单步验证（inference_steps 为 None），多步验证使用第一个步数的结果
        if inference_steps is None and self.is_main_process:
            threshold_cell_accuracy = 0.9  # 可配置的阈值（默认0.9）
            self._check_and_record_threshold(
                step, raw_mse, cell_accuracy, dims, cumulative_flops, threshold_cell_accuracy
            )
        
        # 🆕 只有主进程才写入日志文件，避免多进程竞争
        if not self.is_main_process:
            return
        
        # 🆕 确保目录存在（主进程）
        os.makedirs(os.path.dirname(self.validation_log_path), exist_ok=True)
        
        # 🆕 将当前 step 添加到已记录集合（在写入前）
        if inference_steps is not None:
            # 多步验证：添加到 existing_validation_keys 集合
            self.existing_validation_keys.add((step, inference_steps))
        else:
            self.existing_validation_steps.add(step)
        
        # 🔧 优化：使用追加模式写入，但捕获可能的 OSS 写入错误
        # 如果路径在 OSS 上，写入可能会很慢，但我们已经确保只有主进程写入
        try:
            with open(self.validation_log_path, "a") as f:
                # 🔧 确保 entry 中的所有值都是 JSON 可序列化的（双重保护）
                entry_serializable = self._convert_to_json_serializable(entry)
                f.write(json.dumps(entry_serializable) + "\n")
                f.flush()  # 🔧 强制刷新，确保数据写入（虽然可能慢，但能保证数据不丢失）
        except (IOError, OSError) as e:
            # 🔧 如果写入失败（如 OSS 网络问题），打印警告但不中断训练
            print(f"⚠️  警告: 写入验证日志失败: {e}")
            print(f"   日志路径: {self.validation_log_path}")

    def generate_plots(self):
        train_records = self._load_json_lines(self.train_log_path)
        val_records = self._load_json_lines(self.validation_log_path)
        if train_records:
            # 🆕 兼容数独任务：可能没有 raw_mse，只有 accuracy
            if any("raw_mse" in rec for rec in train_records):
                self._plot_curve(
                    train_records,
                    os.path.join(self.plot_dir, "training_mse_curves.png"),
                    title="Training MSE (Raw vs Accumulated Average)",
                    y_label="Respond MSE",
                )
            else:
                # Sudoku: plot accuracies if present
                if any("cell_accuracy" in rec for rec in train_records):
                    self._plot_simple_curve(
                        train_records,
                        os.path.join(self.plot_dir, "training_cell_accuracy_curves.png"),
                        title="Training Cell Accuracy",
                        y_label="Cell Accuracy",
                        metric_key="cell_accuracy",
                    )
                if any("sudoku_accuracy" in rec for rec in train_records):
                    self._plot_simple_curve(
                        train_records,
                        os.path.join(self.plot_dir, "training_sudoku_accuracy_curves.png"),
                        title="Training Sudoku Accuracy",
                        y_label="Sudoku Accuracy",
                        metric_key="sudoku_accuracy",
                    )
        if val_records:
            # 🆕 检查是否有多个推理步数的记录
            has_multistep = any("inference_steps" in rec for rec in val_records)
            if has_multistep:
                # 多步数模式：为每个步数绘制单独的曲线
                self._plot_multistep_validation_curves(val_records)
            else:
                # 单步模式：根据记录内容选择画 MSE 或 Accuracy
                if any("raw_mse" in rec for rec in val_records):
                    self._plot_curve(
                        val_records,
                        os.path.join(self.plot_dir, "validation_mse_curves.png"),
                        title="Validation MSE (Raw vs Accumulated Average)",
                        y_label="Respond MSE",
                    )
                else:
                    if any("cell_accuracy" in rec for rec in val_records):
                        self._plot_simple_curve(
                            val_records,
                            os.path.join(self.plot_dir, "validation_cell_accuracy_curves.png"),
                            title="Validation Cell Accuracy",
                            y_label="Cell Accuracy",
                            metric_key="cell_accuracy",
                        )
                    if any("sudoku_accuracy" in rec for rec in val_records):
                        self._plot_simple_curve(
                            val_records,
                            os.path.join(self.plot_dir, "validation_sudoku_accuracy_curves.png"),
                            title="Validation Sudoku Accuracy",
                            y_label="Sudoku Accuracy",
                            metric_key="sudoku_accuracy",
                        )
        
        # 🆕 性能-算力对齐：绘制 log10(Cumulative FLOPs) vs Accuracy/MSE 图表
        self._plot_flops_vs_performance(val_records)
        
        # 🆕 导出阈值到 JSON 文件
        self._export_threshold_to_json()

    def _plot_simple_curve(self, records, out_path, title, y_label, metric_key):
        """
        🆕 简单曲线绘制：只画一个指标（用于数独 accuracy 等没有 accumulated_avg 的情况）
        """
        if not records:
            return
        # 仅保留含有该指标的记录
        records = [rec for rec in records if metric_key in rec]
        if not records:
            return
        model_types = sorted({rec.get("model_type", "unknown") for rec in records})
        fig, ax = plt.subplots(figsize=(12, 6))
        for model_type in model_types:
            subset = [rec for rec in records if rec.get("model_type") == model_type and metric_key in rec]
            if not subset:
                continue
            steps = [rec["step"] for rec in subset]
            series = [rec[metric_key] for rec in subset]
            ax.plot(steps, series, label=f"{model_type} {metric_key}", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()

        # OSS 兼容保存逻辑复用 _plot_curve 的实现
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, os.path.basename(out_path))
        try:
            fig.savefig(temp_file_path, dpi=150)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            shutil.move(temp_file_path, out_path)
        except Exception as e:
            print(f"⚠️  Failed to save plot to {out_path}: {e}")
            try:
                fig.savefig(out_path, dpi=150)
            except Exception as e2:
                print(f"❌  Failed to save plot (fallback): {e2}")
        plt.close(fig)

    def _plot_curve(self, records, out_path, title, y_label):
        if not records:
            return
        model_types = sorted({rec["model_type"] for rec in records})
        fig, ax = plt.subplots(figsize=(12, 6))
        for model_type in model_types:
            subset = [rec for rec in records if rec["model_type"] == model_type]
            steps = [rec["step"] for rec in subset]
            # 🆕 兼容：可能某些记录没有 raw_mse（如数独任务）
            subset = [rec for rec in subset if "raw_mse" in rec and "accumulated_avg_mse" in rec]
            if not subset:
                continue
            steps = [rec["step"] for rec in subset]
            raw_series = [rec["raw_mse"] for rec in subset]
            accumulated_avg_series = [rec["accumulated_avg_mse"] for rec in subset]
            ax.plot(steps, raw_series, label=f"{model_type} Raw", alpha=0.5, linestyle='-')
            ax.plot(steps, accumulated_avg_series, label=f"{model_type} Accumulated Avg", linewidth=2, linestyle='-')
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        
        # 🆕 OSS 兼容：先保存到临时文件，再移动到目标路径
        # OSS 挂载不支持 w+b 模式（随机读写），需要先写本地再移动
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, os.path.basename(out_path))
        try:
            # 先保存到本地临时目录
            fig.savefig(temp_file_path, dpi=150)
            # 确保目标文件夹存在
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            # 将保存好的文件移动到 OSS 目标路径（使用 shutil.move，支持 OSS）
            shutil.move(temp_file_path, out_path)
        except Exception as e:
            print(f"⚠️  Failed to save plot to {out_path}: {e}")
            # 如果移动失败，尝试直接保存（某些环境可能支持）
            try:
                fig.savefig(out_path, dpi=150)
            except Exception as e2:
                print(f"❌  Failed to save plot (fallback): {e2}")
        
        plt.close(fig)
    
    def _plot_multistep_validation_curves(self, records):
        """
        🆕 绘制多步数验证曲线：为每个推理步数绘制单独的曲线
        """
        if not records:
            return
        
        # 提取所有推理步数
        inference_steps_list = sorted(set(rec.get("inference_steps") for rec in records if "inference_steps" in rec))
        if not inference_steps_list:
            return
        
        model_types = sorted({rec["model_type"] for rec in records})
        
        # 为每个模型类型绘制一个图
        for model_type in model_types:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # 为每个推理步数绘制曲线
            for inference_steps in inference_steps_list:
                subset = [
                    rec for rec in records 
                    if rec.get("model_type") == model_type and rec.get("inference_steps") == inference_steps
                ]
                if not subset:
                    continue

                # 🆕 兼容数独：多步验证可能记录的是 accuracy 而不是 mse
                if any("raw_mse" in rec for rec in subset):
                    subset_mse = [rec for rec in subset if "raw_mse" in rec and "accumulated_avg_mse" in rec]
                    if not subset_mse:
                        continue
                    steps = [rec["step"] for rec in subset_mse]
                    raw_series = [rec["raw_mse"] for rec in subset_mse]
                    accumulated_avg_series = [rec["accumulated_avg_mse"] for rec in subset_mse]
                    ax.plot(steps, raw_series, 
                           label=f"Step {inference_steps} Raw", 
                           alpha=0.6, linestyle='-', linewidth=1.5)
                    ax.plot(steps, accumulated_avg_series, 
                           label=f"Step {inference_steps} Avg", 
                           linewidth=2.5, linestyle='-')
                elif any("cell_accuracy" in rec for rec in subset):
                    subset_acc = [rec for rec in subset if "cell_accuracy" in rec]
                    steps = [rec["step"] for rec in subset_acc]
                    series = [rec["cell_accuracy"] for rec in subset_acc]
                    ax.plot(steps, series, label=f"Step {inference_steps} cell_acc", linewidth=2.0)
                elif any("sudoku_accuracy" in rec for rec in subset):
                    subset_acc = [rec for rec in subset if "sudoku_accuracy" in rec]
                    steps = [rec["step"] for rec in subset_acc]
                    series = [rec["sudoku_accuracy"] for rec in subset_acc]
                    ax.plot(steps, series, label=f"Step {inference_steps} sudoku_acc", linewidth=2.0)
            
            # 根据曲线内容自动设置标题/纵轴
            if any("raw_mse" in rec for rec in records if rec.get("model_type") == model_type):
                ax.set_title(f"Validation MSE - {model_type} (Multi-Step Inference)")
                ax.set_ylabel("Respond MSE")
            else:
                ax.set_title(f"Validation Accuracy - {model_type} (Multi-Step Inference)")
                ax.set_ylabel("Accuracy")
            ax.set_xlabel("Training Step")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc='best', ncol=2)
            fig.tight_layout()
            
            out_path = os.path.join(self.plot_dir, f"validation_mse_curves_{model_type}_multistep.png")
            
            # 🆕 OSS 兼容：先保存到临时文件，再移动到目标路径
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, os.path.basename(out_path))
            try:
                fig.savefig(temp_file_path, dpi=150)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                shutil.move(temp_file_path, out_path)
            except Exception as e:
                print(f"⚠️  Failed to save plot to {out_path}: {e}")
                try:
                    fig.savefig(out_path, dpi=150)
                except Exception as e2:
                    print(f"❌  Failed to save plot (fallback): {e2}")
            
            plt.close(fig)
        
        # 也生成一个汇总图（所有步数在同一图中，只显示 Raw）
        fig, ax = plt.subplots(figsize=(14, 8))
        for inference_steps in inference_steps_list:
            # 合并所有模型类型（如果只有一个模型类型，就只显示那个）
            all_subset = [
                rec for rec in records 
                if rec.get("inference_steps") == inference_steps
            ]
            if not all_subset:
                continue
            
            steps = [rec["step"] for rec in all_subset]
            raw_series = [rec["raw_mse"] for rec in all_subset]
            
            ax.plot(steps, raw_series, 
                   label=f"Step {inference_steps}", 
                   linewidth=2.5, linestyle='-', marker='o', markersize=3)
        
        ax.set_title("Validation MSE Comparison (Multi-Step Inference)")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Respond MSE")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc='best')
        fig.tight_layout()
        
        out_path = os.path.join(self.plot_dir, "validation_mse_curves_multistep_comparison.png")
        
        # 🆕 OSS 兼容：先保存到临时文件，再移动到目标路径
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, os.path.basename(out_path))
        try:
            fig.savefig(temp_file_path, dpi=150)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            shutil.move(temp_file_path, out_path)
        except Exception as e:
            print(f"⚠️  Failed to save plot to {out_path}: {e}")
            try:
                fig.savefig(out_path, dpi=150)
            except Exception as e2:
                print(f"❌  Failed to save plot (fallback): {e2}")
        
        plt.close(fig)

    def _plot_flops_vs_performance(self, val_records):
        """
        🆕 性能-算力对齐：绘制 log10(Cumulative FLOPs) vs Accuracy/MSE 图表
        
        Args:
            val_records: 验证记录列表
        """
        if not val_records:
            return
        
        # 过滤出包含 cumulative_flops 的记录
        records_with_flops = [rec for rec in val_records if "cumulative_flops" in rec and rec.get("cumulative_flops") is not None]
        if not records_with_flops:
            print("⚠️  没有包含 cumulative_flops 的验证记录，跳过 FLOPs vs Performance 图表")
            return
        
        # 判断任务类型（数独或线性回归）
        has_cell_accuracy = any("cell_accuracy" in rec for rec in records_with_flops)
        has_mse = any("raw_mse" in rec for rec in records_with_flops)
        
        if has_cell_accuracy:
            # 数独任务：绘制 log10(Cumulative FLOPs) vs Cell Accuracy
            self._plot_flops_vs_accuracy(records_with_flops, metric_key="cell_accuracy")
        elif has_mse:
            # 线性回归任务：绘制 log10(Cumulative FLOPs) vs MSE
            self._plot_flops_vs_mse(records_with_flops)
    
    def _plot_flops_vs_accuracy(self, records, metric_key="cell_accuracy"):
        """
        绘制 log10(Cumulative FLOPs) vs Accuracy 图表
        """
        if not records:
            return
        
        model_types = sorted({rec.get("model_type", "unknown") for rec in records})
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for model_type in model_types:
            subset = [rec for rec in records if rec.get("model_type") == model_type and metric_key in rec]
            if not subset:
                continue
            
            # 提取 log10(cumulative_flops) 和 accuracy
            log10_flops = [np.log10(max(float(rec["cumulative_flops"]), 1.0)) for rec in subset]
            accuracies = [rec[metric_key] for rec in subset]
            
            ax.plot(log10_flops, accuracies, label=f"{model_type}", linewidth=2, marker='o', markersize=4)
        
        ax.set_title(f"Performance vs Computational Cost ({metric_key.replace('_', ' ').title()})")
        ax.set_xlabel("log₁₀(Cumulative FLOPs)")
        ax.set_ylabel("Accuracy" if "accuracy" in metric_key else metric_key)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        
        out_path = os.path.join(self.plot_dir, f"flops_vs_{metric_key}.png")
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, os.path.basename(out_path))
        try:
            fig.savefig(temp_file_path, dpi=150)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            shutil.move(temp_file_path, out_path)
        except Exception as e:
            print(f"⚠️  Failed to save plot to {out_path}: {e}")
            try:
                fig.savefig(out_path, dpi=150)
            except Exception as e2:
                print(f"❌  Failed to save plot (fallback): {e2}")
        plt.close(fig)
    
    def _plot_flops_vs_mse(self, records):
        """
        绘制 log10(Cumulative FLOPs) vs MSE 图表
        """
        if not records:
            return
        
        model_types = sorted({rec.get("model_type", "unknown") for rec in records})
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for model_type in model_types:
            subset = [rec for rec in records if rec.get("model_type") == model_type and "raw_mse" in rec]
            if not subset:
                continue
            
            # 提取 log10(cumulative_flops) 和 MSE
            log10_flops = [np.log10(max(float(rec["cumulative_flops"]), 1.0)) for rec in subset]
            mse_values = [rec["raw_mse"] for rec in subset]
            
            ax.plot(log10_flops, mse_values, label=f"{model_type}", linewidth=2, marker='o', markersize=4)
        
        ax.set_title("Performance vs Computational Cost (MSE)")
        ax.set_xlabel("log₁₀(Cumulative FLOPs)")
        ax.set_ylabel("MSE")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        # 可选：使用对数y轴（如果MSE值跨度大）
        # ax.set_yscale('log')
        fig.tight_layout()
        
        out_path = os.path.join(self.plot_dir, "flops_vs_mse.png")
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, os.path.basename(out_path))
        try:
            fig.savefig(temp_file_path, dpi=150)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            shutil.move(temp_file_path, out_path)
        except Exception as e:
            print(f"⚠️  Failed to save plot to {out_path}: {e}")
            try:
                fig.savefig(out_path, dpi=150)
            except Exception as e2:
                print(f"❌  Failed to save plot (fallback): {e2}")
        plt.close(fig)
    
    def _export_threshold_to_json(self):
        """
        🆕 导出阈值到 JSON 文件（用于性能-算力对齐分析）
        """
        if not self.is_main_process:
            return
        
        if self.flops_to_threshold is None:
            # 未达到阈值，不导出
            return
        
        # 构建阈值信息
        threshold_data = {
            "model_type": self.model_type,
            "threshold_metric": self.threshold_metric,  # 'cell_accuracy' 或 'mse'
            "threshold_value": self.threshold_value,  # 达到的实际值
            "flops_to_threshold": self.flops_to_threshold,  # 达到阈值所需的 FLOPs
            "log10_flops_to_threshold": np.log10(float(self.flops_to_threshold)) if self.flops_to_threshold > 0 else 0.0,
            "threshold_reached_step": self.threshold_reached_step,  # 达到阈值的步数
        }
        
        # 保存到 JSON 文件
        json_path = os.path.join(os.path.dirname(self.plot_dir), "threshold_flops.json")
        try:
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(threshold_data, f, indent=2)
            print(f"✅ 阈值 FLOPs 已导出到: {json_path}")
        except Exception as e:
            print(f"⚠️  导出阈值 FLOPs 失败: {e}")

    @staticmethod
    def _load_json_lines(path):
        if not os.path.exists(path):
            return []
        records = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records


# ============================================================
# Training Setup and Configuration Utilities
# ============================================================

def setup_accelerator(training_config):
    """
    初始化 Accelerator（如果可用且启用分布式训练）
    
    Args:
        training_config: 训练配置字典
    
    Returns:
        accelerator: Accelerator 实例或 None
        device: torch.device
        use_distributed: bool
    """
    # 🆕 分布式训练支持
    try:
        from accelerate import Accelerator
        from accelerate.utils import InitProcessGroupKwargs
        from datetime import timedelta
        # 🆕 导入 DDP 配置工具（用于设置 find_unused_parameters）
        try:
            from accelerate.utils import DistributedDataParallelKwargs
            DDP_KWARGS_AVAILABLE = True
        except ImportError:
            try:
                from accelerate import DistributedDataParallelKwargs
                DDP_KWARGS_AVAILABLE = True
            except ImportError:
                DistributedDataParallelKwargs = None
                DDP_KWARGS_AVAILABLE = False
        ACCELERATE_AVAILABLE = True
    except ImportError:
        ACCELERATE_AVAILABLE = False
        Accelerator = None
        InitProcessGroupKwargs = None
        DistributedDataParallelKwargs = None
        DDP_KWARGS_AVAILABLE = False
        timedelta = None
    
    # 自动检测：如果环境变量中有分布式相关变量，则启用分布式训练
    # 或者通过配置显式指定
    env_has_distributed = any([
        os.environ.get("WORLD_SIZE"),
        os.environ.get("RANK"),
        os.environ.get("LOCAL_RANK"),
        os.environ.get("MASTER_ADDR"),
    ])
    use_distributed = (training_config.get("use_distributed", False) or env_has_distributed) and ACCELERATE_AVAILABLE
    
    if use_distributed:
        # 🆕 解决 "unused parameters" 报错的标准做法
        # 使用 DistributedDataParallelKwargs 明确开启 find_unused_parameters
        # 这解决了某些模型层在条件分支中可能不被使用的问题
        # 错误信息: RuntimeError: Expected to have finished reduction in the prior iteration...
        # 原因: 某些模型参数在 forward 过程中可能不被使用（例如条件分支）
        
        if DDP_KWARGS_AVAILABLE and DistributedDataParallelKwargs is not None:
            # 方法1: 使用 DistributedDataParallelKwargs (推荐，最标准的方法)
            try:
                # 🆕 关闭 buffer 同步（Transformer 模型使用 LayerNorm，不需要同步 buffers）
                # 这可以避免 _sync_buffers() 导致的超时，并略微提高训练速度
                ddp_kwargs = DistributedDataParallelKwargs(
                    find_unused_parameters=True,
                    broadcast_buffers=False  # 禁止前向传播时的自动 buffer 同步
                )
                # 🔧 增加分布式超时时间到 1 小时（防止评测时死锁和 NCCL 通信超时）
                # 评测可能需要较长时间，且 NCCL/RCCL 通信在大模型或慢网络下可能需要更长时间
                # 设置为 1 小时（3600秒）是推荐值，既能避免超时，又不会无限等待
                if InitProcessGroupKwargs is not None and timedelta is not None:
                    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))  # 1 小时
                    kwargs_handlers = [ddp_kwargs, process_group_kwargs]
                else:
                    kwargs_handlers = [ddp_kwargs]
                
                accelerator = Accelerator(
                    kwargs_handlers=kwargs_handlers,
                    mixed_precision=training_config.get("mixed_precision", "no")
                )
                print("✅ 使用 DistributedDataParallelKwargs 设置:")
                print("   - find_unused_parameters=True")
                print("   - broadcast_buffers=False (避免 buffer 同步超时)")
                if process_group_kwargs in kwargs_handlers:
                    print("✅ 分布式超时时间已设置为 1 小时")
            except (TypeError, AttributeError) as e:
                # 如果 DistributedDataParallelKwargs 失败，尝试 DDPPlugin
                print(f"⚠️  DistributedDataParallelKwargs 初始化失败: {e}")
                print("   尝试使用 DDPPlugin 作为备选方案...")
                try:
                    from accelerate.utils import DDPPlugin
                    # 🆕 关闭 buffer 同步（Transformer 模型使用 LayerNorm，不需要同步 buffers）
                    ddp_plugin = DDPPlugin(
                        find_unused_parameters=True,
                        broadcast_buffers=False  # 禁止前向传播时的自动 buffer 同步
                    )
                    # 🔧 增加分布式超时时间到 1 小时
                    if InitProcessGroupKwargs is not None and timedelta is not None:
                        process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))  # 1 小时
                        accelerator = Accelerator(
                            ddp_plugin=ddp_plugin,
                            kwargs_handlers=[process_group_kwargs],
                            mixed_precision=training_config.get("mixed_precision", "no")
                        )
                        print("✅ 使用 DDPPlugin 设置:")
                        print("   - find_unused_parameters=True")
                        print("   - broadcast_buffers=False (避免 buffer 同步超时)")
                        print("✅ 分布式超时时间已设置为 1 小时")
                    else:
                        accelerator = Accelerator(
                            ddp_plugin=ddp_plugin,
                            mixed_precision=training_config.get("mixed_precision", "no")
                        )
                        print("✅ 使用 DDPPlugin 设置 find_unused_parameters=True")
                except (ImportError, TypeError, AttributeError):
                    # 最后备选：默认初始化（不推荐，可能仍会报错）
                    accelerator = Accelerator(mixed_precision=training_config.get("mixed_precision", "no"))
                    print(f"❌ 警告: 无法设置 find_unused_parameters=True")
                    print(f"     Accelerate 版本可能较旧或不支持")
                    print(f"     建议升级: pip install --upgrade accelerate>=0.20.0")
        else:
            # 如果 DistributedDataParallelKwargs 不可用，尝试 DDPPlugin
            try:
                from accelerate.utils import DDPPlugin
                # 🆕 关闭 buffer 同步（Transformer 模型使用 LayerNorm，不需要同步 buffers）
                ddp_plugin = DDPPlugin(
                    find_unused_parameters=True,
                    broadcast_buffers=False  # 禁止前向传播时的自动 buffer 同步
                )
                # 🔧 增加分布式超时时间到 1 小时
                if InitProcessGroupKwargs is not None and timedelta is not None:
                    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))  # 1 小时
                    accelerator = Accelerator(
                        ddp_plugin=ddp_plugin,
                        kwargs_handlers=[process_group_kwargs],
                        mixed_precision=training_config.get("mixed_precision", "no")
                    )
                    print("✅ 使用 DDPPlugin 设置:")
                    print("   - find_unused_parameters=True")
                    print("   - broadcast_buffers=False (避免 buffer 同步超时)")
                    print("✅ 分布式超时时间已设置为 1 小时")
                else:
                    accelerator = Accelerator(
                        ddp_plugin=ddp_plugin,
                        mixed_precision=training_config.get("mixed_precision", "no")
                    )
                    print("✅ 使用 DDPPlugin 设置 find_unused_parameters=True")
            except (ImportError, TypeError, AttributeError):
                # 最后备选：默认初始化
                accelerator = Accelerator(mixed_precision=training_config.get("mixed_precision", "no"))
                print(f"⚠️  警告: 无法设置 find_unused_parameters=True")
                print(f"     Accelerate 版本可能较旧，将使用默认设置")
                print(f"     如果遇到 'unused parameters' 错误，请升级 accelerate 到 >= 0.20.0")
        
        # ⚠️ 重要：在分布式模式下，必须使用 accelerator.device
        device = accelerator.device
        
        print(f"\n{'='*60}")
        print(f"🚀 分布式训练已启用 (DDP)")
        print(f"   - 进程数: {accelerator.num_processes}")
        print(f"   - 当前 Rank: {accelerator.process_index}")
        print(f"   - 主进程: {accelerator.is_main_process}")
        print(f"   - 设备: {device}")
        print(f"   - find_unused_parameters: True ✅ (已开启)")
        print(f"{'='*60}\n")
    else:
        accelerator = None
        # ⚠️ 只有在非分布式模式下才手动定义 device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"模式: 单卡/CPU 训练, 设备: {device}")
    
    return accelerator, device, use_distributed


def parse_multi_epoch_config(training_config):
    """
    解析 Multi-epoch 配置
    
    支持两种格式：
    1. 新格式：training.multi_epoch.enabled
    2. 旧格式（数独）：training.use_multi_epoch
    
    Args:
        training_config: 训练配置字典
    
    Returns:
        use_multi_epoch: bool
        num_epochs: int
        steps_per_epoch: int
        shuffle_between_epochs: bool
    """
    multi_epoch_cfg = training_config.get("multi_epoch", {})
    
    # 检查新格式
    if multi_epoch_cfg:
        use_multi_epoch = multi_epoch_cfg.get("enabled", False)
        num_epochs = multi_epoch_cfg.get("num_epochs", 1)
        steps_per_epoch = multi_epoch_cfg.get("steps_per_epoch", training_config.get("train_steps", 50000))
        shuffle_between_epochs = multi_epoch_cfg.get("shuffle_between_epochs", True)
    else:
        # 检查旧格式（数独配置）
        use_multi_epoch = training_config.get("use_multi_epoch", False)
        num_epochs = training_config.get("num_epochs", 1)
        steps_per_epoch = training_config.get("steps_per_epoch", training_config.get("train_steps", 50000))
        shuffle_between_epochs = training_config.get("shuffle_between_epochs", True)
    
    if use_multi_epoch:
        print(f"\n{'='*60}")
        print("🔄 Multi-Epoch Training Enabled")
        print(f"  - Epochs: {num_epochs}")
        print(f"  - Steps per epoch: {steps_per_epoch}")
        print(f"  - Total steps: {num_epochs * steps_per_epoch}")
        print(f"  - Shuffle between epochs: {shuffle_between_epochs}")
        print(f"{'='*60}\n")
    
    return use_multi_epoch, num_epochs, steps_per_epoch, shuffle_between_epochs


def setup_sequence_mode(sequence_mode, training_config, n_prompt, initial_n_respond, model, is_ar_model):
    """
    设置序列模式并打印相关信息
    
    Args:
        sequence_mode: 序列模式字符串
        training_config: 训练配置字典
        n_prompt: prompt 点数
        initial_n_respond: 初始 respond 点数
        model: 模型实例
        is_ar_model: 是否为 AR 模型
    
    Returns:
        fixed_permutation_x: Tensor 或 None
        fixed_permutation_y: Tensor 或 None
        permutation_pool_x: Tensor 或 None
        permutation_pool_y: Tensor 或 None
    """
    # 注意：generate_fixed_permutation, generate_permutation_pool, get_model_attr 已在同一文件中定义
    
    fixed_permutation_x = None
    fixed_permutation_y = None
    permutation_pool_x = None
    permutation_pool_y = None
    
    if sequence_mode in ["fixed_permutation", "fixed_permutation_xy"]:
        permutation_seed = training_config.get("permutation_seed", 42)
        
        # 生成 x 的固定排列（必须）
        fixed_permutation_x = generate_fixed_permutation(n_prompt, initial_n_respond, seed=permutation_seed)
        
        # 根据模式决定是否生成 y 的固定排列
        if sequence_mode == "fixed_permutation_xy":
            # 变体2：同时打乱 x 和 y（使用不同的种子）
            permutation_seed_y = training_config.get("permutation_seed_y", permutation_seed + 1000)
            fixed_permutation_y = generate_fixed_permutation(n_prompt, initial_n_respond, seed=permutation_seed_y)
        
        print(f"\n{'='*60}")
        print("🔒 Fixed-Permutation Non-Sequential ICL Mode Enabled")
        print(f"   - X Permutation Seed: {permutation_seed}")
        
        if sequence_mode == "fixed_permutation":
            print("   - Mode 1: 只打乱 prompt 的 x 顺序，y 保持原顺序")
            print("   - 效果：破坏 prompt 中 (x,y) 的配对关系")
            print("   - Purpose: Test if model can learn the fixed permutation to recover x-y correspondence")
        else:  # fixed_permutation_xy
            print(f"   - Y Permutation Seed: {permutation_seed_y}")
            print("   - Mode 2: 同时独立打乱 prompt 的 x 和 y 顺序")
            print("   - 效果：完全破坏 prompt 中 (x,y) 的配对关系（更具挑战性）")
            print("   - Purpose: Test model's ability to learn both permutations under harder conditions")
        
        print(f"   - X Permutation: {fixed_permutation_x.tolist()[:10]}{'...' if len(fixed_permutation_x) > 10 else ''}")
        if fixed_permutation_y is not None:
            print(f"   - Y Permutation: {fixed_permutation_y.tolist()[:10]}{'...' if len(fixed_permutation_y) > 10 else ''}")
        
        # 模型类型特殊说明
        if is_ar_model:
            # AR 模型的特殊说明
            attention_mode = get_model_attr(model, 'attention_mode', 'causal')
            print(f"\n   📌 AR Model: attention_mode = '{attention_mode}'")
            if attention_mode == "causal":
                print("      ⚠️  WARNING: Causal attention with fixed permutation")
                print("      → Challenge: Model must learn fixed permutation under causal constraint")
            elif attention_mode == "prefix_lm":
                print("      ✅ Prefix-LM: Better suited for fixed permutation learning")
        else:
            # MDM 模型（LLaDA, Dream, SDAR）的说明
            print(f"\n   📌 MDM Model: Bidirectional attention")
            print("      ✅ MDM models use bidirectional attention, better suited for fixed permutation learning")
            print("      → Expected: Better performance than AR models with causal attention")
        
        print(f"{'='*60}\n")
    
    elif sequence_mode in ["pool_permutation", "pool_permutation_xy"]:
        permutation_seed = training_config.get("permutation_seed", 42)
        pool_size = training_config.get("permutation_pool_size", 20)  # 默认池大小为20
        
        # 生成 x 的排列池（必须）
        permutation_pool_x = generate_permutation_pool(n_prompt, pool_size=pool_size, seed=permutation_seed)
        
        # 根据模式决定是否生成 y 的排列池
        if sequence_mode == "pool_permutation_xy":
            # 变体2：同时打乱 x 和 y（使用不同的种子）
            permutation_seed_y = training_config.get("permutation_seed_y", permutation_seed + 1000)
            permutation_pool_y = generate_permutation_pool(n_prompt, pool_size=pool_size, seed=permutation_seed_y)
        
        print(f"\n{'='*60}")
        print("🎯 Pool-Permutation Non-Sequential ICL Mode Enabled")
        print(f"   - Pool Size: {pool_size}")
        print(f"   - X Permutation Pool Seed: {permutation_seed}")
        
        if sequence_mode == "pool_permutation":
            print("   - Mode 1: 只打乱 prompt 的 x 顺序，y 保持原顺序")
            print("   - 效果：破坏 prompt 中 (x,y) 的配对关系")
            print("   - 特点：每次从固定池中随机采样一个排列（介于 fixed 和 random 之间）")
            print("   - Purpose: Test if model can learn to handle multiple fixed permutations")
        else:  # pool_permutation_xy
            print(f"   - Y Permutation Pool Seed: {permutation_seed_y}")
            print("   - Mode 2: 同时独立打乱 prompt 的 x 和 y 顺序")
            print("   - 效果：完全破坏 prompt 中 (x,y) 的配对关系（更具挑战性）")
            print("   - 特点：每次从固定池中随机采样一个排列（介于 fixed 和 random 之间）")
            print("   - Purpose: Test model's ability to handle multiple fixed permutations under harder conditions")
        
        print(f"   - X Pool Sample (first perm): {permutation_pool_x[0].tolist()[:10]}{'...' if len(permutation_pool_x[0]) > 10 else ''}")
        if permutation_pool_y is not None:
            print(f"   - Y Pool Sample (first perm): {permutation_pool_y[0].tolist()[:10]}{'...' if len(permutation_pool_y[0]) > 10 else ''}")
        
        # 模型类型特殊说明
        if is_ar_model:
            attention_mode = get_model_attr(model, 'attention_mode', 'causal')
            print(f"\n   📌 AR Model: attention_mode = '{attention_mode}'")
            if attention_mode == "causal":
                print("      ⚠️  CHALLENGE: Causal attention with pool permutation")
                print("      → Expected: Moderate performance (between fixed and random)")
            elif attention_mode == "prefix_lm":
                print("      ✅ Prefix-LM: Better suited for pool permutation learning")
        else:
            print(f"\n   📌 MDM Model: Bidirectional attention")
            print("      ✅ MDM models use bidirectional attention, better suited for pool permutation learning")
            print("      → Expected: Better performance than AR models with causal attention")
        
        print(f"{'='*60}\n")
    
    elif sequence_mode == "random_permutation":
        print(f"\n{'='*60}")
        print("🎲 Random-Permutation Non-Sequential ICL Mode Enabled")
        print("   - Each batch uses a DIFFERENT random permutation (harder than fixed permutation)")
        print("   - Prompt part: x is randomly shuffled each batch")
        print("   - Prompt part: y keeps original order (or can be shuffled if permute_y=True)")
        print("   - Respond part: UNCHANGED, correct (x,y) pairs maintained")
        print("   - Challenge: Model must learn general correspondence, NOT memorize a specific permutation")
        
        # 模型类型特殊说明
        if is_ar_model:
            attention_mode = get_model_attr(model, 'attention_mode', 'causal')
            print(f"\n   📌 AR Model: attention_mode = '{attention_mode}'")
            if attention_mode == "causal":
                print("      ⚠️⚠️  EXTREME CHALLENGE: Causal attention with random permutation")
                print("      → Expected: Very poor performance (harder than fixed permutation)")
            elif attention_mode == "prefix_lm":
                print("      ⚠️  CHALLENGE: Prefix-LM with random permutation")
                print("      → Expected: Difficult, but better than causal")
        else:
            print(f"\n   📌 MDM Model: Bidirectional attention")
            print("      ⚠️  CHALLENGE: Even MDM models will struggle with random permutation")
            print("      → Test: Can bidirectional attention learn general correspondence?")
        
        print(f"{'='*60}\n")
    
    elif sequence_mode == "non_sequential":
        print(f"\n{'='*60}")
        print("🔀 Non-Sequential ICL Mode Enabled")
        print("   - Prompt and Respond pairs will be shuffled randomly each batch")
        print("   - Model will use respond_position_mask to identify respond pairs")
        
        # 模型类型特殊说明
        if is_ar_model:
            # AR 模型的特殊说明
            attention_mode = get_model_attr(model, 'attention_mode', 'causal')
            print(f"\n   📌 AR Model: attention_mode = '{attention_mode}'")
            if attention_mode == "causal":
                print("      ⚠️  WARNING: Standard causal mask with shuffled sequence")
                print("      → Expected: Significant performance degradation")
                print("      → Purpose: Demonstrate AR's dependence on sequence order")
            elif attention_mode == "prefix_lm":
                print("      ✅ Prefix-LM: Prompt bidirectional, Respond causal")
                print("      → Expected: Better handling of shuffled sequence")
                print("      → Purpose: More fair comparison with MDM")
        else:
            # MDM 模型（LLaDA, Dream, SDAR）的说明
            print(f"\n   📌 MDM Model: Bidirectional attention")
            print("      ✅ MDM models use bidirectional attention, naturally handle shuffled sequences")
            print("      → Expected: Better performance than AR models with causal attention")
        
        print(f"{'='*60}\n")
    
    return fixed_permutation_x, fixed_permutation_y, permutation_pool_x, permutation_pool_y


def generate_output_directory(config, training_config, model_conf, n_prompt, is_sudoku_task, n_respond=None, is_pathfinding_task=False):
    """
    生成输出目录路径

    Args:
        config: 完整配置字典
        training_config: 训练配置字典
        model_conf: 模型配置字典
        n_prompt: prompt 点数
        is_sudoku_task: 是否为数独任务
        n_respond: 数独/路径查找任务的 n_respond（如果不是则为 None）
        is_pathfinding_task: 是否为路径查找任务

    Returns:
        out_dir: 输出目录路径字符串
    """
    if "out_dir" in config:
        out_dir = config["out_dir"]
    else:
        # 自动生成目录名，格式与 run_batch_experiments.py 对齐
        # 格式: {model_type}_{size_key}_D{dim}_P{prompt}_R{respond}_seed{seed}
        # 🆕 数独/路径查找任务使用固定值，标准任务使用 curriculum
        if is_sudoku_task or is_pathfinding_task:
            initial_n_respond = n_respond
            final_n_respond = n_respond
        else:
            initial_n_respond = training_config["curriculum"]["points"]["start"]
            final_n_respond = training_config["curriculum"]["points"]["end"]
        
        # 判断模型尺寸（根据 n_embd 推断，与 run_batch_experiments.py 的逻辑对齐）
        n_embd = model_conf.get('n_embd', 256)
        if n_embd <= 256:
            size_key = 'standard'
        else:
            size_key = 'large'
        
        # 如果有多个 respond 值，使用初始值（大多数情况下 initial == final）
        respond_value = initial_n_respond if initial_n_respond == final_n_respond else f"{initial_n_respond}to{final_n_respond}"
        
        # 构建目录名，包含 seed 信息
        model_type = model_conf['family']
        auto_dir = f"{model_type}_{size_key}_D{model_conf['n_dims']}_P{n_prompt}_R{respond_value}"
        
        # 如果有 random_seed，添加到目录名
        if 'random_seed' in training_config or 'random_seed' in config:
            seed_value = training_config.get('random_seed', config.get('random_seed', 42))
            auto_dir = f"{auto_dir}_seed{seed_value}"
        
        # 🆕 使用绝对路径保存到持久化存储（星云容器）
        # 优先使用环境变量 CHECKPOINT_BASE_DIR，否则使用默认路径
        checkpoint_base = os.environ.get("CHECKPOINT_BASE_DIR", "/checkpoint/522240")
        out_dir = os.path.join(checkpoint_base, auto_dir)
    
    return out_dir


def resume_from_checkpoint(model, optim, training_config, state_path, device, is_sudoku_task, cur=None, use_multi_epoch=False):
    """
    从检查点恢复训练状态
    
    Args:
        model: 模型实例
        optim: 优化器实例
        training_config: 训练配置字典
        state_path: 检查点文件路径
        device: torch.device
        is_sudoku_task: 是否为数独任务
        cur: Curriculum 实例（如果不是数独任务）
        use_multi_epoch: 是否使用多epoch训练
    
    Returns:
        starting_step: int
        starting_epoch: int
    """
    starting_step = 0
    starting_epoch = 0
    
    # 🆕 优先从resume_from_checkpoint加载（如果指定）
    resume_from_checkpoint_path = training_config.get("resume_from_checkpoint", None)
    if resume_from_checkpoint_path and os.path.exists(resume_from_checkpoint_path):
        print(f"[Resume] Loading from external checkpoint: {resume_from_checkpoint_path}")
        try:
            state = torch.load(resume_from_checkpoint_path, map_location=device)
            model.load_state_dict(state["model_state_dict"])
            optim.load_state_dict(state["optimizer_state_dict"])
            starting_step = state.get("train_step", 0)
            starting_epoch = state.get("current_epoch", 0)
            for _ in range(starting_step + 1):
                if not is_sudoku_task and cur is not None:
                    cur.update()
            if use_multi_epoch:
                print(f"[Resume] Successfully resumed from external checkpoint: epoch {starting_epoch}, step {starting_step}")
            else:
                print(f"[Resume] Successfully resumed from external checkpoint: step {starting_step}")
        except (FileNotFoundError, EOFError, RuntimeError) as e:
            print(f"❌ [Resume] Failed to load checkpoint from {resume_from_checkpoint_path}")
            print(f"   Error: {e}")
            print(f"   The checkpoint file may be corrupted or incomplete.")
            print(f"   Solution: Delete the corrupted checkpoint file and restart training from scratch.")
            print(f"   Or specify a different checkpoint path in the config.")
            raise
    elif os.path.exists(state_path):
        print(f"[Resume] Found checkpoint at {state_path}, resuming training...")
        try:
            state = torch.load(state_path, map_location=device)
            model.load_state_dict(state["model_state_dict"])
            optim.load_state_dict(state["optimizer_state_dict"])
            starting_step = state.get("train_step", 0)
            starting_epoch = state.get("current_epoch", 0)  # 🆕 恢复epoch信息
            for _ in range(starting_step + 1):
                if not is_sudoku_task and cur is not None:
                    cur.update()
            if use_multi_epoch:
                print(f"[Resume] Successfully resumed from epoch {starting_epoch}, step {starting_step}")
            else:
                print(f"[Resume] Successfully resumed from step {starting_step}")
        except (FileNotFoundError, EOFError, RuntimeError) as e:
            print(f"❌ [Resume] Failed to load checkpoint from {state_path}")
            print(f"   Error: {e}")
            print(f"   The checkpoint file may be corrupted or incomplete.")
            print(f"   Solution: Delete the corrupted checkpoint file '{state_path}' and restart training from scratch.")
            print(f"   Command: rm {state_path}")
            raise
    
    return starting_step, starting_epoch


def setup_wandb(wandb_cfg, config, use_distributed=False, accelerator=None):
    """
    初始化 WandB
    
    Args:
        wandb_cfg: WandB 配置字典
        config: 完整配置字典
        use_distributed: 是否使用分布式训练
        accelerator: Accelerator 实例（如果使用分布式训练）
    
    Returns:
        should_log_wandb: bool
    """
    # 确定是否需要记录日志（只有主进程且配置开启时才记录）
    if use_distributed:
        should_log_wandb = accelerator.is_main_process and wandb_cfg.get("log", False)
    else:
        should_log_wandb = wandb_cfg.get("log", False)
    
    if should_log_wandb:
        try:
            import wandb
            wandb.init(
                project=wandb_cfg.get("project", "in-context-learning-prompt-respond"),
                entity=wandb_cfg.get("entity", None),
                name=wandb_cfg.get("name", "experiment"),
                notes=wandb_cfg.get("notes", "Prompt-Respond training"),
                config=config,
                resume="allow",  # 允许resume（用于断点续训）
                settings=wandb.Settings(init_timeout=300),  # 增加超时时间到 300 秒
            )
            # 🔧 验证 wandb 是否成功初始化
            if wandb.run is None:
                raise RuntimeError("wandb.init() completed but wandb.run is None")
            if use_distributed:
                print(f"[WandB] 主进程 (Rank {accelerator.process_index}) 已初始化 WandB")
        except Exception as e:
            print(f"⚠️  Wandb initialization failed: {e}")
            print("⚠️  Continuing training without wandb logging...")
            should_log_wandb = False
            wandb_cfg["log"] = False  # 禁用 wandb 日志记录
    
    return should_log_wandb


def check_training_complete(starting_step, total_steps, state_path, device, model, optim, config, use_distributed=False, accelerator=None, mse_logger=None):
    """
    检查训练是否已完成
    
    Args:
        starting_step: 起始步数
        total_steps: 总步数
        state_path: 检查点文件路径
        device: torch.device
        model: 模型实例
        optim: 优化器实例
        config: 完整配置字典
        use_distributed: 是否使用分布式训练
        accelerator: Accelerator 实例（如果使用分布式训练）
        mse_logger: LocalMSELogger 实例
    
    Returns:
        is_complete: bool - 如果训练已完成则返回 True
    """
    if starting_step >= total_steps:
        print(f"\n{'='*70}")
        print(f"✅ 训练已完成！")
        print(f"   当前step: {starting_step}")
        print(f"   目标step: {total_steps}")
        print(f"   训练已完成，无需继续训练")
        print(f"{'='*70}\n")
        
        # 检查最终checkpoint是否已存在且正确
        if os.path.exists(state_path):
            try:
                state = torch.load(state_path, map_location=device)
                final_step = state.get("train_step", 0)
                if final_step >= total_steps:
                    print(f"✅ 最终checkpoint已存在（step {final_step}），跳过保存和评估")
                    print(f"   如需重新评估，请手动运行评估脚本")
                    return True
                else:
                    print(f"⚠️  Checkpoint步数 ({final_step}) 小于目标步数 ({total_steps})")
                    print(f"   将保存最终checkpoint（使用当前checkpoint的状态）")
            except Exception as e:
                print(f"⚠️  无法读取checkpoint: {e}")
                print(f"   将保存最终checkpoint")
        else:
            print(f"⚠️  未找到checkpoint，将保存最终checkpoint")
        
        # 保存最终checkpoint（如果不存在或步数不对）
        # 注意：此时模型状态应该已经从checkpoint加载了
        if mse_logger is not None:
            mse_logger.flush_train(force=True)
        # 🆕 分布式训练：只在主进程保存最终 checkpoint
        if not use_distributed or accelerator.is_main_process:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "train_step": total_steps,
                "config": config,
            }, state_path)
        print(f"[Training Complete] Final checkpoint saved")
        print(f"\n💡 提示: 训练已完成，跳过最终评估")
        print(f"   如需重新评估，请手动运行评估脚本")
        return True
    
    return False


# ============================================================
# Learning Rate Scheduling Utilities
# ============================================================

def get_learning_rate(step, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    """
    计算学习率（warmup + cosine decay）
    
    这是数独任务和 core-nebula 对齐训练的标准学习率调度策略：
    - Warmup 阶段：线性从 0 增加到 learning_rate
    - Cosine 衰减阶段：从 learning_rate 余弦衰减到 min_lr
    
    Args:
        step: 当前训练步数
        learning_rate: 峰值学习率（warmup 后的目标值）
        warmup_iters: warmup 步数
        lr_decay_iters: 学习率衰减的总步数（通常等于 max_steps）
        min_lr: 最小学习率（cosine 衰减的最终值）
    
    Returns:
        lr: 当前步数的学习率
    """
    import math
    
    # Warmup 阶段：线性增加
    if step < warmup_iters:
        return learning_rate * step / max(1, warmup_iters)
    
    # 衰减阶段结束：固定为 min_lr
    if step > lr_decay_iters:
        return min_lr
    
    # Cosine 衰减阶段
    decay_ratio = (step - warmup_iters) / max(1, (lr_decay_iters - warmup_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def extract_classification_model_outputs(results):
    """
    从分类任务模型的 forward 返回值中统一提取 loss, output, respond_masked_indices

    适用于: Sudoku, Pathfinding 等分类任务

    统一协议：所有分类模型在 train_mode=True 时返回：
    - Dream: (loss, logits, None, mask)
    - AR: (loss, logits)
    - MDM (LLaDA): (loss, logits, t, mask)
    - Pathfinding: (loss, logits)

    Args:
        results: 模型 forward 的返回值（tuple 或单个值）

    Returns:
        loss: 损失值（Tensor）
        output: 输出 logits [B, n_respond, seq_len, vocab_size]
        respond_masked_indices: mask 张量 [B, n_respond, seq_len]，或 None
    """
    if isinstance(results, tuple):
        loss = results[0]  # 第一个总是 loss
        output = results[1]  # 第二个是 logits
        # 可选：respond_masked_indices（根据不同模型的返回值格式提取）
        # Dream: (loss, sol_logits, None, sol_mask) -> mask at index 3
        # AR: (loss, sol_logits) -> no mask
        # MDM (LLaDA): (loss, sol_logits, t, mask_in_sol) -> mask at index 3
        if len(results) >= 4:
            # 如果有4个或更多元素，第4个通常是mask
            respond_masked_indices = results[3] if isinstance(results[3], torch.Tensor) and results[3].dtype == torch.bool else None
        elif len(results) == 3:
            # 3个元素：检查第3个是否是bool tensor（可能是mask）
            respond_masked_indices = results[2] if isinstance(results[2], torch.Tensor) and results[2].dtype == torch.bool else None
        else:
            respond_masked_indices = None
    else:
        # 非 tuple 返回值（不应该发生，但兼容处理）
        loss = results
        output = None
        respond_masked_indices = None
        raise ValueError(f"Unexpected model return type: {type(results)}")
    
    return loss, output, respond_masked_indices


def normalize_classification_output(output):
    """
    归一化分类任务模型的 output 形状：[B, n_respond, seq_len, vocab] -> [B, 1, seq_len, vocab]

    适用于: Sudoku, Pathfinding 等分类任务

    统一协议：所有分类模型的 output 应归一化为 [B, 1, seq_len, vocab_size]（取最后一个 respond）
    注：模型内部已统一返回 [B, 1, seq_len, vocab_size]，但保留此逻辑作为兜底

    Args:
        output: 模型输出 logits [B, n_respond, seq_len, vocab_size]

    Returns:
        normalized_output: [B, 1, seq_len, vocab_size]
    """
    if output is not None and output.dim() == 4:
        if output.shape[1] > 1:
            output = output[:, -1:, :, :]  # 取最后一个 respond: [B, n_respond, 81, 10] -> [B, 1, 81, 10]
    return output


# Backward compatibility aliases
extract_sudoku_model_outputs = extract_classification_model_outputs
normalize_sudoku_output = normalize_classification_output


def optimizer_step(loss, optimizer, accelerator=None, scaler=None, clip_grad_norm=None):
    """
    统一的优化器步骤：zero_grad + backward + step（支持分布式和混合精度）
    
    Args:
        loss: 损失值（Tensor）
        optimizer: 优化器实例
        accelerator: Accelerator 实例（分布式训练时使用，可选）
        scaler: GradScaler 实例（混合精度训练时使用，可选）
        clip_grad_norm: 梯度裁剪的最大范数（可选，默认不裁剪）
    
    Returns:
        None
    """
    optimizer.zero_grad(set_to_none=True)
    
    # 反向传播（支持分布式和混合精度）
    if scaler is not None and scaler.is_enabled():
        # 混合精度训练
        scaler.scale(loss).backward()
        if clip_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    elif accelerator is not None:
        # 分布式训练（使用 accelerator）
        accelerator.backward(loss)
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], clip_grad_norm)
        optimizer.step()
    else:
        # 标准训练
        loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], clip_grad_norm)
        optimizer.step()


def nebula_tokens_to_digits(tok81: torch.Tensor) -> torch.Tensor:
    """
    将 core-nebula token IDs 转换为数字（用于 Sudoku 验证）
    
    Token 映射：
    - token 0..8 -> digit 1..9
    - 其他 token（'$', '=', MASK等）-> digit 0（无效）
    
    Args:
        tok81: Token IDs [B, 81] 或 [81]
    
    Returns:
        digits: 数字 [B, 81] 或 [81]，范围 0..9（0 表示无效）
    """
    digits = torch.zeros_like(tok81)
    valid = (tok81 >= 0) & (tok81 <= 8)
    digits[valid] = tok81[valid] + 1
    return digits


def calculate_sudoku_accuracy(pred_digits: torch.Tensor, true_digits: torch.Tensor):
    """
    计算 Sudoku 准确率（cell accuracy 和 sudoku accuracy）
    
    Args:
        pred_digits: 预测数字 [B, 81]
        true_digits: 真实数字 [B, 81]
    
    Returns:
        cell_accuracy: 单元格准确率（所有单元格的平均准确率）
        sudoku_accuracy: 数独准确率（完全正确的数独比例）
    """
    correct = (pred_digits.long() == true_digits.long()).float()  # [B, 81]
    cell_accuracy = correct.mean().item()  # 所有格子的平均准确率
    sudoku_accuracy = (correct.mean(dim=-1) == 1.0).float().mean().item()  # 完全正确的数独比例
    return cell_accuracy, sudoku_accuracy


def calculate_sudoku_accuracy_masked(
    pred_digits: torch.Tensor,
    true_digits: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float]:
    """
    仅在被 mask 的位置上计算 Sudoku 准确率（与 loss 计算范围一致）。
    用于扩散类模型（BAD-AR、LLaDA 等）：loss 只对 masked 位置计算，故 train 指标应对齐。
    
    Args:
        pred_digits: [B, 81] 预测数字
        true_digits: [B, 81] 真实数字
        mask: [B, 81] 或 [B, 1, 81] 布尔型，True 表示该位置参与 loss（被 mask）
    
    Returns:
        cell_accuracy: 仅 masked 位置的平均准确率
        sudoku_accuracy: 仅考虑 masked 位置时「该样本所有 masked 全对」的比例（仅统计有 masked 的样本）
    """
    correct = (pred_digits.long() == true_digits.long()).float()  # [B, 81]
    m = mask.float()
    if m.dim() == 3:
        m = m.squeeze(1)  # [B, 81]
    m = m.to(correct.device)
    n = m.sum()
    if n < 1:
        return 0.0, 0.0
    cell_accuracy = (correct * m).sum().item() / n
    total_masked = m.sum(dim=-1)  # [B]
    correct_masked = (correct * m).sum(dim=-1)  # [B]
    all_correct = (correct_masked >= total_masked - 1e-6) & (total_masked >= 1e-6)
    valid = total_masked >= 1e-6
    sudoku_accuracy = all_correct[valid].float().mean().item() if valid.any() else 0.0
    return cell_accuracy, sudoku_accuracy
