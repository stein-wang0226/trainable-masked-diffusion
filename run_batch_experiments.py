#!/usr/bin/env python3
"""
一站式批量实验脚本（支持多GPU并行）
功能：生成配置 + 并行运行实验 + 记录结果

特性：
- 自动检测GPU数量并并行运行实验
- 支持手动指定GPU列表
- 实时显示进度和结果
- 自动记录失败实验列表
- 支持 'standard' 和 'large' 两种模型尺寸
bash run_batch_experiments.sh --generate-only
bash run_batch_experiments.sh --generate-only --use-all-block-sizes
"""

import os
import sys
import yaml
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# 最优参数加载
# ============================================================

def load_optimal_params(params_file='optimal_params.json'):
    """
    从JSON文件加载最优参数配置
    
    Args:
        params_file: 参数配置文件路径（默认：optimal_params.json）
    
    Returns:
        dict: 包含各模型最优参数的字典
    """
    params_path = Path(params_file)
    if not params_path.exists():
        print(f"⚠️  警告: 未找到最优参数文件 {params_file}，将使用默认值")
        return {}
    
    try:
        with open(params_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config.get('optimal_params', {})
    except Exception as e:
        print(f"⚠️  警告: 加载最优参数文件失败: {e}，将使用默认值")
        return {}

def get_optimal_block_sizes(model_type, dim=None, n_prompt=None, n_respond=None, optimal_params=None, use_all=False):
    """
    根据模型类型和任务维度获取最优 block_size 列表
    
    优先级：
    1. 如果 use_all=True，直接返回 BLOCK_SIZES（生成所有组合）
    2. 从 optimal_params.json 读取（如果存在且包含该模型）
    3. 使用后备默认值 BLOCK_SIZES（如果配置文件不存在或模型未配置）
    
    Args:
        model_type: 模型类型
        dim: 维度（可选，用于判断是否为简单任务）
        n_prompt: prompt 数量（可选）
        n_respond: respond 数量（可选）
        optimal_params: 最优参数字典（如果为None，会从文件加载）
        use_all: 如果为True，忽略最优参数，返回所有 block_sizes
    
    Returns:
        list: block_size 列表
    """
    # 如果 use_all=True，直接返回所有 block_sizes
    if use_all:
        return BLOCK_SIZES
    
    if optimal_params is None:
        optimal_params = load_optimal_params()
    
    # 后备默认值（当配置文件不存在或模型未配置时使用）
    default_block_sizes = BLOCK_SIZES
    
    if model_type not in optimal_params:
        return default_block_sizes
    
    model_params = optimal_params[model_type]
    
    # LLaDABlock 特殊处理：根据任务复杂度选择参数
    if model_type == 'llada_block':
        # 检查是否为简单任务
        simple_threshold = model_params.get('simple_task_threshold', {})
        is_simple = (
            dim is not None and dim <= simple_threshold.get('dim', 10) and
            n_prompt is not None and n_prompt <= simple_threshold.get('prompt', 10) and
            n_respond is not None and n_respond <= simple_threshold.get('respond', 10)
        )
        
        if is_simple and 'block_size_simple' in model_params:
            return model_params['block_size_simple']
        else:
            return model_params.get('block_size', default_block_sizes)
    
    # 其他模型直接返回 block_size
    return model_params.get('block_size', default_block_sizes)

def get_optimal_bpd_k_values(optimal_params=None, use_all=False):
    """
    获取 BPD-AR 的最优 K 值列表
    
    优先级：
    1. 如果 use_all=True，直接返回 BPD_K_VALUES（生成所有组合）
    2. 从 optimal_params.json 读取（如果存在且包含 bpd_ar）
    3. 使用后备默认值 BPD_K_VALUES（如果配置文件不存在或未配置）
    
    Args:
        optimal_params: 最优参数字典（如果为None，会从文件加载）
        use_all: 如果为True，忽略最优参数，返回所有 K 值
    
    Returns:
        list: K 值列表
    """
    # 如果 use_all=True，直接返回所有 K 值
    if use_all:
        return BPD_K_VALUES
    
    if optimal_params is None:
        optimal_params = load_optimal_params()
    
    # 后备默认值（当配置文件不存在或模型未配置时使用）
    default_k_values = BPD_K_VALUES
    
    if 'bpd_ar' not in optimal_params:
        return default_k_values
    
    return optimal_params['bpd_ar'].get('k_values', default_k_values)

# ============================================================
# 实验参数配置
# ============================================================

DIMS = [10,15]
PROMPTS = [20]
RESPONDS = [20]
MODEL_TYPES = ['llada_block','bop_ar','bad_ar','llama','llada']
# MODEL_TYPES = ['bad_ar']  # 默认运行
# AR模型列表（用于判断训练步数）
AR_MODEL_TYPES = ['llama', 'gpt2', 'gptj', 'qwen', 'qwen2', 'qwen2.5', 'llama2', 'llama3']
# SIZE_KEYS = ['standard', 'large'] # 新增：模型尺寸标识
SIZE_KEYS = ['standard','small','big'] # 默认只运行 standard 模型
# 🆕 Block Diffusion 的 block_size 列表（用于 LLaDABlock、BOP-AR 和 RBO-AR 消融实验）
# 参考文档：推荐值 1 (AR-like), 4-10 (medium), 20 (MDM-like)
# None 表示 baseline（不使用 block diffusion，仅用于 LLaDABlock）
# 注：SDAR 已暂时注释（效果不佳，改用 LLaDABlock）
# 🆕 现在从 optimal_params.json 读取最优参数，这里仅作为后备默认值（当配置文件不存在时使用）
BLOCK_SIZES = [2,4,10]  # 后备默认值，实际会从 optimal_params.json 读取
# 🆕 Sequence modes（用于 BOP-AR 和其他模型测试）
# SEQUENCE_MODES = ['sequential']  # 已移除：只使用 sequential 模式，避免混淆
# 所有实验统一使用 sequential 模式（sequence_mode='sequential'）
# 🆕 随机种子列表（支持多种子实验）
RANDOM_SEEDS = [1,2,3]  # 默认1个种子，可以通过命令行参数覆盖
# 🆕 BPD-AR 的 K 值列表（并行度）
# 🆕 现在从 optimal_params.json 读取最优参数，这里仅作为后备默认值（当配置文件不存在时使用）
BPD_K_VALUES = [2]  # 后备默认值，实际会从 optimal_params.json 读取
# Standard 模型大小（作为基线）



SMALL_SIZE = {
    'n_embd': 192,      # 宽度减小 (256 -> 192)
    'n_layers': 8,       # 深度减小 (12 -> 8)
    'n_heads': 6,       # 保持 head_dim = 192/6 = 32，与 Standard 一致
    'mlp_ratio': 4.0,
}
STANDARD_SIZE = {
    'n_embd': 256,
    'n_layers': 12,
    'n_heads': 8,
    'mlp_ratio': 4.0,
}
# big 模型大小（参数量约是 Standard 的 3 倍，更合理）
BIG_SIZE = {
    'n_embd': 384,      # 维度适度增加（256 -> 384）
    'n_layers': 16,     # 层数适度增加（12 -> 18）
    'n_heads': 12,      # 保持 n_embd % n_heads == 0
    'mlp_ratio': 4.0,
}



# Large 模型大小（参数量约是 Standard 的 6 倍，更合理）
LARGE_SIZE = {
    'n_embd': 512,      # 维度适度增加（256 -> 384）
    'n_layers': 20,     # 层数适度增加（12 -> 18）
    'n_heads': 16,      # 保持 n_embd % n_heads == 0
    'mlp_ratio': 4.0,
}
# LARGE_SIZE = {
#     'n_embd': 384,      # 维度适度增加（256 -> 384）
#     'n_layers': 24,     # 层数适度增加（12 -> 18）
#     'n_heads': 8,      # 保持 n_embd % n_heads == 0
#     'mlp_ratio': 4.0,
# }
SIZE_SETTINGS = {
    'standard': STANDARD_SIZE,
    'small': SMALL_SIZE,
    'big': BIG_SIZE,
    'large': LARGE_SIZE,
}

# ============================================================
# 配置模板
# ============================================================

def create_config(model_type, size_key, dim, n_prompt, n_respond, random_seed=42,
                  block_size=None, sequence_mode='sequential'):
    """
    创建实验配置

    Args:
        model_type: 模型类型 ('llama', 'llada', 'llada_block', 'bop_ar', 'rbo_ar', 'ebo_ar', 'bpd_ar', 'bad_ar')
                   注：'sdar' 已暂时停用
        size_key: 模型尺寸 ('standard', 'large', 'big')
        dim: 维度
        n_prompt: prompt 数量
        n_respond: respond 数量
        random_seed: 随机种子
        block_size: Block size（用于 LLaDABlock、BOP-AR、RBO-AR、EBO-AR、BPD-AR、BAD-AR，None=baseline/不使用block diffusion）
                   注：LLaDABlock 支持 None (baseline)，BOP-AR、RBO-AR、EBO-AR、BPD-AR、BAD-AR 需要指定 block_size
        sequence_mode: 序列模式 ('sequential', 'non_sequential')
    """

    # 1. 根据 size_key 获取模型尺寸配置
    size_config = SIZE_SETTINGS[size_key]

    # 模型配置
    # ===== SDAR 已暂时注释（效果不佳，改用 LLaDABlock） =====
    # if model_type == 'sdar':
    #     # 🆕 SDAR 模型配置（支持 Block Diffusion）
    #     if block_size is None:
    #         raise ValueError("SDAR 模型必须指定 block_size 参数")
    #
    #     model_config = {
    #         'family': 'sdar',
    #         'n_dims': dim,
    #         'n_positions': 101,  # 应大于 n_prompt + n_respond * 2
    #         'n_prompt': n_prompt,
    #         'n_respond': n_respond,
    #         'n_embd': size_config['n_embd'],
    #         'n_layers': size_config['n_layers'],
    #         'n_heads': size_config['n_heads'],
    #         'mlp_ratio': size_config['mlp_ratio'],
    #         'mask_epsilon': 0.1,
    #         'loss_weight_type': '1/t',
    #         'train_mask_ratio': 1.0,
    #         'eval_mask_ratio': 1.0,
    #         'eval_mask_mode': 'fixed',
    #         'use_prompt_context': True,
    #         # 🆕 Block Diffusion 参数
    #         'use_block_diffusion': True,
    #         'block_size': block_size,
    #         # 🆕 Block-by-block inference 参数（v2.1 新增）
    #         # 默认启用 block-by-block inference（与 use_block_diffusion 一致）
    #         'use_block_by_block_inference': True,  # 显式启用，确保使用新的 block-by-block 推理逻辑
    #         'inference_steps_per_block': None,  # None: 自动从 inference.steps 计算
    #         'inference': {
    #             'use_multistep_inference': False,
    #             'steps': 10,
    #             'scheduler': 'linear',
    #             'confidence_alg': 'entropy',
    #         }
    #     }
    if model_type == 'sdar':
        raise ValueError("SDAR 模型已暂时停用（效果不佳，改用 LLaDABlock）")
    elif model_type == 'llada_block':
        # 🆕 LLaDABlock 模型配置（支持 Block Diffusion）
        use_block_diff = block_size is not None  # None 表示 baseline 模式

        model_config = {
            'family': 'llada_block',
            'n_dims': dim,
            'n_positions': 101,
            'n_prompt': n_prompt,
            'n_respond': n_respond,
            'n_embd': size_config['n_embd'],
            'n_layers': size_config['n_layers'],
            'n_heads': size_config['n_heads'],
            'mlp_ratio': size_config['mlp_ratio'],
            'block_group_size': 1,
            'mask_epsilon': 0.1,
            'loss_weight_type': '1/t',
            'train_mask_ratio': 1.0,
            'eval_mask_ratio': 1.0,
            'eval_mask_mode': 'fixed',
            'use_prompt_context': True,
            # 🆕 Block Diffusion 参数
            'use_block_diffusion': use_block_diff,
            'inference': {
                'use_multistep_inference': False,
                'steps': 10,
                'scheduler': 'linear',
                'confidence_alg': 'entropy',
            }
        }
        # 只在启用 block diffusion 时添加 block_size 参数
        if use_block_diff:
            model_config['block_size'] = block_size

    elif model_type == 'bop_ar':
        # 🆕 BOP-AR (ScatDiff) 模型配置（Offset-based Autoregressive）
        # 支持 block_size 参数控制垂直生成深度
        model_config = {
            'family': 'bop_ar',
            'n_dims': dim,
            'n_positions': 101,
            'n_prompt': n_prompt,
            'n_respond': n_respond,
            'n_embd': size_config['n_embd'],
            'n_layers': size_config['n_layers'],
            'n_heads': size_config['n_heads'],
            'mlp_ratio': size_config['mlp_ratio'],
            'block_group_size': 1,
            'mask_epsilon': 0.1,
            'loss_weight_type': '1/t',
            'train_mask_ratio': 1.0,
            'eval_mask_ratio': 1.0,
            'eval_mask_mode': 'fixed',
            'use_prompt_context': True,
            'inference': {
                'use_multistep_inference': False,
                'steps': 10,
                'scheduler': 'linear',
                'confidence_alg': 'entropy',
            }
        }
        # 🆕 添加 block_size 参数（如果指定）
        # block_size 控制 ScatDiff 的垂直生成深度
        # block_size=1: 2 layers (x, y)
        # block_size=5: 10 layers
        # block_size=10: 20 layers
        if block_size is not None:
            model_config['block_size'] = block_size

    elif model_type == 'rbo_ar':
        # 🆕 RBO-AR (Random Block-Order Autoregressive) 模型配置
        # 支持 block_size 参数控制块大小（每个块包含 block_size 个点）
        model_config = {
            'family': 'rbo_ar',
            'n_dims': dim,
            'n_positions': 101,
            'n_prompt': n_prompt,
            'n_respond': n_respond,
            'n_embd': size_config['n_embd'],
            'n_layers': size_config['n_layers'],
            'n_heads': size_config['n_heads'],
            'mlp_ratio': size_config['mlp_ratio'],
            'block_group_size': 1,
            'mask_epsilon': 0.1,
            'loss_weight_type': '1/t',
            'train_mask_ratio': 1.0,
            'eval_mask_ratio': 1.0,
            'eval_mask_mode': 'fixed',
            'use_prompt_context': True,
            'inference': {
                'use_multistep_inference': False,
                'steps': 10,
                'scheduler': 'linear',
                'confidence_alg': 'entropy',
            }
        }
        # 🆕 添加 block_size 参数（如果指定）
        # block_size 控制每个块包含的点数（每个块有 2*block_size 个 position）
        # block_size=1: 最细粒度随机排序
        # block_size=4: 中等粒度随机排序
        # block_size=10: 粗粒度随机排序
        if block_size is not None:
            model_config['block_size'] = block_size
        # RBO-AR 默认使用随机优先级
        model_config['random_order'] = True
        model_config['priority_seed'] = None  # None=每次随机

    elif model_type == 'ebo_ar':
        # 🆕 EBO-AR (Entropy-based Block-Order Autoregressive) 模型配置
        # EBO-AR 使用与 RBO-AR 相同的模型结构，但推理时使用熵引导优先级
        # 支持 block_size 参数控制块大小（每个块包含 block_size 个点）
        model_config = {
            'family': 'rbo_ar',  # EBO-AR 使用 RBO-AR 模型，通过 evaluation.use_ebo_inference 区分
            'n_dims': dim,
            'n_positions': 101,
            'n_prompt': n_prompt,
            'n_respond': n_respond,
            'n_embd': size_config['n_embd'],
            'n_layers': size_config['n_layers'],
            'n_heads': size_config['n_heads'],
            'mlp_ratio': size_config['mlp_ratio'],
            'block_group_size': 1,
            'mask_epsilon': 0.1,
            'loss_weight_type': '1/t',
            'train_mask_ratio': 1.0,
            'eval_mask_ratio': 1.0,
            'eval_mask_mode': 'fixed',
            'use_prompt_context': True,
            'inference': {
                'use_multistep_inference': False,
                'steps': 10,
                'scheduler': 'linear',
                'confidence_alg': 'entropy',
            }
        }
        # 🆕 添加 block_size 参数（如果指定）
        # block_size 控制每个块包含的点数（每个块有 2*block_size 个 position）
        if block_size is not None:
            model_config['block_size'] = block_size
        # EBO-AR 训练时使用随机优先级（与 RBO-AR 相同）
        model_config['random_order'] = True
        model_config['priority_seed'] = None  # None=每次随机
        # 🆕 EBO-AR 推理参数
        model_config['num_probe_samples'] = 3  # 回归任务方差估计的采样次数

    elif model_type == 'bpd_ar':
        # 🆕 BPD-AR (Block-wise Parallel Diffusion) 模型配置
        # BPD-AR 使用与 RBO-AR 相同的模型结构，但推理时使用块级并行加速
        # 支持 block_size 参数控制块大小（每个块包含 block_size 个点）
        # 注意：bpd_k 参数由 evaluation 部分控制（不在 model 部分）
        model_config = {
            'family': 'rbo_ar',  # BPD-AR 使用 RBO-AR 模型，通过 evaluation.use_bpd 区分
            'n_dims': dim,
            'n_positions': 101,
            'n_prompt': n_prompt,
            'n_respond': n_respond,
            'n_embd': size_config['n_embd'],
            'n_layers': size_config['n_layers'],
            'n_heads': size_config['n_heads'],
            'mlp_ratio': size_config['mlp_ratio'],
            'block_group_size': 1,
            'mask_epsilon': 0.1,
            'loss_weight_type': '1/t',
            'train_mask_ratio': 1.0,
            'eval_mask_ratio': 1.0,
            'eval_mask_mode': 'fixed',
            'use_prompt_context': True,
            'inference': {
                'use_multistep_inference': False,
                'steps': 10,
                'scheduler': 'linear',
                'confidence_alg': 'entropy',
            }
        }
        # 🆕 添加 block_size 参数（如果指定）
        # block_size 控制每个块包含的点数（每个块有 2*block_size 个 position）
        if block_size is not None:
            model_config['block_size'] = block_size
        # BPD-AR 训练时使用随机优先级（与 RBO-AR 相同）
        model_config['random_order'] = True
        model_config['priority_seed'] = None  # None=每次随机
        # 🆕 BPD-AR 推理参数
        model_config['num_probe_samples'] = 3  # 回归任务方差估计的采样次数

    elif model_type == 'bad_ar':
        # 🆕 BAD-AR (Block-level Autoregressive Diffusion) 模型配置
        # BAD-AR 结合了块间 Diffusion 和块内 AR 的混合范式
        # 支持 block_size 参数控制块大小（每个块包含 block_size 个点）
        if block_size is None:
            raise ValueError("BAD-AR 模型必须指定 block_size 参数")
        
        model_config = {
            'family': 'bad_ar',
            'n_dims': dim,
            'n_positions': 101,
            'n_prompt': n_prompt,
            'n_respond': n_respond,
            'n_embd': size_config['n_embd'],
            'n_layers': size_config['n_layers'],
            'n_heads': size_config['n_heads'],
            'mlp_ratio': size_config['mlp_ratio'],
            'block_group_size': 1,
            'mask_epsilon': 0.1,
            'loss_weight_type': '1/t',
            'train_mask_ratio': 1.0,
            'eval_mask_ratio': 1.0,
            'eval_mask_mode': 'fixed',
            'use_prompt_context': True,
            'inference': {
                'use_multistep_inference': False,
                'steps': 10,
                'scheduler': 'linear',
                'confidence_alg': 'entropy',
            },
            'block_size': block_size,  # BAD-AR 核心参数：每个块包含的点数
        }

    elif model_type == 'llada':
        model_config = {
            'family': 'llada',
            'n_dims': dim,
            # n_positions 应大于 n_prompt + n_respond * 2
            'n_positions': 101,
            'n_prompt': n_prompt,
            'n_respond': n_respond,
            'n_embd': size_config['n_embd'],
            'n_layers': size_config['n_layers'],
            'n_heads': size_config['n_heads'],
            'mlp_ratio': size_config['mlp_ratio'],
            'block_group_size': 1,
            'mask_epsilon': 0.1,
            'loss_weight_type': '1/t',
            'train_mask_ratio': 1.0,
            'eval_mask_ratio': 1.0,
            'eval_mask_mode': 'fixed',
            'use_prompt_context': True,
            'inference': {
                'use_multistep_inference': False,
                'steps': 10,
                'scheduler': 'linear',
                'confidence_alg': 'entropy',
            }
        }
    else:  # llama
        model_config = {
            'family': 'llama',
            'n_dims': dim,
            'n_positions': 101,
            'n_prompt': n_prompt,
            'n_respond': n_respond,
            'n_embd': size_config['n_embd'],
            'n_layer': size_config['n_layers'], # 注意：llama 使用 n_layer
            'n_head': size_config['n_heads'], # 注意：llama 使用 n_head
        }

    # 训练配置
    # 根据模型类型和维度设置不同的训练步数
    if model_type in AR_MODEL_TYPES:
        # AR模型：与MDM（DLM）模型一致，根据维度设置
        # D=10: 100万步, D=20: 200万步, D=30: 300万步
        dim_to_steps = {
            10: 1000000,   # D=10: 100万步
            15: 1500000,   # D=15: 150万步
            20: 2000000,   # D=20: 200万步
            30: 3000000,   # D=30: 300万步
        }
        train_steps = dim_to_steps.get(dim, 1000000)  # 默认100万步
    # elif model_type in ['sdar', 'llada_block', 'bop_ar']:  # SDAR已注释
    elif model_type in ['llada_block', 'bop_ar', 'rbo_ar', 'ebo_ar', 'bpd_ar', 'bad_ar']:
        # LLaDABlock, BOP-AR, RBO-AR, EBO-AR, BPD-AR, BAD-AR模型：与AR和DLM模型一致，根据维度设置
        dim_to_steps = {
            10: 1000000,   # D=10: 100万步
            15: 1500000,   # D=15: 150万步
            20: 2000000,   # D=20: 200万步
            30: 3000000,   # D=30: 300万步
        }
        train_steps = dim_to_steps.get(dim, 1000000)  # 默认100万步
    else:
        # DLM模型：根据维度设置
        # D=10: 100万步, D=20: 200万步, D=30: 300万步
        dim_to_steps = {
            10: 1000000,   # D=10: 100万步
            15: 1500000,   # D=15: 150万步
            20: 2000000,   # D=20: 200万步
            30: 3000000,   # D=30: 300万步
        }
        train_steps = dim_to_steps.get(dim, 1000000)  # 默认100万步

    training_config = {
        'task': 'linear_regression',
        'data': 'gaussian',
        'batch_size': 64,
        'learning_rate': 0.0001,
        'weight_decay': 0.0,
        'train_steps': train_steps,
        'save_every_steps': 5000,
        'w_type': 'gaussian',
        'num_tasks': None,
        'num_training_examples': None,
        'sequence_mode': sequence_mode,  # 🆕 支持不同的序列模式
        'task_kwargs': {},
        'log_interval': 200,
        'validation': {
            'batch_count': 50,
            'batch_size': 64,
            'seed': 1234,
            'eval_every_steps': 5000,
        },
        'curriculum': {
            'dims': {'start': dim, 'end': dim, 'inc': 0, 'interval': 10000000},
            'points': {'start': n_respond, 'end': n_respond, 'inc': 0, 'interval': 10000000}
        },
        'random_seed': random_seed,  # 🆕 添加随机种子到训练配置
    }

    # 实验名称（包含种子信息、block_size、sequence_mode）
    name_parts = [model_type]

    # 🆕 LLaDABlock, BOP-AR, RBO-AR, EBO-AR, BAD-AR 模型：在名称中包含 block_size（如果使用）
    if model_type in ['llada_block', 'bop_ar', 'rbo_ar', 'ebo_ar', 'bad_ar']:
        if block_size is not None:
            name_parts.append(f"bs{block_size}")
        elif model_type == 'llada_block':  # LLaDABlock支持baseline
            name_parts.append("baseline")
        # BOP-AR, RBO-AR, BAD-AR 如果没有指定 block_size，默认不添加 baseline 标记（将使用默认值）

    # 添加基本信息
    name_parts.extend([size_key, f"D{dim}", f"P{n_prompt}", f"R{n_respond}"])

    # 添加序列模式（如果不是默认的 sequential）
    if sequence_mode != 'sequential':
        name_parts.append(sequence_mode)

    # 添加种子
    name_parts.append(f"seed{random_seed}")

    exp_name = "_".join(name_parts)

    # WandB group 和 notes
    # ===== SDAR WandB 配置已注释（效果不佳，改用 LLaDABlock） =====
    # if model_type == 'sdar' and block_size is not None:
    #     wandb_group = f"{model_type}_{size_key}_bs{block_size}"
    #     wandb_notes = f"{size_key.capitalize()} experiment: {model_type.upper()} Block Diffusion (block_size={block_size}), D={dim}, P={n_prompt}, R={n_respond}, {sequence_mode}, seed={random_seed}"
    #     wandb_tags = [model_type, size_key, 'block_diffusion', f'block_size_{block_size}', sequence_mode, f'D{dim}', f'P{n_prompt}', f'R{n_respond}', f'seed{random_seed}']
    if model_type == 'llada_block':
        if block_size is not None:
            wandb_group = f"{model_type}_{size_key}_bs{block_size}"
            wandb_notes = f"{size_key.capitalize()} experiment: LLaDABlock (block_size={block_size}), D={dim}, P={n_prompt}, R={n_respond}, {sequence_mode}, seed={random_seed}"
            wandb_tags = [model_type, size_key, 'block_diffusion', f'block_size_{block_size}', sequence_mode, f'D{dim}', f'P{n_prompt}', f'R{n_respond}', f'seed{random_seed}']
        else:
            wandb_group = f"{model_type}_{size_key}_baseline"
            wandb_notes = f"{size_key.capitalize()} experiment: LLaDABlock baseline, D={dim}, P={n_prompt}, R={n_respond}, {sequence_mode}, seed={random_seed}"
            wandb_tags = [model_type, size_key, 'baseline', sequence_mode, f'D{dim}', f'P{n_prompt}', f'R{n_respond}', f'seed{random_seed}']
    elif model_type == 'bop_ar':
        # 🆕 BOP-AR (ScatDiff): 包含 block_size 信息
        if block_size is not None:
            wandb_group = f"{model_type}_{size_key}_bs{block_size}"
            wandb_notes = f"{size_key.capitalize()} experiment: BOP-AR (ScatDiff, block_size={block_size}), D={dim}, P={n_prompt}, R={n_respond}, {sequence_mode}, seed={random_seed}"
            wandb_tags = [model_type, size_key, 'scatdiff', 'offset_autoregressive', f'block_size_{block_size}', sequence_mode, f'D{dim}', f'P{n_prompt}', f'R{n_respond}', f'seed{random_seed}']
        else:
            wandb_group = f"{model_type}_{size_key}"
            wandb_notes = f"{size_key.capitalize()} experiment: BOP-AR (ScatDiff, default), D={dim}, P={n_prompt}, R={n_respond}, {sequence_mode}, seed={random_seed}"
            wandb_tags = [model_type, size_key, 'scatdiff', 'offset_autoregressive', sequence_mode, f'D{dim}', f'P{n_prompt}', f'R{n_respond}', f'seed{random_seed}']
    elif model_type == 'rbo_ar':
        # 🆕 RBO-AR (Random Block-Order): 包含 block_size 信息
        if block_size is not None:
            wandb_group = f"{model_type}_{size_key}_bs{block_size}"
            wandb_notes = f"{size_key.capitalize()} experiment: RBO-AR (Random Block-Order, block_size={block_size}), D={dim}, P={n_prompt}, R={n_respond}, {sequence_mode}, seed={random_seed}"
            wandb_tags = [model_type, size_key, 'random_block_order', 'rbo_ar', f'block_size_{block_size}', sequence_mode, f'D{dim}', f'P{n_prompt}', f'R{n_respond}', f'seed{random_seed}']
        else:
            wandb_group = f"{model_type}_{size_key}"
            wandb_notes = f"{size_key.capitalize()} experiment: RBO-AR (Random Block-Order, default), D={dim}, P={n_prompt}, R={n_respond}, {sequence_mode}, seed={random_seed}"
            wandb_tags = [model_type, size_key, 'random_block_order', 'rbo_ar', sequence_mode, f'D{dim}', f'P{n_prompt}', f'R{n_respond}', f'seed{random_seed}']
    elif model_type == 'ebo_ar':
        # 🆕 EBO-AR (Entropy-based Block-Order): 包含 block_size 信息
        if block_size is not None:
            wandb_group = f"{model_type}_{size_key}_bs{block_size}"
            wandb_notes = f"{size_key.capitalize()} experiment: EBO-AR (Entropy-based Block-Order, block_size={block_size}), D={dim}, P={n_prompt}, R={n_respond}, {sequence_mode}, seed={random_seed}, num_probe_samples=3"
            wandb_tags = [model_type, size_key, 'entropy_based', 'ebo_ar', f'block_size_{block_size}', sequence_mode, f'D{dim}', f'P{n_prompt}', f'R{n_respond}', f'seed{random_seed}', 'probe_samples_3']
        else:
            wandb_group = f"{model_type}_{size_key}"
            wandb_notes = f"{size_key.capitalize()} experiment: EBO-AR (Entropy-based Block-Order, default), D={dim}, P={n_prompt}, R={n_respond}, {sequence_mode}, seed={random_seed}, num_probe_samples=3"
            wandb_tags = [model_type, size_key, 'entropy_based', 'ebo_ar', sequence_mode, f'D{dim}', f'P{n_prompt}', f'R{n_respond}', f'seed{random_seed}', 'probe_samples_3']
    elif model_type == 'bpd_ar':
        # 🆕 BPD-AR (Block-wise Parallel Diffusion): 包含 block_size 和 K 值信息
        if block_size is not None:
            wandb_group = f"{model_type}_{size_key}_bs{block_size}"
            wandb_notes = f"{size_key.capitalize()} experiment: BPD-AR (Block-wise Parallel Diffusion, block_size={block_size}), D={dim}, P={n_prompt}, R={n_respond}, {sequence_mode}, seed={random_seed}, num_probe_samples=3"
            wandb_tags = [model_type, size_key, 'parallel_diffusion', 'bpd_ar', f'block_size_{block_size}', sequence_mode, f'D{dim}', f'P{n_prompt}', f'R{n_respond}', f'seed{random_seed}', 'probe_samples_3']
        else:
            wandb_group = f"{model_type}_{size_key}"
            wandb_notes = f"{size_key.capitalize()} experiment: BPD-AR (Block-wise Parallel Diffusion, default), D={dim}, P={n_prompt}, R={n_respond}, {sequence_mode}, seed={random_seed}, num_probe_samples=3"
            wandb_tags = [model_type, size_key, 'parallel_diffusion', 'bpd_ar', sequence_mode, f'D{dim}', f'P{n_prompt}', f'R{n_respond}', f'seed{random_seed}', 'probe_samples_3']
    elif model_type == 'bad_ar':
        # 🆕 BAD-AR (Block-level Autoregressive Diffusion): 包含 block_size 信息
        if block_size is not None:
            wandb_group = f"{model_type}_{size_key}_bs{block_size}"
            wandb_notes = f"{size_key.capitalize()} experiment: BAD-AR (Block-level Autoregressive Diffusion, block_size={block_size}), D={dim}, P={n_prompt}, R={n_respond}, {sequence_mode}, seed={random_seed}"
            wandb_tags = [model_type, size_key, 'block_diffusion', 'intra_block_ar', f'block_size_{block_size}', sequence_mode, f'D{dim}', f'P{n_prompt}', f'R{n_respond}', f'seed{random_seed}']
        else:
            wandb_group = f"{model_type}_{size_key}"
            wandb_notes = f"{size_key.capitalize()} experiment: BAD-AR (Block-level Autoregressive Diffusion, default), D={dim}, P={n_prompt}, R={n_respond}, {sequence_mode}, seed={random_seed}"
            wandb_tags = [model_type, size_key, 'block_diffusion', 'intra_block_ar', sequence_mode, f'D{dim}', f'P{n_prompt}', f'R{n_respond}', f'seed{random_seed}']
    else:
        wandb_group = f"{model_type}_{size_key}"
        wandb_notes = f"{size_key.capitalize()} experiment: {model_type.upper()}, D={dim}, P={n_prompt}, R={n_respond}, {sequence_mode}, seed={random_seed}"
        wandb_tags = [model_type, size_key, sequence_mode, f'D{dim}', f'P{n_prompt}', f'R{n_respond}', f'seed{random_seed}']

    # 🆕 使用 OSS 挂载路径（集群环境）
    # 统一路径格式
    # out_dir = f\"/path/to/your/batch_checkpoints_selected/{exp_name}\"  # TODO: customize if needed
    # out_dir = f\"/path/to/your/batch_checkpoints_block_size_comp/{exp_name}\"  # TODO: customize if needed
    out_dir = f\"./outputs/batch_checkpoints_model_size_comp/{exp_name}\"

    # WandB 配置
    wandb_config = {
        'log': True,
        # 项目名区分，或使用统一项目
        'project': "in-context-learning-prompt-respond-bs_comp", # todo modify block_size_comp
        'name': exp_name,
        'group': wandb_group,
        'log_every_steps': 200,
        'notes': wandb_notes,
        'tags': wandb_tags
    }

    # 组装完整配置
    evaluation_config = {'use_autoregressive_eval': (model_type == 'llama')}
    # 🆕 EBO-AR: 添加 use_ebo_inference 参数
    if model_type == 'ebo_ar':
        evaluation_config['use_ebo_inference'] = True
    # 🆕 BPD-AR: 添加 use_bpd 参数（注意：bpd_k 由 create_bpd_config 脚本处理）
    elif model_type == 'bpd_ar':
        # BPD-AR 在配置文件中已经设置了 use_bpd 和 bpd_k
        # 这里保持默认 evaluation_config 即可
        pass
    
    return {
        'model': model_config,
        'training': training_config,
        'out_dir': out_dir,
        'wandb': wandb_config,
        'evaluation': evaluation_config
    }, exp_name


# ============================================================
# 日志重定向类
# ============================================================

class Tee:
    """同时输出到控制台和文件的类"""
    def __init__(self, file_path):
        self.file = open(file_path, 'a', encoding='utf-8')
        self.stdout = sys.stdout
    
    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()
    
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()


# ============================================================
# GPU 检测和并行运行函数
# ============================================================

def detect_available_gpus():
    """
    检测系统中可用的计算设备数量（GPU/NPU）
    支持集群环境（通过环境变量）
    支持阿里 NPU 环境
    
    Returns:
        list: 可用设备 ID列表，如 [0, 1, 2, 3]
    """
    # 🆕 优先检查 CUDA_VISIBLE_DEVICES（集群环境常用）
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible:
        try:
            # 解析 CUDA_VISIBLE_DEVICES，如 "0,1,2,3" 或 "0-3"
            if ',' in cuda_visible:
                # 格式: "0,1,2,3"
                gpu_ids = [int(x.strip()) for x in cuda_visible.split(',') if x.strip()]
            elif '-' in cuda_visible:
                # 格式: "0-3" -> [0,1,2,3]
                start, end = map(int, cuda_visible.split('-'))
                gpu_ids = list(range(start, end + 1))
            else:
                # 单个数字
                gpu_ids = [int(cuda_visible.strip())]
            
            if gpu_ids:
                # 返回逻辑设备 ID（从 0 开始）
                return list(range(len(gpu_ids)))
        except (ValueError, AttributeError):
            pass  # 如果解析失败，继续其他检测方法
    
    # 🆕 检查 NPU 相关环境变量（阿里云 NPU）
    # 阿里 NPU 可能使用 ASCEND_RT_VISIBLE_DEVICES 或类似变量
    npu_visible = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '') or \
               os.environ.get('NPU_VISIBLE_DEVICES', '') or \
               os.environ.get('DEVICE_ID', '')
    
    if npu_visible:
        try:
            # 尝试解析 NPU 设备
            if ',' in npu_visible:
                npu_ids = [int(x.strip()) for x in npu_visible.split(',') if x.strip()]
            elif '-' in npu_visible:
                start, end = map(int, npu_visible.split('-'))
                npu_ids = list(range(start, end + 1))
            else:
                npu_ids = [int(npu_visible.strip())]
            
            if npu_ids:
                return list(range(len(npu_ids)))
        except (ValueError, AttributeError):
            pass
    
    # 🆕 检查集群资源分配（通过环境变量推断）
    # 某些集群可能通过其他环境变量指定设备数量
    cluster_devices = os.environ.get('WORLD_SIZE', '') or \
                     os.environ.get('NPU_WORLD_SIZE', '') or \
                     os.environ.get('DEVICE_NUM', '')
    
    if cluster_devices:
        try:
            num_devices = int(cluster_devices)
            if num_devices > 0:
                return list(range(num_devices))
        except ValueError:
            pass
    
    # 回退到 nvidia-smi 检测（仅 NVIDIA GPU）
    try:
        # 执行 nvidia-smi -L 并统计行数，即为 GPU 数量
        gpu_list_output = subprocess.check_output(['nvidia-smi', '-L'], 
                                                  stderr=subprocess.DEVNULL).decode('utf-8')
        available_gpus = [i for i, line in enumerate(gpu_list_output.strip().split('\n')) if line]
        return available_gpus
    except (FileNotFoundError, subprocess.CalledProcessError):
        # 如果 nvidia-smi 不可用，可能是 NPU 环境
        # 🆕 对于 NPU 环境，如果检测不到设备，返回 [0] 表示单设备运行
        # 这样至少可以串行运行实验
        print("⚠️  未检测到 NVIDIA GPU，可能是 NPU 环境，将使用单设备模式")
        return [0]  # 返回单设备，允许串行运行


def run_single_experiment(config_path, gpu_ids=None, gpus_per_exp=1, use_distributed=False):
    """
    单个实验的运行函数，在单独的线程中执行。
    支持单卡和多卡分布式训练。
    
    Args:
        config_path: 配置文件路径
        gpu_ids: GPU ID列表（如果为None，则不限制GPU）
        gpus_per_exp: 每个实验使用的GPU数量（默认1，单卡训练）
        use_distributed: 是否使用分布式训练（accelerate launch）
    
    Returns:
        dict: 包含实验结果的字典
    """
    exp_name = config_path.stem
    timestamp = datetime.now().strftime('%H:%M:%S')
    
    if gpu_ids is not None:
        if len(gpu_ids) > 1:
            print(f"[{timestamp}] 🚀 GPUs {gpu_ids}: 启动 {exp_name} (分布式训练)", flush=True)
        else:
            print(f"[{timestamp}] 🚀 GPU {gpu_ids[0]}: 启动 {exp_name}", flush=True)
    else:
        print(f"[{timestamp}] 🚀 启动 {exp_name} (无GPU限制)", flush=True)
    
    exp_start = datetime.now()
    success = False
    error_msg = None
    
    try:
        # 设置环境变量，确保子进程只使用分配的 GPU
        env = os.environ.copy()
        if gpu_ids is not None:
            env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
        
        # 🆕 检测 ROCM 环境（AMD GPU）
        is_rocm_env = False
        try:
            import torch
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                is_rocm_env = True
            elif 'rocm' in torch.__version__.lower():
                is_rocm_env = True
        except:
            pass
        
        # 🔧 在 ROCM 环境下设置必要的环境变量（支持多卡训练，增强稳定性）
        if is_rocm_env:
            # 确保 ROCM 看到正确的设备（与 CUDA_VISIBLE_DEVICES 保持一致）
            if 'CUDA_VISIBLE_DEVICES' in env:
                env['HIP_VISIBLE_DEVICES'] = env['CUDA_VISIBLE_DEVICES']
            # 减少日志输出，防止日志缓冲区溢出导致的段错误（SIGSEGV）
            # 使用 'VERSION' 级别而不是 'INFO'，只显示版本信息，减少输出
            env['NCCL_DEBUG'] = 'VERSION'
            # 保持网络和 P2P 通信启用
            env.setdefault('NCCL_IB_DISABLE', '0')
            env.setdefault('NCCL_P2P_DISABLE', '0')
            # PyTorch 在 ROCM 环境下会自动使用 RCCL，但明确设置更安全
            # 可选：针对部分 AMD 卡的稳定性补丁（如果需要可以取消注释）
            # env.setdefault('HSA_OVERRIDE_GFX_VERSION', '10.3.0')
        
        # 🆕 检测是否在平台分布式环境中（由平台 launcher=accelerate 启动）
        # 如果已经在分布式环境中，不要再次调用 accelerate launch，避免嵌套启动
        world_size = int(env.get('WORLD_SIZE', '1'))
        is_platform_distributed = world_size > 1
        
        # 🆕 多卡分布式训练：使用 accelerate launch
        # 但如果在平台分布式环境中，不要再次启动 accelerate launch
        if use_distributed and len(gpu_ids) > 1 and not is_platform_distributed:
            # 使用 accelerate launch 进行分布式训练
            accelerate_config = 'src/conf/accelerate_configs/ddp_multi_gpu.yaml'
            
            # 🆕 为每个实验生成唯一的端口（避免冲突）
            # 使用实验名称的哈希值来生成端口号
            import hashlib
            port_base = 29500
            port_offset = int(hashlib.md5(exp_name.encode()).hexdigest()[:4], 16) % 1000
            master_port = port_base + port_offset
            
            # 构建 accelerate launch 命令
            cmd = [
                'accelerate', 'launch',
                '--config_file', accelerate_config,
                '--num_processes', str(len(gpu_ids)),
                '--num_machines', '1',
                '--machine_rank', '0',
                '--main_process_ip', '127.0.0.1',
                '--main_process_port', str(master_port),
                'src/train_prompt_respond.py',
                '--config', str(config_path)
            ]
        else:
            # 单卡训练：直接运行训练脚本
            cmd = [sys.executable, '-u', 'src/train_prompt_respond.py', '--config', str(config_path)]
        
        # 使用 subprocess.run 执行命令
        # 将输出重定向到日志文件（实时写入）
        # 🆕 日志文件也保存到 OSS 路径（与 out_dir 一致）
        # 注：SDAR 特殊路径已移除，统一使用 batch_checkpoints
        log_file = Path(f\"./outputs/batch_checkpoints/{exp_name}/training.log\")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w') as log_f:
            result = subprocess.run(
                cmd, 
                check=True, 
                env=env, 
                stdout=log_f,  # 实时输出到文件
                stderr=subprocess.STDOUT,
                text=True
            )
        success = True
        timestamp = datetime.now().strftime('%H:%M:%S')
        if gpu_ids is not None:
            if len(gpu_ids) > 1:
                print(f"[{timestamp}] ✅ GPUs {gpu_ids}: 完成 {exp_name}", flush=True)
            else:
                print(f"[{timestamp}] ✅ GPU {gpu_ids[0]}: 完成 {exp_name}", flush=True)
        else:
            print(f"[{timestamp}] ✅ 完成 {exp_name}", flush=True)
        
    except subprocess.CalledProcessError as e:
        timestamp = datetime.now().strftime('%H:%M:%S')
        if gpu_ids is not None:
            if len(gpu_ids) > 1:
                print(f"[{timestamp}] ❌ GPUs {gpu_ids}: 失败 {exp_name}", flush=True)
            else:
                print(f"[{timestamp}] ❌ GPU {gpu_ids[0]}: 失败 {exp_name}", flush=True)
        else:
            print(f"[{timestamp}] ❌ 失败 {exp_name}", flush=True)
        print(f"    错误代码: {e.returncode}", flush=True)
        # 从日志文件读取错误信息
        try:
            # 注：SDAR 特殊路径已移除，统一使用本地 batch_checkpoints 目录
            log_file = Path(f"./outputs/batch_checkpoints/{exp_name}/training.log")
            if log_file.exists():
                with open(log_file, 'r') as f:
                    log_content = f.read()
                    log_preview = log_content[-500:] if len(log_content) > 500 else log_content
                    print(f"    日志 (最后部分): \n{log_preview}", flush=True)
        except:
            pass
        error_msg = f"Return code: {e.returncode}"
        success = False
    
    except Exception as e:
        timestamp = datetime.now().strftime('%H:%M:%S')
        if gpu_ids is not None:
            if len(gpu_ids) > 1:
                print(f"[{timestamp}] ❌ GPUs {gpu_ids}: 异常 {exp_name}", flush=True)
            else:
                print(f"[{timestamp}] ❌ GPU {gpu_ids[0]}: 异常 {exp_name}", flush=True)
        else:
            print(f"[{timestamp}] ❌ 异常 {exp_name}", flush=True)
        print(f"    错误: {str(e)}", flush=True)
        error_msg = str(e)
        success = False
    
    exp_end = datetime.now()
    duration = (exp_end - exp_start).total_seconds()
    
    return {
        'name': exp_name,
        'success': success,
        'duration_minutes': duration / 60,
        'gpu_ids': gpu_ids,  # 🆕 改为 gpu_ids（列表）
        'error': error_msg
    }


# ============================================================
# 主程序
# ============================================================

def main():
    # 🆕 检测是否在分布式环境中（由平台 launcher=accelerate 启动）
    # 如果已经在分布式环境中，不要再次启动分布式训练，避免嵌套启动
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    
    is_distributed_env = world_size > 1
    
    if is_distributed_env:
        print(f"⚠️  检测到分布式环境: WORLD_SIZE={world_size}, RANK={rank}, LOCAL_RANK={local_rank}")
        print("⚠️  平台已通过 launcher=accelerate 启动，将跳过批量实验管理逻辑")
        print("⚠️  当前进程将直接运行单个实验（由平台分配）")
        # 在分布式环境中，应该只运行单个实验
        # 这里需要根据实际情况调整逻辑
        # 暂时先退出，提示用户使用单实验模式
        print("❌ 错误: 在分布式环境中，run_batch_experiments.py 不能作为批量管理器")
        print("💡 建议: 如果要在分布式环境中运行，请直接调用 src/train_prompt_respond.py")
        print("💡 或者: 使用单卡模式（--gpus-per-exp 1）进行批量实验")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description='批量运行标准和大型实验（支持多GPU并行）')
    parser.add_argument('--generate-only', action='store_true',
                       help='只生成配置文件，不运行')
    parser.add_argument('--run-only', action='store_true',
                       help='只运行已有配置，不重新生成')
    parser.add_argument('--model', choices=['llama', 'llada', 'llada_block', 'bop_ar', 'rbo_ar', 'ebo_ar', 'bpd_ar', 'bad_ar'],
                       help='只运行指定模型 (llama/llada/llada_block/bop_ar/rbo_ar/ebo_ar/bpd_ar/bad_ar)')
    parser.add_argument('--dim', type=int, choices=DIMS,
                       help='只运行指定维度')
    parser.add_argument('--size', choices=SIZE_KEYS,
                       help='只运行指定模型尺寸 (standard/large/big)')
    parser.add_argument('--gpus', type=str, default=None,
                       help='手动指定GPU列表，用逗号分隔，如: 0,1,2,3 (默认自动检测)')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='最大并行数（默认等于GPU数量）')
    parser.add_argument('--limit', type=int,
                       help='限制运行实验数量')
    parser.add_argument('--start-from', type=int, default=0,
                       help='从第N个实验开始')
    parser.add_argument('--daemon', action='store_true',
                       help='后台运行模式（使用nohup，终端断开不中断）')
    parser.add_argument('--log-file', type=str, default=None,
                       help='日志文件路径（默认：experiments_batch/run.log）')
    parser.add_argument('--show-output', action='store_true',
                       help='后台运行时也在终端显示输出（使用tee）')
    parser.add_argument('--seeds', type=str, default=None,
                       help='随机种子列表，用逗号分隔，如: 42,123,456 (默认使用RANDOM_SEEDS)')
    parser.add_argument('--block-sizes', type=str, default=None,
                       help='Block size 列表（用于 LLaDABlock 和 BOP-AR），用逗号分隔，如: 1,4,10 或 None,1,4,10 (None=baseline for LLaDABlock)')
    parser.add_argument('--bpd-k-values', type=str, default=None,
                       help='BPD-AR 的 K 值列表（并行度），用逗号分隔，如: 1,2,4 (默认从最优参数读取)')
    parser.add_argument('--use-all-block-sizes', action='store_true',
                       help='忽略 optimal_params.json，使用所有 block_sizes 生成配置（生成所有组合）')
    parser.add_argument('--gpus-per-exp', type=int, default=1,
                       help='每个实验使用的GPU数量（默认1，单卡训练；>1时启用分布式训练）')
    parser.add_argument('--use-distributed', action='store_true',
                       help='启用分布式训练（使用 accelerate launch），需要 --gpus-per-exp > 1')
    args = parser.parse_args()
    
    # 🆕 解析种子列表
    if args.seeds:
        try:
            args.seeds = [int(s.strip()) for s in args.seeds.split(',')]
        except ValueError:
            print("❌ 错误: --seeds 参数格式不正确，应为逗号分隔的数字，如: 42,123,456")
            return
    else:
        args.seeds = RANDOM_SEEDS
    
    # 🆕 加载最优参数配置（如果未使用 --use-all-block-sizes）
    if args.use_all_block_sizes:
        print("🔄 使用所有 block_sizes 生成配置（忽略 optimal_params.json）")
        optimal_params = {}  # 设置为空，但函数会使用 use_all 参数
    else:
        optimal_params = load_optimal_params()
        if optimal_params:
            print("✅ 已加载最优参数配置 (optimal_params.json)")
            # 显示各模型的最优参数
            for model_type, params in optimal_params.items():
                if 'block_size' in params:
                    print(f"   {model_type}: block_size={params['block_size']}")
                if 'k_values' in params:
                    print(f"   {model_type}: k_values={params['k_values']}")
        else:
            print("⚠️  使用默认参数配置")
    
    # 🆕 解析 block_size 列表（用于 LLaDABlock 和 BOP-AR 模型）
    # 支持 "None" 字符串表示 baseline 模式（仅 LLaDABlock）
    # 如果用户未指定，则从最优参数配置读取
    user_specified_block_sizes = args.block_sizes is not None
    if args.block_sizes:
        try:
            parsed_block_sizes = []
            for s in args.block_sizes.split(','):
                s = s.strip()
                if s.lower() == 'none' or s == '':
                    parsed_block_sizes.append(None)
                else:
                    parsed_block_sizes.append(int(s))
            args.block_sizes = parsed_block_sizes
            print(f"✅ 使用命令行指定的 block_sizes: {args.block_sizes}")
        except ValueError:
            print("❌ 错误: --block-sizes 参数格式不正确，应为逗号分隔的数字或 None，如: None,1,4,10")
            return
    else:
        # 未指定时，不设置 args.block_sizes（将在生成配置时根据模型类型从最优参数动态获取）
        args.block_sizes = None
        print(f"💡 未指定 --block-sizes，将在生成配置时根据模型类型从最优参数读取")
    
    # 🆕 解析 BPD-AR 的 K 值（如果用户未指定，从最优参数读取或使用所有值）
    user_specified_bpd_k = args.bpd_k_values is not None
    if args.bpd_k_values:
        try:
            args.bpd_k_values = [int(k.strip()) for k in args.bpd_k_values.split(',')]
            print(f"✅ 使用命令行指定的 BPD-AR K 值: {args.bpd_k_values}")
        except ValueError:
            print("❌ 错误: --bpd-k-values 参数格式不正确，应为逗号分隔的数字，如: 1,2,4")
            return
    else:
        # 从最优参数读取或使用所有值
        args.bpd_k_values = get_optimal_bpd_k_values(optimal_params, use_all=args.use_all_block_sizes)
        if args.use_all_block_sizes:
            print(f"💡 BPD-AR K 值: {args.bpd_k_values} (使用所有值)")
        else:
            print(f"💡 BPD-AR K 值: {args.bpd_k_values} (从最优参数读取)")
    
    # ============================================================
    # Daemon 模式处理：如果指定了 --daemon，使用 nohup 重新启动
    # ============================================================
    if args.daemon and not os.environ.get('RUN_BATCH_DAEMON'):
        # 设置环境变量标记，避免无限递归
        env = os.environ.copy()
        env['RUN_BATCH_DAEMON'] = '1'
        
        # 构建命令（-u 参数确保无缓冲输出）
        cmd = [sys.executable, '-u', __file__]
        if args.generate_only:
            cmd.append('--generate-only')
        if args.run_only:
            cmd.append('--run-only')
        if args.model:
            cmd.extend(['--model', args.model])
        if args.dim:
            cmd.extend(['--dim', str(args.dim)])
        if args.size:
            cmd.extend(['--size', args.size])
        if args.gpus:
            cmd.extend(['--gpus', args.gpus])
        if args.max_workers:
            cmd.extend(['--max-workers', str(args.max_workers)])
        if args.limit:
            cmd.extend(['--limit', str(args.limit)])
        if args.start_from:
            cmd.extend(['--start-from', str(args.start_from)])
        if args.log_file:
            cmd.extend(['--log-file', args.log_file])
        
        # 确定日志文件路径
        output_dir = Path('src/conf/experiments_batch')
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = args.log_file or str(output_dir / 'run.log')
        
        # 使用 nohup 启动
        print(f"🔄 启动后台模式...")
        print(f"📝 日志文件: {log_file}")
        
        if args.show_output:
            print(f"💡 输出将同时显示在终端和日志文件")
        else:
            print(f"💡 提示: 使用 'tail -f {log_file}' 查看实时日志")
        
        print(f"💡 提示: 使用 'ps aux | grep run_batch_experiments' 查看进程状态\n")
        
        with open(log_file, 'a') as log:
            log.write(f"\n{'='*70}\n")
            log.write(f"启动时间: {datetime.now().isoformat()}\n")
            log.write(f"命令: {' '.join(cmd)}\n")
            log.write(f"{'='*70}\n\n")
        
        # 如果指定了 --show-output，提示使用更好的方案
        if args.show_output:
            print(f"💡 提示: 要在终端看到实时输出，推荐使用以下方式之一：")
            print(f"   1. 使用 screen: screen -S experiments")
            print(f"   2. 使用 tmux: tmux new -s experiments")
            print(f"   3. 在另一个终端查看日志: tail -f {log_file}")
            print(f"\n   继续使用 nohup 模式（输出只到日志文件）...\n")
        
        # 使用 nohup 启动，输出重定向到日志文件
        with open(log_file, 'a') as log:
            process = subprocess.Popen(
                ['nohup'] + cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid  # 创建新的进程组
            )
        
        print(f"✅ 后台进程已启动 (PID: {process.pid})")
        if args.show_output:
            print(f"📝 日志文件: {log_file}")
            print(f"💡 在另一个终端运行: tail -f {log_file} 查看实时输出")
        else:
            print(f"📝 查看日志: tail -f {log_file}")
        print(f"🛑 停止进程: kill {process.pid}\n")
        
        return
    
    # ============================================================
    # 日志文件处理（非 daemon 模式）
    # ============================================================
    # 🆕 默认输出目录（通用）
    output_dir = Path('src/conf/experiments_batch') # todo modify D10
    output_dir.mkdir(parents=True, exist_ok=True)

    # 🆕 LLaDABlock 专用目录
    llada_block_output_dir = Path('src/conf/llada_block')
    llada_block_output_dir.mkdir(parents=True, exist_ok=True)

    # 🆕 BOP-AR 专用目录
    bop_ar_output_dir = Path('src/conf/BOP-AR')
    bop_ar_output_dir.mkdir(parents=True, exist_ok=True)

    # 🆕 RBO-AR 专用目录
    rbo_ar_output_dir = Path('src/conf/RBO-AR')
    rbo_ar_output_dir.mkdir(parents=True, exist_ok=True)

    # 🆕 EBO-AR 专用目录
    ebo_ar_output_dir = Path('src/conf/EBO-AR')
    ebo_ar_output_dir.mkdir(parents=True, exist_ok=True)

    # 🆕 BPD-AR 专用目录
    bpd_ar_output_dir = Path('src/conf/BPD-AR')
    bpd_ar_output_dir.mkdir(parents=True, exist_ok=True)

    # 🆕 BAD-AR 专用目录
    bad_ar_output_dir = Path('src/conf/BAD-AR')
    bad_ar_output_dir.mkdir(parents=True, exist_ok=True)

    
    
    # 如果指定了日志文件，设置重定向
    log_tee = None
    if args.log_file:
        log_file_path = Path(args.log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        log_tee = Tee(str(log_file_path))
        sys.stdout = log_tee
        print(f"📝 日志已重定向到: {args.log_file}\n")
    
    # ============================================================
    # 步骤 1: 生成配置文件
    # ============================================================
    if not args.run_only:
        print(f"\n{'='*70}")
        print("📝 生成实验配置文件...")
        print(f"{'='*70}\n")

        experiments = []
        # 🆕 获取种子列表（支持命令行参数覆盖）
        seeds_to_use = args.seeds if hasattr(args, 'seeds') and args.seeds else RANDOM_SEEDS

        # 🆕 如果用户指定了模型类型，使用指定的模型；否则使用默认的 MODEL_TYPES
        model_types_to_process = [args.model] if args.model else MODEL_TYPES

        for model_type in model_types_to_process:

            for size_key in SIZE_KEYS: # 循环新增：模型尺寸
                if args.size and size_key != args.size:
                    continue
                # ===== SDAR 实验生成已注释（效果不佳，改用 LLaDABlock） =====
                for dim in DIMS:
                    if args.dim and dim != args.dim:
                        continue

                    for n_prompt in PROMPTS:
                        for n_respond in RESPONDS:
                            # 跳过不合理的组合，例如总长度大于 60
                            if n_prompt + n_respond > 60:
                                continue

                            # 🆕 为每个种子生成配置
                            for random_seed in seeds_to_use:
                                # 🆕 SDAR 模型：为每个 block_size 生成配置（视作不同的模型变体）
                                # if model_type == 'sdar':
                                #     for block_size in args.block_sizes:
                                #         config, exp_name = create_config(
                                #             model_type, size_key, dim, n_prompt, n_respond,
                                #             random_seed, block_size=block_size
                                #         )
                                #         config_path = output_dir / f"{exp_name}.yaml"
                                #
                                #         # 保存配置
                                #         with open(config_path, 'w') as f:
                                #             # 使用 sort_keys=False 保持键的写入顺序，提高可读性
                                #             yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                                #
                                #         experiments.append({
                                #             'name': exp_name,
                                #             'path': str(config_path),
                                #             'model': model_type,
                                #             'size': size_key,
                                #             'dim': dim,
                                #             'prompt': n_prompt,
                                #             'respond': n_respond,
                                #             'seed': random_seed,
                                #             'block_size': block_size,  # 🆕 记录 block_size
                                #         })

                                # 🆕 LLaDABlock 模型：为每个 block_size 生成配置（包括 baseline）
                                # 配置文件保存到专用目录 src/conf/llada_block/
                                if model_type == 'llada_block':
                                    # 🆕 根据任务维度动态获取最优 block_size
                                    # 如果用户通过命令行指定了，优先使用用户指定的值
                                    if user_specified_block_sizes:
                                        optimal_block_sizes = args.block_sizes
                                    else:
                                        optimal_block_sizes = get_optimal_block_sizes(
                                            model_type, dim, n_prompt, n_respond, optimal_params,
                                            use_all=args.use_all_block_sizes
                                        )
                                    for block_size in optimal_block_sizes:
                                        config, exp_name = create_config(
                                            model_type, size_key, dim, n_prompt, n_respond,
                                            random_seed, block_size=block_size
                                        )
                                        # 🆕 LLaDABlock 使用专用目录
                                        config_path = llada_block_output_dir / f"{exp_name}.yaml"

                                        # 保存配置
                                        with open(config_path, 'w') as f:
                                            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                                        experiments.append({
                                            'name': exp_name,
                                            'path': str(config_path),
                                            'model': model_type,
                                            'size': size_key,
                                            'dim': dim,
                                            'prompt': n_prompt,
                                            'respond': n_respond,
                                            'seed': random_seed,
                                            'block_size': block_size,  # None=baseline, or specific value
                                        })

                                # 🆕 BOP-AR 模型：为每个 block_size 生成配置（类似 LLaDABlock）
                                # BOP-AR (ScatDiff) 使用 block_size 控制垂直生成深度
                                # 配置文件保存到专用目录 src/conf/BOP-AR/
                                # 注: 暂时只生成 sequential 模式（non_sequential 需等 MDM/AR 对比基准）
                                elif model_type == 'bop_ar':
                                    # 🆕 从最优参数读取 block_size
                                    if user_specified_block_sizes:
                                        optimal_block_sizes = args.block_sizes
                                    else:
                                        optimal_block_sizes = get_optimal_block_sizes(
                                            model_type, dim, n_prompt, n_respond, optimal_params,
                                            use_all=args.use_all_block_sizes
                                        )
                                    for block_size in optimal_block_sizes:
                                        # 🔧 只生成 sequential 模式（暂不做 non_sequential 对比）
                                        sequence_mode = 'sequential'
                                        config, exp_name = create_config(
                                            model_type, size_key, dim, n_prompt, n_respond,
                                            random_seed, block_size=block_size, sequence_mode=sequence_mode
                                        )
                                        # 🆕 BOP-AR 使用专用目录
                                        config_path = bop_ar_output_dir / f"{exp_name}.yaml"

                                        # 保存配置
                                        with open(config_path, 'w') as f:
                                            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                                        experiments.append({
                                            'name': exp_name,
                                            'path': str(config_path),
                                            'model': model_type,
                                            'size': size_key,
                                            'dim': dim,
                                            'prompt': n_prompt,
                                            'respond': n_respond,
                                            'seed': random_seed,
                                            'block_size': block_size,  # ScatDiff block_size
                                            'sequence_mode': sequence_mode,
                                        })

                                # 🆕 RBO-AR 模型：为每个 block_size 生成配置（类似 BOP-AR）
                                # RBO-AR (Random Block-Order) 使用 block_size 控制块大小
                                # 配置文件保存到专用目录 src/conf/RBO-AR/
                                elif model_type == 'rbo_ar':
                                    # 🆕 从最优参数读取 block_size
                                    if user_specified_block_sizes:
                                        optimal_block_sizes = args.block_sizes
                                    else:
                                        optimal_block_sizes = get_optimal_block_sizes(
                                            model_type, dim, n_prompt, n_respond, optimal_params,
                                            use_all=args.use_all_block_sizes
                                        )
                                    for block_size in optimal_block_sizes:
                                        # 🔧 只生成 sequential 模式
                                        sequence_mode = 'sequential'
                                        config, exp_name = create_config(
                                            model_type, size_key, dim, n_prompt, n_respond,
                                            random_seed, block_size=block_size, sequence_mode=sequence_mode
                                        )
                                        # 🆕 RBO-AR 使用专用目录
                                        config_path = rbo_ar_output_dir / f"{exp_name}.yaml"

                                        # 保存配置
                                        with open(config_path, 'w') as f:
                                            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                                        experiments.append({
                                            'name': exp_name,
                                            'path': str(config_path),
                                            'model': model_type,
                                            'size': size_key,
                                            'dim': dim,
                                            'prompt': n_prompt,
                                            'respond': n_respond,
                                            'seed': random_seed,
                                            'block_size': block_size,  # RBO-AR block_size
                                            'sequence_mode': sequence_mode,
                                        })

                                # 🆕 EBO-AR 模型：为每个 block_size 生成配置（类似 RBO-AR）
                                # EBO-AR (Entropy-based Block-Order) 使用 block_size 控制块大小
                                # 配置文件保存到专用目录 src/conf/EBO-AR/
                                elif model_type == 'ebo_ar':
                                    # 🆕 从最优参数读取 block_size
                                    if user_specified_block_sizes:
                                        optimal_block_sizes = args.block_sizes
                                    else:
                                        optimal_block_sizes = get_optimal_block_sizes(
                                            model_type, dim, n_prompt, n_respond, optimal_params,
                                            use_all=args.use_all_block_sizes
                                        )
                                    for block_size in optimal_block_sizes:
                                        # 🔧 只生成 sequential 模式
                                        sequence_mode = 'sequential'
                                        config, exp_name = create_config(
                                            model_type, size_key, dim, n_prompt, n_respond,
                                            random_seed, block_size=block_size, sequence_mode=sequence_mode
                                        )
                                        # 🆕 EBO-AR 使用专用目录
                                        config_path = ebo_ar_output_dir / f"{exp_name}.yaml"

                                        # 保存配置
                                        with open(config_path, 'w') as f:
                                            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                                        experiments.append({
                                            'name': exp_name,
                                            'path': str(config_path),
                                            'model': model_type,
                                            'size': size_key,
                                            'dim': dim,
                                            'prompt': n_prompt,
                                            'respond': n_respond,
                                            'seed': random_seed,
                                            'block_size': block_size,  # EBO-AR block_size
                                            'sequence_mode': sequence_mode,
                                        })

                                # 其他模型：正常生成配置
                                elif model_type == 'bpd_ar':
                                    # 🆕 BPD-AR 模型：为每个 block_size 和 K 值生成配置
                                    # BPD-AR (Block-wise Parallel Diffusion) 使用 K 值控制并行度
                                    # 配置文件保存到专用目录 src/conf/BPD-AR/
                                    # 🆕 从最优参数读取 block_size 和 K 值
                                    if user_specified_block_sizes:
                                        optimal_block_sizes = args.block_sizes
                                    else:
                                        optimal_block_sizes = get_optimal_block_sizes(
                                            model_type, dim, n_prompt, n_respond, optimal_params,
                                            use_all=args.use_all_block_sizes
                                        )
                                    optimal_k_values = args.bpd_k_values
                                    for block_size in optimal_block_sizes:
                                        for bpd_k in optimal_k_values:
                                            # 🔧 只生成 sequential 模式
                                            sequence_mode = 'sequential'
                                            config, exp_name = create_config(
                                                model_type, size_key, dim, n_prompt, n_respond,
                                                random_seed, block_size=block_size, sequence_mode=sequence_mode
                                            )

                                            # 🆕 为 BPD-AR 配置添加 K 值信息到 evaluation 部分
                                            # 注意：BPD-AR 配置已经由 src/conf/BPD-AR/ 中的配置文件定义
                                            # 这里的配置只是作为参考，实际会被覆盖
                                            if bpd_k == 1:
                                                config['evaluation'] = {
                                                    'use_autoregressive_eval': False,
                                                    'use_ebo_inference': True,
                                                }
                                            else:
                                                config['evaluation'] = {
                                                    'use_autoregressive_eval': False,
                                                    'use_bpd': True,
                                                    'bpd_k': bpd_k,
                                                }

                                            # 修改 out_dir 和 wandb 名称以包含 K 值
                                            k_str = f'_k{bpd_k}'
                                            base_exp_name = exp_name
                                            exp_name = f'bpd_ar_k{bpd_k}_bs{block_size}_{size_key}_D{dim}_P{n_prompt}_R{n_respond}_seed{random_seed}'

                                            config['out_dir'] = f"./outputs/batch_checkpoints/{exp_name}"
                                            config['wandb']['name'] = exp_name
                                            config['wandb']['group'] = f"bpd_ar_k{bpd_k}_{size_key}_bs{block_size}"

                                            # 🆕 BPD-AR 使用专用目录
                                            config_path = bpd_ar_output_dir / f"{exp_name}.yaml"

                                            # 保存配置
                                            with open(config_path, 'w') as f:
                                                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                                            experiments.append({
                                                'name': exp_name,
                                                'path': str(config_path),
                                                'model': model_type,
                                                'size': size_key,
                                                'dim': dim,
                                                'prompt': n_prompt,
                                                'respond': n_respond,
                                                'seed': random_seed,
                                                'block_size': block_size,  # BPD-AR block_size
                                                'bpd_k': bpd_k,  # BPD-AR K 值
                                                'sequence_mode': sequence_mode,
                                            })

                                # 🆕 BAD-AR 模型：为每个 block_size 生成配置（类似 RBO-AR）
                                # BAD-AR (Block-level Autoregressive Diffusion) 使用 block_size 控制块大小
                                # 配置文件保存到专用目录 src/conf/BAD-AR/
                                elif model_type == 'bad_ar':
                                    # 🆕 从最优参数读取 block_size
                                    if user_specified_block_sizes:
                                        optimal_block_sizes = args.block_sizes
                                    else:
                                        optimal_block_sizes = get_optimal_block_sizes(
                                            model_type, dim, n_prompt, n_respond, optimal_params,
                                            use_all=args.use_all_block_sizes
                                        )
                                    for block_size in optimal_block_sizes:
                                        # 🔧 只生成 sequential 模式
                                        sequence_mode = 'sequential'
                                        config, exp_name = create_config(
                                            model_type, size_key, dim, n_prompt, n_respond,
                                            random_seed, block_size=block_size, sequence_mode=sequence_mode
                                        )
                                        # 🆕 BAD-AR 使用专用目录
                                        config_path = bad_ar_output_dir / f"{exp_name}.yaml"

                                        # 保存配置
                                        with open(config_path, 'w') as f:
                                            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                                        experiments.append({
                                            'name': exp_name,
                                            'path': str(config_path),
                                            'model': model_type,
                                            'size': size_key,
                                            'dim': dim,
                                            'prompt': n_prompt,
                                            'respond': n_respond,
                                            'seed': random_seed,
                                            'block_size': block_size,  # BAD-AR block_size
                                            'sequence_mode': sequence_mode,
                                        })

                                # 其他模型：正常生成配置
                                else:
                                    config, exp_name = create_config(
                                        model_type, size_key, dim, n_prompt, n_respond, random_seed
                                    )
                                    config_path = output_dir / f"{exp_name}.yaml"

                                    # 保存配置
                                    with open(config_path, 'w') as f:
                                        # 使用 sort_keys=False 保持键的写入顺序，提高可读性
                                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                                    experiments.append({
                                        'name': exp_name,
                                        'path': str(config_path),
                                        'model': model_type,
                                        'size': size_key, # 记录 size
                                        'dim': dim,
                                        'prompt': n_prompt,
                                        'respond': n_respond,
                                        'seed': random_seed,  # 🆕 记录种子
                                    })

        print(f"✅ 生成了 {len(experiments)} 个配置文件")
        print(f"📁 位置: {output_dir}\n")

        if args.generate_only:
            print("提示: 使用 --run-only 参数来运行这些实验")
            return
    
    # ============================================================
    # 步骤 2: 运行实验（并行）
    # ============================================================
    
    # 1. 检测或指定可用的 GPU
    if args.gpus:
        # 手动指定GPU
        try:
            available_gpus = [int(g.strip()) for g in args.gpus.split(',')]
            num_gpus = len(available_gpus)
            print(f"✅ 使用指定的 {num_gpus} 个 GPU: {available_gpus}")
        except ValueError:
            print("❌ 错误: --gpus 参数格式不正确，应为逗号分隔的数字，如: 0,1,2,3")
            return
    else:
        # 自动检测GPU
        available_gpus = detect_available_gpus()
        num_gpus = len(available_gpus)
        
        if num_gpus == 0:
            print("⚠️ 警告: 未检测到计算设备（可能是 NPU 环境）. 任务将串行运行.")
            num_gpus = 1
            available_gpus = [None]  # 使用 None 表示不限制设备
        else:
            print(f"✅ 自动检测到 {num_gpus} 个计算设备: {available_gpus}")
    
    # 🆕 检测 ROCM/AMD GPU 环境（用于设置环境变量，但不禁用分布式训练）
    is_rocm = False
    try:
        import torch
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            is_rocm = True
        elif 'rocm' in torch.__version__.lower():
            is_rocm = True
    except:
        pass
    
    # 检查环境变量
    if not is_rocm:
        is_rocm = 'ROCM' in os.environ or 'HIP' in os.environ or 'rocm' in os.environ.get('LD_LIBRARY_PATH', '').lower()
    
    if is_rocm:
        print("✅ 检测到 ROCM/AMD GPU 环境，将使用 RCCL 作为通信后端")
        # 🔧 设置 ROCM 相关的环境变量（减少日志输出，防止日志缓冲区溢出导致段错误）
        os.environ.setdefault('NCCL_DEBUG', 'VERSION')  # 使用 VERSION 级别减少日志输出
        os.environ.setdefault('NCCL_IB_DISABLE', '0')
        os.environ.setdefault('NCCL_P2P_DISABLE', '0')
        # 确保使用 RCCL（PyTorch 会自动处理，但明确设置更安全）
        os.environ.setdefault('NCCL_BACKEND', 'RCCL')
    
    # 🆕 多卡训练模式检查
    gpus_per_exp = args.gpus_per_exp
    use_distributed = args.use_distributed
    
    if gpus_per_exp > 1:
        if not use_distributed:
            print("⚠️  警告: --gpus-per-exp > 1 但未启用 --use-distributed，将自动启用分布式训练")
            use_distributed = True
        
        if num_gpus < gpus_per_exp:
            print(f"⚠️  警告: 需要 {gpus_per_exp} 个设备，但只检测到 {num_gpus} 个")
            print(f"💡 提示: 在 NPU 环境下，如果实际有多个设备但检测不到，可以：")
            print(f"   1. 设置环境变量 CUDA_VISIBLE_DEVICES=0,1,2,3,... (根据实际设备数)")
            print(f"   2. 或者使用 --gpus 参数手动指定设备数量，如: --gpus 0,1,2,3,4,5,6,7")
            print(f"   3. 或者降低 --gpus-per-exp 参数（例如改为 1）")
            
            # 🆕 对于 NPU 环境，如果检测不到设备，允许用户手动指定或自动降级
            if args.gpus:
                # 用户已手动指定，使用用户指定的
                print(f"✅ 使用手动指定的设备: {available_gpus}")
                # 重新计算
                num_gpus = len(available_gpus)
                if num_gpus < gpus_per_exp:
                    print(f"❌ 错误: 手动指定的设备数 ({num_gpus}) 仍小于需要的设备数 ({gpus_per_exp})")
                    print(f"💡 建议: 降低 --gpus-per-exp 参数或增加 --gpus 参数中的设备数量")
                    return
            else:
                # 如果没有手动指定，且检测不到足够设备，自动降级为单设备模式
                print(f"⚠️  自动降级为单设备模式（--gpus-per-exp=1）")
                gpus_per_exp = 1
                use_distributed = False
        
        # 计算可以并行运行的实验数量
        max_parallel_exps = num_gpus // gpus_per_exp
        print(f"📊 多卡训练模式: 每个实验使用 {gpus_per_exp} 个GPU")
        print(f"📊 最多可并行运行 {max_parallel_exps} 个实验")
    else:
        max_parallel_exps = num_gpus
        print(f"📊 单卡训练模式: 每个实验使用 1 个GPU")
    
    # 应用 max_workers 限制（如果指定）
    if args.max_workers:
        max_parallel_exps = min(max_parallel_exps, args.max_workers)
        print(f"📊 限制最大并行数为: {max_parallel_exps}")
    
    # 获取所有配置文件
    # 🆕 根据模型类型从不同目录读取配置
    if args.model == 'llada_block':
        # LLaDABlock 模型：只从 llada_block 目录读取
        config_files = sorted(llada_block_output_dir.glob("*.yaml"))
        print(f"📁 从 LLaDABlock 专用目录读取配置: {llada_block_output_dir}")
    elif args.model == 'bop_ar':
        # BOP-AR 模型：只从 BOP-AR 目录读取
        config_files = sorted(bop_ar_output_dir.glob("*.yaml"))
        print(f"📁 从 BOP-AR 专用目录读取配置: {bop_ar_output_dir}")
    elif args.model == 'rbo_ar':
        # RBO-AR 模型：只从 RBO-AR 目录读取
        config_files = sorted(rbo_ar_output_dir.glob("*.yaml"))
        print(f"📁 从 RBO-AR 专用目录读取配置: {rbo_ar_output_dir}")
    elif args.model == 'ebo_ar':
        # EBO-AR 模型：只从 EBO-AR 目录读取
        config_files = sorted(ebo_ar_output_dir.glob("*.yaml"))
        print(f"📁 从 EBO-AR 专用目录读取配置: {ebo_ar_output_dir}")
    elif args.model == 'bpd_ar':
        # BPD-AR 模型：只从 BPD-AR 目录读取
        config_files = sorted(bpd_ar_output_dir.glob("*.yaml"))
        print(f"📁 从 BPD-AR 专用目录读取配置: {bpd_ar_output_dir}")
    elif args.model == 'bad_ar':
        # BAD-AR 模型：只从 BAD-AR 目录读取
        config_files = sorted(bad_ar_output_dir.glob("*.yaml"))
        print(f"📁 从 BAD-AR 专用目录读取配置: {bad_ar_output_dir}")
    elif args.model and args.model not in ['llada_block', 'bop_ar', 'rbo_ar', 'ebo_ar', 'bpd_ar', 'bad_ar']:
        # 其他指定模型：从 experiments_batch 读取
        config_files = sorted(output_dir.glob("*.yaml"))
        print(f"📁 从通用目录读取配置: {output_dir}")
    else:
        # 未指定模型：从所有目录都读取
        config_files = sorted(output_dir.glob("*.yaml")) + \
                       sorted(llada_block_output_dir.glob("*.yaml")) + \
                       sorted(bop_ar_output_dir.glob("*.yaml")) + \
                       sorted(rbo_ar_output_dir.glob("*.yaml")) + \
                       sorted(ebo_ar_output_dir.glob("*.yaml")) + \
                       sorted(bpd_ar_output_dir.glob("*.yaml")) + \
                       sorted(bad_ar_output_dir.glob("*.yaml"))
        print(f"📁 从多个目录读取配置: {output_dir}, {llada_block_output_dir}, {bop_ar_output_dir}, {rbo_ar_output_dir}, {ebo_ar_output_dir}, {bpd_ar_output_dir}, {bad_ar_output_dir}")
    
    # 应用过滤
    if args.model:
        config_files = [f for f in config_files if args.model in f.stem]
    if args.dim:
        config_files = [f for f in config_files if f'_D{args.dim}_' in f.stem]
    if args.size: # 新增过滤
        config_files = [f for f in config_files if f'_{args.size}_' in f.stem]
    
    # 应用范围
    config_files = config_files[args.start_from:]
    if args.limit:
        config_files = config_files[:args.limit]
    
    print(f"\n{'='*70}", flush=True)
    if gpus_per_exp > 1:
        print(f"🚀 开始并行运行实验 (分布式训练: {gpus_per_exp} GPU/实验, 最多 {max_parallel_exps} 个并行)...", flush=True)
    else:
        print(f"🚀 开始并行运行实验 (单卡训练: 1 GPU/实验, 最多 {max_parallel_exps} 个并行)...", flush=True)
    print(f"📋 共 {len(config_files)} 个实验待运行", flush=True)
    print(f"{'='*70}\n", flush=True)
    
    # 运行实验
    results = []
    start_time = datetime.now()
    
    # 2. 使用 ThreadPoolExecutor 限制并发数
    with ThreadPoolExecutor(max_workers=max_parallel_exps) as executor:
        
        # 3. 提交所有任务到执行器
        futures = []
        
        # 🆕 GPU 分配逻辑：根据 gpus_per_exp 分组分配
        if gpus_per_exp > 1:
            # 多卡模式：将 GPU 分组分配给实验
            gpu_groups = []
            for i in range(0, len(available_gpus), gpus_per_exp):
                group = available_gpus[i:i+gpus_per_exp]
                if len(group) == gpus_per_exp:  # 只使用完整的组
                    gpu_groups.append(group)
            
            # 循环使用 GPU 组
            gpu_group_cycler = iter((gpu_groups * (len(config_files) // len(gpu_groups) + 1))[:len(config_files)])
            
            for config_path in config_files:
                gpu_group = next(gpu_group_cycler)
                future = executor.submit(run_single_experiment, config_path, gpu_group, gpus_per_exp, use_distributed)
                futures.append(future)
        else:
            # 单卡模式：原有逻辑
            gpu_cycler = iter((available_gpus * (len(config_files) // num_gpus + 1))[:len(config_files)])
            for config_path in config_files:
                gpu_id = next(gpu_cycler)
                # 单卡模式：gpu_ids 是单个元素的列表
                gpu_ids = [gpu_id] if gpu_id is not None else None
                future = executor.submit(run_single_experiment, config_path, gpu_ids, gpus_per_exp, False)
                futures.append(future)
        
        # 4. 实时收集结果
        # as_completed 允许我们按任务完成的顺序获取结果
        completed_count = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                completed_count += 1
                
                # 实时更新进度
                success_count = sum(1 for r in results if r['success'])
                remaining = len(futures) - completed_count
                print(f"\n--- 实时进度: {completed_count}/{len(futures)} 完成 ({success_count} 成功, {remaining} 剩余) ---", flush=True)
                
            except Exception as e:
                # 捕获调度器级别的异常 (不常见)
                print(f"⚠️ 调度器发生错误: {e}", flush=True)
                results.append({
                    'name': 'unknown',
                    'success': False,
                    'duration_minutes': 0,
                    'gpu_ids': None,
                    'error': f"Scheduler error: {str(e)}"
                })
    
    # ============================================================
    # 总结
    # ============================================================
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    success_count = sum(1 for r in results if r['success'])
    fail_count = len(results) - success_count
    
    print(f"\n{'='*70}", flush=True)
    print("📊 实验总结", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"总实验数: {len(results)}", flush=True)
    print(f"成功: {success_count}", flush=True)
    print(f"失败: {fail_count}", flush=True)
    print(f"总耗时: {total_duration/3600:.2f} 小时", flush=True)
    if len(results) > 0:
        print(f"平均每个: {total_duration/len(results)/60:.1f} 分钟", flush=True)
        if num_gpus > 1:
            print(f"并行加速: 约 {num_gpus}x (理论值)", flush=True)
    
    # 保存失败实验列表
    if fail_count > 0:
        failed_experiments = [r['name'] for r in results if not r['success']]
        failed_file = output_dir / 'failed_experiments.txt'
        with open(failed_file, 'w') as f:
            for name in failed_experiments:
                f.write(f"{name}\n")
        print(f"\n❌ 失败实验列表已保存到: {failed_file}", flush=True)
        print(f"   可以手动重新运行失败的实验", flush=True)
    
    # 保存详细结果到JSON文件
    results_file = output_dir / 'experiment_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'total': len(results),
            'success': success_count,
            'failed': fail_count,
            'total_duration_hours': total_duration / 3600,
            'results': results
        }, f, indent=2)
    print(f"📄 详细结果已保存到: {results_file}", flush=True)
    
    print(f"{'='*70}\n", flush=True)
    
    # 恢复 stdout 并关闭日志文件
    if log_tee:
        sys.stdout = log_tee.stdout
        log_tee.close()
        print(f"✅ 日志已保存到: {args.log_file}")


if __name__ == '__main__':
    main()

