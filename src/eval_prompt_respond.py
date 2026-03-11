"""
Evaluation Script for Prompt-Respond Models
===========================================

评估 Prompt-Respond 模型的性能
"""

import os
import sys
import yaml
import torch
import numpy as np
import inspect
from tqdm import tqdm

# ✅ 确保能 import 到仓库根下的 dllm 包和本地模块（向后兼容：只在路径不存在时添加）
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
repo_root_norm = os.path.normpath(repo_root)
dllm_path = os.path.join(repo_root, "dllm")
dllm_path_norm = os.path.normpath(dllm_path)

# 检查路径是否已存在（使用规范化路径比较，兼容不同路径表示）
if not any(os.path.normpath(p) == repo_root_norm for p in sys.path):
    sys.path.insert(0, repo_root)
if not any(os.path.normpath(p) == dllm_path_norm for p in sys.path):
    sys.path.insert(0, dllm_path)

from models_prompt_respond import build_model_prompt_respond
from samplers import get_data_sampler
from tasks import get_task_sampler
# 🆕 导入 get_model_attr 用于安全访问模型属性（兼容 DDP）
try:
    from train_utils import get_model_attr
except ImportError:
    # 如果导入失败，定义简单的 fallback
    def get_model_attr(model, attr_name, default=None):
        if hasattr(model, 'module'):
            return getattr(model.module, attr_name, default)
        return getattr(model, attr_name, default)
from train_utils import generate_fixed_permutation, apply_fixed_permutation, apply_random_permutation, generate_permutation_pool, apply_pool_permutation, generate_permutation_pool, apply_pool_permutation
import torch.nn.functional as F


def _bpd_ar_inference_sudoku(
    model,
    xs,
    ys,
    n_prompt,
    n_respond,
    device,
    num_steps=20,
    k_per_step=4,
    respond_position_mask=None,
):
    """
    BPD-AR (Best-Path Decoding with Autoregressive) 推理：熵引导的自适应推理
    
    对于数独任务，不一次性预测 81 个格子，而是：
    1. 每步根据 Logits 计算每个格子的预测熵（Entropy）
    2. 优先填充熵最小的 K 个格子（最确定的预测）
    3. 迭代 20 步完成一盘数独
    
    Args:
        model: SudokuLLaDA 模型
        xs: [B, n_points, 81] - quiz 部分
        ys: [B, n_points, 81] - solution 部分（用于构造序列，推理时会被 mask）
        n_prompt: prompt 数量
        n_respond: respond 数量
        device: 设备
        num_steps: 迭代步数（默认 20）
        k_per_step: 每步填充的格子数（默认 4，81 / 20 ≈ 4）
        respond_position_mask: [B, n_points] boolean mask（保留兼容性）
    
    Returns:
        pred: [B, n_respond, 81, 10] - 最终的 logits
        mask: [B, n_respond, 81] - boolean mask（所有位置都已填充，全 False）
    """
    B, n_points, d = xs.shape
    assert d == 81, f"数独任务要求 d=81，实际为 {d}"
    
    # 获取实际模型（兼容 DDP）
    actual_model = model.module if hasattr(model, 'module') else model
    
    # 初始化：所有 respond 部分的 solution 都使用 MASK token (10)
    # 创建一个可变的 ys_copy，用于在迭代中更新
    ys_copy = ys.clone()  # [B, n_points, 81]
    
    # 初始化 respond 部分的 solution 为 MASK token
    ys_copy[:, n_prompt:, :] = 10  # MASK token
    
    # 初始化 mask 状态：所有 respond 部分的 solution 格子都未填充
    # filled_mask: [B, n_respond, 81] - True 表示已填充，False 表示未填充（mask）
    filled_mask = torch.zeros(B, n_respond, 81, dtype=torch.bool, device=device)
    
    # 迭代填充
    final_logits = None
    for step in range(num_steps):
        # 调用模型 forward（推理模式）
        # 模型会根据 ys_copy 中的 MASK token 进行推理
        output = actual_model(xs, ys_copy, train_mode=False, respond_position_mask=respond_position_mask)
        
        if isinstance(output, tuple):
            logits, _ = output  # logits: [B, n_respond, 81, 10]
        else:
            logits = output  # logits: [B, n_respond, 81, 10]
        
        final_logits = logits  # 保存最后一次的 logits
        
        # 计算每个格子的预测熵（Entropy）
        probs = F.softmax(logits, dim=-1)  # [B, n_respond, 81, 10]
        log_probs = F.log_softmax(logits, dim=-1)  # [B, n_respond, 81, 10]
        entropy = -(probs * log_probs).sum(dim=-1)  # [B, n_respond, 81] - 每个格子的熵
        
        # 只对未填充的格子计算熵
        # 将已填充的格子的熵设为 inf，这样不会被选中
        entropy_masked = entropy.clone()
        entropy_masked[filled_mask] = float('inf')  # 已填充的格子设为 inf
        
        # 对每个 batch 和每个 respond，选择熵最小的 K 个格子
        for b in range(B):
            for r in range(n_respond):
                # 获取未填充格子的索引
                unfilled_indices = (~filled_mask[b, r]).nonzero(as_tuple=True)[0]  # [num_unfilled]
                
                if len(unfilled_indices) == 0:
                    # 所有格子都已填充，跳过
                    continue
                
                # 计算这一步要填充的格子数（不超过剩余未填充的格子数）
                k = min(k_per_step, len(unfilled_indices))
                
                # 获取未填充格子的熵
                unfilled_entropy = entropy_masked[b, r, unfilled_indices]  # [num_unfilled]
                
                # 选择熵最小的 K 个格子
                _, topk_indices = torch.topk(unfilled_entropy, k, largest=False)  # [k]
                selected_cell_indices = unfilled_indices[topk_indices]  # [k]
                
                # 填充选中的格子：使用 argmax 预测值
                pred_digits = torch.argmax(logits[b, r, selected_cell_indices, :], dim=-1)  # [k]
                
                # 更新 ys_copy 中 respond 部分的 solution
                ys_copy[b, n_prompt + r, selected_cell_indices] = pred_digits
                
                # 更新 filled_mask
                filled_mask[b, r, selected_cell_indices] = True
        
        # 检查是否所有格子都已填充
        if filled_mask.all():
            break
    
    # 如果还有未填充的格子，使用最后一次的 argmax 预测填充
    if not filled_mask.all():
        remaining_mask = ~filled_mask  # [B, n_respond, 81]
        remaining_pred = torch.argmax(final_logits, dim=-1)  # [B, n_respond, 81]
        for b in range(B):
            for r in range(n_respond):
                remaining_cells = remaining_mask[b, r].nonzero(as_tuple=True)[0]
                if len(remaining_cells) > 0:
                    ys_copy[b, n_prompt + r, remaining_cells] = remaining_pred[b, r, remaining_cells]
    
    # 返回最终的 logits 和 mask（所有位置都已填充，mask 全 False）
    final_mask = torch.zeros(B, n_respond, 81, dtype=torch.bool, device=device)
    
    return final_logits, final_mask


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


def eval_model_prompt_respond(
    model,
    task_sampler,
    data_sampler,
    n_prompt,
    n_respond,
    n_dims,
    batch_size=64,
    num_eval_examples=1280,
    use_autoregressive_eval=False,
    fixed_batches=None,
    task_sampler_args=None,
    sequence_mode="sequential",
    inference_steps_list=None,  # 🆕 可选：推理步数列表，如 [1, 5, 10, 20]
    permutation_seed=42,  # 🆕 Fixed Permutation: x 的排列种子
    permutation_seed_y=None,  # 🆕 Fixed Permutation: y 的排列种子（可选）
    permute_y=False,  # 🆕 Random Permutation: 是否也打乱 y（默认 False）
    permutation_pool_size=20,  # 🆕 Pool Permutation: 排列池大小
    use_ebo_inference=None,  # 🆕 EBO-AR: 是否使用熵引导推理（None 时从模型或配置读取）
    use_bpd_ar_sudoku=True,  # 🆕 BPD-AR: 数独任务是否使用 BPD-AR 推理（默认 True）
    bpd_ar_steps=20,  # 🆕 BPD-AR: 迭代步数（默认 20）
    bpd_ar_k_per_step=4,  # 🆕 BPD-AR: 每步填充的格子数（默认 4）
):
    """
    评估 Prompt-Respond 模型
    
    Args:
        model: 训练好的模型
        task_sampler: task sampler
        data_sampler: data sampler
        n_prompt: prompt 长度
        n_respond: respond 长度
        n_dims: 特征维度
        batch_size: batch size
        num_eval_examples: 评估样本总数
        use_autoregressive_eval: 如果为 True，AR 模型使用自回归预测（避免 label 泄漏）
                                 如果为 False，使用原方案（直接使用真实标签）
        fixed_batches: 可选的、预定义的数据/task seed 列表（用于固定 validation set）
        task_sampler_args: 额外的参数（如 seeds）会传给 task_sampler
        sequence_mode: "sequential" or "non_sequential" (for Non-Sequential ICL)
        inference_steps_list: 可选，如果提供，会在每个步数下分别评估，返回每个步数的结果
                             格式: [1, 5, 10, 20] 或 None（使用模型默认步数）
    
    Returns:
        如果 inference_steps_list 为 None: 返回单个 results 字典
        如果 inference_steps_list 不为 None: 返回 {step: results} 字典
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 🆕 如果指定了多个推理步数，分别评估
    if inference_steps_list is not None:
        all_results = {}
        original_steps = None
        
        # 保存原始推理步数（如果模型支持）
        if hasattr(model, 'inference_steps'):
            original_steps = model.inference_steps
        
        print(f"\n{'='*70}")
        print(f"🔄 多步数评估模式: 将测试 {len(inference_steps_list)} 个推理步数")
        print(f"   步数列表: {inference_steps_list}")
        print(f"{'='*70}\n")
        
        for steps in inference_steps_list:
            print(f"\n📊 评估推理步数: {steps}")
            print("-" * 70)
            
            # 临时修改模型的推理步数
            if hasattr(model, 'inference_steps'):
                model.inference_steps = steps
            
            # 执行评估（递归调用，但不传入 inference_steps_list）
            results = eval_model_prompt_respond(
                model=model,
                task_sampler=task_sampler,
                data_sampler=data_sampler,
                n_prompt=n_prompt,
                n_respond=n_respond,
                n_dims=n_dims,
                batch_size=batch_size,
                num_eval_examples=num_eval_examples,
                use_autoregressive_eval=use_autoregressive_eval,
                fixed_batches=fixed_batches,
                task_sampler_args=task_sampler_args,
                sequence_mode=sequence_mode,
                inference_steps_list=None,  # 递归调用时不再多步评估
                permutation_seed=permutation_seed,  # 传递排列种子
                permutation_seed_y=permutation_seed_y,  # 传递 y 的排列种子
                permute_y=permute_y,  # 🆕 传递 permute_y 参数
                permutation_pool_size=permutation_pool_size,  # 🆕 传递排列池大小
                use_bpd_ar_sudoku=use_bpd_ar_sudoku,  # 🆕 传递 BPD-AR 参数
                bpd_ar_steps=bpd_ar_steps,  # 🆕 传递 BPD-AR 步数
                bpd_ar_k_per_step=bpd_ar_k_per_step,  # 🆕 传递 BPD-AR K 值
            )
            
            all_results[steps] = results

            # 打印当前步数的结果
            print(f"\n✅ 步数 {steps} 的结果:")
            if 'cell_accuracy' in results:
                # 数独任务：显示准确率
                print(f"   Cell Accuracy: {results['cell_accuracy']['mean']:.4f}")
                print(f"   Sudoku Accuracy: {results['sudoku_accuracy']['mean']:.4f}")
            else:
                # 非数独任务：显示 MSE
                print(f"   Respond MSE: Mean={results['respond_mse']['mean']:.6f}, "
                      f"Std={results['respond_mse']['std']:.6f}")

        # 恢复原始推理步数
        if original_steps is not None:
            model.inference_steps = original_steps

        # 打印汇总结果
        print(f"\n{'='*70}")
        print("📊 多步数评估汇总结果")
        print(f"{'='*70}")

        # 检查是否是数独任务（通过第一个结果判断）
        first_result = list(all_results.values())[0]
        is_sudoku_summary = 'cell_accuracy' in first_result

        if is_sudoku_summary:
            # 数独任务：显示准确率汇总
            print(f"{'步数':<10} {'Cell Accuracy':<20} {'Sudoku Accuracy':<20}")
            print("-" * 70)
            for steps in sorted(all_results.keys()):
                r = all_results[steps]
                print(f"{steps:<10} {r['cell_accuracy']['mean']:<20.4f} {r['sudoku_accuracy']['mean']:<20.4f}")
        else:
            # 非数独任务：显示 MSE 汇总
            print(f"{'步数':<10} {'Respond MSE (Mean)':<20} {'Respond MSE (Std)':<20}")
            print("-" * 70)
            for steps in sorted(all_results.keys()):
                r = all_results[steps]
                print(f"{steps:<10} {r['respond_mse']['mean']:<20.6f} {r['respond_mse']['std']:<20.6f}")
        print("=" * 70)
        
        return all_results
    
    # === 原有的单步评估逻辑 ===
    
    total_points = n_prompt + n_respond
    all_prompt_mse = []
    all_respond_mse = []
    all_overall_mse = []

    if fixed_batches is not None:
        num_batches = len(fixed_batches)
    else:
        num_batches = num_eval_examples // batch_size

    batch_means = []

    # 🆕 判断模型类型（安全访问 model.family，兼容 DDP 包装）
    model_family = get_model_attr(model, 'family')
    is_ar_model = model_family is not None and model_family in ["gpt2", "gptj", "llama", "llama2", "llama3", "qwen", "qwen2", "qwen2.5"]

    # 🆕 检测数独任务（基于 n_dims == 81 或模型 family 以 "sudoku_" 开头）
    # 也兼容旧的 810 维格式（向后兼容）
    is_sudoku_task = (n_dims == 81) or (n_dims == 810) or (model_family and model_family.startswith('sudoku_'))

    # 🆕 检测路径查找任务（基于模型 family 以 "pathfinding_" 开头）
    is_pathfinding_task = (model_family and model_family.startswith('pathfinding_'))

    # 🆕 如果是数独或路径查找任务，收集预测和真实值用于准确率计算
    all_predictions = [] if (is_sudoku_task or is_pathfinding_task) else None
    all_targets = [] if (is_sudoku_task or is_pathfinding_task) else None
    
    # 🆕 Fixed Permutation: 生成固定排列（与训练时保持一致）
    fixed_permutation_x = None
    fixed_permutation_y = None
    if sequence_mode in ["fixed_permutation", "fixed_permutation_xy"]:
        # 生成与训练时相同的固定排列
        fixed_permutation_x = generate_fixed_permutation(n_prompt, n_respond, seed=permutation_seed)
        if sequence_mode == "fixed_permutation_xy":
            if permutation_seed_y is None:
                permutation_seed_y = permutation_seed + 1000  # 默认值
            fixed_permutation_y = generate_fixed_permutation(n_prompt, n_respond, seed=permutation_seed_y)
    
    # 🆕 Pool Permutation: 生成排列池（与训练时保持一致）
    permutation_pool_x = None
    permutation_pool_y = None
    if sequence_mode in ["pool_permutation", "pool_permutation_xy"]:
        # 生成与训练时相同的排列池
        permutation_pool_x = generate_permutation_pool(n_prompt, pool_size=permutation_pool_size, seed=permutation_seed)
        if sequence_mode == "pool_permutation_xy":
            if permutation_seed_y is None:
                permutation_seed_y = permutation_seed + 1000  # 默认值
            permutation_pool_y = generate_permutation_pool(n_prompt, pool_size=permutation_pool_size, seed=permutation_seed_y)
    
    # 🆕 显示当前推理步数（如果使用多步推理）
    inference_info = ""
    if hasattr(model, 'use_multistep_inference') and model.use_multistep_inference:
        if hasattr(model, 'inference_steps'):
            inference_info = f" (推理步数: {model.inference_steps})"
    
    eval_mode_str = "autoregressive (no label leakage)" if (is_ar_model and use_autoregressive_eval) else "original (with label leakage for AR)"
    print(f"Evaluating model on {num_eval_examples} examples...")
    print(f"Evaluation mode: {eval_mode_str}{inference_info}")
    if sequence_mode in ["fixed_permutation", "fixed_permutation_xy"]:
        print(f"Fixed Permutation mode: {sequence_mode} (seed_x={permutation_seed}, seed_y={permutation_seed_y})")
    elif sequence_mode == "random_permutation":
        print(f"Random Permutation mode: Each batch uses a different random permutation (permute_y={permute_y})")
    elif sequence_mode in ["pool_permutation", "pool_permutation_xy"]:
        print(f"Pool Permutation mode: {sequence_mode} (pool_size={permutation_pool_size}, seed_x={permutation_seed}, seed_y={permutation_seed_y})")
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches)):
            # 生成数据
            data_seeds = None
            batch_task_args = {}
            if fixed_batches is not None:
                batch_seeds = fixed_batches[batch_idx]
                data_seeds = batch_seeds.get("data_seeds")
                if "task_seeds" in batch_seeds:
                    batch_task_args["seeds"] = batch_seeds["task_seeds"]

            xs = data_sampler.sample_xs(total_points, batch_size, n_dims, seeds=data_seeds)
            combined_task_args = {}
            if task_sampler_args:
                combined_task_args.update(task_sampler_args)
            combined_task_args.update(batch_task_args)
            task = task_sampler(**combined_task_args)
            ys = task.evaluate(xs)
            
            # === 🆕 Non-Sequential ICL: 打乱 Prompt-Respond Pairs ===
            respond_position_mask = None
            if sequence_mode == "non_sequential":
                xs, ys, respond_position_mask = shuffle_prompt_respond_pairs(xs, ys, n_prompt, n_respond)
            elif sequence_mode in ["fixed_permutation", "fixed_permutation_xy"]:
                # 🆕 Fixed Permutation: 应用固定排列（与训练时保持一致）
                xs, ys, respond_position_mask = apply_fixed_permutation(
                    xs, ys, n_prompt, n_respond,
                    fixed_permutation_x,
                    fixed_permutation_y
                )
            elif sequence_mode == "random_permutation":
                # 🆕 Random Permutation: 每个batch使用不同的随机排列
                xs, ys, respond_position_mask = apply_random_permutation(
                    xs, ys, n_prompt, n_respond,
                    permute_y=permute_y  # 使用函数参数传入的 permute_y
                )
            elif sequence_mode in ["pool_permutation", "pool_permutation_xy"]:
                # 🆕 Pool Permutation: 从固定排列池中随机采样（与训练时保持一致）
                xs, ys, respond_position_mask = apply_pool_permutation(
                    xs, ys, n_prompt, n_respond, 
                    permutation_pool_x, 
                    permutation_pool_y
                )
            
            xs, ys = xs.to(device), ys.to(device)
            if respond_position_mask is not None:
                respond_position_mask = respond_position_mask.to(device)
            
            # 前向传播
            # 所有模型都只返回respond部分的预测
            if is_ar_model:
                # 🆕 AR 模型：支持 Non-Sequential（根据 attention_mode）
                # 注意：原来的 TransformerModelPromptRespond 不支持 train_mode 参数
                # 只有 sudoku 的 AR 模型才支持 train_mode
                # 检查模型是否支持 train_mode 参数
                forward_sig = inspect.signature(model.forward)
                supports_train_mode = 'train_mode' in forward_sig.parameters
                supports_use_autoregressive_eval = 'use_autoregressive_eval' in forward_sig.parameters
                
                if use_autoregressive_eval:
                    # 自回归评估：使用预测值，避免 label 泄漏
                    # 注意：autoregressive eval 在 non_sequential 模式下可能不完全支持
                    # 如果遇到问题，可以设置 use_autoregressive_eval=False
                    # 仅在模型 forward 支持 use_autoregressive_eval 参数时才传入（避免 SudokuAR 等报错）
                    if supports_train_mode:
                        if supports_use_autoregressive_eval:
                            model_output = model(
                                xs, ys,
                                train_mode=False,  # Sudoku AR 模型需要这个参数
                                use_autoregressive_eval=True,
                                respond_position_mask=respond_position_mask
                            )
                        else:
                            model_output = model(
                                xs, ys,
                                train_mode=False,  # Sudoku AR 模型需要这个参数
                                respond_position_mask=respond_position_mask
                            )
                    else:
                        if supports_use_autoregressive_eval:
                            model_output = model(
                                xs, ys,
                                use_autoregressive_eval=True,
                                respond_position_mask=respond_position_mask
                            )
                        else:
                            model_output = model(
                                xs, ys,
                                respond_position_mask=respond_position_mask
                            )
                else:
                    # 原方案：直接使用真实标签（可能有 label 泄漏）
                    if supports_train_mode:
                        if supports_use_autoregressive_eval:
                            model_output = model(
                                xs, ys,
                                train_mode=False,  # Sudoku AR 模型需要这个参数
                                use_autoregressive_eval=False,
                                respond_position_mask=respond_position_mask
                            )
                        else:
                            model_output = model(
                                xs, ys,
                                train_mode=False,  # Sudoku AR 模型需要这个参数
                                respond_position_mask=respond_position_mask
                            )
                    else:
                        if supports_use_autoregressive_eval:
                            model_output = model(
                                xs, ys,
                                use_autoregressive_eval=False,
                                respond_position_mask=respond_position_mask
                            )
                        else:
                            model_output = model(
                                xs, ys,
                                respond_position_mask=respond_position_mask
                            )
                
                # 处理模型输出（可能是 tuple 或单个 tensor）
                if isinstance(model_output, tuple):
                    if len(model_output) == 2:
                        pred, loss_mask = model_output  # pred: [B, n_respond] 或 [B, n_respond, D]
                    else:
                        # 如果 tuple 长度不对，可能是异常情况
                        raise ValueError(f"AR模型返回的tuple长度异常: {len(model_output)}, 期望2")
                else:
                    pred = model_output  # pred: [B, n_respond] 或 [B, n_respond, D]
                    loss_mask = None
                mask = None  # AR 模型不使用 mask
            else:
                # 🆕 MDM 模型
                # 检查是否支持 train_mode 参数（兼容 DDP 包装）
                actual_model = model.module if hasattr(model, 'module') else model
                forward_sig = inspect.signature(actual_model.forward)
                supports_train_mode = 'train_mode' in forward_sig.parameters
                supports_ebo_inference = 'use_ebo_inference' in forward_sig.parameters
                
                # 🆕 数独任务：默认使用 BPD-AR 推理（熵引导的自适应推理）
                # 但对 sudoku_dream：模型内部已经封装了 DreamSampler 多步采样，
                # 直接调用 model(..., train_mode=False) 即可，避免使用 BPD-AR 的 10/MASK 逻辑。
                if is_sudoku_task and not is_ar_model and use_bpd_ar_sudoku and not (model_family == "sudoku_dream"):
                    # BPD-AR: 迭代填充，每步选择熵最小的 K 个格子
                    pred, mask = _bpd_ar_inference_sudoku(
                        model=model,
                        xs=xs,
                        ys=ys,
                        n_prompt=n_prompt,
                        n_respond=n_respond,
                        device=device,
                        num_steps=bpd_ar_steps,  # 迭代步数
                        k_per_step=bpd_ar_k_per_step,  # 每步填充的格子数
                        respond_position_mask=respond_position_mask,
                    )
                else:
                    # 非数独任务或 AR 模型：使用标准推理
                    # Get use_ebo_inference: function parameter > model attribute > config > default False
                    eval_use_ebo = use_ebo_inference
                    if eval_use_ebo is None:
                        # Try to get from model attribute
                        eval_use_ebo = getattr(actual_model, 'use_ebo_inference', None)
                        if eval_use_ebo is None:
                            # Default to False
                            eval_use_ebo = False
                    
                    if supports_train_mode:
                        if supports_ebo_inference:
                            output = model(xs, ys, train_mode=False, respond_position_mask=respond_position_mask, use_ebo_inference=eval_use_ebo)
                        else:
                            output = model(xs, ys, train_mode=False, respond_position_mask=respond_position_mask)  # 可能是 pred 或 (pred, mask)
                    else:
                        # 如果模型不支持 train_mode，只传入 respond_position_mask
                        output = model(xs, ys, respond_position_mask=respond_position_mask)  # 可能是 pred 或 (pred, mask)
                    if isinstance(output, tuple):
                        if len(output) == 2:
                            pred, mask = output  # [B, n_respond] 或 [B, n_respond, D], [B, n_respond]
                        else:
                            # 如果 tuple 只有一个元素，可能是特殊情况
                            pred = output[0]
                            mask = None
                    else:
                        pred = output  # [B, n_respond] 或 [B, n_respond, D]
                        mask = None  # 如果没有mask信息，则对所有位置计算MSE
            
            # 🔧 检查 pred 是否为 tensor（防止解包失败导致 pred 仍是 tuple）
            if not isinstance(pred, torch.Tensor):
                raise TypeError(f"模型输出 pred 应该是 tensor，但得到了 {type(pred)}: {pred}")
            
            pred = pred.cpu()
            ys = ys.cpu()
            if mask is not None:
                mask = mask.cpu()
            
            # 🔧 获取实际的 batch 大小（可能小于 batch_size，比如最后一个 batch）
            actual_batch_size = pred.shape[0]
            
            # 🔧 修复：根据模式正确提取 respond 真实值
            if respond_position_mask is not None:
                # Non-sequential: 使用 mask 提取真实的 respond 值
                respond_position_mask_cpu = respond_position_mask.cpu()
                respond_true = torch.stack([
                    ys[i, respond_position_mask_cpu[i]]
                    for i in range(ys.shape[0])
                ])
            else:
                # Sequential: 直接切片
                respond_true = ys[:, n_prompt:]
            
            # 所有模型都不预测prompt部分
            prompt_mse = torch.zeros(actual_batch_size)  # 不计算prompt MSE，使用实际batch大小

            # 🆕 数独和路径查找任务：跳过 MSE 计算（分类任务，使用准确率指标）
            if is_sudoku_task or is_pathfinding_task:
                # 分类任务，pred 是 logits，不能直接计算 MSE，使用占位值 0
                respond_mse = torch.zeros(actual_batch_size)
                overall_mse = torch.zeros(actual_batch_size)
            # Respond 部分：只对被mask的位置计算MSE（如果提供了mask）
            elif mask is not None:
                # 只对被mask的位置计算MSE
                respond_diff = (pred - respond_true) ** 2  # [B, n_respond] or [B, n_respond, D]
                # 对于高维输出（如Sudoku的810维），需要扩展mask维度以匹配
                if respond_diff.dim() == 3:  # [B, n_respond, D]
                    # 先对每个位置的所有维度求平均：[B, n_respond, D] -> [B, n_respond]
                    mse_per_position = respond_diff.mean(dim=-1)  # [B, n_respond]
                    # 只保留被 mask 的位置
                    mask_float = mask.float()  # [B, n_respond]
                    masked_mse = mse_per_position * mask_float  # [B, n_respond]
                    mask_counts = mask_float.sum(dim=1)  # [B]
                    # 对每个样本的被 mask 位置求平均
                    respond_mse = masked_mse.sum(dim=1) / (mask_counts + 1e-8)  # [B]
                else:  # [B, n_respond]
                    mask_float = mask.float()
                    masked_mse = respond_diff * mask_float  # [B, n_respond]
                    mask_counts = mask_float.sum(dim=1)  # [B]
                    respond_mse = masked_mse.sum(dim=1) / (mask_counts + 1e-8)  # [B]
                # Overall = Respond MSE（因为只预测respond部分）
                overall_mse = respond_mse
            else:
                # 如果没有mask信息，则对所有位置计算MSE（向后兼容）
                # 🔧 确保返回的是 [B] 形状，而不是标量
                if pred.dim() == 3:  # [B, n_respond, D]
                    respond_mse = ((pred - respond_true) ** 2).mean(dim=(1, 2))  # [B]
                elif pred.dim() == 2:  # [B, n_respond]
                    respond_mse = ((pred - respond_true) ** 2).mean(dim=1)  # [B]
                else:
                    # 如果 pred 是 1维或0维，需要特殊处理
                    respond_mse = ((pred - respond_true) ** 2).mean()
                    if respond_mse.dim() == 0:
                        respond_mse = respond_mse.unsqueeze(0)  # 确保是 [1] 而不是标量
                # Overall = Respond MSE（因为只预测respond部分）
                overall_mse = respond_mse
            
            # 🔧 确保所有 MSE 张量都是 1维的 [B]，而不是标量
            if prompt_mse.dim() == 0:
                prompt_mse = prompt_mse.unsqueeze(0)
            if respond_mse.dim() == 0:
                respond_mse = respond_mse.unsqueeze(0)
            if overall_mse.dim() == 0:
                overall_mse = overall_mse.unsqueeze(0)
            
            all_prompt_mse.append(prompt_mse)
            all_respond_mse.append(respond_mse)
            all_overall_mse.append(overall_mse)
            batch_means.append(float(respond_mse.mean().item()))

            # 🆕 如果是数独或路径查找任务，收集预测和真实值
            if is_sudoku_task or is_pathfinding_task:
                # pred 可能是 logits [B, n_respond, seq_len, vocab_size] 或已经是预测值
                # 检查维度来判断
                if pred.dim() == 4:  # Logits format
                    all_predictions.append(pred)
                elif pred.dim() == 3:  # Already predictions
                    all_predictions.append(pred)
                else:
                    if is_sudoku_task:
                        raise ValueError(f"数独任务的 pred 维度异常: {pred.shape}")
                    else:
                        raise ValueError(f"路径查找任务的 pred 维度异常: {pred.shape}")

                all_targets.append(respond_true)
    
    # 聚合结果
    all_prompt_mse = torch.cat(all_prompt_mse, dim=0).numpy()
    all_respond_mse = torch.cat(all_respond_mse, dim=0).numpy()
    all_overall_mse = torch.cat(all_overall_mse, dim=0).numpy()

    # 🆕 计算数独任务的准确率指标
    sudoku_metrics = None
    if is_sudoku_task and all_predictions:
        # 合并所有batch的预测和真实值
        all_predictions_tensor = torch.cat(all_predictions, dim=0)[:num_eval_examples]
        all_targets_tensor = torch.cat(all_targets, dim=0)[:num_eval_examples]  # [N, n_respond, 81]

        # 计算数独指标（使用 tasks_sudoku 中的函数）
        from tasks_sudoku import sudoku_accuracy

        # 检查 all_predictions_tensor 的格式
        if all_predictions_tensor.dim() == 4:  # Logits format: [N, n_respond, 81, 10]
            # sudoku_accuracy 期望 logits 格式
            metrics = sudoku_accuracy(all_predictions_tensor, all_targets_tensor)
        elif all_predictions_tensor.dim() == 3:  # Already predictions: [N, n_respond, 81]
            # 需要转换为 one-hot logits 格式
            # 或者直接计算准确率
            N, n_resp, n_cells = all_predictions_tensor.shape
            # 创建 one-hot logits（将预测值转为确定的 logits）
            pred_logits = torch.zeros(N, n_resp, n_cells, 10, dtype=torch.float32)
            # 使用 scatter_ 创建 one-hot
            pred_values = all_predictions_tensor.long().unsqueeze(-1)  # [N, n_resp, 81, 1]
            pred_logits.scatter_(-1, pred_values, 1.0)  # 在预测的类别上设为1
            pred_logits = pred_logits * 100  # 放大 logits 使 argmax 明确
            metrics = sudoku_accuracy(pred_logits, all_targets_tensor)
        else:
            raise ValueError(f"数独预测张量维度异常: {all_predictions_tensor.shape}")

        sudoku_metrics = {
            'cell_accuracy': metrics['cell_accuracy'].item() if torch.is_tensor(metrics['cell_accuracy']) else metrics['cell_accuracy'],
            'sudoku_accuracy': metrics['board_accuracy'].item() if torch.is_tensor(metrics['board_accuracy']) else metrics['board_accuracy'],
        }

    # 🆕 计算路径查找任务的准确率指标
    pathfinding_metrics = None
    if is_pathfinding_task and all_predictions:
        # 合并所有batch的预测和真实值
        all_predictions_tensor = torch.cat(all_predictions, dim=0)[:num_eval_examples]
        all_targets_tensor = torch.cat(all_targets, dim=0)[:num_eval_examples]

        # 计算路径查找指标（使用 tasks_pathfinding 中的函数）
        from tasks_pathfinding import pathfinding_accuracy

        # pathfinding_accuracy 期望 logits 格式: [B, n_respond, path_len, vocab_size]
        # 和 targets 格式: [B, n_points, path_len]
        if all_predictions_tensor.dim() == 4:  # Logits format
            # 需要构造完整的 ys 格式 [B, n_points, path_len]
            # all_targets_tensor 是 [B, n_respond, path_len]，需要扩展为 [B, n_points, path_len]
            # 简化处理：假设 n_prompt=0，则 n_points = n_respond
            metrics = pathfinding_accuracy(all_predictions_tensor, all_targets_tensor)
        elif all_predictions_tensor.dim() == 3:  # Already predictions
            # 创建 one-hot logits
            B, n_resp, path_len = all_predictions_tensor.shape
            vocab_size = int(all_predictions_tensor.max().item()) + 1
            pred_logits = torch.zeros(B, n_resp, path_len, vocab_size, dtype=torch.float32)
            pred_values = all_predictions_tensor.long().unsqueeze(-1)
            pred_logits.scatter_(-1, pred_values, 1.0)
            pred_logits = pred_logits * 100
            metrics = pathfinding_accuracy(pred_logits, all_targets_tensor)
        else:
            raise ValueError(f"路径查找预测张量维度异常: {all_predictions_tensor.shape}")

        pathfinding_metrics = {
            'node_accuracy': metrics['node_accuracy'].item() if torch.is_tensor(metrics['node_accuracy']) else metrics['node_accuracy'],
            'path_accuracy': metrics['path_accuracy'].item() if torch.is_tensor(metrics['path_accuracy']) else metrics['path_accuracy'],
        }

    results = {
        "prompt_mse": {
            "mean": float(np.mean(all_prompt_mse)),
            "std": float(np.std(all_prompt_mse)),
            "median": float(np.median(all_prompt_mse)),
        },
        "respond_mse": {
            "mean": float(np.mean(all_respond_mse)),
            "std": float(np.std(all_respond_mse)),
            "median": float(np.median(all_respond_mse)),
            "batch_means": batch_means,
        },
        "overall_mse": {
            "mean": float(np.mean(all_overall_mse)),
            "std": float(np.std(all_overall_mse)),
            "median": float(np.median(all_overall_mse)),
        },
    }

    # 🆕 添加数独任务的准确率指标（格式适配 train_prompt_respond.py 的期望）
    if sudoku_metrics is not None:
        results["cell_accuracy"] = {
            "mean": sudoku_metrics["cell_accuracy"],
            "std": 0.0,  # 暂不计算标准差
            "median": sudoku_metrics["cell_accuracy"],
        }
        results["sudoku_accuracy"] = {
            "mean": sudoku_metrics["sudoku_accuracy"],
            "std": 0.0,  # 暂不计算标准差
            "median": sudoku_metrics["sudoku_accuracy"],
        }
        results["rule_violation_rate"] = {
            "mean": 0.0,  # eval_sudoku 没有计算违规率，可以后续添加
            "std": 0.0,
            "median": 0.0,
        }

    # 🆕 添加路径查找任务的准确率指标
    if pathfinding_metrics is not None:
        results["node_accuracy"] = {
            "mean": pathfinding_metrics["node_accuracy"],
            "std": 0.0,
            "median": pathfinding_metrics["node_accuracy"],
        }
        results["path_accuracy"] = {
            "mean": pathfinding_metrics["path_accuracy"],
            "std": 0.0,
            "median": pathfinding_metrics["path_accuracy"],
        }

    return results


def load_model_from_checkpoint(checkpoint_path, config_path=None):
    """
    从 checkpoint 加载模型
    
    Args:
        checkpoint_path: 模型 checkpoint 路径
        config_path: 配置文件路径（可选，如果 checkpoint 中包含 config 则不需要）
    
    Returns:
        model: 加载好的模型
        config: 配置
    """
    print(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 获取 config
    if "config" in checkpoint:
        config = checkpoint["config"]
    elif config_path is not None:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("Config not found in checkpoint and config_path not provided")
    
    # 构建模型
    model = build_model_prompt_respond(config["model"])
    
    # 加载权重
    model.load_state_dict(checkpoint["model_state_dict"])
    
    print(f"Model loaded successfully (trained for {checkpoint.get('train_step', 'unknown')} steps)")
    
    return model, config


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Prompt-Respond ICL Models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (optional)")
    parser.add_argument("--num_eval_examples", type=int, default=1280, help="Number of evaluation examples")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--use_autoregressive_eval", type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=None,
                        help="Use autoregressive evaluation for AR models (avoids label leakage). "
                             "If not specified, will read from config file. "
                             "If False, uses original scheme (with label leakage for AR).")
    args = parser.parse_args()
    
    # Load model
    model, config = load_model_from_checkpoint(args.checkpoint, args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Get parameters
    n_prompt = config["model"]["n_prompt"]
    n_respond = config["model"]["n_respond"]
    n_dims = config["model"]["n_dims"]
    task_name = config["training"]["task"]
    data_name = config["training"]["data"]
    w_type = config["training"].get("w_type", "gaussian")
    
    # Get use_autoregressive_eval from config or command line (command line has priority)
    use_autoregressive_eval = args.use_autoregressive_eval
    if use_autoregressive_eval is None:
        # 从配置文件读取（如果存在 evaluation 部分）
        if "evaluation" in config and "use_autoregressive_eval" in config["evaluation"]:
            use_autoregressive_eval = config["evaluation"]["use_autoregressive_eval"]
        else:
            # 默认值：False（使用原方案）
            use_autoregressive_eval = False
    
    # Build samplers
    # 🆕 支持从配置透传 data_kwargs（如 unit_norm），确保 train/eval 一致
    data_sampler = get_data_sampler(
        data_name,
        n_dims=n_dims,
        **config.get("training", {}).get("data_kwargs", {}),
    )
    task_sampler = get_task_sampler(
        task_name,
        n_dims,
        args.batch_size,
        w_type=w_type,
        **config["training"].get("task_kwargs", {}),
    )
    
    # Get use_ebo_inference from config
    use_ebo_inference = None
    if "evaluation" in config and "use_ebo_inference" in config["evaluation"]:
        use_ebo_inference = config["evaluation"]["use_ebo_inference"]
    
    # Evaluate
    results = eval_model_prompt_respond(
        model,
        task_sampler,
        data_sampler,
        n_prompt,
        n_respond,
        n_dims,
        batch_size=args.batch_size,
        num_eval_examples=args.num_eval_examples,
        use_autoregressive_eval=use_autoregressive_eval,
        use_ebo_inference=use_ebo_inference,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    print(f"Prompt MSE:  Mean={results['prompt_mse']['mean']:.6f}, "
          f"Std={results['prompt_mse']['std']:.6f}, "
          f"Median={results['prompt_mse']['median']:.6f}")
    print(f"Respond MSE: Mean={results['respond_mse']['mean']:.6f}, "
          f"Std={results['respond_mse']['std']:.6f}, "
          f"Median={results['respond_mse']['median']:.6f}")
    print(f"Overall MSE: Mean={results['overall_mse']['mean']:.6f}, "
          f"Std={results['overall_mse']['std']:.6f}, "
          f"Median={results['overall_mse']['median']:.6f}")
    print("=" * 60)
    
    # Save results
    output_dir = os.path.dirname(args.checkpoint)
    results_path = os.path.join(output_dir, "eval_results.yaml")
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"\nResults saved to {results_path}")

