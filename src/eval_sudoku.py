"""
数独任务评估脚本
================

评估训练好的数独模型在测试集上的表现。

使用方法：
    python src/eval_sudoku.py \
        --checkpoint checkpoints/sudoku_llada_*/state.pt \
        --num_eval_examples 500 \
        --batch_size 32
"""

import os
import sys
import argparse
import torch
import yaml
import numpy as np
from tqdm import tqdm

# 导入数独模块
from samplers_sudoku import get_sudoku_data_sampler
from tasks_sudoku import get_sudoku_task_sampler
from models_prompt_respond_sudoku import build_sudoku_model


def evaluate_sudoku_model(
    model,
    data_sampler,
    task_sampler,
    n_prompt,
    n_respond,
    batch_size,
    num_eval_examples,
    device,
    is_ar_model=False,
):
    """
    评估数独模型
    
    Args:
        model: 训练好的模型
        data_sampler: 数独数据采样器
        task_sampler: 数独任务采样器
        n_prompt: prompt数量
        n_respond: respond数量
        batch_size: 批大小
        num_eval_examples: 总评估样本数
        device: 设备
        is_ar_model: 是否是AR模型
    
    Returns:
        dict: 评估结果
    """
    model.eval()
    
    total_points = n_prompt + n_respond
    num_batches = (num_eval_examples + batch_size - 1) // batch_size
    
    # 收集所有预测和真实值
    all_predictions = []
    all_targets = []
    
    print(f"\n开始评估 (共 {num_batches} 个batch)...")
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="评估进度"):
            # 采样数据
            xs = data_sampler.sample_xs(total_points, batch_size)
            task = task_sampler()
            ys = task.evaluate(xs)
            
            # 模型预测
            xs_device = xs.to(device)
            ys_device = ys.to(device)
            
            if is_ar_model:
                pred_y = model(xs_device, ys_device, train_mode=False)
            else:
                pred_y, _ = model(xs_device, ys_device, train_mode=False)
            
            # 提取 respond 部分的真实值
            true_y = ys[:, n_prompt:, :].cpu()
            pred_y = pred_y.cpu()
            
            all_predictions.append(pred_y)
            all_targets.append(true_y)
    
    # 合并所有batch
    all_predictions = torch.cat(all_predictions, dim=0)[:num_eval_examples]
    all_targets = torch.cat(all_targets, dim=0)[:num_eval_examples]
    
    # 计算指标
    results = compute_sudoku_metrics(all_predictions, all_targets)
    
    return results


def compute_sudoku_metrics(predictions, targets):
    """
    计算数独任务的评估指标（One-Hot编码）
    
    Args:
        predictions: [N, n_respond, 810] 预测值（logits）
        targets: [N, n_respond, 810] 真实值（one-hot）
    
    Returns:
        dict: 评估指标
    """
    from tasks_sudoku import decode_sudoku_logits, decode_sudoku_onehot
    
    # 解码为类别
    pred_digits = decode_sudoku_logits(predictions)  # [N, n_respond, 81]
    true_digits = decode_sudoku_onehot(targets)  # [N, n_respond, 81]
    
    # 格子级别的准确率
    correct_cells = (pred_digits == true_digits).float()
    cell_accuracy = correct_cells.mean().item()
    
    # 数独级别的准确率（每个数独的81个格子都正确）
    sudoku_accuracy_per_example = correct_cells.mean(dim=2)  # [N, n_respond]
    sudoku_perfect = (sudoku_accuracy_per_example == 1.0).float()
    sudoku_accuracy = sudoku_perfect.mean().item()
    
    # MSE（归一化值）
    mse = ((predictions - targets) ** 2).mean().item()
    
    # 每个位置的准确率（用于分析哪些位置更难）
    per_position_accuracy = correct_cells.mean(dim=(0, 1))  # [81]
    
    # 统计完全正确的数独数量
    total_sudokus = predictions.shape[0] * predictions.shape[1]
    perfect_sudokus = sudoku_perfect.sum().item()
    
    results = {
        'cell_accuracy': cell_accuracy,
        'sudoku_accuracy': sudoku_accuracy,
        'mse': mse,
        'total_sudokus': total_sudokus,
        'perfect_sudokus': int(perfect_sudokus),
        'per_position_accuracy': per_position_accuracy.numpy(),
    }
    
    return results


def print_results(results):
    """打印评估结果"""
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    
    print(f"格子准确率 (Cell Accuracy): {results['cell_accuracy']:.4f}")
    print(f"  → 平均每个格子预测正确的概率")
    
    print(f"\n数独准确率 (Sudoku Accuracy): {results['sudoku_accuracy']:.4f}")
    print(f"  → 完全正确的数独比例")
    print(f"  → 完全正确: {results['perfect_sudokus']}/{results['total_sudokus']}")
    
    print(f"\nMSE (归一化值): {results['mse']:.6f}")
    
    # 分析最难和最容易的位置
    per_pos_acc = results['per_position_accuracy']
    easiest_positions = np.argsort(per_pos_acc)[-5:]
    hardest_positions = np.argsort(per_pos_acc)[:5]
    
    print(f"\n最容易的5个位置 (准确率最高):")
    for pos in easiest_positions[::-1]:
        row, col = pos // 9, pos % 9
        print(f"  位置 [{row}, {col}] (索引 {pos}): {per_pos_acc[pos]:.4f}")
    
    print(f"\n最难的5个位置 (准确率最低):")
    for pos in hardest_positions:
        row, col = pos // 9, pos % 9
        print(f"  位置 [{row}, {col}] (索引 {pos}): {per_pos_acc[pos]:.4f}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="评估数独 ICL 模型")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--data_path", type=str, default="diffusion-vs-ar/data/data/sudoku_test.csv",
                        help="测试数据集路径")
    parser.add_argument("--num_eval_examples", type=int, default=500, help="评估样本数")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)")
    args = parser.parse_args()
    
    # 设置设备
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 加载检查点
    print(f"\n加载检查点: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"错误: 检查点不存在: {args.checkpoint}")
        return
    
    state = torch.load(args.checkpoint, map_location=device)
    config = state["config"]
    
    print(f"训练步数: {state.get('train_step', 'unknown')}")
    print(f"模型类型: {config['model']['family']}")
    
    # 构建模型
    print("\n构建模型...")
    model = build_sudoku_model(config["model"])
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    
    # 判断模型类型
    is_ar_model = hasattr(model, 'family') and model.family in [
        "gpt2", "gptj", "llama", "llama2", "llama3", "qwen", "qwen2", "qwen2.5"
    ]
    
    print(f"模型类型: {'AR' if is_ar_model else 'MDM'}")
    
    # 创建数据采样器和任务采样器
    print(f"\n加载数据: {args.data_path}")
    data_sampler = get_sudoku_data_sampler(
        data_path=args.data_path,
        n_dims=810  # One-Hot编码
    )
    
    task_sampler = get_sudoku_task_sampler(
        n_dims=81,
        batch_size=args.batch_size,
        data_sampler=data_sampler
    )
    
    n_prompt = config["model"]["n_prompt"]
    n_respond = config["model"]["n_respond"]
    
    print(f"Prompt: {n_prompt}, Respond: {n_respond}")
    
    # 评估模型
    results = evaluate_sudoku_model(
        model=model,
        data_sampler=data_sampler,
        task_sampler=task_sampler,
        n_prompt=n_prompt,
        n_respond=n_respond,
        batch_size=args.batch_size,
        num_eval_examples=args.num_eval_examples,
        device=device,
        is_ar_model=is_ar_model,
    )
    
    # 打印结果
    print_results(results)
    
    # 保存结果
    out_dir = os.path.dirname(args.checkpoint)
    results_path = os.path.join(out_dir, "eval_results.yaml")
    
    # 转换numpy数组为列表（以便保存为YAML）
    results_to_save = results.copy()
    results_to_save['per_position_accuracy'] = results_to_save['per_position_accuracy'].tolist()
    
    with open(results_path, 'w') as f:
        yaml.dump(results_to_save, f)
    
    print(f"结果已保存到: {results_path}")


if __name__ == "__main__":
    main()

