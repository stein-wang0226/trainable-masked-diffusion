'''
从配置文件和命令行参数中加载模型及其相关设置。
定义多种数据生成方法和任务评估逻辑。
聚合并保存评估结果，支持不同的任务和策略
'''
import json
import os
import sys

from munch import Munch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import yaml
import math

import models
from samplers import get_data_sampler, sample_transformation,rand_select_sampler

from tasks import get_task_sampler
"""
测试数据中xs_p 策略
类似自回归？模拟逐步暴露的信息量，测试模型对部分已知信息的利用能力和对未知信息的泛化能力。
"""
'''
读取模型运行路径（run_path）下的配置文件（config.yaml）和模型权重。
支持加载最新的权重（state.pt）或特定步骤的模型（model_{step}.pt）。
关键逻辑：
使用 torch.load 加载模型的状态字典，并通过 model.load_state_dict 恢复模型。
'''
def get_model_from_run(run_path,w_type='add',step=-1, only_conf=False):
    # todo 改的是models12中的config.yaml
    if w_type=="add":
        config_path = os.path.join(run_path, "config.yaml")
    elif w_type=="gaussian":
        config_path = os.path.join(run_path, "config_w_g.yaml")
    elif w_type=="uniform":
        config_path = os.path.join(run_path, "config_w_u.yaml")
    else:
        raise ValueError("w_type must be 'add' or 'gaussian' or 'uniform'.")

    print("run_path:", run_path)

    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        conf = Munch.fromDict(yaml.safe_load(fp)) # todo 从yaml中读取conf
    if only_conf:
        return None, conf
    model = models.build_model(conf.model)

    if step == -1:
        state_path = os.path.join(run_path, "state.pt")
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    return model, conf


# Functions for evaluation

'''
任务评估 batch
功能：

对一个批次的数据进行模型评估。
如果提供 xs_p，则组合训练数据和测试数据进行逐点评估。
逻辑：
调用 task_sampler 生成任务。
根据模型支持的设备（cuda 或 cpu）调整计算。
如果没有 xs_p，直接评估并计算损失。
如果提供了 xs_p，对测试样本逐点评估，并计算逐点评估指标。
'''



# def eval_batch(model, task_sampler, xs, xs_p=None):
#     task = task_sampler()
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     def safe_forward(model, xs, ys, **kwargs):
#         """
#         通用推理前向：
#         - 自动检测模型类型（AR 或 非AR）
#         - 自动关闭 dropout / mask / noise
#         - 自动解包 (loss, pred)
#         """
#         try:
#             # 优先尝试显式 eval 模式
#             output = model(xs.to("cuda"), ys.to("cuda"), train_mode=False, **kwargs)
#         except TypeError:
#             # 对旧版 AR 模型（无 train_mode 参数）
#             output = model(xs.to("cuda"), ys.to("cuda"), **kwargs)

#         # 兼容 (loss, pred) 返回
#         if isinstance(output, tuple):
#             output = output[-1]
#         return output.detach()

#     model.eval()
#     with torch.no_grad():
#         if xs_p is None:
#             # === 多点并行预测 ===
#             ys = task.evaluate(xs)
#             pred = safe_forward(model, xs, ys)
#             metrics = task.get_metric()(pred.cpu(), ys)
#             # 🧠 === Debug 区：保存部分预测与真实值 ===
#             if torch.rand(1).item() < 0.04:  # 只打印/保存 1% 的 batch，防止太多
#                 pred_np = pred[:4].cpu().numpy()
#                 ys_np = ys[:4].cpu().numpy()
#                 print("\n[Eval Debug] === Sample Predictions ===")
#                 for i in range(min(4, len(pred_np))):
#                     print(f"Pred[{i}]: {pred_np[i, :5]} | True[{i}]: {ys_np[i, :5]}")
#                 np.savez("eval_debug_samples.npz", pred=pred_np, true=ys_np)
#                 # ↑ 每次会覆盖保存最新一次 eval 的预测样本

#         else:
#             # === 逐点评估（兼容自回归 Transformer） ===
#             b_size, n_points, _ = xs.shape
#             metrics = torch.zeros(b_size, n_points)
#             for i in range(n_points):
#                 xs_comb = torch.cat((xs[:, :i, :], xs_p[:, i:, :]), dim=1)
#                 ys = task.evaluate(xs_comb)
#                 pred = safe_forward(model, xs_comb, ys, inds=[i])
#                 metrics[:, i] = task.get_metric()(pred.cpu(), ys)[:, i]

#     model.train()
#     return metrics

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os

def eval_batch(model, task_sampler, xs, xs_p=None, debug_save=True, debug_plot=True, save_dir="./eval_debug"):
    """
    增强版 eval_batch:
    - 打印 & 保存预测与真实标签
    - 自动绘制散点图 y_true vs y_pred
    - 保存为 PNG 文件（带时间戳或 step）
    - 🎯 新增：同时计算全 mask (100%) 和配置 mask ratio 两种模式的 MSE
    """
    task = task_sampler()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_dir, exist_ok=True)

    def safe_forward(model, xs, ys, **kwargs):
        try:
            output = model(xs.to(device), ys.to(device), train_mode=False, **kwargs)
        except TypeError:
            output = model(xs.to(device), ys.to(device), **kwargs)
        if isinstance(output, tuple):
            output = output[-1]
        return output.detach()

    model.eval()
    with torch.no_grad():
        if xs_p is None:
            ys = task.evaluate(xs)
            
            # 🎯 检查模型是否支持两种评估模式（LLaDA 和 Dream 都支持）
            supports_dual_eval = hasattr(model, 'train_eval_mask_mode')
            
            if supports_dual_eval:
                # === 模式1：全 mask (100%) 评估 ===
                # 临时保存当前配置
                original_mode = model.train_eval_mask_mode
                original_fixed_ratio = getattr(model, 'fixed_mask_ratio', 0.5)
                
                # 临时设置为 curriculum 模式（推理时使用 t=1.0，全 mask）
                model.train_eval_mask_mode = "curriculum"
                pred_full_mask = safe_forward(model, xs, ys)
                metrics_full_mask = task.get_metric()(pred_full_mask.cpu(), ys)
                
                # === 模式2：按配置的 mask ratio 评估 ===
                # 根据 original_mode 决定推理时的 mask ratio：
                # - 如果 original_mode == "curriculum" → 使用训练时的 mask ratio（max_t）
                #   需要临时设置为 fixed 模式，fixed_mask_ratio = max_t
                # - 如果 original_mode == "fixed" → 使用 fixed_mask_ratio
                if original_mode == "curriculum":
                    # 计算当前训练进度对应的 max_t（curriculum 的 mask ratio）
                    if hasattr(model, '_compute_max_t'):
                        # 使用模型的辅助方法（优先）
                        max_t = model._compute_max_t()
                    else:
                        # 兼容旧代码：手动计算
                        if model.curriculum_schedule == "linear":
                            max_t = 0.5 + 0.5 * model.training_progress
                        elif model.curriculum_schedule == "cosine":
                            max_t = 0.5 + 0.5 * (1 - math.cos(model.training_progress * math.pi)) / 2
                        else:  # exponential
                            max_t = 0.5 + 0.5 * (model.training_progress ** 2)
                    
                    # 临时设置为 fixed 模式，借助fixed_mask_ratio使用 curriculum 的 mask ratio
                    model.train_eval_mask_mode = "fixed" 
                    model.fixed_mask_ratio = max_t
                else:
                    # original_mode == "fixed"，恢复原配置
                    model.train_eval_mask_mode = original_mode
                    if hasattr(model, 'fixed_mask_ratio'):
                        model.fixed_mask_ratio = original_fixed_ratio
                
                pred_config_mask = safe_forward(model, xs, ys)
                metrics_config_mask = task.get_metric()(pred_config_mask.cpu(), ys)
                
                # 恢复原配置（用于后续可能的使用）
                model.train_eval_mask_mode = original_mode
                if hasattr(model, 'fixed_mask_ratio'):
                    model.fixed_mask_ratio = original_fixed_ratio
                
                # 返回两种 metrics（字典格式）
                metrics = {
                    "full_mask": metrics_full_mask,
                    "config_mask": metrics_config_mask
                }
                pred = pred_config_mask  # 用于后续 debug 和绘图
            else:
                # 不支持双模式评估的模型，使用原有逻辑
                pred = safe_forward(model, xs, ys)
                metrics = task.get_metric()(pred.cpu(), ys)

            # 🧠 Debug: 打印前几个样本
            if torch.rand(1).item() < 0.05:  # 5% 概率打印
                pred_np = pred[:4].cpu().numpy()
                ys_np = ys[:4].cpu().numpy()
                print("\n[Eval Debug] === Sample Predictions vs Labels ===")
                for i in range(min(4, len(pred_np))):
                    print(f"Sample[{i}]:")
                    print(f"  y_pred: {np.array2string(pred_np[i, :10], precision=4)}")
                    print(f"  y_true: {np.array2string(ys_np[i, :10], precision=4)}")
                
                # 🎯 如果模型支持两种评估模式，同时打印两种模式的对比
                if supports_dual_eval:
                    pred_full_np = pred_full_mask[:4].cpu().numpy()
                    print("\n[Eval Debug] === Full Mask (100%) vs Config Mask Comparison ===")
                    for i in range(min(4, len(pred_np))):
                        print(f"Sample[{i}]:")
                        print(f"  Full Mask Pred: {np.array2string(pred_full_np[i, :10], precision=4)}")
                        print(f"  Config Mask Pred: {np.array2string(pred_np[i, :10], precision=4)}")
                        print(f"  True: {np.array2string(ys_np[i, :10], precision=4)}")

            # ✅ 保存样本
            if debug_save:
                np.savez(os.path.join(save_dir, "eval_debug_samples.npz"),
                         y_pred=pred.cpu().numpy(),
                         y_true=ys.cpu().numpy())
                # print(f"[Eval Debug] Saved all samples to {save_dir}/eval_debug_samples.npz")

            # ✅ 绘图部分
            if debug_plot:
                y_true = ys.cpu().numpy().flatten()
                y_pred = pred.cpu().numpy().flatten()
                plt.figure(figsize=(5, 5))
                sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, s=15, color="royalblue", edgecolor=None)
                max_val = max(y_true.max(), y_pred.max())
                min_val = min(y_true.min(), y_pred.min())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x (ideal)")
                plt.xlabel("True $y$")
                plt.ylabel("Predicted $\hat{y}$")
                plt.title("Predicted vs True Scatter Plot")
                plt.legend()
                plt.tight_layout()
                save_path = os.path.join(save_dir, "eval_pred_vs_true.png")
                plt.savefig(save_path, dpi=200)
                plt.close()
                # print(f"[Eval Debug] Saved scatter plot to {save_path}")

        else:
            # === 自回归评估（rare case）===
            b_size, n_points, _ = xs.shape
            metrics = torch.zeros(b_size, n_points)
            for i in range(n_points):
                xs_comb = torch.cat((xs[:, :i, :], xs_p[:, i:, :]), dim=1)
                ys = task.evaluate(xs_comb)
                pred = safe_forward(model, xs_comb, ys, inds=[i])
                metrics[:, i] = task.get_metric()(pred.cpu(), ys)[:, i]

    model.train()
    return metrics



# Functions for generating different kinds of train/test data
# 定义数据生成策略
# 通过不同的生成策略，模拟各种任务难度和特征分布
'''
5种用于生成训练和测试数据的strategy
gen_standard：标准训练数据生成，无特殊变化。
gen_opposite_quadrants：生成符号相反的训练和测试样本。
gen_random_quadrants：生成随机象限的训练样本。
gen_orthogonal_train_test：生成训练集和测试集正交的样本。
gen_overlapping_train_test：生成部分重叠的训练和测试样本。
'''
# 无 xs_p
def gen_standard(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)

    return xs, None

# 生成 符号相反 的训练集和测试集
def gen_opposite_quadrants(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = -xs_train_pre

    return xs_train_pre, xs_test_post

# 生成 随机象限 的训练集和测试集 每个维度随机符号
def gen_random_quadrants(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = xs

    return xs_train_pre, xs_test_post

# 生成训练集和测试集，使得测试样本与训练样本的特征在向量空间中是 正交的。
def gen_orthogonal_train_test(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    n_dim = xs.shape[2]
    n_points = min(n_points, n_dim)
    # raise ValueError("number of points should be at most the dimension.")
    xs_train_pre = xs
    xs_test_post = torch.zeros(xs.shape)
    for i in range(n_points):
        xs_test_post_i = xs[:, i : i + 1, :]
        xs_train_pre_i = xs[:, :i, :]
        _, _, Vt = torch.linalg.svd(xs_train_pre_i, full_matrices=False)
        xs_train_pre_i_projection = Vt.transpose(1, 2) @ Vt
        xs_test_post_i_orthogonalized = (
            xs_test_post_i - xs_test_post_i @ xs_train_pre_i_projection
        )
        xs_test_post_i_normalized = (
            xs_test_post_i_orthogonalized
            * xs_test_post_i.norm(dim=2).unsqueeze(2)
            / xs_test_post_i_orthogonalized.norm(dim=2).unsqueeze(2)
        )

        xs_test_post[:, i : i + 1, :] = xs_test_post_i_normalized

    return xs_train_pre, xs_test_post


def gen_overlapping_train_test(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    xs_train_pre = xs
    xs_test_post = xs.clone()
    b_size = xs.shape[0]
    for i in range(1, n_points):
        xs_train_pre_i = xs[:, :i, :]
        perm = torch.stack([torch.randperm(i) for _ in range(b_size)]).unsqueeze(dim=1)
        ind_mat = (perm == 0) + 0.0
        xs_test_post[:, i : i + 1, :] = ind_mat @ xs_train_pre_i


    return xs_train_pre, xs_test_post



'''
聚合评估结果
功能：
对评估结果metrics进行统计分析，计算均值、标准差和引导法（Bootstrap）置信区间。

输入：
metrics：形状为 [num_eval, n_points] 的张量，表示多个批次的逐点评估结果。
输出： 一个包含以下字段的字典：
mean：逐点的平均值。
std：逐点的标准差。
bootstrap_low 和 bootstrap_high：置信区间的上下界。
'''

def aggregate_metrics(metrics, bootstrap_trials=1000):
    """
    Takes as input a tensor of shape (num_eval, n_points) and returns a dict with
    per-point mean, stddev, and bootstrap limits
    """
    results = {}
    results["mean"] = metrics.mean(dim=0)
    results["std"] = metrics.std(dim=0, unbiased=True)
    n = len(metrics)
    bootstrap_indices = torch.randint(n, size=(bootstrap_trials, n))
    bootstrap_means = metrics[bootstrap_indices].mean(dim=1).sort(dim=0)[0]
    results["bootstrap_low"] = bootstrap_means[int(0.05 * bootstrap_trials), :]
    results["bootstrap_high"] = bootstrap_means[int(0.95 * bootstrap_trials), :]

    return {k: v.tolist() for k, v in results.items()}

'''
评估整个模型
功能：

执行模型的整体评估，支持多种任务、数据生成策略和配置。
对多个批次的数据调用 eval_batch，并聚合结果。

关键逻辑：
初始化data sampler task_sampler。
动态加载数据生成函数：
generating_func = globals()[f"gen_{prompting_strategy}"]
根据 prompting_strategy 动态选择数据生成策略。
循环执行 eval_batch 并收集结果。
使用 aggregate_metrics 统计和保存结果。
'''
def eval_model( #
    # 参数匹配 kwargs key, 从 kwargs 中解析得到参数
    model,
    task_name,
    data_name,
    n_dims,
    n_points,
    prompting_strategy,
    num_eval_examples=1280,
    batch_size=64,
    If_shift_w_distribution=False, # 默认false ， yaml传入true 启用 w1 + w2
    eval_w_type="add",
    data_sampler_kwargs={},
    task_sampler_kwargs={},
):
    """
    Evaluate a model on a task with a variety of strategies.
       Args:
       - task: which base task we are evaluating on. E.g., "linear_regression"
       - prompting_strategy: how to construct the prompt, e.g., "random_quadrants"
       - num_eval_examples: total number of examples to evaluate on
       - **sampler_kwargs: remaining arguments to pass directly to the sampler
    """

    assert num_eval_examples % batch_size == 0
    # todo data sampler

    data_sampler = get_data_sampler(data_name, n_dims, **data_sampler_kwargs)

    # todo   (w1+w2)x  if
    if If_shift_w_distribution:
        task_sampler = get_task_sampler(
            task_name, n_dims, batch_size,w_type=eval_w_type, **task_sampler_kwargs
        )
    else:
        task_sampler = get_task_sampler(
            task_name, n_dims, batch_size, **task_sampler_kwargs
        )

    all_metrics = []
    all_metrics_full_mask = []  # 🎯 全 mask 模式的 metrics
    all_metrics_config_mask = []  # 🎯 配置 mask 模式的 metrics
    
    supports_dual_eval = hasattr(model, 'train_eval_mask_mode')

    generating_func = globals()[f"gen_{prompting_strategy}"] # 根据变量prompting_strategy选择 data生成strategy function
    for i in range(num_eval_examples // batch_size):
        # 根据strategy生成 xs 和 xs_p  (符号相反、随机象限..)
        xs, xs_p = generating_func(data_sampler, n_points, batch_size)

        metrics = eval_batch(model, task_sampler, xs, xs_p)
        
        if supports_dual_eval and isinstance(metrics, dict):
            # 支持双模式评估的模型返回字典，包含两种模式的 metrics
            all_metrics_full_mask.append(metrics["full_mask"])
            all_metrics_config_mask.append(metrics["config_mask"])
            all_metrics.append(metrics["config_mask"])  # 保持向后兼容
        else:
            # 不支持双模式评估的模型，使用原有逻辑
            all_metrics.append(metrics)

    metrics = torch.cat(all_metrics, dim=0)
    results = aggregate_metrics(metrics)
    
    # 🎯 如果模型支持双模式评估，同时聚合两种模式的 metrics
    if supports_dual_eval and len(all_metrics_full_mask) > 0:
        metrics_full_mask = torch.cat(all_metrics_full_mask, dim=0)
        metrics_config_mask = torch.cat(all_metrics_config_mask, dim=0)
        results_full_mask = aggregate_metrics(metrics_full_mask)
        results_config_mask = aggregate_metrics(metrics_config_mask)
        
        # 合并结果
        results = {
            "full_mask": results_full_mask,
            "config_mask": results_config_mask,
            "mean": results_config_mask["mean"],  # 保持向后兼容，默认返回 config_mask
        }
        
        # 🧠 === 额外统计输出 ===
        mean_full = np.mean(results_full_mask['mean'])
        mean_config = np.mean(results_config_mask['mean'])
        print(f"[Eval Summary] Full Mask (100%): Mean={mean_full:.6f}, Std={np.std(results_full_mask['mean']):.6f}")
        print(f"[Eval Summary] Config Mask: Mean={mean_config:.6f}, Std={np.std(results_config_mask['mean']):.6f}")
    else:
        # 🧠 === 额外统计输出 ===
        print(f"[Eval Summary] Mean={np.mean(results['mean']):.6f}, Std={np.std(results['mean']):.6f}")

    return results



'''
自动化评估构建
功能：
根据配置（conf）生成所有支持的评估策略。
包括标准评估、随机象限、正交训练测试等。
用途：
批量管理评估任务，便于扩展和多策略比较。
'''
# todo 根据配置（conf） 读取参数
def build_evals(conf):# 学习 domain shift
    n_dims = conf.model.n_dims
    n_points = conf.training.curriculum.points.end
    batch_size = conf.training.batch_size

    task_name = conf.training.task
    data_name = conf.training.data

    If_shift_w_distribution = conf.eval.If_shift_w_distribution
    eval_w_type = conf.eval.eval_w_type
    # 创建评估任务的基础配置，所有任务共享这些参数。
    # 如果具体任务有附加需求，可以在后续阶段覆盖这些参数。
    base_kwargs = {
        "task_name": task_name,
        "n_dims": n_dims,
        "n_points": n_points,
        "batch_size": batch_size,
        "data_name": data_name,
        "prompting_strategy": "standard",
        # todo eval from shifted distribution
        "If_shift_w_distribution":If_shift_w_distribution,
        "eval_w_type": eval_w_type,
    }
    evaluation_kwargs = {}
    # 默认的标准评估任务，其prompting_strategy为"standard"
    evaluation_kwargs["standard"] = {"prompting_strategy": "standard"} #
    #  如果任务名称不是linear_regression：添加一个linear_regression的评估任务，用于与其他任务比较.
    #遍历当前的evaluation_kwargs，将基础参数base_kwargs
    # 合并到每个任务的配置中, 返回更新后的evaluation_kwargs。
    if task_name != "linear_regression":
        if task_name in ["relu_2nn_regression"]:
            evaluation_kwargs["linear_regression"] = {"task_name": "linear_regression"}
        for name, kwargs in evaluation_kwargs.items():
            # allow kwargs to override base_kwargs values
            evaluation_kwargs[name] = base_kwargs.copy()
            evaluation_kwargs[name].update(kwargs)
        return evaluation_kwargs # 非linear


    # 生成prompt 的strategy
    for strategy in [
        "random_quadrants",
        "orthogonal_train_test",
        "overlapping_train_test",
    ]:
        evaluation_kwargs[strategy] = {"prompting_strategy": strategy}

    for method in ["half_subspace", "skewed"]:
        if "subspace" in method:
            eigenvals = torch.zeros(n_dims)
            eigenvals[: n_dims // 2] = 1
        else:
            eigenvals = 1 / (torch.arange(n_dims) + 1)

        scale = sample_transformation(eigenvals, normalize=True)
        evaluation_kwargs[f"{method}"] = {
            "data_sampler_kwargs": {"scale": scale},
        }

    for dim in ["x", "y"]:
        for scale in [0.333, 0.5, 2, 3]:
            if dim == "x":
                eigenvals = scale * torch.ones(n_dims)
                t = sample_transformation(eigenvals)
                scaling_args = {"data_sampler_kwargs": {"scale": t}}
            else:
                eigenvals = scale * torch.ones(n_dims)
                scaling_args = {"task_sampler_kwargs": {"scale": scale}}

            evaluation_kwargs[f"scale-{dim}={scale}"] = scaling_args

    evaluation_kwargs[f"noisyLR"] = {
        "task_sampler_kwargs": {"renormalize_ys": True, "noise_std": 1},
        "task_name": "noisy_linear_regression",
    }

    for name, kwargs in evaluation_kwargs.items():
        # allow kwargs to override base_kwargs values
        evaluation_kwargs[name] = base_kwargs.copy()
        evaluation_kwargs[name].update(kwargs)

    return evaluation_kwargs
"""
return evaluation_kwargs like:
{
    "standard": {
        "task_name": "relu_2nn_regression",
        "n_dims": 20,
        "n_points": 40,
        "batch_size": 64,
        "data_name": "gaussian",
        "prompting_strategy": "standard",
        "eval_w_type": "weight_shift"
    },
    "linear_regression": {
        "task_name": "linear_regression",
        "n_dims": 20,
        "n_points": 40,
        "batch_size": 64,
        "data_name": "gaussian",
        "prompting_strategy": "standard",
        "eval_w_type": "weight_shift"
    }
}
...
"""
def compute_evals(all_models, evaluation_kwargs, save_path=None, recompute=False):
    try:
        with open(save_path) as fp:
            all_metrics = json.load(fp)
    except Exception:
        all_metrics = {}

    for eval_name, kwargs in tqdm(evaluation_kwargs.items()): # 最后一个 error
        metrics = {}
        if eval_name in all_metrics and not recompute:
            metrics = all_metrics[eval_name]
        for model in all_models:
            if model.name in metrics and not recompute:
                continue

            metrics[model.name] = eval_model(model, **kwargs)
        all_metrics[eval_name] = metrics
        if save_path is not None:
            with open(save_path, "w") as fp:
                json.dump(all_metrics, fp, indent=2)
    # 保存评估指标
    if save_path is not None:
        with open(save_path, "w") as fp:
            json.dump(all_metrics, fp, indent=2)

    return all_metrics


def get_run_metrics(
    run_path, step=-1, cache=True, skip_model_load=False, skip_baselines=False,w_type="add",
):
    if skip_model_load:
        # todo 不同conf
        _, conf = get_model_from_run(run_path,w_type=w_type, only_conf=True)
        all_models = []
    else:
        model, conf = get_model_from_run(run_path,w_type=w_type, step=step)
        model = model.cuda().eval()
        all_models = [model]
        if not skip_baselines: #
            all_models += models.get_relevant_baselines(conf.training.task)
    evaluation_kwargs = build_evals(conf) # 根据conf解析的每个task的参数
    # write result into metrics.json
    if not cache:
        save_path = None
    elif step == -1:
        save_path = os.path.join(run_path, f"metrics_{w_type}.json") # 结果保存路径
    else:
        save_path = os.path.join(run_path, f"metrics_{w_type}_{step}.json")

    recompute = False
    if save_path is not None and os.path.exists(save_path):
        checkpoint_created = os.path.getmtime(run_path)
        cache_created = os.path.getmtime(save_path)
        if checkpoint_created > cache_created:
            recompute = True

    all_metrics = compute_evals(all_models, evaluation_kwargs, save_path, recompute)
    return all_metrics



def conf_to_model_name(conf):
    if conf.model.family == "gpt2" or conf.model.family == "llada" or conf.model.family == "gptJ":
        return {
            (3, 2): "Transformer-xs",
            (6, 4): "Transformer-small",
            (12, 8): "Transformer",
        }[(conf.model.n_layer, conf.model.n_head)]
    else:
        return conf.wandb.name


def baseline_names(name):
    if "OLS" in name:
        return "Least Squares"
    if name == "averaging":
        return "Averaging"
    if "NN" in name:
        k = name.split("_")[1].split("=")[1]
        return f"{k}-Nearest Neighbors"
    if "lasso" in name:
        alpha = name.split("_")[1].split("=")[1]
        return f"Lasso (alpha={alpha})"
    if "gd" in name:
        return "2-layer NN, GD"
    if "decision_tree" in name:
        return "Greedy Tree Learning"
    if "xgboost" in name:
        return "XGBoost"
    return name
'''
运行目录管理
'''

def read_run_dir(run_dir):
    all_runs = {}
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        for run_id in os.listdir(task_dir):
            run_path = os.path.join(task_dir, run_id)
            _, conf = get_model_from_run(run_path, only_conf=True)
            params = {}
            params["run_id"] = run_id
            params["task"] = task
            params["model"] = conf_to_model_name(conf)
            params["kwargs"] = "_".join(
                f"{k}={v}" for k, v in conf.training.task_kwargs.items()
            )
            num_tasks = (
                conf.training.num_tasks if "num_tasks" in conf.training else None
            )
            params["num_tasks"] = num_tasks if num_tasks is not None else -1
            num_examples = (
                conf.training.num_training_examples
                if "num_training_examples" in conf.training
                else None
            )
            params["num_examples"] = num_examples if num_examples is not None else -1
            params["n_dims"] = conf.model.n_dims
            params["n_layer"] = conf.model.n_layer
            params["n_head"] = conf.model.n_head
            params["run_name"] = conf.wandb.name

            for k, v in params.items():
                if k not in all_runs:
                    all_runs[k] = []
                all_runs[k].append(v)

    df = pd.DataFrame(all_runs).sort_values("run_name")
    # assert len(df) == len(df.run_name.unique())
    if len(df) != len(df.run_name.unique()):
        print(f"Warning: Found {len(df) - len(df.run_name.unique())} duplicate run_name(s).")

    return df

if __name__ == "__main__":
    run_dir = sys.argv[1]
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        print(f"Evaluating task {task}")
        for run_id in tqdm(os.listdir(task_dir)):
            run_path = os.path.join(run_dir, task, run_id)
            metrics = get_run_metrics(run_path)



            