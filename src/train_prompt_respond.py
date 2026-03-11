"""
Training Script for Prompt-Respond Models
=========================================

本文件是本仓库中 **Prompt-Respond ICL / Sudoku / ** 统一的训练入口脚本：

- 数据生成：prompt (n_prompt) + respond (k)
- AR Model: 自回归预测 respond 部分，只在 respond 计算 loss
- MDM Model: 只对 respond 部分 mask，只在 respond 计算 loss

主要功能：
- `train()` 函数：主训练循环
- 训练监控：WandB + 本地日志
- 验证评估：固定验证集 + 多步数评估
- 支持：Multi-Epoch、Non-Sequential ICL、Multi-Step Inference

使用方式（示例）：
    python src/train_prompt_respond.py --config path/to/your_config.yaml
"""

import os
import sys
import time  # 🔧 添加 time 模块，用于统计验证耗时

# 🔧 关键修复：必须在任何 torch.distributed 调用之前设置 NCCL 超时环境变量
# 这些环境变量必须在导入 torch 之前设置，否则不会生效
os.environ.setdefault("NCCL_TIMEOUT", "3600000")  # 1小时（毫秒）
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

import yaml
import torch
import numpy as np
import inspect  # 🆕 用于检查模型 forward 函数签名
from tqdm import tqdm
import wandb

# 🆕 分布式训练支持（已迁移到 train_utils.py）

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

# Import models
from models_prompt_respond import build_model_prompt_respond
# 🆕 数独任务专用模型
try:
    from models_prompt_respond_sudoku import build_sudoku_model
except ImportError:
    build_sudoku_model = None
# 🆕 路径查找任务专用模型
try:
    from models_prompt_respond_pathfinding import build_pathfinding_model
except ImportError:
    build_pathfinding_model = None

# Import data & task samplers
from samplers import get_data_sampler
from tasks import get_task_sampler
from curriculum import Curriculum

# 🆕 Import utilities from separate modules
from train_utils import (
    set_seed,
    sample_seeds,
    build_validation_seed_batches,
    build_epoch_seed_pool,
    shuffle_prompt_respond_pairs,
    generate_fixed_permutation,
    apply_fixed_permutation,
    apply_random_permutation,
    generate_permutation_pool,
    get_model_attr,  # 🆕 添加 DDP 兼容的模型属性访问函数
    apply_pool_permutation,
    train_step_ar,
    train_step_mdm,
    LocalMSELogger,
    extract_respond_values,
    setup_accelerator,
    parse_multi_epoch_config,
    setup_sequence_mode,
    generate_output_directory,
    resume_from_checkpoint,
    setup_wandb,
    check_training_complete,
    get_learning_rate,  # 🏗️ 统一的学习率调度函数
    extract_classification_model_outputs,  # 🏗️ 统一的分类模型返回值提取（Sudoku, Pathfinding 等）
    normalize_classification_output,  # 🏗️ 统一的分类模型 output 归一化
    optimizer_step,  # 🏗️ 统一的优化器步骤封装
    nebula_tokens_to_digits,  # 🏗️ core-nebula token 转数字
    calculate_sudoku_accuracy,  # 🏗️ Sudoku 准确率计算
    calculate_sudoku_accuracy_masked,  # 🏗️ 仅 masked 位置的准确率（与 loss 对齐）
    profile_model_flops,  # 🆕 FLOPs 性能分析函数
)

def train(model, config):
    """Main training loop for Prompt-Respond models"""
    
    print("=== Parsed config (brief) ===")
    print("model:\n", yaml.dump(config["model"], sort_keys=False))
    print("training:\n", yaml.dump(config["training"], sort_keys=False))
    print("wandb:\n", yaml.dump(config["wandb"], sort_keys=False))
    
    training = config["training"]
    wandb_cfg = config["wandb"]

    # ============================================================
    # 🆕 Sudoku "core-nebula compatible" mode (B) - config switch
    # ============================================================
    # This mode bypasses the prompt-respond ICL formulation and trains/evals in the
    # same (x,y) token format as dllm-pathfinding/core-nebula:
    # - sequence = (quiz_with_$ + '=') + solution
    # - x = sequence[:-1]
    # - y = sequence[1:], with prefix positions masked as -100
    #
    # Backward compatible: default remains the existing ICL sudoku pipeline.
    sudoku_format = training.get("sudoku_format", "icl")
    # ============================================================
    # (B) core-nebula aligned trainer for Dream Sudoku (REMOVED)
    # ============================================================
    # NOTE: v3 训练闭环已迁移合并进通用训练外壳（统一协议）。
    # 所有 Sudoku 模型现在使用统一的训练流程（见下方 is_sudoku_task 分支）。
    # 已删除的代码块：core_nebula_dream_trainer_csv


    # ============================================================
    # core-nebula aligned trainer for Sudoku AR (REMOVED)
    # ============================================================
    # NOTE: v3 训练闭环已迁移合并进通用训练外壳（统一协议）。
    # 已删除的代码块：core_nebula_ar_trainer


    # ============================================================
    # (B) core-nebula compatible token-format trainer (generic LM)
    # ============================================================
    if training.get("task_type") == "sudoku" and sudoku_format == "core_nebula":
        import pandas as pd
        from torch.utils.data import Dataset, DataLoader
        import torch.nn.functional as F

        try:
            from dllm_pathfinding.core_nebula.tokenizer import SudokuTokenizer  # type: ignore
        except Exception:
            # Fallback to local path import (repo layout)
            import os as _os, sys as _sys
            _sys.path.append(_os.path.join(_os.path.dirname(__file__), "..", "dllm-pathfinding", "core-nebula"))
            from tokenizer import SudokuTokenizer  # type: ignore

        class _SudokuNebulaCsvDataset(Dataset):
            def __init__(self, csv_path: str):
                # Preserve leading zeros for quizzes/solutions
                self.df = pd.read_csv(csv_path, dtype={"quizzes": str, "solutions": str})
                self.tokenizer = SudokuTokenizer()

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                q = str(self.df.iloc[idx]["quizzes"]).strip()
                a = str(self.df.iloc[idx]["solutions"]).strip()
                q = q.zfill(81)
                a = a.zfill(81)
                # core-nebula uses '$' for blanks, digits 1-9, and '=' delimiter
                q = q.replace("0", "$")
                prefix_str = q + "="
                target_str = a
                prefix = self.tokenizer.encode(prefix_str)
                target = self.tokenizer.encode(target_str)
                full = torch.tensor(prefix + target, dtype=torch.long)
                x = full[:-1]
                y = full[1:].clone()
                y[: len(prefix) - 1] = -100
                return x, y

        def _pad_collate(batch):
            xs, ys = zip(*batch)
            max_len = max(x.size(0) for x in xs)
            pad_id = SudokuTokenizer().encoder["$"]  # safe: '$' exists
            x_pad = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
            y_pad = torch.full((len(xs), max_len), -100, dtype=torch.long)
            for i, (x, y) in enumerate(zip(xs, ys)):
                x_pad[i, : x.size(0)] = x
                y_pad[i, : y.size(0)] = y
            return x_pad, y_pad

        # Rebuild a vocab-sized LM head model in-place:
        # Expect user to set model.family to a causal LM family supported by build_model_prompt_respond,
        # OR pass a HF-style model that supports (input_ids)->logits. Here we use the provided model as-is.
        # The only requirement: model(xs) returns logits [B, L, vocab].
        train_ds = _SudokuNebulaCsvDataset(training["data_path"])
        val_cfg = training.get("validation", {})
        val_ds = _SudokuNebulaCsvDataset(val_cfg.get("data_path", training["data_path"]))
        train_loader = DataLoader(train_ds, batch_size=training["batch_size"], shuffle=True, collate_fn=_pad_collate)
        val_loader = DataLoader(val_ds, batch_size=val_cfg.get("batch_size", training["batch_size"]), shuffle=False, collate_fn=_pad_collate)

        device = next(model.parameters()).device
        optim = torch.optim.Adam(model.parameters(), lr=training["learning_rate"], weight_decay=training.get("weight_decay", 0.0))
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

        def _token_acc(logits, y):
            preds = torch.argmax(logits, dim=-1)
            mask = (y != -100)
            correct = (preds[mask] == y[mask]).float().mean().item() if mask.any() else 0.0
            return correct

        def _seq_acc(logits, y):
            preds = torch.argmax(logits, dim=-1)
            mask = (y != -100)
            correct_tokens = (preds == y) | ~mask
            return torch.all(correct_tokens, dim=1).float().mean().item()

        total_steps = training["train_steps"]
        eval_every = val_cfg.get("eval_every_steps", 1000)
        model.train()
        for step in tqdm(range(total_steps), ncols=120, miniters=50, mininterval=2.0):
            x, y = next(iter(train_loader))
            x, y = x.to(device), y.to(device)
            logits = model(x)  # expect [B, L, vocab]
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            optim.zero_grad()
            loss.backward()
            optim.step()
            if step % max(1, training.get("log_interval", 50)) == 0:
                with torch.no_grad():
                    ta = _token_acc(logits, y)
                    sa = _seq_acc(logits, y)
                tqdm.write(f"[core_nebula] step={step} loss={loss.item():.4f} token_acc={ta:.3f} seq_acc={sa:.3f}")
            if step > 0 and step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    accs, saccs = [], []
                    for vx, vy in val_loader:
                        vx, vy = vx.to(device), vy.to(device)
                        v_logits = model(vx)
                        accs.append(_token_acc(v_logits, vy))
                        saccs.append(_seq_acc(v_logits, vy))
                tqdm.write(f"[core_nebula][val] step={step} token_acc={sum(accs)/len(accs):.3f} seq_acc={sum(saccs)/len(saccs):.3f}")
                model.train()
        return
    
    # 🆕 检测任务类型（需要在早期检测，因为后面会用到）
    is_sudoku_task = training.get("task_type") == "sudoku" or "sudoku" in training.get("data_path", "").lower()
    is_pathfinding_task = training.get("task_type") == "pathfinding" or "pathfinding" in training.get("data_path", "").lower()
    log_interval = max(1, training.get("log_interval", 100))
    wandb_log_interval = max(log_interval, wandb_cfg.get("log_every_steps", log_interval))
    
    # 🆕 初始化 Accelerator（如果可用且启用分布式训练）
    accelerator, device, use_distributed = setup_accelerator(training)
    
    model.to(device).train()
    
    # 🆕 设置随机种子（从config或默认值42）
    random_seed = training.get("random_seed", config.get("random_seed", 42))
    set_seed(random_seed)
    print(f"\n{'='*60}")
    print(f"🎲 Random Seed: {random_seed}")
    print(f"{'='*60}\n")
    
    # 🆕 读取 sequence_mode 配置
    sequence_mode = training.get("sequence_mode", "sequential")  # "sequential", "non_sequential", "fixed_permutation", "fixed_permutation_xy", "random_permutation", "pool_permutation", "pool_permutation_xy"
    valid_modes = ["sequential", "non_sequential", "fixed_permutation", "fixed_permutation_xy", "random_permutation", "pool_permutation", "pool_permutation_xy"]
    assert sequence_mode in valid_modes, \
        f"Invalid sequence_mode: {sequence_mode}. Must be one of {valid_modes}."
    
    # 判断是否是 AR 模型
    # 🆕 安全地访问 model.family（兼容 DDP 包装）
    model_family = get_model_attr(model, 'family')
    is_ar_model = model_family is not None and model_family in [
        "gpt2", "gptj", "llama", "llama2", "llama3", "qwen", "qwen2", "qwen2.5",
    ]
    
    # 🆕 提前读取 n_prompt（用于生成固定排列）
    n_prompt = config["model"]["n_prompt"]
    
    # Curriculum（数独和路径查找任务不需要）- 需要在设置序列模式之前确定 initial_n_respond
    if is_sudoku_task or is_pathfinding_task:
        # 数独/路径查找任务：固定 n_respond，不使用 curriculum
        cur = None
        n_respond = config["model"]["n_respond"]  # 使用固定值
        initial_n_respond = n_respond
    else:
        # 标准任务：使用 curriculum
        if "curriculum" not in training:
            raise KeyError("'curriculum' is required for non-sudoku/pathfinding tasks. Please add curriculum configuration or set task_type: sudoku/pathfinding")
        cur = Curriculum(training["curriculum"])
        n_respond = None  # 由 curriculum 动态控制
        initial_n_respond = training["curriculum"]["points"]["start"]
    
    # 🆕 设置序列模式并生成相应的排列
    fixed_permutation_x, fixed_permutation_y, permutation_pool_x, permutation_pool_y = setup_sequence_mode(
        sequence_mode, training, n_prompt, initial_n_respond, model, is_ar_model
    )
    
    # 🆕 Multi-epoch 配置
    use_multi_epoch, num_epochs, steps_per_epoch, shuffle_between_epochs = parse_multi_epoch_config(training)
    multi_epoch_cfg = training.get("multi_epoch", {})
    
    # Optimizer
    # - Non-sudoku/pathfinding: keep legacy Adam (do not break linear regression ICL)
    # - Sudoku/Pathfinding (all families): AdamW + optional warmup/cosine schedule (nebula-style)
    if is_sudoku_task or is_pathfinding_task:
        learning_rate = float(training["learning_rate"])
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=float(training.get("weight_decay", 0.01)),
        )
        warmup_iters = int(training.get("warmup_steps", 1000))
        min_lr = float(training.get("min_lr", 1e-5))
    else:
        optim = torch.optim.Adam(
            model.parameters(),
            lr=training["learning_rate"],
            weight_decay=training.get("weight_decay", 0),
        )
    
    # 🆕 使用 Accelerator 准备模型和优化器（分布式训练）
    if use_distributed:
        # 🆕 设置 NCCL/RCCL 环境变量（在分布式训练开始前）
        # 这些环境变量需要在进程组初始化之前设置
        import os
        # PyTorch 的 NCCL 环境变量（使用 TORCH_ 前缀）
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
        # NCCL 超时设置
        os.environ.setdefault("NCCL_IB_TIMEOUT", "22")  # InfiniBand 超时 22 秒
        # NCCL 调试（可选，生产环境可设为 WARN）
        if os.environ.get("NCCL_DEBUG") is None:
            os.environ.setdefault("NCCL_DEBUG", "INFO")
        # 分布式调试（可选，生产环境可关闭）
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "INFO")
        
        model, optim = accelerator.prepare(model, optim)
        
        # 🆕 验证 DDP 设置：检查模型是否被正确包装，并确认 find_unused_parameters 设置
        import torch.nn as nn
        if isinstance(model, nn.parallel.DistributedDataParallel):
            # 如果模型是 DDP 包装的，检查 find_unused_parameters 状态
            # 注意：find_unused_parameters 在 DDP 初始化时设置，这里只是验证
            print(f"[Distributed] 模型已通过 DDP 包装")
            print(f"[Distributed] 注意: find_unused_parameters 应在 Accelerator 初始化时通过 DDPPlugin 设置")
        elif hasattr(model, 'module'):
            # Accelerate 可能使用其他包装方式
            print(f"[Distributed] 模型已通过 Accelerate 包装 (检测到 model.module)")
        
        print(f"[Distributed] 模型和优化器已通过 Accelerator 准备")
    
    bsz = training["batch_size"]
    # 🆕 数独/路径查找任务：n_dims 根据任务类型设置
    if is_sudoku_task:
        n_dims = 81  # 数独任务：quiz 部分为 81 维
    elif is_pathfinding_task:
        # 路径查找任务：edge_len = (path_len - 1) * degree * 2 + 2
        degree = config["model"].get("degree", 2)
        path_len = config["model"].get("path_len", 3)
        n_dims = (path_len - 1) * degree * 2 + 2  # edges + query
    else:
        n_dims = config["model"]["n_dims"]
    # n_prompt 已在前面定义（用于 Fixed Permutation）
    
    # Validation config (fixed set)
    validation_cfg = training.get("validation", {})
    validation_batch_count = validation_cfg.get("batch_count", 50)
    validation_batch_size = validation_cfg.get("batch_size", bsz)
    validation_seed = validation_cfg.get("seed", 1234)
    validation_eval_steps = validation_cfg.get("eval_every_steps", training.get("eval_every_steps", 1000))

    # Data & Task Samplers
    if is_sudoku_task:
        # 🆕 数独任务：使用数独专用的采样器
        try:
            from samplers_sudoku import get_sudoku_data_sampler
            from tasks_sudoku import get_sudoku_task_sampler
        except ImportError:
            raise ImportError("数独任务需要导入 samplers_sudoku 和 tasks_sudoku 模块")

        data_path = training.get("data_path", "diffusion-vs-ar/data/data/sudoku_train.csv")
        data_sampler = get_sudoku_data_sampler(
            data_path=data_path,
            n_dims=81,  # 数独任务：quiz 部分为 81 维
            **training.get("data_kwargs", {}),
        )
        task_sampler = get_sudoku_task_sampler(
            n_dims=81,  # 数独任务：quiz 部分为 81 维
            batch_size=bsz,
            data_sampler=data_sampler,
            **training.get("task_kwargs", {}),
        )
    elif is_pathfinding_task:
        # 🆕 路径查找任务：使用路径查找专用的采样器
        try:
            from samplers_pathfinding import get_pathfinding_data_sampler
            from tasks_pathfinding import get_pathfinding_task_sampler
        except ImportError:
            raise ImportError("路径查找任务需要导入 samplers_pathfinding 和 tasks_pathfinding 模块")

        data_path = training.get("data_path", "dllm-pathfinding/core-nebula/data/datasets/graphs/deg_2_path_3_nodes_10_train_200000.txt")
        num_nodes = config["model"].get("num_nodes", 10)
        degree = config["model"].get("degree", 2)
        path_len = config["model"].get("path_len", 3)

        data_sampler = get_pathfinding_data_sampler(
            data_path=data_path,
            n_dims=n_dims,
            num_nodes=num_nodes,
            degree=degree,
            path_len=path_len,
            **training.get("data_kwargs", {}),
        )
        task_sampler = get_pathfinding_task_sampler(
            n_dims=n_dims,
            batch_size=bsz,
            data_sampler=data_sampler,
            **training.get("task_kwargs", {}),
        )
    else:
        # 标准任务：使用标准采样器
        if "data" not in training:
            raise KeyError("'data' is required for non-sudoku/pathfinding tasks")
        if "task" not in training:
            raise KeyError("'task' is required for non-sudoku/pathfinding tasks")
        
        # 🆕 支持通过 training.data_kwargs 传递数据采样器参数（如 unit_norm）
        data_sampler = get_data_sampler(
            training["data"],
            n_dims=n_dims,
            **training.get("data_kwargs", {}),
        )
        task_sampler = get_task_sampler(
            training["task"],
            n_dims,
            bsz,
            w_type=training.get("w_type", "gaussian"),
            num_tasks=training.get("num_tasks"),
            **training.get("task_kwargs", {}),
        )

    validation_batches = build_validation_seed_batches(
        validation_batch_count, validation_batch_size, validation_seed
    )
    validation_num_examples = validation_batch_count * validation_batch_size
    
    # 🆕 Multi-epoch: 预生成固定数据集种子池
    if use_multi_epoch:
        epoch_seed_pool = build_epoch_seed_pool(
            num_epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=bsz,
            base_seed=42,
            shuffle_between_epochs=shuffle_between_epochs
        )
        print(f"[Multi-Epoch] Generated seed pool with {len(epoch_seed_pool)} entries")
    else:
        epoch_seed_pool = None

    # Output directory
    model_conf = config["model"]
    out_dir = generate_output_directory(config, training, model_conf, n_prompt, is_sudoku_task, n_respond)
    config["out_dir"] = out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    # 🔧 关键优化：日志先写本地临时路径，避免 OSS 写入慢导致死锁
    # OSS 挂载盘（ossfs）每写一行都会触发 HTTP POST，网络慢时会卡死 Rank 0
    # 解决方案：先写本地硬盘，定期同步到 OSS（在checkpoint保存时），训练结束后再最终同步
    local_jsonl_dir = "./tmp_logs"
    os.makedirs(local_jsonl_dir, exist_ok=True)
    
    # 本地临时路径（快速写入）
    local_train_log_path = os.path.join(local_jsonl_dir, "train_mse.jsonl")
    local_validation_log_path = os.path.join(local_jsonl_dir, "validation_mse.jsonl")
    
    # OSS 最终路径（定期同步和最终同步）
    oss_log_dir = os.path.join(out_dir, "logs")
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(oss_log_dir, exist_ok=True)
    
    oss_train_log_path = os.path.join(oss_log_dir, "train_mse.jsonl")
    oss_validation_log_path = os.path.join(oss_log_dir, "validation_mse.jsonl")
    
    # 🔧 定义日志同步函数（在checkpoint保存时调用，避免训练失败丢失日志）
    def sync_logs_to_oss():
        """将本地日志同步到OSS（仅在主进程执行，带超时保护）
        
        🆕 修复：使用追加模式合并日志，而不是覆盖，避免丢失已有日志
        """
        if not use_distributed or accelerator.is_main_process:
            import shutil
            import time
            import json
            start_time = time.time()
            try:
                # 🔧 优化：添加超时保护，避免OSS网络慢导致长时间阻塞
                # 如果OSS很慢，最多等待30秒，然后继续训练
                timeout = 30  # 秒
                
                # 🆕 同步训练日志：使用追加模式合并，而不是覆盖
                if os.path.exists(local_train_log_path):
                    try:
                        # 🆕 读取本地日志的所有新记录（只追加 OSS 中没有的记录）
                        local_records = []
                        if os.path.exists(local_train_log_path):
                            with open(local_train_log_path, 'r') as f:
                                for line in f:
                                    line = line.strip()
                                    if line:
                                        try:
                                            local_records.append(json.loads(line))
                                        except json.JSONDecodeError:
                                            continue
                        
                        # 🆕 读取 OSS 上已有的记录（如果存在）
                        oss_existing_steps = set()
                        if os.path.exists(oss_train_log_path):
                            with open(oss_train_log_path, 'r') as f:
                                for line in f:
                                    line = line.strip()
                                    if line:
                                        try:
                                            record = json.loads(line)
                                            if 'step' in record:
                                                oss_existing_steps.add(record['step'])
                                        except json.JSONDecodeError:
                                            continue
                        
                        # 🆕 只追加本地日志中 OSS 没有的记录
                        new_records = [r for r in local_records if r.get('step', -1) not in oss_existing_steps]
                        
                        if new_records:
                            # 追加模式写入 OSS
                            with open(oss_train_log_path, 'a') as f:
                                for record in new_records:
                                    f.write(json.dumps(record) + "\n")
                                f.flush()
                            elapsed = time.time() - start_time
                            if elapsed > 5:
                                print(f"⚠️  训练日志同步耗时 {elapsed:.1f}秒: 追加了 {len(new_records)} 条新记录到 {oss_train_log_path}")
                            else:
                                print(f"✅ 训练日志已同步到 OSS: 追加了 {len(new_records)} 条新记录")
                        else:
                            print(f"ℹ️  训练日志无需同步: 所有记录已存在")
                    except Exception as e:
                        print(f"⚠️  训练日志同步失败: {e}，继续训练")
                
                # 🆕 同步验证日志：使用追加模式合并，而不是覆盖
                if os.path.exists(local_validation_log_path):
                    try:
                        # 🆕 读取本地日志的所有新记录
                        local_records = []
                        if os.path.exists(local_validation_log_path):
                            with open(local_validation_log_path, 'r') as f:
                                for line in f:
                                    line = line.strip()
                                    if line:
                                        try:
                                            local_records.append(json.loads(line))
                                        except json.JSONDecodeError:
                                            continue
                        
                        # 🆕 读取 OSS 上已有的记录（如果存在）
                        oss_existing_keys = set()
                        if os.path.exists(oss_validation_log_path):
                            with open(oss_validation_log_path, 'r') as f:
                                for line in f:
                                    line = line.strip()
                                    if line:
                                        try:
                                            record = json.loads(line)
                                            step = record.get('step', -1)
                                            inference_steps = record.get('inference_steps', None)
                                            # 对于多步验证，使用 (step, inference_steps) 作为唯一标识
                                            if inference_steps is not None:
                                                oss_existing_keys.add((step, inference_steps))
                                            else:
                                                oss_existing_keys.add((step, None))
                                        except json.JSONDecodeError:
                                            continue
                        
                        # 🆕 只追加本地日志中 OSS 没有的记录
                        new_records = []
                        for r in local_records:
                            step = r.get('step', -1)
                            inference_steps = r.get('inference_steps', None)
                            key = (step, inference_steps)
                            if key not in oss_existing_keys:
                                new_records.append(r)
                        
                        if new_records:
                            # 追加模式写入 OSS
                            with open(oss_validation_log_path, 'a') as f:
                                for record in new_records:
                                    f.write(json.dumps(record) + "\n")
                                f.flush()
                            elapsed = time.time() - start_time
                            if elapsed > 5:
                                print(f"⚠️  验证日志同步耗时 {elapsed:.1f}秒: 追加了 {len(new_records)} 条新记录到 {oss_validation_log_path}")
                            else:
                                print(f"✅ 验证日志已同步到 OSS: 追加了 {len(new_records)} 条新记录")
                        else:
                            print(f"ℹ️  验证日志无需同步: 所有记录已存在")
                    except Exception as e:
                        print(f"⚠️  验证日志同步失败: {e}，继续训练")
                
                total_elapsed = time.time() - start_time
                if total_elapsed > 10:
                    print(f"⚠️  日志同步总耗时 {total_elapsed:.1f}秒，可能影响训练速度")
                    
            except Exception as e:
                print(f"⚠️  警告: 同步日志到 OSS 失败: {e}")
                print(f"   本地日志仍保留在: {local_jsonl_dir}")
                print(f"   训练将继续进行，日志将在下次checkpoint时重试同步")
    
    # is_ar_model 已在前面定义
    model_type = "AR" if is_ar_model else "MDM"
    # 🆕 只有主进程才写入日志，避免多进程竞争导致数据覆盖
    is_main_process_for_logger = not use_distributed or accelerator.is_main_process
    # 🔧 使用本地路径初始化 Logger（快速写入，避免 OSS 卡死）
    mse_logger = LocalMSELogger(
        local_train_log_path, local_validation_log_path, plot_dir, log_interval, model_type,
        is_main_process=is_main_process_for_logger
    )
    state_path = os.path.join(out_dir, "state.pt")
    print(f"[Output Directory] {out_dir}")
    
    # Resume logic
    starting_step, starting_epoch = resume_from_checkpoint(
        model, optim, training, state_path, device, is_sudoku_task, cur, use_multi_epoch
    )
    
    # 🆕 Initialize wandb（简化：一个run，一条完整曲线）
    should_log_wandb = setup_wandb(wandb_cfg, config, use_distributed, accelerator)
    
    # Training loop
    pool_size = training.get("num_training_examples", None)
    
    # 🆕 计算总步数：如果使用multi-epoch，则覆盖train_steps
    if use_multi_epoch:
        total_steps = num_epochs * steps_per_epoch
    else:
        # 🏗️ 统一使用 train_steps（向后兼容：如果存在 max_steps，优先使用，但已废弃）
        total_steps = int(training.get("train_steps", training.get("max_steps", 0)))
        if total_steps == 0:
            raise ValueError("必须指定 train_steps 或 max_steps（已废弃，建议使用 train_steps）")
    
    # 🆕 检查训练是否已完成，避免不必要的覆盖
    if check_training_complete(starting_step, total_steps, state_path, device, model, optim, config, 
                                use_distributed, accelerator, mse_logger):
        return
    
    pbar = tqdm(
        range(starting_step, total_steps),
        initial=starting_step,
        total=total_steps,
        ncols=120,  # 设置进度条宽度
        miniters=50,  # 最小更新间隔（步数）：每50步更新一次
        mininterval=2.0,  # 最小更新间隔（秒）：至少2秒更新一次
        leave=True,  # 训练结束后保留进度条
    )
    
    # 🆕 用于跟踪当前epoch
    current_epoch = starting_epoch

    # 🏗️ 使用统一的学习率调度函数（从 train_utils 导入）
    # 注意：_get_lr 已迁移到 train_utils.get_learning_rate，这里使用别名保持兼容
    _get_lr = get_learning_rate

    # ============================================================
    # v3: nebula-token Dream trainer merged into unified shell (DEPRECATED)
    # ============================================================
    # NOTE: This branch is deprecated. All Sudoku models now use the unified ICL training
    # protocol below. The code has been removed.
    # 已删除的代码块：use_nebula_v3


    # 🆕 sudoku/pathfinding LR schedule applied in unified loop
    if is_sudoku_task or is_pathfinding_task:
        lr_decay_iters = int(total_steps)

    # 🆕 初始化累计算力跟踪
    cumulative_training_flops = 0
    train_step_flops = None  # 将在 step==10 时测量
    single_inference_flops = None  # 🆕 单次推理FLOPs，将在 step==10 时测量

    for step in pbar:
        if is_sudoku_task or is_pathfinding_task:
            # 🏗️ 使用统一的学习率调度函数
            lr = get_learning_rate(step, float(training["learning_rate"]), warmup_iters, lr_decay_iters, min_lr)
            for pg in optim.param_groups:
                pg["lr"] = lr
        # 🆕 Multi-epoch: 计算当前epoch和epoch内步数
        if use_multi_epoch:
            current_epoch = step // steps_per_epoch
            step_in_epoch = step % steps_per_epoch
            
            # 检测epoch切换
            if step > starting_step and step_in_epoch == 0:
                print(f"\n{'='*60}")
                print(f"🔄 Epoch {current_epoch} Started (Total Step: {step})")
                print(f"{'='*60}\n")
        else:
            step_in_epoch = step
        
        data_sampler_args, task_sampler_args = {}, {}
        
        # 🆕 Multi-epoch: 从预生成的种子池中获取种子
        if use_multi_epoch and epoch_seed_pool is not None:
            seed_entry = epoch_seed_pool[step]
            data_sampler_args["seeds"] = seed_entry["data_seeds"]
            task_sampler_args["seeds"] = seed_entry["task_seeds"]
        elif pool_size is not None:
            # 原始逻辑：从pool_size中随机采样
            assert pool_size >= bsz
            seeds = sample_seeds(pool_size, bsz)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]
        
        # === 数据采样：prompt + respond ===
        # 使用 curriculum 控制的当前 respond 数量
        # 🆕 数独/路径查找任务使用固定值，标准任务使用 curriculum
        if is_sudoku_task:
            current_n_respond = n_respond
            current_n_dims = 81  # 数独任务：quiz 部分为 81 维
        elif is_pathfinding_task:
            current_n_respond = n_respond
            current_n_dims = n_dims  # 路径查找任务：使用计算的 n_dims
        else:
            current_n_respond = cur.n_points
            current_n_dims = cur.n_dims_truncated
        
        total_points = n_prompt + current_n_respond
        xs = data_sampler.sample_xs(total_points, bsz, current_n_dims, **data_sampler_args)
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs)
        loss_func = task.get_training_metric()
        
        # === 🆕 Non-Sequential ICL: 打乱 Prompt-Respond Pairs ===
        respond_position_mask = None
        if sequence_mode == "non_sequential":
            xs, ys, respond_position_mask = shuffle_prompt_respond_pairs(xs, ys, n_prompt, current_n_respond)
        elif sequence_mode in ["fixed_permutation", "fixed_permutation_xy"]:
            # 🆕 Fixed Permutation: 使用固定排列
            # 模式1：只打乱 x（fixed_permutation_y=None）
            # 模式2：同时打乱 x 和 y（fixed_permutation_y 不为 None）
            xs, ys, respond_position_mask = apply_fixed_permutation(
                xs, ys, n_prompt, current_n_respond, 
                fixed_permutation_x, 
                fixed_permutation_y
            )
        elif sequence_mode == "random_permutation":
            # 🆕 Random Permutation: 每个batch使用不同的随机排列（比fixed更难）
            permute_y = training.get("permute_y", False)  # 是否也打乱y（可选）
            xs, ys, respond_position_mask = apply_random_permutation(
                xs, ys, n_prompt, current_n_respond, 
                permute_y=permute_y
            )
        elif sequence_mode in ["pool_permutation", "pool_permutation_xy"]:
            # 🆕 Pool Permutation: 从固定排列池中随机采样（介于fixed和random之间）
            xs, ys, respond_position_mask = apply_pool_permutation(
                xs, ys, n_prompt, current_n_respond, 
                permutation_pool_x, 
                permutation_pool_y
            )
        
        # 🆕 FLOPs 测量：在 step==10 时测量单步训练 FLOPs
        if step == 10 and train_step_flops is None and (not use_distributed or accelerator.is_main_process):
            print(f"\n[FLOPs Profiling] 正在测量单步训练 FLOPs (step {step})...")
            try:
                sample_batch = (xs.to(device), ys.to(device))
                flops_dict = profile_model_flops(
                    model, sample_batch, optim, loss_func,
                    is_sudoku_task=is_sudoku_task,
                    is_pathfinding_task=is_pathfinding_task,
                    respond_position_mask=respond_position_mask.to(device) if respond_position_mask is not None else None,
                    accelerator=accelerator if use_distributed else None,
                    config=config
                )
                train_step_flops = flops_dict['train_step_flops']
                forward_flops = flops_dict['forward_flops']
                single_inference_flops = flops_dict['single_inference_flops']  # 🆕 保存为全局变量
                print(f"  ✅ 单步训练总 FLOPs: {train_step_flops:.2e}")
                print(f"  ✅ 单次前向传播 FLOPs: {forward_flops:.2e}")
                print(f"  ✅ 单次推理 FLOPs: {single_inference_flops:.2e}")
                print(f"  ✅ 模型家族: {flops_dict['model_family']}")
                if flops_dict['is_ar_model']:
                    print(f"  ✅ AR 模型，推理需要 {flops_dict.get('n_respond', '?')} 次前向传播")
                else:
                    print(f"  ✅ Diff/MDM 模型，推理需要 {flops_dict.get('sampling_steps', '?')} 步采样")
            except Exception as e:
                print(f"  ⚠️  FLOPs 测量失败: {e}")
                import traceback
                traceback.print_exc()
                train_step_flops = None
        
        # === 训练步骤 ===
        if is_sudoku_task or is_pathfinding_task:
            # ============================================================
            # 🏗️ 统一 Sudoku/Pathfinding 训练协议
            # ============================================================
            # 协议：
            # 1. 所有模型在 train_mode=True 时统一调用：
            #    results = model(xs, ys, train_mode=True, respond_position_mask=...)
            # 2. 模型内部负责：
            #    - Token 化和序列拼接
            #    - 如果是扩散模型：采样 t 和 Mask
            #    - 如果是 AR 模型：Causal Mask
            # 3. 返回值统一：
            #    - 至少返回 (loss, output)
            #    - output: [B, n_respond, seq_len, vocab_size]
            # ============================================================
            
            # DEPRECATED: icl_nebula_token_trainer branch (REMOVED)
            # NOTE: This branch is mathematically equivalent to unified protocol.
            # The unified protocol (model forward) does the same operations internally.
            # 已删除的代码块：icl_nebula_token_trainer
            
            # ============================================================
            # 🏗️ 协议化训练步骤（所有 Sudoku 模型）
            # ============================================================
            optim.zero_grad(set_to_none=True)
            
            # 统一调用模型 forward
            results = model(
                xs.to(device), 
                ys.to(device), 
                train_mode=True, 
                respond_position_mask=respond_position_mask.to(device) if respond_position_mask is not None else None
            )

            # 🏗️ 使用统一的返回值提取函数
            loss, output, respond_masked_indices = extract_classification_model_outputs(results)

            # 🏗️ 使用统一的 output 归一化函数
            output = normalize_classification_output(output)

            # 🏗️ 使用统一的优化器步骤函数
            optimizer_step(loss, optim, accelerator=accelerator if use_distributed else None)
            
            # 转换为标量用于日志（保持与旧代码兼容）
            loss = loss.item()
            if output is not None:
                output = output.detach()
        else:
            # 标准任务：使用通用训练步骤
            if is_ar_model:
                # 🆕 AR 模型：支持 Non-Sequential（根据 attention_mode）
                loss, output, respond_masked_indices = train_step_ar(
                    model, xs.to(device), ys.to(device), 
                    optim, loss_func,
                    respond_position_mask=respond_position_mask.to(device) if respond_position_mask is not None else None,
                    accelerator=accelerator if use_distributed else None  # 🆕 传递 accelerator
                )
            else:
                loss, output, respond_masked_indices = train_step_mdm(
                    model, xs.to(device), ys.to(device), 
                    optim, loss_func, 
                    respond_position_mask=respond_position_mask.to(device) if respond_position_mask is not None else None,
                    accelerator=accelerator if use_distributed else None  # 🆕 传递 accelerator
                )
        
        # 🆕 更新累计算力（每个训练步）
        # 注意：train_step_flops 已经是整个 batch 的总 FLOPs（在 profile_model_flops 中测量时使用的 sample_batch 的 batch size 就是 bsz）
        # 因此不需要再乘以 bsz，否则会重复计算导致虚高 bsz 倍
        if train_step_flops is not None:
            cumulative_training_flops += train_step_flops
        
        # === 打印样本（每 2000 步）===
        # 所有模型都只预测respond部分
        if step % 2000 == 0:
            # 临时清空进度条描述，避免干扰打印
            pbar.set_description("")
            respond_pred = output.detach().cpu().numpy()
            
            # 🔧 使用辅助函数：提取 respond 真实值
            respond_true = extract_respond_values(ys, n_prompt, respond_position_mask, actual_n_respond=current_n_respond).detach().cpu().numpy()
            
            print(f"\n[Step {step}] Sample predictions (first 2 examples) - {model_type} Model:")
            for i in range(min(2, bsz)):
                print(f"  Example {i}:")
                print(f"    Respond part (indices {n_prompt}-{total_points-1}):")
                print(f"      Pred: {respond_pred[i]}")
                print(f"      True: {respond_true[i]}")
            print()  # 空行分隔，然后进度条会继续显示
        
        # === 计算指标（用于监控）===
        # 所有模型都只预测respond部分
        with torch.no_grad():
            respond_pred = output.cpu()
            
            # 🔧 使用辅助函数：提取 respond 真实值
            respond_true = extract_respond_values(ys, n_prompt, respond_position_mask, actual_n_respond=current_n_respond).cpu()
            
            # 🆕 数独任务：只计算准确率指标（分类任务，不使用 MSE）
            cell_accuracy = None
            sudoku_accuracy = None
            node_accuracy = None
            path_accuracy = None
            respond_mse = None  # 数独和pathfinding任务不使用 MSE

            if is_sudoku_task:
                try:
                    from tasks_sudoku import decode_sudoku_logits, decode_sudoku_onehot

                    # 解码为类别
                    # respond_pred: [B, n_respond, 81, 10] (logits)
                    # respond_true: [B, n_respond, 81] (solution)
                    pred_digits = decode_sudoku_logits(respond_pred)  # [B, n_respond, 81]
                    true_digits = respond_true.long()  # [B, n_respond, 81] (已经是 solution，直接使用)

                    # 如果 pred_digits 有 n_respond 维度，取最后一个（或平均）
                    if pred_digits.dim() == 3:
                        pred_digits = pred_digits[:, -1, :]  # [B, 81]
                    if true_digits.dim() == 3:
                        true_digits = true_digits[:, -1, :]  # [B, 81]

                    # 🏗️ 仅监控指标：扩散类模型 loss 只对 masked 算，train acc 对齐则仅 masked 统计（只读 mask，不改任何训练/推理逻辑）
                    if respond_masked_indices is not None:
                        mask_cpu = respond_masked_indices.cpu().float()
                        if mask_cpu.dim() == 3:
                            mask_cpu = mask_cpu[:, -1, :]  # [B, 81]，与 pred/true 对应
                        if mask_cpu.sum() >= 1:
                            cell_accuracy, sudoku_accuracy = calculate_sudoku_accuracy_masked(
                                pred_digits, true_digits, mask_cpu
                            )
                        else:
                            cell_accuracy, sudoku_accuracy = 0.0, 0.0
                    else:
                        cell_accuracy, sudoku_accuracy = calculate_sudoku_accuracy(pred_digits, true_digits)
                except Exception as e:
                    import traceback
                    print(f"⚠️  警告: 计算数独准确率时出错: {e}")
                    print(f"    Traceback: {traceback.format_exc()}")
            elif is_pathfinding_task:
                try:
                    from tasks_pathfinding import decode_pathfinding_logits

                    # 解码为节点 ID
                    # respond_pred: [B, n_respond, path_len, vocab_size] (logits)
                    # respond_true: [B, n_respond, path_len] (path nodes)
                    pred_nodes = decode_pathfinding_logits(respond_pred)  # [B, n_respond, path_len]
                    true_nodes = respond_true.long()  # [B, n_respond, path_len]

                    # 如果 pred_nodes 有 n_respond 维度，取最后一个
                    if pred_nodes.dim() == 3:
                        pred_nodes = pred_nodes[:, -1, :]  # [B, path_len]
                    if true_nodes.dim() == 3:
                        true_nodes = true_nodes[:, -1, :]  # [B, path_len]

                    # 计算节点级别和路径级别准确率
                    correct = (pred_nodes == true_nodes).float()
                    node_accuracy = correct.mean()  # 节点准确率
                    path_correct = (correct.mean(dim=-1) == 1.0).float()
                    path_accuracy = path_correct.mean()  # 路径准确率（完全正确）
                except Exception as e:
                    import traceback
                    print(f"⚠️  警告: 计算路径查找准确率时出错: {e}")
                    print(f"    Traceback: {traceback.format_exc()}")
            else:
                # 非数独任务：计算 MSE（回归任务）
                # 🔧 对于 MDM 模型，只对 masked 位置计算 MSE（与训练 loss 保持一致）
                # 对于 AR 模型，对所有 respond 位置计算 MSE
                if respond_masked_indices is not None:
                    # MDM 模型：只对 masked 位置计算 MSE
                    respond_masked_indices_cpu = respond_masked_indices.cpu().bool()
                    # 对每个样本，只计算 masked 位置的 MSE
                    masked_squared_errors = (respond_pred - respond_true) ** 2  # [B, n_respond, D]
                    # 🔧 修复：先对每个位置的所有维度求平均，再对 masked 位置求平均
                    # 这样得到的 MSE 与 AR 模型的计算方式一致（每个维度的平均误差）
                    if masked_squared_errors.dim() == 3:  # [B, n_respond, D]
                        # 先对每个位置的所有维度求平均：[B, n_respond, D] -> [B, n_respond]
                        masked_errors_per_position = masked_squared_errors.mean(dim=-1)
                        # 只保留 masked 位置的误差
                        masked_errors_masked_only = masked_errors_per_position * respond_masked_indices_cpu.float()
                        num_masked = respond_masked_indices_cpu.float().sum().item()
                        if num_masked > 0:
                            respond_mse = masked_errors_masked_only.sum().item() / num_masked
                        else:
                            respond_mse = 0.0
                    else:  # [B, n_respond] - 已经是每个位置的平均误差
                        masked_errors_masked_only = masked_squared_errors * respond_masked_indices_cpu.float()
                        num_masked = respond_masked_indices_cpu.float().sum().item()
                        if num_masked > 0:
                            respond_mse = masked_errors_masked_only.sum().item() / num_masked
                        else:
                            respond_mse = 0.0
                else:
                    # AR 模型：对所有 respond 位置计算 MSE
                    respond_mse = ((respond_pred - respond_true) ** 2).mean().item()

        # 🆕 记录到本地日志（只在验证时记录，与验证日志频率保持一致）
        # 🔧 修改：减少训练日志写入频率，只在验证时记录（与验证日志同步）
        if step % validation_eval_steps == 0:
            mse_logger.record_train(
                step=step,
                raw_mse=respond_mse,  # 数独和pathfinding任务时为 None
                dims=current_n_dims,
                respond_points=current_n_respond,
                prompt_points=n_prompt,
                model_type=model_type,
                cell_accuracy=cell_accuracy if is_sudoku_task else None,
                sudoku_accuracy=sudoku_accuracy if is_sudoku_task else None,
                node_accuracy=node_accuracy if is_pathfinding_task else None,
                path_accuracy=path_accuracy if is_pathfinding_task else None,
                cumulative_flops=cumulative_training_flops if train_step_flops is not None else None,  # 🆕 累计算力
                single_inference_flops=single_inference_flops,  # 🆕 单次推理FLOPs
                train_loss=loss,  # 🆕 训练 loss
            )
        
        # === 计算 Pointwise Loss 和 Excess Loss（类似 train_dllm.py）===
        # 只对 respond 部分计算 pointwise loss
        respond_output = output.to(device)
        
        # 🔧 使用辅助函数：提取 respond 真实值
        respond_ys = extract_respond_values(ys, n_prompt, respond_position_mask, actual_n_respond=current_n_respond).to(device)
        
        pointwise_tags = list(range(current_n_respond))  # respond 部分的位置索引（从0开始）
        pointwise_metric = task.get_metric()
        metric_val = pointwise_metric(respond_output, respond_ys)
        
        # 🔧 处理数独任务返回字典的情况
        if isinstance(metric_val, dict):
            # 数独任务：返回的是字典，提取 cell_accuracy 作为 pointwise 指标
            # cell_accuracy 已经是标量张量，直接使用
            pointwise_loss = metric_val.get('cell_accuracy', torch.tensor(0.0, device=device))
            # 确保是标量张量（如果已经是标量则不变）
            if pointwise_loss.dim() > 0:
                pointwise_loss = pointwise_loss.mean()
        else:
            # 回归任务：返回的是 Tensor，计算位置误差的平均值
            pointwise_loss = metric_val.mean(dim=0)
        
        # Baseline loss：对于 respond 部分，baseline 应该考虑 prompt 已经提供的信息
        # 原始代码中 baseline 是 max(n_dims - i, 0)，其中 i 是序列中的位置索引（从0开始）
        # 在 prompt-respond 设置中，respond 部分在完整序列中的位置是从 n_prompt 开始的
        # 但考虑到 prompt 已经提供了 n_prompt 个示例，respond 部分的 baseline 应该更小
        # 这里我们使用 respond 部分的相对位置（从 0 开始）来计算 baseline，类似于原始代码
        # 这样 baseline 表示：如果只使用前 (n_dims - i) 个维度，能记住多少信息
        baseline_loss = (
            sum(max(current_n_dims - i, 0) for i in range(current_n_respond)) / current_n_respond
            if current_n_respond > 0 else 1.0
        )
        
        # 如果 baseline_loss 为 0（当 n_dims 很小时），使用一个小的默认值避免除零
        if baseline_loss == 0:
            baseline_loss = 1.0
        
        excess_loss = loss / baseline_loss if baseline_loss > 0 else loss
        pointwise_loss_np = pointwise_loss.detach().cpu().flatten().numpy()
        
        # === Logging ===
        # 只在特定步数更新进度条描述，避免频繁刷新
        if step % log_interval == 0:  # 每log_interval步更新一次进度条描述
            if is_sudoku_task and cell_accuracy is not None:
                # 🆕 数独任务：只显示准确率指标（分类任务，不使用 MSE）
                desc = f"Loss:{loss:.3f} Acc:{cell_accuracy:.3f} SudokuAcc:{sudoku_accuracy:.3f}"
            elif is_pathfinding_task and cell_accuracy is not None:
                # 🆕 路径查找任务：显示准确率指标（分类任务，不使用 MSE）
                desc = f"Loss:{loss:.3f} Acc:{cell_accuracy:.3f} PathAcc:{sudoku_accuracy:.3f}"
            elif respond_mse is not None:
                # 标准回归任务：显示MSE
                desc = f"Loss:{loss:.3f} MSE:{respond_mse:.3f} dim:{current_n_dims} k:{current_n_respond}"
            else:
                # 🔧 如果 respond_mse 为 None（可能是某些任务类型），只显示 loss
                desc = f"Loss:{loss:.3f} dim:{current_n_dims} k:{current_n_respond}"
            
            # 🆕 如果使用multi-epoch，显示epoch信息
            if use_multi_epoch:
                desc = f"E{current_epoch}/{num_epochs} " + desc
            # 如果是MDM模型，显示mask ratio
            train_mask_ratio = get_model_attr(model, 'train_mask_ratio')
            if train_mask_ratio is not None:
                desc += f" m:{train_mask_ratio:.2f}"
            pbar.set_description(desc)
        
        # 🆕 只在主进程记录 WandB 日志（使用 should_log_wandb 替代 wandb_cfg.get("log", False)）
        # 🔧 确保所有 wandb.log 调用都在主进程块内，避免多进程同时调用导致问题
        if not use_distributed or accelerator.is_main_process:
            if should_log_wandb and step % wandb_log_interval == 0:
                log_dict = {
                    # Training metrics
                    "train/loss": loss,
                    "train/excess_loss": excess_loss,
                }
                
                # 🆕 数独任务：只记录准确率指标（不使用 MSE）
                if is_sudoku_task and cell_accuracy is not None:
                    log_dict.update({
                        "train/cell_accuracy": cell_accuracy,
                        "train/sudoku_accuracy": sudoku_accuracy,
                    })
                elif is_pathfinding_task and node_accuracy is not None:
                    # 🆕 Pathfinding 任务：只记录准确率指标（不使用 MSE）
                    log_dict.update({
                        "train/node_accuracy": node_accuracy,
                        "train/path_accuracy": path_accuracy,
                    })
                else:
                    # 非数独任务：记录 MSE
                    log_dict.update({
                        "train/mse_raw": respond_mse,
                        "train/mse_accumulated_avg": mse_logger.train_accumulated_avg,
                    })
                
                # 添加其他指标
                log_dict.update({
                    # Curriculum
                    "curriculum/n_dims": current_n_dims,
                    "curriculum/respond_points": current_n_respond,
                })
                
                # 🆕 添加epoch信息（如果使用multi-epoch）- 用于在WandB中标注epoch
                if use_multi_epoch:
                    log_dict["train/epoch"] = current_epoch + 1  # 1-based for clarity (1-20)
                
                # 添加mask ratio（如果是MDM模型）
                train_mask_ratio = get_model_attr(model, 'train_mask_ratio')
                eval_mask_ratio = get_model_attr(model, 'eval_mask_ratio')
                if train_mask_ratio is not None:
                    log_dict["mask_ratio/train"] = train_mask_ratio
                    log_dict["mask_ratio/eval"] = eval_mask_ratio
                
                # 🆕 添加累计算力到 WandB 日志
                if train_step_flops is not None and cumulative_training_flops > 0:
                    log_dict["stats/cumulative_flops"] = cumulative_training_flops
                    log_dict["stats/log10_cumulative_flops"] = np.log10(float(cumulative_training_flops)) if cumulative_training_flops > 0 else 0.0
                # 🆕 添加单次推理FLOPs到 WandB 日志
                if single_inference_flops is not None:
                    log_dict["stats/single_inference_flops"] = single_inference_flops
                    log_dict["stats/log10_single_inference_flops"] = np.log10(float(single_inference_flops)) if single_inference_flops > 0 else 0.0
                
                # 🆕 记录 WandB 日志（只在主进程）
                if wandb.run is not None:
                    wandb.log(log_dict, step=step)
        
        # === Checkpoint ===
        # 🆕 分布式训练：只在主进程保存 checkpoint
        should_save = step % training.get("save_every_steps", 20000) == 0 and step > 0
        if use_distributed:
            should_save = should_save and accelerator.is_main_process
        
        if should_save:
            checkpoint_data = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "train_step": step,
                "config": config,
            }
            # 🆕 保存epoch信息（如果使用multi-epoch）
            if use_multi_epoch:
                checkpoint_data["current_epoch"] = current_epoch
                checkpoint_data["step_in_epoch"] = step_in_epoch
            
            # 保存最新的checkpoint（用于resume）
            torch.save(checkpoint_data, state_path)
            
            # 🔧 关键：在保存checkpoint时同步日志到OSS，避免训练失败丢失日志
            # 这样即使训练中途失败，也能保留到最近一次checkpoint的日志
            mse_logger.flush_train(force=True)  # 确保日志已刷新到文件
            sync_logs_to_oss()
            
            # 🆕 Multi-epoch: 在指定epoch完成时保存checkpoint（例如：1ep, 5ep, 10ep, 20ep）
            if use_multi_epoch and step_in_epoch == steps_per_epoch - 1:
                # 当前epoch完成
                save_at_epochs = multi_epoch_cfg.get("save_at_epochs", [])
                if current_epoch + 1 in save_at_epochs:  # current_epoch是0-based，所以+1
                    epoch_num = current_epoch + 1
                    epoch_checkpoint_path = os.path.join(out_dir, f"state_epoch_{epoch_num}.pt")
                    torch.save(checkpoint_data, epoch_checkpoint_path)
                    print(f"\n[Checkpoint] Saved epoch {epoch_num} checkpoint: {epoch_checkpoint_path} (step {step})")
            
            if use_multi_epoch:
                print(f"\n[Checkpoint] Saved at epoch {current_epoch}, step {step}")
            else:
                print(f"\n[Checkpoint] Saved at step {step}")
        
        # === Validation Evaluation (固定验证集) ===
        # 🆕 允许在 step 0 也进行 validation，记录初始模型性能（baseline）
        if step % validation_eval_steps == 0:
            # 🆕 分布式训练：在评测前强制所有进程同步，防止死锁
            # 原因：如果只有主进程跑评测，其他进程会进入下一轮训练，导致步调不一致
            if use_distributed:
                accelerator.wait_for_everyone()

            # 🆕 只在主进程打印和运行评测（避免重复输出）
            if not use_distributed or accelerator.is_main_process:
                print(f"\n[Validation] Running fixed validation set at step {step}...")
            
            # 🔧 记录验证开始时间（用于统计耗时）
            validation_start_time = time.time()
            
            model.eval()

            # 🆕 检查是否需要在多个推理步数下评估
            eval_config = config.get("evaluation", {})
            inference_steps_list = eval_config.get("inference_steps_list", None)

            # ============================================================
            # Sudoku validation: ALWAYS use generate() for accuracy
            # ============================================================
            if is_sudoku_task:
                # Only main process runs generation & logging, but keep barriers to avoid drift.
                val_cell_acc = None
                val_sudoku_acc = None
                if not use_distributed or accelerator.is_main_process:
                    model_core = model.module if hasattr(model, "module") else model
                    model_family = get_model_attr(model, "family")

                    # 🏗️ 使用统一的 token 转数字函数
                    _nebula_tokens_to_digits = nebula_tokens_to_digits

                    total_cell = 0.0
                    total_sudoku = 0.0
                    cnt = 0

                    # iterate over fixed validation batches (seeded)
                    for batch in validation_batches:
                        data_sampler_args_v = {"seeds": batch["data_seeds"]}
                        task_sampler_args_v = {"seeds": batch["task_seeds"]}

                        total_points = n_prompt + current_n_respond
                        xs_v = data_sampler.sample_xs(total_points, validation_batch_size, current_n_dims, **data_sampler_args_v)
                        task_v = task_sampler(**task_sampler_args_v)
                        ys_v = task_v.evaluate(xs_v)

                        # apply same sequence_mode permutations as training
                        respond_position_mask_v = None
                        if sequence_mode == "non_sequential":
                            xs_v, ys_v, respond_position_mask_v = shuffle_prompt_respond_pairs(xs_v, ys_v, n_prompt, current_n_respond)
                        elif sequence_mode in ["fixed_permutation", "fixed_permutation_xy"]:
                            xs_v, ys_v, respond_position_mask_v = apply_fixed_permutation(
                                xs_v, ys_v, n_prompt, current_n_respond, fixed_permutation_x, fixed_permutation_y
                            )
                        elif sequence_mode == "random_permutation":
                            permute_y = training.get("permute_y", False)
                            xs_v, ys_v, respond_position_mask_v = apply_random_permutation(
                                xs_v, ys_v, n_prompt, current_n_respond, permute_y=permute_y
                            )
                        elif sequence_mode in ["pool_permutation", "pool_permutation_xy"]:
                            xs_v, ys_v, respond_position_mask_v = apply_pool_permutation(
                                xs_v, ys_v, n_prompt, current_n_respond, permutation_pool_x, permutation_pool_y
                            )

                        xs_v = xs_v.to(device)
                        ys_v = ys_v.to(device)
                        if respond_position_mask_v is not None:
                            respond_position_mask_v = respond_position_mask_v.to(device)

                        # 🏗️ 提取 ground truth（所有模型都需要）
                        # 对于数字格式模型（llada, ar 等）：从 ys_v 中提取
                        gt = extract_respond_values(ys_v, n_prompt, respond_position_mask_v, actual_n_respond=current_n_respond)[:, -1, :]  # [B, 81] - 数字格式

                        # build prefix and generate
                        if model_family == "sudoku_dream":
                            if hasattr(model_core, "_build_full_sequence"):
                                full_seq, prefix_len = model_core._build_full_sequence(xs_v, ys_v)
                            else:
                                full_seq, prefix_len = model_core._build_full_sequence_icl(xs_v, ys_v)
                            
                            # 🏗️ 对齐 v3 分支：从 token 序列中提取 ground truth（与 v3 分支一致）
                            true_full = full_seq  # [B, 163] - token 序列
                            true_sol = true_full[:, -81:]  # [B, 81] - token IDs
                            
                            prefix = full_seq[:, :prefix_len]
                            gen = model_core.generate(prefix, max_new_tokens=81)
                            # align to full sequence length if sampler returns longer canvas
                            if gen.dim() == 2 and gen.shape[1] >= full_seq.shape[1]:
                                gen = gen[:, -full_seq.shape[1] :]
                            pred_tok = gen[:, -81:]  # [B, 81] - token IDs
                            
                            # 🏗️ 对齐 v3 分支：直接比较 token IDs（与 v3 分支一致）
                            # 注意：这里使用 token IDs 比较，而不是转换为数字格式
                            pred = pred_tok  # [B, 81] - token IDs
                            gt = true_sol  # [B, 81] - token IDs (覆盖之前设置的 gt)
                        else:
                            pred = model_core.generate(
                                {"xs": xs_v, "ys": ys_v, "respond_position_mask": respond_position_mask_v},
                                max_new_tokens=81,
                            )
                            # allow returning logits
                            if pred.dim() == 4:
                                pred = torch.argmax(pred[:, -1, :, :], dim=-1)
                            elif pred.dim() == 3:
                                pred = torch.argmax(pred, dim=-1)

                        # 🏗️ 计算准确率
                        # 对于 sudoku_dream：pred 和 gt 都是 token IDs，直接比较
                        # 对于其他模型：pred 和 gt 都是数字格式，使用统一的准确率计算函数
                        if model_family == "sudoku_dream":
                            # 🏗️ 对齐 v3 分支：直接比较 token IDs
                            correct = (pred.long() == gt.long()).float()  # [B, 81]
                            cell_acc = correct.mean().item()
                            sudoku_acc = (correct.mean(dim=-1) == 1.0).float().mean().item()
                        else:
                            # 其他模型：使用统一的准确率计算函数（数字格式）
                            cell_acc, sudoku_acc = calculate_sudoku_accuracy(pred, gt)
                        total_cell += cell_acc
                        total_sudoku += sudoku_acc
                        cnt += 1

                    val_cell_acc = total_cell / max(cnt, 1)
                    val_sudoku_acc = total_sudoku / max(cnt, 1)

                    # 🆕 添加FLOPs信息到验证输出
                    flops_str = ""
                    if train_step_flops is not None and cumulative_training_flops > 0:
                        flops_str = f"  FLOPs: {cumulative_training_flops:.2e} (log10: {np.log10(float(cumulative_training_flops)):.2f})"
                    print(f"[Validation @ step {step}] Cell Accuracy: {val_cell_acc:.4f}  Sudoku Accuracy: {val_sudoku_acc:.4f}{flops_str}")

                    if should_log_wandb:
                        val_wandb_dict = {"validation/cell_accuracy": val_cell_acc, "validation/sudoku_accuracy": val_sudoku_acc}
                        # 🆕 添加累计算力到 WandB 日志
                        if train_step_flops is not None and cumulative_training_flops > 0:
                            val_wandb_dict["stats/cumulative_flops"] = cumulative_training_flops
                            val_wandb_dict["stats/log10_cumulative_flops"] = np.log10(float(cumulative_training_flops)) if cumulative_training_flops > 0 else 0.0
                        # 🆕 添加单次推理FLOPs到 WandB 日志
                        if single_inference_flops is not None:
                            val_wandb_dict["stats/single_inference_flops"] = single_inference_flops
                            val_wandb_dict["stats/log10_single_inference_flops"] = np.log10(float(single_inference_flops)) if single_inference_flops > 0 else 0.0
                        if wandb.run is not None:
                            wandb.log(val_wandb_dict, step=step)
                    mse_logger.record_validation(
                        step=step,
                        raw_mse=None,
                        dims=current_n_dims,
                        respond_points=current_n_respond,
                        batch_means=[],
                        model_type=model_type,
                        cell_accuracy=val_cell_acc,
                        sudoku_accuracy=val_sudoku_acc,
                        cumulative_flops=cumulative_training_flops if train_step_flops is not None else None,  # 🆕 累计算力
                        single_inference_flops=single_inference_flops,  # 🆕 单次推理FLOPs
                    )

                if use_distributed:
                    accelerator.wait_for_everyone()
                model.train()
                if use_distributed:
                    accelerator.wait_for_everyone()
                continue

            # ============================================================
            # Non-sudoku validation (legacy path)
            # ============================================================
            from eval_prompt_respond import eval_model_prompt_respond
            
            if inference_steps_list is not None:
                # 🆕 多步数评估模式：在一个 run 中记录多个 validation MSE
                # 🔧 修复死锁问题：所有进程都必须运行评估，因为 DDP 模型的前向传播需要所有进程参与
                # 只有主进程需要打印结果和保存数据，但所有进程都需要执行模型前向传播
                eval_results_dict = eval_model_prompt_respond(
                    model=model,
                    task_sampler=task_sampler,
                    data_sampler=data_sampler,
                    n_prompt=n_prompt,
                    n_respond=current_n_respond,
                    n_dims=current_n_dims,
                    batch_size=validation_batch_size,
                    num_eval_examples=validation_num_examples,
                    use_autoregressive_eval=is_ar_model and config.get("evaluation", {}).get("use_autoregressive_eval", False),
                    fixed_batches=validation_batches,
                    sequence_mode=sequence_mode,
                    inference_steps_list=inference_steps_list,  # 🆕 传入步数列表
                    permutation_seed=training.get("permutation_seed", 42) if sequence_mode in ["fixed_permutation", "fixed_permutation_xy", "pool_permutation", "pool_permutation_xy"] else 42,
                    permutation_seed_y=training.get("permutation_seed_y", None) if sequence_mode in ["fixed_permutation_xy", "pool_permutation_xy"] else None,
                    permute_y=training.get("permute_y", False) if sequence_mode == "random_permutation" else False,  # 🆕 Random Permutation: 传递 permute_y
                    permutation_pool_size=training.get("permutation_pool_size", 20) if sequence_mode in ["pool_permutation", "pool_permutation_xy"] else 20,  # 🆕 Pool Permutation: 传递排列池大小
                )
                
                # 🆕 分布式训练：评测后同步，确保所有进程步调一致
                if use_distributed:
                    accelerator.wait_for_everyone()
                
                # 🆕 记录每个步数的结果到 WandB（一个 run，多个 validation MSE）
                # 只在主进程记录（避免重复输出）
                if not use_distributed or accelerator.is_main_process:
                    wandb_log_dict = {}
                    print(f"[Validation @ step {step}]")
                    for steps in sorted(eval_results_dict.keys()):
                        results = eval_results_dict[steps]
                        
                        if is_sudoku_task:
                            # 🆕 数独任务：只记录准确率（不使用 MSE）
                            step_val_cell_acc = None
                            step_val_sudoku_acc = None
                            # 尝试从 results 中获取
                            if 'cell_accuracy' in results:
                                step_val_cell_acc = results['cell_accuracy']['mean']
                                step_val_sudoku_acc = results.get('sudoku_accuracy', {}).get('mean', 0.0)
                                # 🆕 添加FLOPs信息到验证输出（多步验证时，只在第一步显示FLOPs）
                                flops_str = ""
                                if steps == sorted(eval_results_dict.keys())[0] and train_step_flops is not None and cumulative_training_flops > 0:
                                    flops_str = f"  FLOPs: {cumulative_training_flops:.2e} (log10: {np.log10(float(cumulative_training_flops)):.2f})"
                                print(f"  Step {steps}: Cell Accuracy: {step_val_cell_acc:.4f}, Sudoku Accuracy: {step_val_sudoku_acc:.4f}{flops_str}")
                                wandb_log_dict[f"validation/cell_accuracy_step{steps}"] = step_val_cell_acc
                                wandb_log_dict[f"validation/sudoku_accuracy_step{steps}"] = step_val_sudoku_acc
                            
                            # 🆕 为每个步数分别记录到本地日志文件（只记录准确率）
                            mse_logger.record_validation(
                                step=step,
                                raw_mse=None,  # 数独任务不使用 MSE
                                dims=current_n_dims,
                                respond_points=current_n_respond,
                                batch_means=[],  # 数独任务不使用 batch_means
                                model_type=model_type,
                                inference_steps=steps,
                                cell_accuracy=step_val_cell_acc,
                                sudoku_accuracy=step_val_sudoku_acc,
                                cumulative_flops=cumulative_training_flops if train_step_flops is not None else None,  # 🆕 累计算力
                                single_inference_flops=single_inference_flops,  # 🆕 单次推理FLOPs
                            )
                        else:
                            # 非数独任务：记录 MSE
                            mean_mse = results['respond_mse']['mean']
                            std_mse = results['respond_mse']['std']
                            median_mse = results['respond_mse']['median']
                            # 🆕 添加FLOPs信息到验证输出（多步验证时，只在第一步显示FLOPs）
                            flops_str = ""
                            if steps == sorted(eval_results_dict.keys())[0] and train_step_flops is not None and cumulative_training_flops > 0:
                                flops_str = f"  FLOPs: {cumulative_training_flops:.2e} (log10: {np.log10(float(cumulative_training_flops)):.2f})"
                            print(f"  Step {steps}: Respond MSE: Mean={mean_mse:.6f}, Std={std_mse:.6f}, Median={median_mse:.6f}{flops_str}")
                            wandb_log_dict[f"validation/mse_step{steps}"] = mean_mse
                            
                            # 记录到本地日志文件
                            batch_means = results['respond_mse'].get("batch_means", [])
                            mse_logger.record_validation(
                                step=step,
                                raw_mse=mean_mse,
                                dims=current_n_dims,
                                respond_points=current_n_respond,
                                batch_means=batch_means,
                                model_type=model_type,
                                inference_steps=steps,
                                cumulative_flops=cumulative_training_flops if train_step_flops is not None else None,  # 🆕 累计算力
                                single_inference_flops=single_inference_flops,  # 🆕 单次推理FLOPs
                            )
                
                # 🆕 一次性记录所有步数的结果到 WandB（只在主进程）
                if not use_distributed or accelerator.is_main_process:
                    if should_log_wandb:
                        # 🆕 添加累计算力到多步验证日志
                        if train_step_flops is not None and cumulative_training_flops > 0:
                            wandb_log_dict["stats/cumulative_flops"] = cumulative_training_flops
                            wandb_log_dict["stats/log10_cumulative_flops"] = np.log10(float(cumulative_training_flops)) if cumulative_training_flops > 0 else 0.0
                        # 🆕 添加单次推理FLOPs到多步验证日志
                        if single_inference_flops is not None:
                            wandb_log_dict["stats/single_inference_flops"] = single_inference_flops
                            wandb_log_dict["stats/log10_single_inference_flops"] = np.log10(float(single_inference_flops)) if single_inference_flops > 0 else 0.0
                        if wandb.run is not None:
                            wandb.log(wandb_log_dict, step=step)
                
                # 使用第一个步数的结果作为主要指标（用于 accumulated_avg）
                # 🔧 注意：数独任务不使用 MSE，所以这里只对非数独任务更新 accumulated_avg
                if not is_sudoku_task:
                    default_steps = inference_steps_list[0] if inference_steps_list else None
                    eval_results = eval_results_dict.get(default_steps, list(eval_results_dict.values())[0])
                    if 'respond_mse' in eval_results:
                        mean_mse = eval_results['respond_mse']['mean']
                        
                        # 更新 accumulated_avg（基于第一个步数，只在主进程）
                        if not use_distributed or accelerator.is_main_process:
                            mse_logger.validation_accumulated_avg += (mean_mse - mse_logger.validation_accumulated_avg) / mse_logger.validation_count
                
            else:
                # 单步评估模式（原有逻辑）
                # 🔧 修复死锁问题：所有进程都必须运行评估，因为 DDP 模型的前向传播需要所有进程参与
                # 只有主进程需要打印结果和保存数据，但所有进程都需要执行模型前向传播
                eval_results = eval_model_prompt_respond(
                    model=model,
                    task_sampler=task_sampler,
                    data_sampler=data_sampler,
                    n_prompt=n_prompt,
                    n_respond=current_n_respond,
                    n_dims=current_n_dims,
                    batch_size=validation_batch_size,
                    num_eval_examples=validation_num_examples,
                    use_autoregressive_eval=is_ar_model and config.get("evaluation", {}).get("use_autoregressive_eval", False),
                    fixed_batches=validation_batches,
                    sequence_mode=sequence_mode,
                    permutation_seed=training.get("permutation_seed", 42) if sequence_mode in ["fixed_permutation", "fixed_permutation_xy", "pool_permutation", "pool_permutation_xy"] else 42,
                    permutation_seed_y=training.get("permutation_seed_y", None) if sequence_mode in ["fixed_permutation_xy", "pool_permutation_xy"] else None,
                    permute_y=training.get("permute_y", False) if sequence_mode == "random_permutation" else False,  # 🆕 Random Permutation: 传递 permute_y
                    permutation_pool_size=training.get("permutation_pool_size", 20) if sequence_mode in ["pool_permutation", "pool_permutation_xy"] else 20,  # 🆕 Pool Permutation: 传递排列池大小
                )
                
                # 🆕 分布式训练：评测后同步，确保所有进程步调一致
                if use_distributed:
                    accelerator.wait_for_everyone()
                
                # 🆕 只在主进程打印（避免重复输出）
                if not use_distributed or accelerator.is_main_process:
                    validation_elapsed = time.time() - validation_start_time
                    if validation_elapsed > 60:
                        print(f"⚠️  验证评估耗时 {validation_elapsed:.1f}秒，可能影响训练速度")
                    print(f"[Validation @ step {step}]")
                
                # 🆕 数独任务：只计算并显示准确率（不使用 MSE）
                val_log_dict = {}
                val_cell_acc = None
                val_sudoku_acc = None
                val_node_acc = None
                val_path_acc = None

                if is_sudoku_task:
                    # 🔧 修复死锁问题：直接使用 eval_results 中已计算的准确率，避免重复运行模型前向传播
                    # eval_model_prompt_respond 已经计算了数独准确率并返回在 results 中
                    try:
                        if 'cell_accuracy' in eval_results:
                            val_cell_acc = eval_results['cell_accuracy']['mean']
                            val_sudoku_acc = eval_results.get('sudoku_accuracy', {}).get('mean', 0.0)
                            # 🆕 添加FLOPs信息到验证输出
                            flops_str = ""
                            if train_step_flops is not None and cumulative_training_flops > 0:
                                flops_str = f"  FLOPs: {cumulative_training_flops:.2e} (log10: {np.log10(float(cumulative_training_flops)):.2f})"
                            print(f"  Cell Accuracy: {val_cell_acc:.4f}")
                            print(f"  Sudoku Accuracy: {val_sudoku_acc:.4f}{flops_str}")
                            val_log_dict.update({
                                "validation/cell_accuracy": val_cell_acc,
                                "validation/sudoku_accuracy": val_sudoku_acc,
                            })
                            # 🆕 添加累计算力到验证日志
                            if train_step_flops is not None and cumulative_training_flops > 0:
                                val_log_dict["stats/cumulative_flops"] = cumulative_training_flops
                                val_log_dict["stats/log10_cumulative_flops"] = np.log10(float(cumulative_training_flops)) if cumulative_training_flops > 0 else 0.0
                                # 🆕 添加单次推理FLOPs
                                if single_inference_flops is not None:
                                    val_log_dict["stats/single_inference_flops"] = single_inference_flops
                                    val_log_dict["stats/log10_single_inference_flops"] = np.log10(float(single_inference_flops)) if single_inference_flops > 0 else 0.0
                        else:
                            print(f"  ⚠️  警告: eval_results 中未找到 cell_accuracy，可能评估函数未正确计算数独指标")
                    except Exception as e:
                        import traceback
                        error_msg = str(e)
                        error_traceback = traceback.format_exc()
                        print(f"  ⚠️  警告: 从 eval_results 读取准确率时出错: {error_msg}")
                        print(f"  📋 详细错误信息:\n{error_traceback}")
                elif is_pathfinding_task:
                    # 🆕 Pathfinding 任务：显示和记录准确率
                    try:
                        if 'node_accuracy' in eval_results:
                            val_node_acc = eval_results['node_accuracy']['mean']
                            val_path_acc = eval_results.get('path_accuracy', {}).get('mean', 0.0)
                            # 🆕 添加FLOPs信息到验证输出
                            flops_str = ""
                            if train_step_flops is not None and cumulative_training_flops > 0:
                                flops_str = f"  FLOPs: {cumulative_training_flops:.2e} (log10: {np.log10(float(cumulative_training_flops)):.2f})"
                            print(f"  Node Accuracy: {val_node_acc:.4f}")
                            print(f"  Path Accuracy: {val_path_acc:.4f}{flops_str}")
                            val_log_dict.update({
                                "validation/node_accuracy": val_node_acc,
                                "validation/path_accuracy": val_path_acc,
                            })
                            # 🆕 添加累计算力到验证日志
                            if train_step_flops is not None and cumulative_training_flops > 0:
                                val_log_dict["stats/cumulative_flops"] = cumulative_training_flops
                                val_log_dict["stats/log10_cumulative_flops"] = np.log10(float(cumulative_training_flops)) if cumulative_training_flops > 0 else 0.0
                                # 🆕 添加单次推理FLOPs
                                if single_inference_flops is not None:
                                    val_log_dict["stats/single_inference_flops"] = single_inference_flops
                                    val_log_dict["stats/log10_single_inference_flops"] = np.log10(float(single_inference_flops)) if single_inference_flops > 0 else 0.0
                        else:
                            print(f"  ⚠️  警告: eval_results 中未找到 node_accuracy，可能评估函数未正确计算路径查找指标")
                    except Exception as e:
                        import traceback
                        error_msg = str(e)
                        error_traceback = traceback.format_exc()
                        print(f"  ⚠️  警告: 从 eval_results 读取准确率时出错: {error_msg}")
                        print(f"  📋 详细错误信息:\n{error_traceback}")
                else:
                    # 非数独任务：显示和记录 MSE
                    # 🔧 只在主进程访问 mse_logger，避免多进程竞争
                    if not use_distributed or accelerator.is_main_process:
                        mean_mse = eval_results['respond_mse']['mean']
                        std_mse = eval_results['respond_mse']['std']
                        median_mse = eval_results['respond_mse']['median']
                        # 🆕 添加FLOPs信息到验证输出
                        flops_str = ""
                        if train_step_flops is not None and cumulative_training_flops > 0:
                            flops_str = f"  FLOPs: {cumulative_training_flops:.2e} (log10: {np.log10(float(cumulative_training_flops)):.2f})"
                        print(f"  Respond MSE: Mean={mean_mse:.6f}, Std={std_mse:.6f}, Median={median_mse:.6f}{flops_str}")
                        val_log_dict.update({
                            "validation/mse_raw": mean_mse,
                            "validation/mse_accumulated_avg": mse_logger.validation_accumulated_avg,
                        })
                
                # 🆕 添加累计算力到验证日志
                if train_step_flops is not None and cumulative_training_flops > 0:
                    val_log_dict["stats/cumulative_flops"] = cumulative_training_flops
                    val_log_dict["stats/log10_cumulative_flops"] = np.log10(float(cumulative_training_flops)) if cumulative_training_flops > 0 else 0.0
                    # 🆕 添加单次推理FLOPs
                    if single_inference_flops is not None:
                        val_log_dict["stats/single_inference_flops"] = single_inference_flops
                        val_log_dict["stats/log10_single_inference_flops"] = np.log10(float(single_inference_flops)) if single_inference_flops > 0 else 0.0

                # 🆕 只在主进程记录 WandB 验证日志和本地日志
                if not use_distributed or accelerator.is_main_process:
                    if should_log_wandb and wandb.run is not None:
                        wandb.log(val_log_dict, step=step)

                    # 🆕 记录到本地日志
                    if is_sudoku_task:
                        # 数独任务：只记录准确率（不记录 MSE）
                        mse_logger.record_validation(
                            step=step,
                            raw_mse=None,  # 数独任务不使用 MSE
                            dims=current_n_dims,
                            respond_points=current_n_respond,
                            batch_means=[],  # 数独任务不使用 batch_means
                            model_type=model_type,
                            cell_accuracy=val_cell_acc,
                            sudoku_accuracy=val_sudoku_acc,
                            cumulative_flops=cumulative_training_flops if train_step_flops is not None else None,  # 🆕 累计算力
                            single_inference_flops=single_inference_flops,  # 🆕 单次推理FLOPs
                        )
                    elif is_pathfinding_task:
                        # 🆕 Pathfinding 任务：只记录准确率（不记录 MSE）
                        mse_logger.record_validation(
                            step=step,
                            raw_mse=None,  # Pathfinding 任务不使用 MSE
                            dims=current_n_dims,
                            respond_points=current_n_respond,
                            batch_means=[],  # Pathfinding 任务不使用 batch_means
                            model_type=model_type,
                            node_accuracy=val_node_acc,
                            path_accuracy=val_path_acc,
                            cumulative_flops=cumulative_training_flops if train_step_flops is not None else None,  # 🆕 累计算力
                            single_inference_flops=single_inference_flops,  # 🆕 单次推理FLOPs
                        )
                    else:
                        # 非数独任务：记录 MSE
                        batch_means = eval_results['respond_mse'].get("batch_means", [])
                        mean_mse = eval_results['respond_mse']['mean']
                        mse_logger.record_validation(
                            step=step,
                            raw_mse=mean_mse,
                            dims=current_n_dims,
                            respond_points=current_n_respond,
                            batch_means=batch_means,
                            model_type=model_type,
                            cumulative_flops=cumulative_training_flops if train_step_flops is not None else None,  # 🆕 累计算力
                            single_inference_flops=single_inference_flops,  # 🆕 单次推理FLOPs
                        )

            # 🔧 关键同步点：确保主进程完成所有耗时操作（wandb.log, mse_logger.record_validation等）
            # 之后，所有进程才进入下一轮训练，防止死锁
            if use_distributed:
                accelerator.wait_for_everyone()
            
            model.train()
            
            # 🔧 额外保护：在 model.train() 后再次同步，确保所有进程都准备好进行下一轮训练
            if use_distributed:
                accelerator.wait_for_everyone()
        
        # Update curriculum（数独/路径查找任务不需要）
        if not is_sudoku_task and not is_pathfinding_task:
            cur.update()
    
    # Final save
    mse_logger.flush_train(force=True)
    # 🆕 分布式训练：只在主进程保存最终 checkpoint
    if not use_distributed or accelerator.is_main_process:
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "train_step": total_steps,
            "config": config,
        }, state_path)
        # 🔧 保存最终checkpoint时也同步日志
        sync_logs_to_oss()
    print(f"\n[Training Complete] Final checkpoint saved")
    
    # === Final Evaluation ===
    # 🔧 分布式训练：在最终评估前强制所有进程同步，防止死锁
    if use_distributed:
        accelerator.wait_for_everyone()
    
    if not use_distributed or accelerator.is_main_process:
        print("\n[Final Evaluation] Running end-of-training evaluation...")
    
    model.eval()
    
    from eval_prompt_respond import eval_model_prompt_respond
    
    # 🆕 检查是否需要在多个推理步数下评估
    eval_config = config.get("evaluation", {})
    inference_steps_list = eval_config.get("inference_steps_list", None)
    
    # 最终评估使用 curriculum 的结束值（最大 respond 数量）
    # 🆕 数独和路径查找任务使用固定值，标准任务使用 curriculum
    if is_sudoku_task or is_pathfinding_task:
        final_n_respond = n_respond
    else:
        final_n_respond = training["curriculum"]["points"]["end"]
    
    if inference_steps_list is not None:
        # 🆕 多步数评估模式
        # 🔧 修复死锁问题：所有进程都必须运行评估，因为 DDP 模型的前向传播需要所有进程参与
        # 只有主进程需要打印结果和保存数据，但所有进程都需要执行模型前向传播
        final_results_dict = eval_model_prompt_respond(
            model=model,
            task_sampler=task_sampler,
            data_sampler=data_sampler,
            n_prompt=n_prompt,
            n_respond=final_n_respond,
            n_dims=n_dims,
            batch_size=64,
            num_eval_examples=1280,
            use_autoregressive_eval=is_ar_model and config.get("evaluation", {}).get("use_autoregressive_eval", False),
            sequence_mode=sequence_mode,
            inference_steps_list=inference_steps_list,
            permutation_seed=training.get("permutation_seed", 42) if sequence_mode in ["fixed_permutation", "fixed_permutation_xy"] else 42,
            permutation_seed_y=training.get("permutation_seed_y", None) if sequence_mode == "fixed_permutation_xy" else None,
            permute_y=training.get("permute_y", False) if sequence_mode == "random_permutation" else False,  # 🆕 Random Permutation: 传递 permute_y
        )
        
        # 🆕 分布式训练：最终评测后同步
        if use_distributed:
            accelerator.wait_for_everyone()
        
        # 记录每个步数的最终结果到 WandB（只在主进程）
        if not use_distributed or accelerator.is_main_process:
            wandb_final_log = {}
            print("\n" + "=" * 70)
            print("Final Evaluation Results (Multi-Step):")
            print("=" * 70)
            for steps in sorted(final_results_dict.keys()):
                results = final_results_dict[steps]
                mean_mse = results['respond_mse']['mean']
                std_mse = results['respond_mse']['std']
                print(f"  Step {steps}: Respond MSE: Mean={mean_mse:.6f}, Std={std_mse:.6f}")
                wandb_final_log[f"eval/final_respond_mse_step{steps}"] = mean_mse
            
            print("=" * 70 + "\n")
            
            # 🆕 只在主进程记录和结束 WandB
            if not use_distributed or accelerator.is_main_process:
                if should_log_wandb and wandb.run is not None:
                    wandb.log(wandb_final_log)
                    wandb.finish()
    else:
        # 单步评估模式（原有逻辑）
        # 🔧 修复死锁问题：所有进程都必须运行评估，因为 DDP 模型的前向传播需要所有进程参与
        # 只有主进程需要打印结果和保存数据，但所有进程都需要执行模型前向传播
        final_results = eval_model_prompt_respond(
            model=model,
            task_sampler=task_sampler,
            data_sampler=data_sampler,
            n_prompt=n_prompt,
            n_respond=final_n_respond,
            n_dims=n_dims,
            batch_size=64,
            num_eval_examples=1280,
            use_autoregressive_eval=is_ar_model and config.get("evaluation", {}).get("use_autoregressive_eval", False),
            sequence_mode=sequence_mode,
            permutation_seed=training.get("permutation_seed", 42) if sequence_mode in ["fixed_permutation", "fixed_permutation_xy"] else 42,
            permutation_seed_y=training.get("permutation_seed_y", None) if sequence_mode == "fixed_permutation_xy" else None,
            permute_y=training.get("permute_y", False) if sequence_mode == "random_permutation" else False,  # 🆕 Random Permutation: 传递 permute_y
        )
        
        # 🆕 分布式训练：最终评测后同步
        if use_distributed:
            accelerator.wait_for_everyone()
        
        # 打印最终评估结果（只在主进程）
        if not use_distributed or accelerator.is_main_process:
            print(f"\n{'='*60}")
            print("Final Evaluation Results:")
            print(f"{'='*60}")
            print(f"Respond MSE: Mean={final_results['respond_mse']['mean']:.6f}, "
                  f"Std={final_results['respond_mse']['std']:.6f}")
            print(f"{'='*60}\n")
        
        # 🆕 Log到wandb（只在主进程）
        if not use_distributed or accelerator.is_main_process:
            if should_log_wandb and wandb.run is not None:
                wandb.log({
                    # "eval/final_prompt_mse": final_results['prompt_mse']['mean'],
                    "eval/final_respond_mse": final_results['respond_mse']['mean'],
                    # "eval/final_overall_mse": final_results['overall_mse']['mean'],
                })
                wandb.finish()

    # 🆕 生成绘图（只在主进程，避免多进程同时写入 OSS 导致错误）
    # OSS 挂载不支持多进程同时写入，且绘图操作只需要执行一次
    if not use_distributed or accelerator.is_main_process:
        print("\n[Plot] Generating training curves...")
        mse_logger.generate_plots()
        print("[Plot] Training curves generated successfully")
        
        # 🆕 性能-算力对齐：将阈值 FLOPs 记录到 WandB
        if should_log_wandb and mse_logger.flops_to_threshold is not None and wandb.run is not None:
            wandb_threshold_dict = {
                "performance/flops_to_threshold": mse_logger.flops_to_threshold,
                "performance/log10_flops_to_threshold": np.log10(float(mse_logger.flops_to_threshold)) if mse_logger.flops_to_threshold > 0 else 0.0,
                "performance/threshold_metric": mse_logger.threshold_metric,
                "performance/threshold_value": mse_logger.threshold_value,
                "performance/threshold_reached_step": mse_logger.threshold_reached_step,
            }
            wandb.log(wandb_threshold_dict, step=total_steps)
            print(f"✅ 阈值 FLOPs 已记录到 WandB: {mse_logger.flops_to_threshold:.2e} (metric: {mse_logger.threshold_metric}, value: {mse_logger.threshold_value:.4f})")
        elif mse_logger.flops_to_threshold is None:
            print("ℹ️  训练期间未达到阈值，未记录阈值 FLOPs")
        
        # 🔧 最终同步：确保所有日志都已同步到OSS（训练过程中已在checkpoint保存时定期同步）
        # 这里做最后一次同步，确保没有遗漏
        sync_logs_to_oss()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Prompt-Respond ICL Models")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--random-seed", type=int, default=None, 
                       help="Random seed (overrides config if provided)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for checkpoints (overrides config if provided)")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 🆕 如果命令行提供了随机种子，覆盖config中的值
    if args.random_seed is not None:
        if "training" not in config:
            config["training"] = {}
        config["training"]["random_seed"] = args.random_seed
        # 也保存到顶层config，方便其他地方使用
        config["random_seed"] = args.random_seed
    
    # 🆕 如果命令行提供了输出目录，覆盖config中的值
    if args.output_dir is not None:
        config["out_dir"] = args.output_dir
    
    # Build model
    # 🆕 检测是否为数独任务，使用对应的模型构建函数
    training = config.get("training", {})
    is_sudoku_task = training.get("task_type") == "sudoku" or "sudoku" in training.get("data_path", "").lower()
    is_pathfinding_task = training.get("task_type") == "pathfinding" or "pathfinding" in training.get("data_path", "").lower()

    if is_sudoku_task:
        if build_sudoku_model is None:
            raise ImportError("数独任务需要 models_prompt_respond_sudoku 模块，但导入失败")
        model = build_sudoku_model(config["model"])
    elif is_pathfinding_task:
        if build_pathfinding_model is None:
            raise ImportError("路径查找任务需要 models_prompt_respond_pathfinding 模块，但导入失败")
        # 🆕 路径查找任务：计算 n_dims 并添加到 model config
        degree = config["model"].get("degree", 2)
        path_len = config["model"].get("path_len", 3)
        n_dims = (path_len - 1) * degree * 2 + 2  # edges + query
        config["model"]["n_dims"] = n_dims
        model = build_pathfinding_model(config["model"])
    else:
        model = build_model_prompt_respond(config["model"])
    
    # Train
    train(model, config)

