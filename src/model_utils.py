"""
Model Utilities for Prompt-Respond Models
=========================================

辅助函数：
- CART 权重计算
- 序列组合函数
- Prefix-LM 注意力 mask 创建
- 推理调度器创建
- Sudoku 协议工具函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import re


def cart_weight(
    masked_indices: torch.Tensor, t: torch.Tensor, p: float = 0.3
) -> torch.Tensor:
    """
    Optimized CART weight computation using matrix operations.
    
    CART (Context-Aware Reconstruction Training) weights consider the distance
    between masked positions and unmasked positions, using a geometric distribution.
    
    Args:
        masked_indices (torch.Tensor): (b, l) bool tensor indicating masked positions.
        t (torch.Tensor): (b,) time steps (0-1 sampled uniformly). Not directly used in CART.
        p (float): Parameter of geometric distribution (0 < p <= 1). Default 0.3.
    
    Returns:
        torch.Tensor: (b, l) float tensor of weights.
    """
    b, l = masked_indices.shape
    device = masked_indices.device
    
    idx = torch.arange(l, device=device)
    dist_matrix = (idx[None, :] - idx[:, None]).abs() - 1
    dist_matrix = torch.clamp(dist_matrix, min=0)  # (l, l)
    geo_matrix = (
        torch.log(torch.tensor(p, device=device))
        + (dist_matrix - 1).clamp(min=0) * torch.log(torch.tensor(1 - p, device=device))
    ).exp() * 0.5  # Ensure numerical stability
    geo_matrix.masked_fill_(dist_matrix == 0, 0.0)  # ignore distance = 0
    
    valid_mask = (~masked_indices).float()  # (b, l), 1 = unmasked
    weights = valid_mask @ geo_matrix.T  # (b, l)
    weights = weights * masked_indices.float()
    return weights


def combine_xs_ys(xs_b, ys_b):
    """
    Interleave (x_i, y_i) -> zs
    
    Args:
        xs_b: [B, T, D]
        ys_b: [B, T]
    
    Returns:
        zs: [B, 2T, D]
    """
    bsize, points, dim = xs_b.shape
    ys_b_wide = torch.cat(
        (ys_b.view(bsize, points, 1),
         torch.zeros(bsize, points, dim - 1, device=ys_b.device, dtype=xs_b.dtype)),
        dim=2,
    )
    zs = torch.stack((xs_b, ys_b_wide), dim=2).view(bsize, 2 * points, dim)
    return zs


def create_prefix_lm_mask(
    B, total_points, n_prompt, respond_position_mask, device
):
    """
    创建 Prefix-LM 风格的注意力 mask
    
    - Prompt pairs: 可以互相看到（双向注意力）
    - Respond pairs: 只能看到 Prompt + 之前的 Respond（因果注意力）
    
    Args:
        B: batch size
        total_points: n_prompt + n_respond
        n_prompt: prompt 长度
        respond_position_mask: [B, total_points] boolean tensor 或 None
        device: torch device
    
    Returns:
        attention_mask: [B, 2*total_points, 2*total_points]
    """
    seq_len = 2 * total_points
    
    if respond_position_mask is None:
        # Sequential 模式：前 n_prompt 个 pairs 是 prompt
        prompt_seq_len = 2 * n_prompt
        
        # 创建基础因果 mask（下三角矩阵）
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        
        # Prompt 部分互相可见（设置为全 1）
        mask[:prompt_seq_len, :prompt_seq_len] = 1.0
        
        # 扩展到 batch
        attention_mask = mask.unsqueeze(0).expand(B, -1, -1)
    
    else:
        # Non-Sequential 模式：根据 respond_position_mask 确定 prompt 位置
        # 初始化为因果 mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        attention_mask = mask.unsqueeze(0).expand(B, -1, -1).clone()
        
        # 为每个 batch 单独处理
        for b in range(B):
            prompt_mask = ~respond_position_mask[b]  # [total_points]
            prompt_positions = prompt_mask.nonzero(as_tuple=True)[0]  # prompt 位置索引
            
            # 收集所有 prompt 的序列索引（x 和 y 位置）
            prompt_seq_indices = []
            for pos in prompt_positions:
                prompt_seq_indices.append(pos.item() * 2)      # x 位置
                prompt_seq_indices.append(pos.item() * 2 + 1)  # y 位置
            
            # Prompt 部分互相可见（双向）
            for i in prompt_seq_indices:
                for j in prompt_seq_indices:
                    attention_mask[b, i, j] = 1.0
    
    return attention_mask


def create_inference_scheduler(inference_config):
    """
    根据配置创建推理调度器
    
    Args:
        inference_config: 推理配置字典，包含：
            - scheduler: "linear", "cosine", "logsnr" 等
            - log_snr_min: (可选) Log SNR 调度器的最小值
            - log_snr_max: (可选) Log SNR 调度器的最大值
    
    Returns:
        scheduler: BaseAlphaScheduler 实例，如果 use_multistep_inference=False 则返回 None
    """
    use_multistep = inference_config.get("use_multistep_inference", False)
    if not use_multistep:
        return None
    
    from dllm.core.schedulers import (
        LinearAlphaScheduler,
        CosineAlphaScheduler,
        LogSNRAlphaScheduler,
    )
    
    scheduler_type = inference_config.get("scheduler", "linear").lower()
    
    if scheduler_type == "linear":
        return LinearAlphaScheduler()
    elif scheduler_type == "cosine":
        return CosineAlphaScheduler()
    elif scheduler_type in ["logsnr", "log_snr", "log-snr"]:
        # 支持可选的 log SNR 参数
        log_snr_min = inference_config.get("log_snr_min", -10.0)
        log_snr_max = inference_config.get("log_snr_max", 10.0)
        return LogSNRAlphaScheduler(
            log_snr_min=log_snr_min,
            log_snr_max=log_snr_max,
        )
    else:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. "
            f"Supported: 'linear', 'cosine', 'logsnr'"
        )


def compute_actual_n_respond(total_points, n_prompt, respond_position_mask=None):
    """
    计算实际的 respond 数量（支持动态 curriculum）
    
    Args:
        total_points: 总点数
        n_prompt: prompt 点数
        respond_position_mask: [B, total_points] boolean tensor 或 None
    
    Returns:
        actual_n_respond: 实际的 respond 数量
    """
    if respond_position_mask is not None:
        # Non-sequential mode: respond positions are marked by respond_position_mask
        actual_n_respond = respond_position_mask[0].sum().item()  # 所有 batch 应该一致
    else:
        # Sequential mode: respond positions are at the end
        actual_n_respond = total_points - n_prompt
    return actual_n_respond


def sample_timestep_for_mdm(
    batch_size, 
    device, 
    train_mode, 
    mask_epsilon, 
    train_mask_ratio, 
    eval_mask_ratio, 
    eval_mask_mode
):
    """
    为 Masked Diffusion 模型采样时间步
    
    Args:
        batch_size: batch size
        device: torch device
        train_mode: True for training, False for inference
        mask_epsilon: minimum mask ratio
        train_mask_ratio: training mask ratio
        eval_mask_ratio: evaluation mask ratio
        eval_mask_mode: "fixed" (固定mask ratio) 或 "sample" (随机采样mask)
    
    Returns:
        t_scalar: [batch_size] timestep tensor
    """
    if train_mode:
        t_scalar = mask_epsilon + (train_mask_ratio - mask_epsilon) * torch.rand(batch_size, device=device)
    else:
        # 评估模式：根据 eval_mask_mode 决定 mask 策略
        # 支持新命名 ("fixed"/"sample") 和旧命名 ("full"/"bernoulli") 以保持向后兼容
        if eval_mask_mode in ["sample", "bernoulli"]:
            # "sample" 模式：随机采样 t，使用和训练一样的 Bernoulli mask
            t_scalar = mask_epsilon + (train_mask_ratio - mask_epsilon) * torch.rand(batch_size, device=device)
        elif eval_mask_mode in ["fixed", "full"]:
            # "fixed" 模式：使用固定的 eval_mask_ratio（通常为 1.0，即全 mask）
            t_scalar = torch.ones(batch_size, device=device) * eval_mask_ratio
        else:
            raise ValueError(
                f"Unknown eval_mask_mode: {eval_mask_mode}. "
                f"Supported: 'fixed' (or 'full'), 'sample' (or 'bernoulli')"
            )
    return t_scalar


def generate_masked_indices_for_mdm(
    batch_size,
    total_points,
    n_prompt,
    actual_n_respond,
    t_scalar,
    device,
    use_prompt_context,
    respond_position_mask=None
):
    """
    为 Masked Diffusion 模型生成 masked indices
    
    Args:
        batch_size: batch size
        total_points: 总点数
        n_prompt: prompt 点数
        actual_n_respond: 实际 respond 点数
        t_scalar: [batch_size] timestep tensor
        device: torch device
        use_prompt_context: 是否使用 prompt 作为上下文
        respond_position_mask: [B, total_points] boolean tensor 或 None
    
    Returns:
        masked_indices: [batch_size, total_points] boolean tensor
    """
    masked_indices = torch.zeros(batch_size, total_points, device=device, dtype=torch.bool)
    
    if respond_position_mask is not None:
        # Non-sequential mode: use respond_position_mask to identify respond positions
        respond_positions = respond_position_mask.to(device)  # [B, total_points]
        prompt_positions = ~respond_positions  # [B, total_points]
        
        # 如果 use_prompt_context=False，prompt 部分也全部 mask（无 context 模式）
        if not use_prompt_context:
            masked_indices = prompt_positions.clone()
        
        # 对 respond 部分随机 mask
        respond_mask_prob = torch.rand(batch_size, total_points, device=device) < t_scalar[:, None]
        masked_indices = masked_indices | (respond_positions & respond_mask_prob)
        
    else:
        # Sequential mode: respond positions are at the end
        respond_start = n_prompt
        
        # 如果 use_prompt_context=False，prompt 部分也全部 mask（无 context 模式）
        if not use_prompt_context:
            prompt_mask = torch.ones(batch_size, n_prompt, device=device, dtype=torch.bool)
            masked_indices[:, :n_prompt] = prompt_mask
        
        # 对 respond 部分随机 mask（使用实际的 respond 数量）
        respond_mask_prob = torch.rand(batch_size, actual_n_respond, device=device) < t_scalar[:, None]
        masked_indices[:, respond_start:respond_start+actual_n_respond] = respond_mask_prob
    
    return masked_indices


def extract_respond_predictions(pred_y_all, n_prompt, actual_n_respond, respond_position_mask=None):
    """
    从完整预测中提取 respond 部分的预测
    
    Args:
        pred_y_all: [B, total_points] 完整预测
        n_prompt: prompt 点数
        actual_n_respond: 实际 respond 点数
        respond_position_mask: [B, total_points] boolean tensor 或 None
    
    Returns:
        pred_y: [B, actual_n_respond] respond 部分的预测
    """
    batch_size = pred_y_all.shape[0]
    device = pred_y_all.device
    
    if respond_position_mask is not None:
        # Non-sequential mode: extract predictions at respond positions
        respond_positions = respond_position_mask.to(device)
        pred_y = torch.stack([pred_y_all[i, respond_positions[i]] for i in range(batch_size)])
    else:
        # Sequential mode: respond positions are at the end
        respond_start = n_prompt
        pred_y = pred_y_all[:, respond_start:respond_start+actual_n_respond]
    
    return pred_y


def extract_respond_masked_indices(masked_indices, n_prompt, actual_n_respond, respond_position_mask=None):
    """
    从完整 masked_indices 中提取 respond 部分的 mask
    
    Args:
        masked_indices: [B, total_points] boolean tensor
        n_prompt: prompt 点数
        actual_n_respond: 实际 respond 点数
        respond_position_mask: [B, total_points] boolean tensor 或 None
    
    Returns:
        respond_masked_indices: [B, actual_n_respond] boolean tensor
    """
    batch_size = masked_indices.shape[0]
    device = masked_indices.device
    
    if respond_position_mask is not None:
        # Non-sequential mode: extract masked indices at respond positions
        respond_positions = respond_position_mask.to(device)
        respond_masked_indices = torch.stack([masked_indices[i, respond_position_mask[i]] for i in range(batch_size)])
    else:
        # Sequential mode
        respond_start = n_prompt
        respond_masked_indices = masked_indices[:, respond_start:respond_start+actual_n_respond]
    
    return respond_masked_indices


def extract_respond_targets(target, n_prompt, actual_n_respond, respond_position_mask=None):
    """
    从完整 target 中提取 respond 部分的目标值
    
    Args:
        target: [B, total_points] 完整目标值
        n_prompt: prompt 点数
        actual_n_respond: 实际 respond 点数
        respond_position_mask: [B, total_points] boolean tensor 或 None
    
    Returns:
        respond_target: [B, actual_n_respond] respond 部分的目标值
        respond_masked_indices: [B, actual_n_respond] boolean tensor (同时从 masked_indices 提取)
    """
    batch_size = target.shape[0]
    device = target.device
    
    if respond_position_mask is not None:
        # Non-sequential mode: extract targets at respond positions
        respond_positions = respond_position_mask.to(device)
        respond_target = torch.stack([target[i, respond_positions[i]] for i in range(batch_size)])
    else:
        # Sequential mode
        respond_start = n_prompt
        respond_target = target[:, respond_start:respond_start+actual_n_respond]
    
    return respond_target


def compute_multistep_confidence(
    pred_y_respond, 
    ys_pred, 
    step, 
    confidence_alg, 
    device
):
    """
    计算多步推理的 confidence
    
    Args:
        pred_y_respond: [B, actual_n_respond] 当前预测
        ys_pred: [B, actual_n_respond] 上一步的预测
        step: 当前步数
        confidence_alg: confidence 算法 ("entropy" 或 "random")
        device: torch device
    
    Returns:
        confidence: [B, actual_n_respond] confidence scores
    """
    batch_size, actual_n_respond = pred_y_respond.shape
    
    if confidence_alg == "entropy":
        # 使用预测值的稳定性作为 confidence
        if step == 0:
            confidence = -torch.abs(pred_y_respond)
        else:
            # 后续步：使用预测值的变化（变化越小，confidence 越高）
            pred_change = torch.abs(pred_y_respond - ys_pred)
            confidence = -pred_change
    elif confidence_alg == "random":
        # 随机选择（用于对比实验）
        confidence = torch.rand(batch_size, actual_n_respond, device=device)
    else:
        raise ValueError(f"Unknown inference_confidence_alg: {confidence_alg}")
    
    return confidence


# ============================================================
# Sudoku Unified Protocol Utilities (163-Token + Nebula Tokenizer)
# ============================================================

def digits_to_nebula_tokens(digits: torch.Tensor) -> torch.Tensor:
    """
    Map digits 0..9 to core-nebula sudoku tokens (统一对齐 SudokuDream):
      1..9 -> 0..8
      0 -> 9 ('$')
    
    This is the standard token mapping used by all Sudoku models for fair comparison.
    
    Args:
        digits: Tensor of shape [..., 81] with values in [0, 9]
    
    Returns:
        tokens: Tensor of same shape with values in [0, 9]
            - 0..8: digits 1..9
            - 9: digit 0 (blank, represented as '$')
    """
    if digits.dtype not in (torch.long, torch.int32, torch.int64):
        digits = digits.long()
    
    out = digits.clone()
    mask_zero = (out == 0)
    mask_nonzero = ~mask_zero
    out[mask_nonzero] = out[mask_nonzero] - 1  # 1..9 -> 0..8
    out[mask_zero] = 9  # 0 -> '$' (id 9)
    return out.long()


def vocab_logits_to_digit_logits(vocab_logits: torch.Tensor) -> torch.Tensor:
    """
    Convert vocab logits [B,81,12] to digit logits [B,1,81,10].
    
    Nebula vocab mapping:
    - token 0..8 -> digit 1..9
    - token 9 ('$') -> invalid for solutions
    - token 10 ('=') -> invalid for solutions
    - token 11 (MASK) -> invalid for solutions
    - digit 0 is always invalid for solutions -> -inf
    
    Args:
        vocab_logits: [B, 81, 12] logits over Nebula vocab (0-11)
    
    Returns:
        digit_logits: [B, 1, 81, 10] logits over digits (0-9)
            - digit 0: always -inf (invalid)
            - digit 1..9: from vocab tokens 0..8
    """
    B = vocab_logits.shape[0]
    out = torch.full((B, 1, 81, 10), -1e9, device=vocab_logits.device, dtype=vocab_logits.dtype)
    # digit 1..9 from token 0..8
    out[:, 0, :, 1:10] = vocab_logits[:, :, 0:9]
    out[:, 0, :, 0] = -1e9  # digit 0 is invalid
    return out


def build_sudoku_163_sequence(
    xs: torch.Tensor,
    ys: torch.Tensor,
    n_prompt: int,
    n_respond: int,
    device: torch.device
) -> tuple[torch.Tensor, int]:
    """
    构建 163-token 统一协议序列：
      (Q1 '=' A1) (Q2 '=' A2) ... (Q_prompt '=' A_prompt) (Q_target '=') (A_target)
    
    - 每个完整示例长度：163 tokens (81 Q + 1 '=' + 81 A)
    - Target prefix 长度：n_prompt*163 + 82
    
    Args:
        xs: [B, n_points, 81] quiz digits (0-9)
        ys: [B, n_points, 81] solution digits (0-9)
        n_prompt: number of prompt examples
        n_respond: number of respond examples
        device: torch device
    
    Returns:
        full_sequence: [B, n_prompt*163 + 163] token sequence
        prefix_len: n_prompt*163 + 82 (length of prefix including Q_target '=')
    """
    B, n_points, _ = xs.shape
    assert n_points == n_prompt + n_respond, \
        f"Expected n_points == {n_prompt} + {n_respond}, got {n_points}"
    
    eq = torch.full((B, 1), 10, dtype=torch.long, device=device)  # '=' token id (Nebula vocab)
    
    parts = []
    # Prompt examples: full (Q '=' A)
    for i in range(n_points - 1):
        q_i = digits_to_nebula_tokens(xs[:, i, :].long())  # [B, 81]
        a_i = digits_to_nebula_tokens(ys[:, i, :].long())  # [B, 81]
        parts.extend([q_i, eq, a_i])
    
    # Target example: (Q '=') + A
    q_t = digits_to_nebula_tokens(xs[:, -1, :].long())  # [B, 81]
    a_t = digits_to_nebula_tokens(ys[:, -1, :].long())  # [B, 81]
    
    prefix_tokens = torch.cat(parts + [q_t, eq], dim=1)  # [B, n_prompt*163 + 82]
    prefix_len = int(prefix_tokens.shape[1])
    full = torch.cat([prefix_tokens, a_t], dim=1)  # [B, n_prompt*163 + 163]
    return full, prefix_len


# ============================================================
# Sudoku Model Configuration Extraction
# ============================================================

def extract_sudoku_config(extra_dict: dict) -> tuple[dict, dict]:
    """
    从 extra 中提取数独特定配置，并返回清理后的字典
    
    这个函数用于解决参数冲突问题：当配置文件中包含的参数也在 super().__init__() 
    中硬编码时，会导致 "got multiple values for keyword argument" 错误。
    
    Args:
        extra_dict: 包含所有额外参数的字典（通常来自 **extra）
    
    Returns:
        (sudoku_config, clean_extra):
            - sudoku_config: 提取的数独特定配置
            - clean_extra: 清理后的 extra 字典（已移除可能冲突的参数）
    """
    # 创建副本以避免修改原始字典
    extra = extra_dict.copy()
    
    # 提取数独特定配置
    sudoku_config = {
        'use_multistep_inference': extra.pop('use_multistep_inference', False),
        'inference_steps': extra.pop('inference_steps', 20),
        'inference_k_per_step': extra.pop('inference_k_per_step', 4),
        'inference_confidence_alg': extra.pop('inference_confidence_alg', 'entropy'),
        'num_timesteps': extra.pop('num_timesteps', 20),
        'loss_mode': extra.pop('loss_mode', 'composite'),
        'alpha': extra.pop('alpha', 0.25),
        'gamma': extra.pop('gamma', 1.0),
        'mask_prob_override': extra.pop('mask_prob_override', None),
        'mask_prob_min': extra.pop('mask_prob_min', 0.0),
        'mask_prob_max': extra.pop('mask_prob_max', 1.0),
        't_sampling_power': extra.pop('t_sampling_power', 1.0),
        'use_coordinate_embedding': extra.pop('use_coordinate_embedding', True),  # 默认启用坐标编码
    }
    
    # 彻底清理掉可能导致父类冲突的 Key
    # 这些参数通常在 super().__init__() 中硬编码，如果也在 extra 中会出现冲突
    conflict_keys = [
        'mask_epsilon',
        'loss_weight_type',
        'train_mask_ratio',
        'eval_mask_ratio',
        'eval_mask_mode',
        'use_prompt_context',
        'training_strategy',
    ]
    for k in conflict_keys:
        extra.pop(k, None)
    
    return sudoku_config, extra


# ============================================================
# Sudoku Loss Computation Utilities
# ============================================================

def compute_sudoku_ce_loss(
    vocab_logits: torch.Tensor,
    target_digits: torch.Tensor,
    mask: torch.Tensor,
    use_ignore_index: bool = False,
) -> torch.Tensor:
    """
    计算 Sudoku 任务的 Cross-Entropy Loss（使用 Nebula vocab）
    
    Args:
        vocab_logits: [B, 81, 12] Nebula vocab logits
        target_digits: [B, 81] target digits (0-9)
        mask: [B, 81] boolean mask indicating valid positions
        use_ignore_index: If True, use ignore_index=-100 (Dream-style), else use manual mask (default)
    
    Returns:
        loss: scalar tensor
    """
    import torch.nn.functional as F
    import torch.nn as nn
    
    target_tokens = digits_to_nebula_tokens(target_digits.long())  # [B, 81]
    
    if use_ignore_index:
        # Dream-style: 使用 ignore_index=-100 自动忽略不需要的位置
        # 将未 mask 的位置设置为 -100
        target_with_ignore = target_tokens.clone()
        target_with_ignore[~mask] = -100  # 未 mask 的位置设为 -100
        
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(
            vocab_logits.reshape(-1, 12),  # [B*81, 12]
            target_with_ignore.reshape(-1)  # [B*81]
        )
        return loss
    else:
        # 原方式: 手动 mask 加权平均
        ce_loss = F.cross_entropy(
            vocab_logits.reshape(-1, 12),  # [B*81, 12]
            target_tokens.reshape(-1),  # [B*81]
            reduction='none'
        )
        ce_loss = ce_loss.reshape(target_digits.shape)  # [B, 81]
        return (ce_loss * mask.float()).sum() / (mask.sum() + 1e-8)


def compute_sudoku_composite_loss(
    vocab_logits: torch.Tensor,
    target_digits: torch.Tensor,
    mask: torch.Tensor,
    t: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 1.0,
    num_timesteps: int = 20,
    use_continuous_timestep: bool = False,
) -> torch.Tensor:
    """
    计算 Sudoku 任务的 Composite Loss (CE + Focal + Time Weight)
    
    Args:
        vocab_logits: [B, 81, 12] Nebula vocab logits
        target_digits: [B, 81] target digits (0-9)
        mask: [B, 81] boolean mask
        t: [B] timestep tensor
        alpha: Focal Loss alpha parameter
        gamma: Focal Loss gamma parameter
        num_timesteps: number of timesteps (for discrete timestep mode)
        use_continuous_timestep: whether using continuous timestep
    
    Returns:
        loss: scalar tensor
    """
    import torch.nn.functional as F
    
    # 1. 基础 CE Loss
    target_tokens = digits_to_nebula_tokens(target_digits.long())  # [B, 81]
    ce_loss = F.cross_entropy(
        vocab_logits.reshape(-1, 12),  # [B*81, 12]
        target_tokens.reshape(-1),  # [B*81]
        reduction='none'
    )
    ce_loss = ce_loss.reshape(target_digits.shape)  # [B, 81]
    
    # 2. Focal Weight
    sol_digit_logits = vocab_logits_to_digit_logits(vocab_logits)  # [B, 1, 81, 10]
    with torch.no_grad():
        probs = F.softmax(sol_digit_logits.squeeze(1), dim=-1)  # [B, 81, 10]
        target_p = probs.gather(2, target_digits.unsqueeze(-1)).squeeze(-1)  # [B, 81]
        focal_w = alpha * (1 - target_p) ** gamma  # [B, 81]
    
    # 3. Time Weight
    if use_continuous_timestep:
        time_w = (1.0 - t).reshape(-1, 1)  # [B, 1]
    else:
        time_w = (num_timesteps - t.float()).reshape(-1, 1)  # [B, 1]
    
    # Note: time_w is [B, 1], which will broadcast to [B, 81] when multiplied with [B, 81] tensors
    # This matches the original implementation in SudokuBOPAR/SudokuRBOAR (Line 1781, 2205)
    # Using direct multiplication allows PyTorch's automatic broadcasting: [B, 1] * [B, 81] -> [B, 81]
    return (ce_loss * focal_w * time_w * mask.float()).sum() / (mask.sum() + 1e-8)


def inject_sudoku_coordinates(
    embeds: torch.Tensor,
    total_points: int,
    coord_emb: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """
    将坐标编码注入到 Sudoku 序列的 Quiz 和 Answer 位置
    
    Args:
        embeds: [B, seq_len, n_embd] embeddings
        total_points: 总点数（每个点包含 163 tokens）
        coord_emb: SudokuCoordinateEmbedding 实例（如果为 None，则跳过注入）
        device: torch device
    
    Returns:
        embeds: [B, seq_len, n_embd] embeddings with coordinates injected (if coord_emb is not None)
    """
    if coord_emb is None:
        return embeds
    coords = coord_emb(device)  # [81, n_embd]
    UNIT_LEN = 163
    for i in range(total_points):
        offset = i * UNIT_LEN
        embeds[:, offset:offset + 81, :] += coords.unsqueeze(0)  # Quiz区
        embeds[:, offset + 82:offset + UNIT_LEN, :] += coords.unsqueeze(0)  # Answer区
    return embeds

