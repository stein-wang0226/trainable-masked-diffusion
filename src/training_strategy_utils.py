"""
训练策略辅助函数
用于支持diffusion-vs-ar风格的训练

使用方法：在models_prompt_respond.py中导入并使用
"""

import torch


def sample_timestep_with_strategy(
    b, device, train_mode, training_strategy,
    num_timesteps, mask_epsilon, train_mask_ratio,
    eval_mask_ratio, eval_mask_mode
):
    """
    根据训练策略采样timestep或mask ratio

    Args:
        b: batch size
        device: device
        train_mode: True for training, False for eval
        training_strategy: 训练策略配置字典
        num_timesteps: 时间步总数（用于timestep模式）
        mask_epsilon, train_mask_ratio: ratio模式参数
        eval_mask_ratio, eval_mask_mode: 评估模式参数

    Returns:
        t_scalar: [B] mask比例
        t: [B] 时间步（整数，用于loss权重计算）
    """
    if train_mode:
        if training_strategy['mask_mode'] == 'timestep':
            # 🆕 Timestep模式（diffusion-vs-ar风格）
            t = torch.randint(0, num_timesteps, (b,), device=device)
            mask_schedule = training_strategy.get('mask_schedule', 'linear')
            if mask_schedule == 'linear':
                t_scalar = (t + 1).float() / num_timesteps  # (t+1)/T
            else:
                raise ValueError(f"Unknown mask_schedule: {mask_schedule}")
        else:
            # 原有：Ratio模式（向后兼容）
            t_scalar = mask_epsilon + (
                train_mask_ratio - mask_epsilon
            ) * torch.rand(b, device=device)
            # 为了统一loss权重计算，计算伪时间步
            t = (t_scalar * num_timesteps - 1).clamp(min=0).long()
    else:
        # 评估模式（使用原有逻辑）
        if eval_mask_mode == "fixed":
            t_scalar = torch.ones(b, device=device) * eval_mask_ratio
        elif eval_mask_mode == "sample":
            t_scalar = mask_epsilon + (
                train_mask_ratio - mask_epsilon
            ) * torch.rand(b, device=device)
        else:
            raise ValueError(f"Unknown eval_mask_mode: {eval_mask_mode}")
        # 计算伪时间步
        t = (t_scalar * num_timesteps - 1).clamp(min=0).long()

    return t_scalar, t


def compute_loss_weight_with_strategy(
    per_sample_loss, t, t_scalar, training_strategy, num_timesteps
):
    """
    根据训练策略计算loss权重

    Args:
        per_sample_loss: [B, n_respond] 每个样本的loss（MSE或CE）
        t: [B] 时间步
        t_scalar: [B] mask比例
        training_strategy: 训练策略配置字典
        num_timesteps: 时间步总数

    Returns:
        weighted_loss: scalar，加权后的loss
    """
    loss_reweight_config = training_strategy.get('loss_reweighting', {})

    # Token-level reweighting（可选，用于focal loss）
    if loss_reweight_config.get('enable_token_reweight', False):
        alpha = loss_reweight_config.get('alpha', 0.25)
        gamma = loss_reweight_config.get('gamma', 1.0)
        # Focal loss: α * (1 - exp(-loss))^γ * loss
        per_sample_loss = alpha * (1 - torch.exp(-per_sample_loss)) ** gamma * per_sample_loss

    # Time-level reweighting
    time_weight_mode = loss_reweight_config.get('time_weight_mode', 'original')
    if time_weight_mode == 'linear':
        # T - t （diffusion-vs-ar默认）
        weight = (num_timesteps - t).float().unsqueeze(1)
    elif time_weight_mode == 'original' or time_weight_mode == '1/t':
        # 1/t （原实现）
        weight = (1.0 / (t_scalar + 1e-8)).unsqueeze(1)
        weight = torch.clamp(weight, min=0.1, max=10.0)
    elif time_weight_mode == 'none':
        # 均匀权重
        weight = torch.ones_like(t_scalar).unsqueeze(1)
    else:
        raise ValueError(f"Unknown time_weight_mode: {time_weight_mode}")

    # 应用权重并计算最终loss
    weighted_loss = (per_sample_loss * weight).mean()

    return weighted_loss
