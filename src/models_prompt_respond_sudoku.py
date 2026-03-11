"""
数独专用模型（基于 diffusion-vs-ar 风格）
==========================================

核心特性：
1. 坐标编码：为每个格子注入 Row/Col/Block 信息
2. 离散时间步训练：采样 t ∈ [0, 19]，使用 Focal Loss 重加权
3. 分类任务：输出 10 个类别的 logits（0-9）
4. 支持 BPD-AR 推理（熵引导的自适应推理）
"""

import torch
from torch import nn
import torch.nn.functional as F
import math
from models_prompt_respond import (
    LLaDAPromptRespond,
    TransformerModelPromptRespond,
    LLaDABlockDiffusion,
    BOPARPromptRespond,
    RBOARPromptRespond,
    BADARPromptRespond,
)
from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig
from dllm.pipelines.llada.models.modeling_llada import LLaDAModel as _LLaDABase

# 🆕 Dream (MDM) imports (used by SudokuDream; kept optional for environments without dllm dream)
try:
    from dllm.pipelines.dream.models.configuration_dream import DreamConfig
    from dllm.pipelines.dream.models.modeling_dream import DreamModel as _DreamBase
    from dllm.core.schedulers import LinearAlphaScheduler
    from dllm.pipelines.dream.sampler import DreamSampler
except Exception:
    DreamConfig = None
    _DreamBase = None
    LinearAlphaScheduler = None
    DreamSampler = None

class SudokuCoordinateEmbedding(nn.Module):
    """
    数独 2D 坐标编码器
    
    为每个格子（81个）注入 Row、Col、Block 的 Embedding。
    这能让模型理解 2D 约束（同行/同列/同宫格不能有重复数字）。
    """
    
    def __init__(self, n_embd):
        super().__init__()
        # 将 embedding 维度分成三部分：Row, Col, Block
        row_dim = n_embd // 3
        col_dim = n_embd // 3
        block_dim = n_embd - row_dim - col_dim
        
        self.row_emb = nn.Embedding(9, row_dim)   # 9 行
        self.col_emb = nn.Embedding(9, col_dim)   # 9 列
        self.blk_emb = nn.Embedding(9, block_dim) # 9 个 3×3 宫格
        
        # 预计算 81 个格子的坐标（在 forward 中计算）
        self._precompute_coordinates()
    
    def _precompute_coordinates(self):
        """预计算所有 81 个格子的坐标"""
        # 格子索引 0-80 对应 9×9 网格
        grid_idx = torch.arange(81)
        rows = grid_idx // 9  # 行索引 0-8
        cols = grid_idx % 9   # 列索引 0-8
        # Block 索引：左上角为 (0,0)，右下角为 (2,2)
        block_rows = rows // 3
        block_cols = cols // 3
        blocks = block_rows * 3 + block_cols  # Block 索引 0-8
        
        self.register_buffer('rows', rows)
        self.register_buffer('cols', cols)
        self.register_buffer('blocks', blocks)
    
    def forward(self, device):
        """
        返回 81 个格子的坐标 embedding
        
        Returns:
            coords: [81, n_embd] 坐标 embedding
        """
        rows = self.rows.to(device)
        cols = self.cols.to(device)
        blocks = self.blocks.to(device)
        
        row_emb = self.row_emb(rows)  # [81, row_dim]
        col_emb = self.col_emb(cols)  # [81, col_dim]
        blk_emb = self.blk_emb(blocks)  # [81, block_dim]
        
        # 拼接为完整的 embedding
        coords = torch.cat([row_emb, col_emb, blk_emb], dim=-1)  # [81, n_embd]
        return coords


# ============================================================
# SudokuMixin: 封装数独特有的逻辑（词表、序列构建、坐标注入、Loss计算）
# ============================================================

class SudokuMixin:
    """
    数独模型 Mixin 类，封装所有共有逻辑
    
    提供：
    - 统一的词表设置（Nebula tokenizer, vocab_size=12）
    - 163-token 序列构建
    - 坐标编码注入
    - Loss 计算（CE 和 Composite）
    - Logits Shift 逻辑
    - Token 映射方法
    """
    
    def setup_sudoku_protocol(self, n_embd: int, use_coordinate_embedding: bool = True):
        """
        设置数独协议相关的组件（词表、embedding 层、坐标编码等）
        
        Args:
            n_embd: embedding 维度
            use_coordinate_embedding: 是否使用坐标编码（默认 True，向后兼容）
        """
        self.vocab_size = 12  # Nebula tokenizer: 0-8 (digits 1-9), 9 ('$'), 10 ('='), 11 (MASK)
        self.mask_token_id = 11
        self.eq_token_id = 10
        
        # 输入输出层
        self._read_in = nn.Embedding(self.vocab_size, n_embd)
        self._read_out = nn.Linear(n_embd, self.vocab_size)
        
        # 坐标编码（可选）
        self.use_coordinate_embedding = use_coordinate_embedding
        if self.use_coordinate_embedding:
            self.coord_emb = SudokuCoordinateEmbedding(n_embd)
        else:
            self.coord_emb = None
        
        # MASK token embedding
        self.mask_embedding = nn.Parameter(torch.randn(1, n_embd))
    
    def _digits_to_nebula_tokens(self, digits: torch.Tensor) -> torch.Tensor:
        """Wrapper for shared digits_to_nebula_tokens utility"""
        from model_utils import digits_to_nebula_tokens
        return digits_to_nebula_tokens(digits)
    
    def _vocab_logits_to_digit_logits(self, vocab_logits: torch.Tensor) -> torch.Tensor:
        """Wrapper for shared vocab_logits_to_digit_logits utility"""
        from model_utils import vocab_logits_to_digit_logits
        return vocab_logits_to_digit_logits(vocab_logits)

    def _prepare_163_sequence(self, xs: torch.Tensor, ys: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, int]:
        """
        构建 163-token 统一协议序列
        
        Args:
            xs: [B, n_points, 81] quiz digits
            ys: [B, n_points, 81] solution digits
            device: torch device
        
        Returns:
            (full_sequence, prefix_len)
        """
        from model_utils import build_sudoku_163_sequence
        return build_sudoku_163_sequence(xs, ys, self.n_prompt, self.n_respond, device)
    
    def _inject_sudoku_coords(self, embeds: torch.Tensor, total_points: int, device: torch.device) -> torch.Tensor:
        """
        将坐标编码注入到 Sudoku 序列的 Quiz 和 Answer 位置（如果启用）
        
        Args:
            embeds: [B, seq_len, n_embd] embeddings
            total_points: 总点数（每个点包含 163 tokens）
            device: torch device
        
        Returns:
            embeds: [B, seq_len, n_embd] embeddings with coordinates injected (if enabled)
        """
        if not getattr(self, 'use_coordinate_embedding', True) or self.coord_emb is None:
            return embeds
        from model_utils import inject_sudoku_coordinates
        return inject_sudoku_coordinates(embeds, total_points, self.coord_emb, device)
    
    def _apply_logits_shift_logic(self, vocab_logits: torch.Tensor) -> torch.Tensor:
        """
        应用 Logits Shift（对齐 Transformer off-by-one 特性）
        
        Args:
            vocab_logits: [B, seq_len, vocab_size] vocab logits
        
        Returns:
            shifted_logits: [B, seq_len, vocab_size] shifted logits
        """
        return torch.cat([vocab_logits[:, :1, :], vocab_logits[:, :-1, :]], dim=1)
    
    def _compute_sudoku_loss(
        self,
        vocab_logits: torch.Tensor,
        target_digits: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor = None,
        mode: str = "ce",
    ) -> torch.Tensor:
        """
        统一计算 Sudoku Loss（CE 或 Composite）
        
        Args:
            vocab_logits: [B, 81, 12] Nebula vocab logits
            target_digits: [B, 81] target digits (0-9)
            mask: [B, 81] boolean mask
            t: [B] timestep tensor (for composite loss)
            mode: "ce" (manual mask), "ce_ignore_index" (Dream-style ignore_index), or "composite"
        
        Returns:
            loss: scalar tensor
        """
        from model_utils import compute_sudoku_ce_loss, compute_sudoku_composite_loss
        
        if mode == "ce":
            # 原方式: 手动 mask 加权平均
            return compute_sudoku_ce_loss(vocab_logits, target_digits, mask, use_ignore_index=False)
        elif mode == "ce_ignore_index":
            # Dream-style: 使用 ignore_index=-100
            return compute_sudoku_ce_loss(vocab_logits, target_digits, mask, use_ignore_index=True)
        else:
            # Composite loss
            alpha = getattr(self, 'alpha', 0.25)
            gamma = getattr(self, 'gamma', 1.0)
            num_timesteps = getattr(self, 'num_timesteps', 20)
            use_continuous_timestep = getattr(self, 'use_continuous_timestep', False)
            
            if t is None:
                raise ValueError("t is required for composite loss")
            
            return compute_sudoku_composite_loss(
                vocab_logits, target_digits, mask, t,
                alpha, gamma, num_timesteps, use_continuous_timestep
            )


class SudokuLLaDA(LLaDAPromptRespond, SudokuMixin):
    """
    数独专用 LLaDA 模型（基于 LLaDAPromptRespond + SudokuMixin）
    
    与标准 LLaDAPromptRespond 的区别：
    1. 输入：使用 Embedding 处理离散值（0-10，10 为 MASK）
    2. 坐标编码：为每个格子注入 Row/Col/Block 信息
    3. 输出：10 个类别的 logits（分类任务）
    4. 损失：Cross-Entropy Loss + Focal Loss 重加权 + 时间重加权
    5. 训练策略：离散时间步 t ∈ [0, 19]
    """
    
    def __init__(
        self,
        n_dims=81,  # 数独任务：quiz 部分为 81 维
        n_positions=1000,
        n_embd=256,
        n_layer=12,
        n_head=8,
        n_prompt=5,
        n_respond=1,
        *,
        mlp_ratio=4.0,
        block_group_size=1,
        # 数独任务特定参数
        num_timesteps=20,  # 离散时间步数
        alpha=0.25,  # Focal Loss 参数
        gamma=1.0,  # Focal Loss 参数
        # 🆕 Mask ratio / timestep sampling controls (for convergence ablations)
        # - If mask_prob_override is set, it will be used directly (e.g., 1.0 = full-mask training).
        # - Otherwise, sample t and compute base mask_prob=(t+1)/T, then map it to [mask_prob_min, mask_prob_max].
        mask_prob_override=None,  # Optional[float]
        mask_prob_min=0.0,        # float in [0,1]
        mask_prob_max=1.0,        # float in [0,1]
        t_sampling_power=1.0,     # float > 0; <1 biases toward larger t (higher mask), >1 biases toward smaller t
        # 🆕 Inference controls (align inference with denoising training)
        use_multistep_inference=False,
        inference_steps=20,
        inference_k_per_step=4,
        inference_scheduler=None,  # 🆕 动态 scheduler（可选）
        inference_confidence_alg="entropy",  # "entropy" (supported)
        # 🆕 Loss ablation switch
        # - "composite": CE + focal + time_weight (default, better for LLaDA)
        # - "ce": plain masked cross-entropy only (for Dream alignment experiments)
        loss_mode="composite",
        # 🏗️ Logits Shift 控制（对齐 Dream 和 core-nebula）
        apply_logits_shift=False,  # 是否应用 Logits Shift（默认 False，保持向后兼容）
        verify_shift_alignment=False,  # 是否验证 Shift 对齐（开发阶段启用）
        # 🏗️ 连续时间步采样（对齐 Dream）
        use_continuous_timestep=False,  # 是否使用连续时间步 + LinearAlphaScheduler（默认 False，保持向后兼容）
        time_epsilon=1e-3,  # 连续时间步的最小值（对齐 Dream）
        **extra,
    ):
        # 🏗️ 使用统一的参数过滤函数
        from model_utils import extract_sudoku_config
        sudoku_cfg, clean_extra = extract_sudoku_config(extra)
        
        # 使用 sudoku_cfg 中的值（如果存在），否则使用显式参数
        final_use_multistep = sudoku_cfg.get('use_multistep_inference', use_multistep_inference)
        final_inference_steps = sudoku_cfg.get('inference_steps', inference_steps)
        final_inference_k_per_step = sudoku_cfg.get('inference_k_per_step', inference_k_per_step)
        final_inference_scheduler = sudoku_cfg.get('inference_scheduler', inference_scheduler)
        final_inference_confidence_alg = sudoku_cfg.get('inference_confidence_alg', inference_confidence_alg)
        final_loss_mode = sudoku_cfg.get('loss_mode', loss_mode)
        final_alpha = sudoku_cfg.get('alpha', alpha)
        final_gamma = sudoku_cfg.get('gamma', gamma)
        final_num_timesteps = sudoku_cfg.get('num_timesteps', num_timesteps)
        
        # 使用 clean_extra 中的 training_strategy（如果存在），否则使用硬编码的值
        training_strategy_extra = clean_extra.pop('training_strategy', None)
        if training_strategy_extra is not None:
            final_training_strategy = training_strategy_extra
        else:
            final_training_strategy = {
                'mask_mode': 'timestep',
                'num_timesteps': final_num_timesteps,
                'loss_reweighting': {
                    'enable_token_reweight': True,  # 启用 token 重加权（Focal Loss）
                    'time_weight_mode': 'linear',  # 时间重加权：Weight = (T - t)
                }
            }
        
        # 从 clean_extra 中提取父类参数
        mask_epsilon = clean_extra.pop('mask_epsilon', 1e-3)
        loss_weight_type = clean_extra.pop('loss_weight_type', "1/t")
        train_mask_ratio = clean_extra.pop('train_mask_ratio', 0.5)
        eval_mask_ratio = clean_extra.pop('eval_mask_ratio', 1.0)
        eval_mask_mode = clean_extra.pop('eval_mask_mode', "fixed")
        use_prompt_context_extra = clean_extra.pop('use_prompt_context', True)
        
        # 先调用父类初始化
        super().__init__(
            n_dims=81,  # 数独任务：quiz 部分为 81 维
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_prompt=n_prompt,
            n_respond=n_respond,
            mlp_ratio=mlp_ratio,
            block_group_size=block_group_size,
            mask_epsilon=mask_epsilon,
            loss_weight_type=loss_weight_type,
            train_mask_ratio=train_mask_ratio,  # 会被离散时间步覆盖
            eval_mask_ratio=eval_mask_ratio,
            eval_mask_mode=eval_mask_mode,
            use_prompt_context=use_prompt_context_extra,
            use_multistep_inference=final_use_multistep,
            inference_steps=final_inference_steps,
            inference_confidence_alg=final_inference_confidence_alg,
            training_strategy=final_training_strategy,
            **clean_extra,
        )
        
        self.name = "sudoku_llada"
        # vocab_size 将在 setup_sudoku_protocol() 中统一设置为 12
        
        # 数独任务特定参数
        self.num_timesteps = final_num_timesteps
        self.alpha = final_alpha
        self.gamma = final_gamma

        # Mask ratio controls
        self.mask_prob_override = sudoku_cfg.get('mask_prob_override', mask_prob_override)
        self.mask_prob_min = float(sudoku_cfg.get('mask_prob_min', mask_prob_min))
        self.mask_prob_max = float(sudoku_cfg.get('mask_prob_max', mask_prob_max))
        self.t_sampling_power = float(sudoku_cfg.get('t_sampling_power', t_sampling_power))

        # Inference controls
        self.use_multistep_inference = bool(final_use_multistep)
        self.inference_steps = int(final_inference_steps)
        self.inference_k_per_step = int(final_inference_k_per_step)
        self.inference_confidence_alg = str(final_inference_confidence_alg)

        # 🆕 动态 scheduler 支持（向后兼容）
        if final_inference_scheduler is not None:
            # 模式1: 使用动态 scheduler（类似 ICL）
            from dllm.core.schedulers import LinearAlphaScheduler
            if isinstance(final_inference_scheduler, str):
                if final_inference_scheduler == "linear":
                    self.inference_scheduler = LinearAlphaScheduler()
                else:
                    raise ValueError(f"Unknown scheduler: {final_inference_scheduler}")
            else:
                self.inference_scheduler = final_inference_scheduler
            self.use_dynamic_scheduler = True
            self.inference_k_per_step = None  # 不使用固定 k
            print(f"  [Inference] Using dynamic scheduler: {type(self.inference_scheduler).__name__}")
        else:
            # 模式2: 使用固定 k_per_step（当前默认行为）
            self.inference_scheduler = None
            self.use_dynamic_scheduler = False
            self.inference_k_per_step = int(final_inference_k_per_step)
            print(f"  [Inference] Using fixed k_per_step: {self.inference_k_per_step}")

        # Loss mode
        self.loss_mode = str(final_loss_mode)
        assert self.loss_mode in {"composite", "ce", "ce_ignore_index"}, \
            f"Unknown loss_mode={self.loss_mode}. Use 'composite', 'ce', or 'ce_ignore_index'."
        
        # 🏗️ Logits Shift 控制参数（对齐 Dream）
        self.apply_logits_shift = bool(apply_logits_shift)
        self.verify_shift_alignment = bool(verify_shift_alignment)
        if self.apply_logits_shift and not self.verify_shift_alignment:
            import warnings
            warnings.warn(
                "apply_logits_shift=True: Logits shift is enabled for LLaDA. "
                "This aligns with Dream model and may improve performance."
            )
        
        # 🏗️ 连续时间步采样（对齐 Dream）
        self.use_continuous_timestep = bool(use_continuous_timestep)
        self.time_epsilon = float(time_epsilon)
        if self.use_continuous_timestep:
            if LinearAlphaScheduler is None:
                raise ImportError("use_continuous_timestep=True requires LinearAlphaScheduler from dllm.core.schedulers")
            self.scheduler = LinearAlphaScheduler()
            import warnings
            warnings.warn(
                "use_continuous_timestep=True: Using continuous timestep sampling with LinearAlphaScheduler. "
                "This aligns with Dream model. Discrete timestep parameters (num_timesteps, etc.) will be ignored."
            )
        
        # 🏗️ 使用 Mixin 设置数独协议（这会创建 self._read_in, self._read_out, self.coord_emb, self.mask_embedding）
        use_coord_emb = sudoku_cfg.get('use_coordinate_embedding', True)  # 从 sudoku_cfg 读取
        self.setup_sudoku_protocol(n_embd, use_coordinate_embedding=use_coord_emb)

        print(f"[SudokuLLaDA] Initialized:")
        print(f"  vocab_size: {self.vocab_size} (0-8: digits 1-9, 9: '$', 10: '=', 11: MASK)")
        print(f"  num_timesteps: {self.num_timesteps}")
        print(f"  use_coordinate_embedding: {use_coord_emb}")
        print(f"  Focal Loss: alpha={self.alpha}, gamma={self.gamma}")
        print(f"  Coordinate Embedding: {'enabled' if self.use_coordinate_embedding else 'disabled'}")
        if self.mask_prob_override is not None:
            print(f"  Mask prob override: {self.mask_prob_override}")
        else:
            print(f"  Mask prob range: [{self.mask_prob_min}, {self.mask_prob_max}], t_sampling_power={self.t_sampling_power}")
        if self.use_multistep_inference:
            print(f"  Multi-step inference: enabled (steps={self.inference_steps}, k_per_step={self.inference_k_per_step}, alg={self.inference_confidence_alg})")
        print(f"  Loss mode: {self.loss_mode}")
    
    def _digits_to_nebula_tokens(self, digits: torch.Tensor) -> torch.Tensor:
        """Wrapper for shared digits_to_nebula_tokens utility"""
        from model_utils import digits_to_nebula_tokens
        return digits_to_nebula_tokens(digits)
    
    def _vocab_logits_to_digit_logits(self, vocab_logits: torch.Tensor) -> torch.Tensor:
        """Wrapper for shared vocab_logits_to_digit_logits utility"""
        from model_utils import vocab_logits_to_digit_logits
        return vocab_logits_to_digit_logits(vocab_logits)

    def _build_full_sequence(self, xs: torch.Tensor, ys: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Wrapper for shared build_sudoku_163_sequence utility"""
        from model_utils import build_sudoku_163_sequence
        return build_sudoku_163_sequence(
            xs, ys, self.n_prompt, self.n_respond, xs.device
        )


    @torch.no_grad()
    def _multistep_inference_discrete(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """
        Multi-step denoising inference for Sudoku (163-token protocol).
        Iteratively fills MASK positions by selecting low-entropy cells (BPD-style).

        Returns:
            final_logits: [B, n_respond, 81, 10]
        """
        B, n_points, _ = xs.shape
        device = xs.device
        n_prompt = self.n_prompt
        n_respond = self.n_respond
        UNIT_LEN = 163

        # 🆕 使用 163-token 统一协议构建序列（使用 Mixin 方法）
        full_sequence, prefix_len = self._prepare_163_sequence(xs, ys, device)  # [B, n_prompt*163 + 163]
        input_ids = full_sequence.clone()

        # 🆕 初始化 Answer 部分为 MASK
        target_board_start = n_prompt * UNIT_LEN
        answer_start = target_board_start + 82
        answer_end = target_board_start + UNIT_LEN
        input_ids[:, answer_start:answer_end] = 11  # 🆕 MASK token (ID 11)

        filled = torch.zeros(B, 81, dtype=torch.bool, device=device)
        final_logits = None

        steps = max(1, int(self.inference_steps))
        k_per_step = max(1, int(self.inference_k_per_step))
        for _ in range(steps):
            if filled.all():
                break

            # Forward pass
            embeds = self._read_in(input_ids)  # [B, seq_len, n_embd]

            # 🆕 坐标编码：注入到 Quiz 和 Answer 位置（使用 Mixin 方法）
            embeds = self._inject_sudoku_coords(embeds, n_points, device)

            # Time conditioning
            t_scalar = torch.ones(B, device=device)
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, input_ids.shape[1], 1))
            embeds = embeds + time_emb

            # Backbone
            dummy_input_ids = torch.zeros(B, input_ids.shape[1], dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)  # [B, seq_len, 12]

            # 🆕 提取 Answer 部分的 logits 并转换为 digit logits
            sol_vocab_logits = vocab_logits[:, answer_start:answer_end, :]  # [B, 81, 12]
            logits = self._vocab_logits_to_digit_logits(sol_vocab_logits)  # [B, 1, 81, 10]
            logits = logits.squeeze(1)  # [B, 81, 10]
            final_logits = logits.unsqueeze(1)  # [B, 1, 81, 10] for return format

            # Entropy over 10 classes
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [B, 81]
            entropy = entropy.masked_fill(filled, float("inf"))

            # Fill top-k lowest entropy per sample
            for b in range(B):
                unfilled = (~filled[b]).nonzero(as_tuple=True)[0]
                if unfilled.numel() == 0:
                    continue
                k = min(k_per_step, unfilled.numel())
                unfilled_entropy = entropy[b, unfilled]
                _, topk = torch.topk(unfilled_entropy, k=k, largest=False)
                cells = unfilled[topk]
                pred_digits = torch.argmax(logits[b, cells, :], dim=-1)
                # 🆕 转换为 Nebula tokens 并更新序列
                pred_tokens = self._digits_to_nebula_tokens(pred_digits.unsqueeze(0)).squeeze(0)
                input_ids[b, answer_start + cells] = pred_tokens
                filled[b, cells] = True

        if final_logits is None:
            # Final forward pass
            embeds = self._read_in(input_ids)
            # 🆕 坐标编码：注入到 Quiz 和 Answer 位置（使用 Mixin 方法）
            embeds = self._inject_sudoku_coords(embeds, n_points, device)
            t_scalar = torch.ones(B, device=device)
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, input_ids.shape[1], 1))
            embeds = embeds + time_emb
            dummy_input_ids = torch.zeros(B, input_ids.shape[1], dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)
            sol_vocab_logits = vocab_logits[:, answer_start:answer_end, :]
            logits = self._vocab_logits_to_digit_logits(sol_vocab_logits)
            final_logits = logits  # [B, 1, 81, 10]

        return final_logits

    def _multistep_inference(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """多步去噪推理（路由方法）"""
        if self.use_dynamic_scheduler:
            return self._multistep_inference_dynamic(xs, ys)
        else:
            return self._multistep_inference_discrete(xs, ys)

    def _multistep_inference_dynamic(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """
        Multi-step denoising inference with dynamic scheduler for Sudoku (163-token protocol).
        Uses get_num_transfer_tokens to dynamically determine unmask count per step.

        Returns:
            final_logits: [B, n_respond, 81, 10]
        """
        from dllm.utils.generation_utils import get_num_transfer_tokens

        B, n_points, _ = xs.shape
        device = xs.device
        n_prompt = self.n_prompt
        n_respond = self.n_respond
        UNIT_LEN = 163

        # 🆕 使用 163-token 统一协议构建序列（使用 Mixin 方法）
        full_sequence, prefix_len = self._prepare_163_sequence(xs, ys, device)  # [B, n_prompt*163 + 163]
        input_ids = full_sequence.clone()

        # 🆕 初始化 Answer 部分为 MASK
        target_board_start = n_prompt * UNIT_LEN
        answer_start = target_board_start + 82
        answer_end = target_board_start + UNIT_LEN
        input_ids[:, answer_start:answer_end] = 11  # 🆕 MASK token (ID 11)

        # 初始化 mask 状态
        masked_indices = torch.ones(B, 81, device=device, dtype=torch.bool)
        initial_mask = masked_indices.clone()

        # 使用 scheduler 计算每步 unmask 的数量
        num_transfer_tokens = get_num_transfer_tokens(
            mask_index=masked_indices,
            steps=self.inference_steps,
            scheduler=self.inference_scheduler,
            stochastic=False,
        )
        effective_steps = num_transfer_tokens.size(1)

        final_logits = None

        # 迭代去噪
        for step in range(effective_steps):
            if masked_indices.sum() == 0:
                break

            # Forward pass
            embeds = self._read_in(input_ids)  # [B, seq_len, n_embd]

            # 🆕 坐标编码：注入到 Quiz 和 Answer 位置（使用 Mixin 方法）
            embeds = self._inject_sudoku_coords(embeds, n_points, device)

            # Time conditioning
            t_scalar = torch.ones(B, device=device)
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, input_ids.shape[1], 1))
            embeds = embeds + time_emb

            # Backbone
            dummy_input_ids = torch.zeros(B, input_ids.shape[1], dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)  # [B, seq_len, 12]

            # 🆕 提取 Answer 部分的 logits 并转换为 digit logits
            sol_vocab_logits = vocab_logits[:, answer_start:answer_end, :]  # [B, 81, 12]
            logits = self._vocab_logits_to_digit_logits(sol_vocab_logits)  # [B, 1, 81, 10]
            logits = logits.squeeze(1)  # [B, 81, 10]
            final_logits = logits.unsqueeze(1)  # [B, 1, 81, 10] for return format

            # 计算置信度
            if self.inference_confidence_alg == "entropy":
                probs = F.softmax(logits, dim=-1)
                confidence = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [B, 81]
            else:
                probs = F.softmax(logits, dim=-1)
                confidence = -probs.max(dim=-1)[0]  # [B, 81]

            confidence = confidence.masked_fill(~masked_indices, float("inf"))

            # 根据 scheduler 决定本步 unmask 多少个位置
            for b in range(B):
                k = num_transfer_tokens[b, step].item()
                if k == 0:
                    continue

                unfilled = masked_indices[b].nonzero(as_tuple=True)[0]
                if unfilled.numel() == 0:
                    continue

                k = min(k, unfilled.numel())
                unfilled_confidence = confidence[b, unfilled]
                _, topk = torch.topk(unfilled_confidence, k=k, largest=False)
                cells = unfilled[topk]
                pred_digits = torch.argmax(logits[b, cells, :], dim=-1)
                # 🆕 转换为 Nebula tokens 并更新序列
                pred_tokens = self._digits_to_nebula_tokens(pred_digits.unsqueeze(0)).squeeze(0)
                input_ids[b, answer_start + cells] = pred_tokens
                masked_indices[b, cells] = False

        if final_logits is None:
            # Final forward pass
            embeds = self._read_in(input_ids)
            # 🆕 坐标编码：注入到 Quiz 和 Answer 位置（使用 Mixin 方法）
            embeds = self._inject_sudoku_coords(embeds, n_points, device)
            t_scalar = torch.ones(B, device=device)
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, input_ids.shape[1], 1))
            embeds = embeds + time_emb
            dummy_input_ids = torch.zeros(B, input_ids.shape[1], dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)
            sol_vocab_logits = vocab_logits[:, answer_start:answer_end, :]
            logits = self._vocab_logits_to_digit_logits(sol_vocab_logits)
            final_logits = logits  # [B, 1, 81, 10]

        return final_logits

    def _compute_mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample mask probability.

        Args:
            t: [B] integer timestep in [0, T-1]
        Returns:
            mask_prob: [B] float in [0,1]
        """
        if self.mask_prob_override is not None:
            mask_prob = torch.full_like(t, float(self.mask_prob_override), dtype=torch.float32)
        else:
            # base in (0,1]
            base = (t.float() + 1.0) / float(self.num_timesteps)
            # map base -> [min, max]
            mask_prob = self.mask_prob_min + (self.mask_prob_max - self.mask_prob_min) * base
        return torch.clamp(mask_prob, 0.0, 1.0)

    def _sample_timesteps(self, B: int, device) -> torch.Tensor:
        """
        Sample discrete timesteps with optional bias.

        t_sampling_power:
          - 1.0: uniform over [0, T-1]
          - <1.0: bias toward larger t (higher mask)
          - >1.0: bias toward smaller t (lower mask)
        """
        if self.num_timesteps <= 1:
            return torch.zeros(B, device=device, dtype=torch.long)
        if self.t_sampling_power == 1.0:
            return torch.randint(0, self.num_timesteps, (B,), device=device)
        # Continuous u -> discrete t (biased)
        u = torch.rand(B, device=device)
        u = torch.clamp(u, 1e-8, 1.0)
        u = u ** self.t_sampling_power
        t = torch.floor(u * self.num_timesteps).long()
        return torch.clamp(t, 0, self.num_timesteps - 1)
    
    def forward(self, xs, ys, train_mode=True, respond_position_mask=None):
        """
        Forward pass for Sudoku task - 使用 163-token 统一协议
        
        Args:
            xs: [B, n_prompt + n_respond, 81] - quiz 部分
            ys: [B, n_prompt + n_respond, 81] - solution 部分
            train_mode: True for training, False for inference
            respond_position_mask: [B, total_points] boolean tensor (保留兼容性，未使用)
        
        Returns:
            If train_mode: (loss, pred_logits, t, mask)
                - loss: 标量
                - pred_logits: [B, 1, 81, 10]
                - t: [B] 时间步
                - mask: [B, 1, 81] boolean mask
            Else: (pred_logits, mask)
                - pred_logits: [B, 1, 81, 10]
                - mask: [B, 1, 81] boolean mask
        """
        B, n_points, d = xs.shape
        device = xs.device
        
        assert d == 81, f"数独任务要求 d=81（quiz 部分），实际为 {d}"
        assert ys.shape == (B, n_points, 81), f"ys 形状应为 [B, n_points, 81]，实际为 {ys.shape}"
        assert n_points == self.n_prompt + self.n_respond, \
            f"SudokuLLaDA expects n_points == {self.n_prompt} + {self.n_respond}, got {n_points}"
        
        # 🆕 使用 163-token 统一协议构建序列（使用 Mixin 方法）
        full_sequence, prefix_len = self._prepare_163_sequence(xs, ys, device)  # [B, n_prompt*163 + 163]
        B, seq_len = full_sequence.shape
        UNIT_LEN = 163
        
        n_prompt = self.n_prompt
        n_respond = self.n_respond
        
        # 🆕 计算 target Answer 部分的位置（163-token 协议）
        target_board_start = n_prompt * UNIT_LEN
        answer_start = target_board_start + 82  # Answer 部分起始位置（跳过 Q + '='）
        answer_end = target_board_start + UNIT_LEN  # Answer 部分结束位置
        
        if train_mode:
            # === 时间步采样策略（支持连续和离散）===
            if self.use_continuous_timestep:
                # 🏗️ 连续时间步采样（对齐 Dream）
                t = self.time_epsilon + (1.0 - self.time_epsilon) * torch.rand(B, device=device)
                # 使用 LinearAlphaScheduler 计算 mask 概率（对齐 Dream）
                alpha_t = self.scheduler(t)  # [B] alpha(t) ∈ [0, 1]
                mask_prob = 1.0 - alpha_t  # [B] p_mask ∈ [0, 1]
            else:
                # 离散时间步训练策略
                t = self._sample_timesteps(B, device)
                mask_prob = self._compute_mask_prob(t)
            
            # 🆕 在 163-token 协议下，只对 Answer 部分应用 mask
            target_part_mask = torch.zeros_like(full_sequence, dtype=torch.bool)
            target_part_mask[:, answer_start:answer_end] = True
            
            rand = torch.rand_like(full_sequence, dtype=torch.float32)
            masked_indices = (rand < mask_prob.reshape(B, 1)) & target_part_mask
            input_ids = full_sequence.clone()
            input_ids[masked_indices] = 11  # 🆕 MASK token (ID 11)
            
            # Embedding
            embeds = self._read_in(input_ids)  # [B, seq_len, n_embd]
            
            # 🆕 坐标编码：注入到 Quiz 和 Answer 位置（使用 Mixin 方法）
            embeds = self._inject_sudoku_coords(embeds, n_points, device)
            
            # Time conditioning
            if self.use_continuous_timestep:
                t_scalar = t  # [B] 已经在 [time_epsilon, 1.0]
            else:
                t_scalar = t.float() / self.num_timesteps  # [B] 归一化到 [0, 1]
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, seq_len, 1))
            embeds = embeds + time_emb
            
            # Backbone forward
            dummy_input_ids = torch.zeros(B, seq_len, dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]  # [B, seq_len, n_embd]
            
            # Readout
            vocab_logits = self._read_out(h)  # [B, seq_len, 12]
            
            # 🏗️ Logits Shift（对齐 Dream 和 core-nebula，使用 Mixin 方法）
            if self.apply_logits_shift:
                vocab_logits = self._apply_logits_shift_logic(vocab_logits)
                if self.verify_shift_alignment:
                    assert vocab_logits.shape == vocab_logits.shape  # 验证形状
            
            # 🆕 提取 Answer 部分的 logits 并转换为 digit logits
            sol_vocab_logits = vocab_logits[:, answer_start:answer_end, :]  # [B, 81, 12]
            sol_logits = self._vocab_logits_to_digit_logits(sol_vocab_logits)  # [B, 1, 81, 10]
            
            # 🆕 计算损失：使用 Mixin 的统一 Loss 计算方法
            respond_solutions = ys[:, -1:, :]  # [B, 1, 81]
            target_digits = respond_solutions.squeeze(1)  # [B, 81]
            mask_in_sol = masked_indices[:, answer_start:answer_end]  # [B, 81]
            
            # 使用 Mixin 的统一 Loss 计算方法
            final_loss = self._compute_sudoku_loss(
                sol_vocab_logits, target_digits, mask_in_sol, t, mode=self.loss_mode
            )
            
            # 统一接口格式
            mask_in_sol = mask_in_sol.unsqueeze(1)  # [B, 1, 81]
            
            return final_loss, sol_logits, t, mask_in_sol
        
        else:
            # === 推理模式（基于 163-token 协议）===
            # 🆕 多步推理：与 mask-去噪训练对齐（默认关闭，保持向后兼容）
            if getattr(self, "use_multistep_inference", False):
                final_logits = self._multistep_inference(xs=xs, ys=ys)
                final_mask = torch.zeros(B, 1, 81, device=device, dtype=torch.bool)
                return final_logits, final_mask

            # 🆕 使用 163-token 统一协议构建序列
            input_ids = full_sequence.clone()
            # 初始化 Answer 部分为 MASK
            input_ids[:, answer_start:answer_end] = 11  # 🆕 MASK token (ID 11)

            # Embedding
            embeds = self._read_in(input_ids)  # [B, seq_len, n_embd]
            
            # 🆕 坐标编码：注入到 Quiz 和 Answer 位置（使用 Mixin 方法）
            embeds = self._inject_sudoku_coords(embeds, n_points, device)
            
            # Time conditioning（使用最大时间步）
            t_scalar = torch.ones(B, device=device)
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, seq_len, 1))
            embeds = embeds + time_emb
            
            # Backbone
            dummy_input_ids = torch.zeros(B, seq_len, dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]  # [B, seq_len, n_embd]
            
            # Readout
            vocab_logits = self._read_out(h)  # [B, seq_len, 12]
            
            # 🆕 提取 Answer 部分的 logits 并转换为 digit logits
            answer_vocab_logits = vocab_logits[:, answer_start:answer_end, :]  # [B, 81, 12]
            sol_logits = self._vocab_logits_to_digit_logits(answer_vocab_logits)  # [B, 1, 81, 10]
            
            # 返回所有位置都被 mask 的标记
            mask = torch.ones(B, 1, 81, device=device, dtype=torch.bool)
            
            return sol_logits, mask

    @torch.no_grad()
    def generate(self, prefix, max_new_tokens=81, **kwargs):
        """
        Unified Sudoku validation API: generate final 81 digits for the target answer.

        prefix can be:
        - {"xs": xs, "ys": ys, "respond_position_mask": mask(optional)}
        - (xs, ys) or (xs, ys, respond_position_mask)
        """
        if isinstance(prefix, dict):
            xs = prefix["xs"]
            ys = prefix["ys"]
            respond_position_mask = prefix.get("respond_position_mask", None)
        elif isinstance(prefix, tuple):
            xs, ys = prefix[0], prefix[1]
            respond_position_mask = prefix[2] if len(prefix) > 2 else None
        else:
            raise TypeError("SudokuLLaDA.generate expects prefix as dict or tuple (xs,ys[,mask]).")

        out = self.forward(xs, ys, train_mode=False, respond_position_mask=respond_position_mask)
        if isinstance(out, tuple):
            pred_logits = out[0]
        else:
            pred_logits = out
        # pred_logits: [B, n_respond, 81, 10] -> take last respond
        pred_digits = torch.argmax(pred_logits[:, -1, :, :], dim=-1)  # [B,81]
        return pred_digits


class SudokuAR(TransformerModelPromptRespond, SudokuMixin):
    """
    数独专用 AR 模型（基于 TransformerModelPromptRespond + SudokuMixin）

    特性：
    1. 逐格生成（Causal Attention）
    2. 离散 Embedding 输入（Nebula vocab: 0-11）
    3. 12-class 分类输出（Nebula vocab）
    4. 🆕 坐标编码（Quiz 和 Answer 位置，基于 163-token 协议）
    5. Focal Loss 训练
    6. 🆕 Unified 163-Token Protocol：与 SudokuDream 对齐
    """

    def __init__(
        self,
        n_dims=81,
        n_positions=1000,
        n_embd=256,
        n_layer=12,
        n_head=8,
        n_prompt=5,
        n_respond=1,
        type="llama",  # 🆕 默认使用 llama backbone，支持多种: "gpt2", "gptJ", "llama", "llama2", "llama3", "qwen", "qwen2", "qwen2.5"
        mlp_ratio=4.0,
        attention_mode="causal",
        # 🆕 Backbone 配置参数（用于加载预训练模型）
        pretrained=False,  # 是否使用预训练模型
        model_name_or_path=None,  # 预训练模型路径或名称
        # 数独特有参数
        alpha=0.25,  # Focal Loss alpha (deprecated, not used when loss_mode="ce")
        gamma=1.0,   # Focal Loss gamma (deprecated, not used when loss_mode="ce")
        # 🆕 Loss mode switch
        loss_mode="ce",  # "ce": plain CE loss (default), "focal": CE + Focal Loss
        **extra,
    ):
        # 调用父类初始化
        super().__init__(
            n_dims=81,  # 固定为 81（quiz 部分）
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            type=type,
            mlp_ratio=mlp_ratio,
            n_prompt=n_prompt,
            n_respond=n_respond,
            attention_mode=attention_mode,
            pretrained=pretrained,
            model_name_or_path=model_name_or_path,
        )

        self.name = "sudoku_ar"
        self.alpha = alpha
        self.gamma = gamma
        self.loss_mode = str(loss_mode)
        assert self.loss_mode in {"ce", "focal"}, f"Unknown loss_mode={self.loss_mode}. Use 'ce' or 'focal'."

        # 🏗️ 使用 Mixin 设置数独协议
        use_coord_emb = extra.pop('use_coordinate_embedding', True)  # SudokuAR 不使用 extract_sudoku_config，直接从 extra 读取
        self.setup_sudoku_protocol(n_embd, use_coordinate_embedding=use_coord_emb)

        print(f"[SudokuAR] Initialized:")
        print(f"  backbone_type: {type}")
        print(f"  pretrained: {pretrained}")
        if pretrained and model_name_or_path:
            print(f"  model_name_or_path: {model_name_or_path}")
        print(f"  vocab_size: {self.vocab_size}")
        print(f"  use_coordinate_embedding: {use_coord_emb}")
        print(f"  Loss mode: {self.loss_mode}")
        if self.loss_mode == "focal":
            print(f"  Focal Loss: alpha={self.alpha}, gamma={self.gamma}")
        print(f"  attention_mode: {self.attention_mode}")
        print(f"  Coordinate Embedding: {'enabled' if self.use_coordinate_embedding else 'disabled'}")
        print(f"  🆕 Using 163-token protocol (81 Q + 1 '=' + 81 A)")

    def forward(self, xs, ys, train_mode=True, respond_position_mask=None):
        """
        Forward pass for Sudoku AR task（基于 163-token 协议）
        
        序列结构：(Q1 '=' A1) (Q2 '=' A2) ... (Q_prompt '=' A_prompt) (Q_target '=' A_target)
        每个 board = 163 tokens: 81 Q + 1 '=' + 81 A

        Args:
            xs: [B, n_prompt + n_respond, 81] - quiz 部分
            ys: [B, n_prompt + n_respond, 81] - solution 部分
            train_mode: True for training
            respond_position_mask: 保留兼容性

        Returns:
            If train_mode: (loss, pred_logits)
            Else: (pred_logits, mask)
        """
        B, n_points, d = xs.shape
        device = xs.device

        assert d == 81, f"数独任务要求 d=81，实际为 {d}"
        assert n_points == self.n_prompt + self.n_respond, \
            f"SudokuAR expects n_points == {self.n_prompt} + {self.n_respond}, got {n_points}"

        # 🆕 使用 163-token 统一协议构建序列（使用 Mixin 方法）
        full_sequence, prefix_len = self._prepare_163_sequence(xs, ys, device)  # [B, n_prompt*163 + 163]
        B, seq_len = full_sequence.shape
        UNIT_LEN = 163

        n_prompt = self.n_prompt
        n_respond = self.n_respond

        # 🆕 Answer 部分位置（基于 163-token 协议）
        target_board_start = n_prompt * UNIT_LEN
        answer_start = target_board_start + 82  # Answer 从索引 82 开始
        answer_end = target_board_start + UNIT_LEN  # Answer 到索引 163 结束

        if train_mode:
            # === 训练模式：使用真实标签 ===
            input_ids = full_sequence.clone()  # [B, seq_len]
            embeds = self._read_in(input_ids)  # [B, seq_len, n_embd]

            # 🆕 坐标编码：注入到 Quiz 和 Answer 位置（使用 Mixin 方法）
            embeds = self._inject_sudoku_coords(embeds, n_points, device)

            # Backbone forward (使用父类的 backbone)
            outputs = self._backbone(inputs_embeds=embeds)
            h = outputs.last_hidden_state  # [B, seq_len, hidden_size]

            # 维度对齐（如果需要）
            h = self._align_proj(h)  # [B, seq_len, n_embd]

            # Readout
            logits = self._read_out(h)  # [B, seq_len, 12] (Nebula vocab)

            # 🆕 提取 Answer 部分的 logits（基于 163-token 协议）
            answer_vocab_logits = logits[:, answer_start:answer_end, :]  # [B, 81, 12]
            
            # 提取真实 solution（转换为 Nebula tokens）
            target_vocab_ids = self._digits_to_nebula_tokens(ys[:, -1, :].long())  # [B, 81]

            # Cross-Entropy Loss + Focal Loss (使用 Nebula vocab)
            ce_loss = F.cross_entropy(
                answer_vocab_logits.reshape(-1, 12),
                target_vocab_ids.reshape(-1),
                reduction='none'
            )
            ce_loss = ce_loss.reshape(B, 81)
            
            # 🆕 转换为 digit logits（用于统一接口）
            sol_logits = self._vocab_logits_to_digit_logits(answer_vocab_logits)  # [B, 1, 81, 10]

            # Loss calculation (controlled by loss_mode)
            if self.loss_mode == "ce":
                # Plain CE loss
                final_loss = ce_loss.mean()
            else:
                # Focal Loss 重加权（使用 Nebula vocab logits）
                with torch.no_grad():
                    probs = F.softmax(answer_vocab_logits, dim=-1)
                    target_probs = probs.gather(2, target_vocab_ids.unsqueeze(-1)).squeeze(-1)
                    focal_weight = self.alpha * (1 - target_probs) ** self.gamma
                # 计算平均 loss
                final_loss = (ce_loss * focal_weight).mean()

            # ✅ 与 train_prompt_respond.py 的 sudoku-AR 分支对齐：只返回 (loss, output)
            return final_loss, sol_logits
        else:
            # === 推理模式：自回归生成（基于 163-token 协议）===
            input_ids = full_sequence.clone()
            # 初始化 Answer 部分为 MASK
            input_ids[:, answer_start:answer_end] = 11  # 🆕 MASK token (ID 11)  # MASK token

            # 逐格生成（从左到右，从上到下）
            for cell_idx in range(81):
                # Embedding
                embeds = self._read_in(input_ids)

                # 🆕 坐标编码：注入到 Quiz 和 Answer 位置（使用 Mixin 方法）
                embeds = self._inject_sudoku_coords(embeds, n_points, device)

                # Forward
                outputs = self._backbone(inputs_embeds=embeds)
                h = outputs.last_hidden_state
                h = self._align_proj(h)
                logits = self._read_out(h)  # [B, seq_len, 12] (Nebula vocab)

                # 🆕 提取当前格子的预测（基于 163-token 协议）
                pred_vocab_ids = logits[:, answer_start + cell_idx, :].argmax(dim=-1)  # [B]
                input_ids[:, answer_start + cell_idx] = pred_vocab_ids

            # 🆕 提取最终预测（基于 163-token 协议）
            embeds = self._read_in(input_ids)
            embeds = self._inject_sudoku_coords(embeds, n_points, device)

            outputs = self._backbone(inputs_embeds=embeds)
            h = outputs.last_hidden_state
            h = self._align_proj(h)
            logits = self._read_out(h)  # [B, seq_len, 12] (Nebula vocab)

            answer_vocab_logits = logits[:, answer_start:answer_end, :]  # [B, 81, 12]
            sol_logits = self._vocab_logits_to_digit_logits(answer_vocab_logits)  # [B, 1, 81, 10]

            mask = torch.ones(B, 1, 81, device=device, dtype=torch.bool)
            return sol_logits, mask

    @torch.no_grad()
    def generate(self, prefix, max_new_tokens=81, **kwargs):
        """
        Unified Sudoku validation API: autoregressively fill target solution and return digits [B,81].
        """
        if isinstance(prefix, dict):
            xs = prefix["xs"]
            ys = prefix["ys"]
            respond_position_mask = prefix.get("respond_position_mask", None)
        elif isinstance(prefix, tuple):
            xs, ys = prefix[0], prefix[1]
            respond_position_mask = prefix[2] if len(prefix) > 2 else None
        else:
            raise TypeError("SudokuAR.generate expects prefix as dict or tuple (xs,ys[,mask]).")

        out = self.forward(xs, ys, train_mode=False, respond_position_mask=respond_position_mask)
        pred_logits = out[0] if isinstance(out, tuple) else out
        pred_digits = torch.argmax(pred_logits[:, -1, :, :], dim=-1)
        return pred_digits


class SudokuLLaDABlock(LLaDABlockDiffusion, SudokuMixin):
    """
    数独专用 LLaDA Block Diffusion 模型（基于 LLaDABlockDiffusion + SudokuMixin）

    特性：
    1. Block-Causal Attention（块间因果，块内双向）
    2. 离散 Embedding 输入（0-10）
    3. 10-class 分类输出（0-9）
    4. 坐标编码
    5. Focal Loss + 时间重加权
    6. 🆕 Sudoku-Aware Block Partitioning：基于 163-token 协议（81 Q + 1 '=' + 81 A）
    7. 🆕 Unified 163-Token Protocol：与 SudokuDream 对齐
    """

    def _digits_to_nebula_tokens(self, digits: torch.Tensor) -> torch.Tensor:
        """Wrapper for shared digits_to_nebula_tokens utility"""
        from model_utils import digits_to_nebula_tokens
        return digits_to_nebula_tokens(digits)
    
    def _vocab_logits_to_digit_logits(self, vocab_logits: torch.Tensor) -> torch.Tensor:
        """Wrapper for shared vocab_logits_to_digit_logits utility"""
        from model_utils import vocab_logits_to_digit_logits
        return vocab_logits_to_digit_logits(vocab_logits)

    def _build_full_sequence(self, xs: torch.Tensor, ys: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Wrapper for shared build_sudoku_163_sequence utility"""
        from model_utils import build_sudoku_163_sequence
        return build_sudoku_163_sequence(
            xs, ys, self.n_prompt, self.n_respond, xs.device
        )

    def _compute_sudoku_block_ids(self, seq_len: int, block_size: int, device: torch.device) -> torch.Tensor:
        """
        计算 Sudoku-Aware Block IDs（基于 163-token 协议）
        
        序列结构：(Q1 '=' A1) (Q2 '=' A2) ... (Q_target '=' A_target)
        每个 board = 163 tokens: 81 Q + 1 '=' + 81 A
        
        Internal Layout per Board:
        - Indices 0-80: Quiz Grid
        - Index 81: The '=' token
        - Indices 82-162: Answer Grid
        
        Args:
            seq_len: 总序列长度（token 数）
            block_size: 块大小（9=行，81=整盘）
            device: torch device
            
        Returns:
            block_ids: [seq_len] tensor，每个 token 的块 ID
        """
        UNIT_LEN = 163  # 每个 board 的长度
        pos = torch.arange(seq_len, device=device)
        board_idx = pos // UNIT_LEN
        pos_in_board = pos % UNIT_LEN
        
        # 状态标记
        is_quiz = (pos_in_board < 81)
        is_eq = (pos_in_board == 81)
        is_ans = (pos_in_board > 81)
        
        internal_block_id = torch.zeros_like(pos)
        
        if block_size == 9:
            # Quiz: 9行 (ID 0-8)
            internal_block_id[is_quiz] = pos_in_board[is_quiz] // 9
            # '=': (ID 9)
            internal_block_id[is_eq] = 9
            # Answer: 9行 (ID 10-18)
            internal_block_id[is_ans] = 10 + (pos_in_board[is_ans] - 82) // 9
            num_blocks_per_unit = 19
        elif block_size == 81:
            # Quiz: 1块 (ID 0)
            internal_block_id[is_quiz] = 0
            # '=': (ID 1)
            internal_block_id[is_eq] = 1
            # Answer: 1块 (ID 2)
            internal_block_id[is_ans] = 2
            num_blocks_per_unit = 3
        else:
            # 其他 block_size：在 Quiz/Answer 内按 block_size 分块
            # Quiz 部分
            quiz_blocks = (81 + block_size - 1) // block_size
            internal_block_id[is_quiz] = pos_in_board[is_quiz] // block_size
            # '=' 单独一块
            internal_block_id[is_eq] = quiz_blocks
            # Answer 部分
            ans_pos_in_board = pos_in_board[is_ans] - 82
            ans_blocks = (81 + block_size - 1) // block_size
            internal_block_id[is_ans] = quiz_blocks + 1 + (ans_pos_in_board // block_size)
            num_blocks_per_unit = quiz_blocks + 1 + ans_blocks
        
        return board_idx * num_blocks_per_unit + internal_block_id

    def _create_block_causal_attention_bias(
        self,
        total_points: int,
        n_prompt: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        创建 Sudoku-Aware Block-Causal Attention Bias（基于 163-token 协议）
        
        序列结构：(Q1 '=' A1) (Q2 '=' A2) ... (Q_prompt '=' A_prompt) (Q_target '=' A_target)
        每个 board = 163 tokens: 81 Q + 1 '=' + 81 A
        
        Args:
            total_points: 总点数（n_prompt + n_respond）
            n_prompt: prompt 点数
            block_size: 块大小（9=行，81=整盘）
            device: torch device
            dtype: dtype for the bias
            
        Returns:
            attention_bias: [1, 1, seq_len, seq_len] 的attention bias
        """
        # 163-token 协议：每个 board = 163 tokens
        UNIT_LEN = 163
        seq_len = total_points * UNIT_LEN  # 总 token 数
        prompt_len = n_prompt * UNIT_LEN   # Prompt 部分的 token 数
        respond_len = seq_len - prompt_len  # Respond 部分的 token 数
        
        # 初始化 bias 为 -inf（默认不能 attend）
        bias = torch.full((1, 1, seq_len, seq_len), float('-inf'), device=device, dtype=dtype)
        
        # === Step 1: Prompt 区域（完全双向 attention）===
        bias[:, :, :prompt_len, :prompt_len] = 0  # Prompt 内所有位置互相可见
        
        # === Step 2: Respond 区域（Sudoku-Aware Block-Causal Attention）===
        if respond_len > 0:
            # 计算每个 token 的 Sudoku block ID
            block_ids = self._compute_sudoku_block_ids(seq_len, block_size, device)
            
            # 向量化构造 attention mask
            # Query 和 Key 的 block IDs
            block_i = block_ids.view(-1, 1)  # [seq_len, 1]
            block_j = block_ids.view(1, -1)  # [1, seq_len]
            
            # 位置索引
            pos_i = torch.arange(seq_len, device=device).view(-1, 1)  # [seq_len, 1]
            pos_j = torch.arange(seq_len, device=device).view(1, -1)  # [1, seq_len]
            
            # 标记 Prompt 和 Respond 区域
            is_prompt_i = pos_i < prompt_len
            is_prompt_j = pos_j < prompt_len
            is_respond_i = pos_i >= prompt_len
            is_respond_j = pos_j >= prompt_len
            
            # 核心规则：
            # 1. Prompt 区域：完全双向（已在 Step 1 设置）
            # 2. Respond -> Prompt：所有 Respond 位置可以看到 Prompt
            bias[:, :, prompt_len:, :prompt_len] = 0
            
            # 3. Respond -> Respond：
            #    - 块间因果：block_j < block_i（k 的块在 q 的块之前）
            #    - 块内双向：block_j == block_i（同一块内所有位置互相可见）
            mask_respond = (
                is_respond_i & is_respond_j &  # 都是 Respond 区域
                ((block_j < block_i) | (block_j == block_i))  # 块间因果或块内双向
            )
            bias[:, :, mask_respond] = 0
        
        return bias

    def __init__(
        self,
        n_dims=81,
        n_positions=1000,
        n_embd=256,
        n_layer=12,
        n_head=8,
        n_prompt=5,
        n_respond=1,
        *,
        mlp_ratio=4.0,
        block_group_size=1,
        # Block Diffusion 参数
        use_block_diffusion=True,
        block_size=9,  # 每行 9 个格子
        # 数独参数
        num_timesteps=20,
        alpha=0.25,
        gamma=1.0,
        # 🆕 Mask controls (same as SudokuLLaDA)
        mask_prob_override=None,
        mask_prob_min=0.0,
        mask_prob_max=1.0,
        t_sampling_power=1.0,
        loss_mode="composite",
        **extra,
    ):
        # 🏗️ 使用统一的参数过滤函数
        from model_utils import extract_sudoku_config
        sudoku_cfg, clean_extra = extract_sudoku_config(extra)
        
        # 使用 sudoku_cfg 中的值（如果存在），否则使用显式参数
        final_use_multistep = sudoku_cfg.get('use_multistep_inference', False)
        final_inference_steps = sudoku_cfg.get('inference_steps', 10)
        final_inference_k_per_step = sudoku_cfg.get('inference_k_per_step', 4)
        final_inference_scheduler = sudoku_cfg.get('inference_scheduler', None)
        final_inference_confidence_alg = sudoku_cfg.get('inference_confidence_alg', 'entropy')
        final_loss_mode = sudoku_cfg.get('loss_mode', loss_mode)
        final_alpha = sudoku_cfg.get('alpha', alpha)
        final_gamma = sudoku_cfg.get('gamma', gamma)
        final_num_timesteps = sudoku_cfg.get('num_timesteps', num_timesteps)
        
        # 使用 clean_extra 中的 training_strategy（如果存在），否则使用硬编码的值
        training_strategy_extra = clean_extra.pop('training_strategy', None)
        if training_strategy_extra is not None:
            final_training_strategy = training_strategy_extra
        else:
            final_training_strategy = {
                'mask_mode': 'timestep',
                'num_timesteps': final_num_timesteps,
                'loss_reweighting': {
                    'enable_token_reweight': True,
                    'time_weight_mode': 'linear',
                }
            }
        
        # 调用父类初始化
        super().__init__(
            n_dims=81,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_prompt=n_prompt,
            n_respond=n_respond,
            mlp_ratio=mlp_ratio,
            block_group_size=block_group_size,
            use_block_diffusion=use_block_diffusion,
            block_size=block_size,
            mask_epsilon=1e-3,
            loss_weight_type="1/t",
            train_mask_ratio=0.5,
            eval_mask_ratio=1.0,
            eval_mask_mode="fixed",
            use_prompt_context=True,
            use_multistep_inference=final_use_multistep,
            inference_steps=final_inference_steps,
            inference_confidence_alg=final_inference_confidence_alg,
            training_strategy=final_training_strategy,
            **clean_extra,
        )

        self.name = "sudoku_llada_block"
        self.num_timesteps = final_num_timesteps
        self.alpha = final_alpha
        self.gamma = final_gamma

        self.mask_prob_override = sudoku_cfg.get('mask_prob_override', mask_prob_override)
        self.mask_prob_min = float(sudoku_cfg.get('mask_prob_min', mask_prob_min))
        self.mask_prob_max = float(sudoku_cfg.get('mask_prob_max', mask_prob_max))
        self.t_sampling_power = float(sudoku_cfg.get('t_sampling_power', t_sampling_power))
        self.loss_mode = str(final_loss_mode)
        assert self.loss_mode in {"composite", "ce", "ce_ignore_index"}, \
            f"Unknown loss_mode={self.loss_mode}. Use 'composite', 'ce', or 'ce_ignore_index'."
        
        # 🆕 设置多步推理参数（父类已设置 inference_steps 和 inference_confidence_alg）
        self.inference_k_per_step = int(final_inference_k_per_step)

        # 🆕 动态 scheduler 支持（向后兼容）
        if final_inference_scheduler is not None:
            # 模式1: 使用动态 scheduler（类似 ICL）
            from dllm.core.schedulers import LinearAlphaScheduler
            if isinstance(final_inference_scheduler, str):
                if final_inference_scheduler == "linear":
                    self.inference_scheduler = LinearAlphaScheduler()
                else:
                    raise ValueError(f"Unknown scheduler: {final_inference_scheduler}")
            else:
                self.inference_scheduler = final_inference_scheduler
            self.use_dynamic_scheduler = True
            self.inference_k_per_step = None  # 不使用固定 k
            print(f"  [Inference] Using dynamic scheduler: {type(self.inference_scheduler).__name__}")
        else:
            # 模式2: 使用固定 k_per_step（当前默认行为）
            self.inference_scheduler = None
            self.use_dynamic_scheduler = False
            self.inference_k_per_step = int(final_inference_k_per_step)
            print(f"  [Inference] Using fixed k_per_step: {self.inference_k_per_step}")

        # 🏗️ 使用 Mixin 设置数独协议
        use_coord_emb = sudoku_cfg.get('use_coordinate_embedding', True)  # 从 sudoku_cfg 读取
        self.setup_sudoku_protocol(n_embd, use_coordinate_embedding=use_coord_emb)

        print(f"[SudokuLLaDABlock] Initialized:")
        print(f"  Coordinate Embedding: {'enabled' if self.use_coordinate_embedding else 'disabled'}")
        print(f"  use_block_diffusion: {use_block_diffusion}")
        print(f"  block_size: {block_size}")
        print(f"  num_timesteps: {num_timesteps}")
        if self.mask_prob_override is not None:
            print(f"  Mask prob override: {self.mask_prob_override}")
        else:
            print(f"  Mask prob range: [{self.mask_prob_min}, {self.mask_prob_max}], t_sampling_power={self.t_sampling_power}")
        print(f"  Loss mode: {self.loss_mode}")

    def _compute_mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        if self.mask_prob_override is not None:
            mask_prob = torch.full_like(t, float(self.mask_prob_override), dtype=torch.float32)
        else:
            base = (t.float() + 1.0) / float(self.num_timesteps)
            mask_prob = self.mask_prob_min + (self.mask_prob_max - self.mask_prob_min) * base
        return torch.clamp(mask_prob, 0.0, 1.0)

    def _sample_timesteps(self, B: int, device) -> torch.Tensor:
        if self.num_timesteps <= 1:
            return torch.zeros(B, device=device, dtype=torch.long)
        if self.t_sampling_power == 1.0:
            return torch.randint(0, self.num_timesteps, (B,), device=device)
        u = torch.rand(B, device=device)
        u = torch.clamp(u, 1e-8, 1.0)
        u = u ** self.t_sampling_power
        t = torch.floor(u * self.num_timesteps).long()
        return torch.clamp(t, 0, self.num_timesteps - 1)

    def forward(self, xs, ys, train_mode=True, respond_position_mask=None):
        """
        Forward pass - 使用 163-token 统一协议
        
        序列结构：(Q1 '=' A1) (Q2 '=' A2) ... (Q_prompt '=' A_prompt) (Q_target '=' A_target)
        每个 board = 163 tokens: 81 Q + 1 '=' + 81 A
        """
        B, n_points, d = xs.shape
        device = xs.device

        assert d == 81, f"数独任务要求 d=81，实际为 {d}"
        assert n_points == self.n_prompt + self.n_respond, \
            f"SudokuLLaDABlock expects n_points == {self.n_prompt} + {self.n_respond}, got {n_points}"

        # 🆕 使用 163-token 统一协议构建序列（使用 Mixin 方法）
        full_sequence, prefix_len = self._prepare_163_sequence(xs, ys, device)  # [B, n_prompt*163 + 163]
        B, seq_len = full_sequence.shape
        UNIT_LEN = 163

        n_prompt = self.n_prompt
        n_respond = self.n_respond

        if train_mode:
            # 时间步采样
            t = self._sample_timesteps(B, device)
            mask_prob = self._compute_mask_prob(t)

            # 🆕 在 163-token 协议下，只对 Answer 部分应用 mask
            # Answer 部分在每个 board 的索引 82-162
            target_part_mask = torch.zeros_like(full_sequence, dtype=torch.bool)
            for board_idx in range(n_prompt, n_points):
                board_start = board_idx * UNIT_LEN
                # Answer 部分：索引 82-162（相对于 board_start）
                answer_start = board_start + 82
                answer_end = board_start + UNIT_LEN
                target_part_mask[:, answer_start:answer_end] = True

            # 生成 mask（只对 Answer 部分）
            rand = torch.rand_like(full_sequence, dtype=torch.float32)
            masked_indices = (rand < mask_prob.reshape(B, 1)) & target_part_mask
            input_ids = full_sequence.clone()
            input_ids[masked_indices] = 11  # 🆕 MASK token (ID 11 in Nebula vocab)

            # Embedding
            embeds = self._read_in(input_ids)

            # 🆕 坐标编码：注入到 Quiz 和 Answer 位置（使用 Mixin 方法）
            embeds = self._inject_sudoku_coords(embeds, n_points, device)

            # Time conditioning
            t_scalar = t.float() / self.num_timesteps
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, seq_len, 1))
            embeds = embeds + time_emb

            # 🆕 创建 Sudoku-Aware Block-Causal Attention Bias（基于 163-token 协议）
            if self.use_block_diffusion:
                attention_bias = self._create_block_causal_attention_bias(
                    total_points=n_points,  # 总点数（每个点包含 163 tokens）
                    n_prompt=n_prompt,     # Prompt 点数（每个点包含 163 tokens）
                    block_size=self.block_size,
                    device=device,
                    dtype=embeds.dtype
                )
                attention_bias = attention_bias.expand(B, -1, -1, -1)
            else:
                # 标准双向 attention
                attention_bias = self._backbone.get_bidirectional_attention_bias(
                    seq_len=seq_len,
                    device=device
                )
            
            dummy_input_ids = torch.zeros(B, seq_len, dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                output_hidden_states=True,
                attention_bias=attention_bias,
            )
            h = out.hidden_states[-1]

            # Readout
            logits = self._read_out(h)  # [B, seq_len, 12] (Nebula vocab)

            # 🆕 提取 Answer 部分的 logits（基于 163-token 协议）
            target_board_start = n_prompt * UNIT_LEN
            answer_start = target_board_start + 82
            answer_end = target_board_start + UNIT_LEN
            answer_vocab_logits = logits[:, answer_start:answer_end, :]  # [B, 81, 12]

            # 🆕 计算 Loss（使用 Mixin 的统一 Loss 计算方法）
            target_digits = ys[:, -1, :]  # [B, 81]
            answer_mask = masked_indices[:, answer_start:answer_end]  # [B, 81]
            
            final_loss = self._compute_sudoku_loss(
                answer_vocab_logits, target_digits, answer_mask, t, mode=self.loss_mode
            )
            
            # 🆕 转换为 digit logits（用于统一接口）
            sol_logits = self._vocab_logits_to_digit_logits(answer_vocab_logits)  # [B, 1, 81, 10]

            # 转换为统一接口格式
            mask_in_sol = answer_mask.unsqueeze(1)  # [B, 1, 81]

            return final_loss, sol_logits, t, mask_in_sol
        else:
            # 🆕 推理模式（基于 163-token 协议）
            # 🆕 多步推理：与 mask-去噪训练对齐（默认关闭，保持向后兼容）
            if getattr(self, "use_multistep_inference", False):
                final_logits = self._multistep_inference(xs=xs, ys=ys)
                final_mask = torch.zeros(B, 1, 81, device=device, dtype=torch.bool)
                return final_logits, final_mask

            # 单步推理
            input_ids = full_sequence.clone()
            target_board_start = n_prompt * UNIT_LEN
            answer_start = target_board_start + 82
            answer_end = target_board_start + UNIT_LEN
            input_ids[:, answer_start:answer_end] = 11  # 🆕 MASK token (ID 11)

            embeds = self._read_in(input_ids)

            # 🆕 坐标编码：注入到 Quiz 和 Answer 位置（使用 Mixin 方法）
            embeds = self._inject_sudoku_coords(embeds, n_points, device)

            t_scalar = torch.ones(B, device=device)
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, seq_len, 1))
            embeds = embeds + time_emb

            if self.use_block_diffusion:
                attention_bias = self._create_block_causal_attention_bias(
                    total_points=n_points,
                    n_prompt=n_prompt,
                    block_size=self.block_size,
                    device=device,
                    dtype=embeds.dtype
                )
                attention_bias = attention_bias.expand(B, -1, -1, -1)
            else:
                attention_bias = self._backbone.get_bidirectional_attention_bias(
                    seq_len=seq_len,
                    device=device
                )

            dummy_input_ids = torch.zeros(B, seq_len, dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                output_hidden_states=True,
                attention_bias=attention_bias,
            )
            h = out.hidden_states[-1]

            logits = self._read_out(h)  # [B, seq_len, 12] (Nebula vocab)
            answer_vocab_logits = logits[:, answer_start:answer_end, :]  # [B, 81, 12]
            sol_logits = self._vocab_logits_to_digit_logits(answer_vocab_logits)  # [B, 1, 81, 10]

            mask = torch.ones(B, 1, 81, device=device, dtype=torch.bool)
            return sol_logits, mask

    @torch.no_grad()
    def _multistep_inference_discrete(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """
        Multi-step denoising inference for Sudoku Block Diffusion (163-token protocol).
        Iteratively fills MASK positions by selecting low-entropy cells (BPD-style).

        Returns:
            final_logits: [B, 1, 81, 10]
        """
        B, n_points, _ = xs.shape
        device = xs.device
        n_prompt = self.n_prompt
        n_respond = self.n_respond
        UNIT_LEN = 163

        # 🆕 使用 163-token 统一协议构建序列
        full_sequence, prefix_len = self._prepare_163_sequence(xs, ys, device)  # [B, n_prompt*163 + 163]
        input_ids = full_sequence.clone()

        # 🆕 初始化 Answer 部分为 MASK
        target_board_start = n_prompt * UNIT_LEN
        answer_start = target_board_start + 82
        answer_end = target_board_start + UNIT_LEN
        input_ids[:, answer_start:answer_end] = 11  # 🆕 MASK token (ID 11)

        filled = torch.zeros(B, 81, dtype=torch.bool, device=device)
        final_logits = None

        steps = max(1, int(getattr(self, 'inference_steps', 10)))
        k_per_step = max(1, int(getattr(self, 'inference_k_per_step', 4)))
        
        for _ in range(steps):
            if filled.all():
                break

            # Forward pass
            embeds = self._read_in(input_ids)  # [B, seq_len, n_embd]

            # 🆕 坐标编码：注入到 Quiz 和 Answer 位置（使用 Mixin 方法）
            embeds = self._inject_sudoku_coords(embeds, n_points, device)

            # Time conditioning
            t_scalar = torch.ones(B, device=device)
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, input_ids.shape[1], 1))
            embeds = embeds + time_emb

            # 🆕 Block attention bias
            if self.use_block_diffusion:
                attention_bias = self._create_block_causal_attention_bias(
                    total_points=n_points,
                    n_prompt=n_prompt,
                    block_size=self.block_size,
                    device=device,
                    dtype=embeds.dtype
                )
                attention_bias = attention_bias.expand(B, -1, -1, -1)
            else:
                attention_bias = self._backbone.get_bidirectional_attention_bias(
                    seq_len=input_ids.shape[1],
                    device=device
                )

            # Backbone
            dummy_input_ids = torch.zeros(B, input_ids.shape[1], dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                output_hidden_states=True,
                attention_bias=attention_bias,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)  # [B, seq_len, 12]

            # 🆕 提取 Answer 部分的 logits 并转换为 digit logits
            sol_vocab_logits = vocab_logits[:, answer_start:answer_end, :]  # [B, 81, 12]
            logits = self._vocab_logits_to_digit_logits(sol_vocab_logits)  # [B, 1, 81, 10]
            logits = logits.squeeze(1)  # [B, 81, 10]
            final_logits = logits.unsqueeze(1)  # [B, 1, 81, 10] for return format

            # Entropy over 10 classes
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [B, 81]
            entropy = entropy.masked_fill(filled, float("inf"))

            # Fill top-k lowest entropy per sample
            for b in range(B):
                unfilled = (~filled[b]).nonzero(as_tuple=True)[0]
                if unfilled.numel() == 0:
                    continue
                k = min(k_per_step, unfilled.numel())
                unfilled_entropy = entropy[b, unfilled]
                _, topk = torch.topk(unfilled_entropy, k=k, largest=False)
                cells = unfilled[topk]
                pred_digits = torch.argmax(logits[b, cells, :], dim=-1)
                # 🆕 转换为 Nebula tokens 并更新序列
                pred_tokens = self._digits_to_nebula_tokens(pred_digits.unsqueeze(0)).squeeze(0)
                input_ids[b, answer_start + cells] = pred_tokens
                filled[b, cells] = True

        if final_logits is None:
            # Final forward pass
            embeds = self._read_in(input_ids)
            # 🆕 坐标编码：注入到 Quiz 和 Answer 位置（使用 Mixin 方法）
            embeds = self._inject_sudoku_coords(embeds, n_points, device)
            t_scalar = torch.ones(B, device=device)
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, input_ids.shape[1], 1))
            embeds = embeds + time_emb
            
            if self.use_block_diffusion:
                attention_bias = self._create_block_causal_attention_bias(
                    total_points=n_points,
                    n_prompt=n_prompt,
                    block_size=self.block_size,
                    device=device,
                    dtype=embeds.dtype
                )
                attention_bias = attention_bias.expand(B, -1, -1, -1)
            else:
                attention_bias = self._backbone.get_bidirectional_attention_bias(
                    seq_len=input_ids.shape[1],
                    device=device
                )
            
            dummy_input_ids = torch.zeros(B, input_ids.shape[1], dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                output_hidden_states=True,
                attention_bias=attention_bias,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)
            sol_vocab_logits = vocab_logits[:, answer_start:answer_end, :]
            logits = self._vocab_logits_to_digit_logits(sol_vocab_logits)
            final_logits = logits  # [B, 1, 81, 10]

        return final_logits

    def _multistep_inference(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """多步去噪推理（路由方法）"""
        if self.use_dynamic_scheduler:
            return self._multistep_inference_dynamic(xs, ys)
        else:
            return self._multistep_inference_discrete(xs, ys)

    def _multistep_inference_dynamic(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """
        Multi-step denoising inference with dynamic scheduler for Sudoku Block Diffusion (163-token protocol).
        Uses get_num_transfer_tokens to dynamically determine unmask count per step.

        Returns:
            final_logits: [B, 1, 81, 10]
        """
        from dllm.utils.generation_utils import get_num_transfer_tokens

        B, n_points, _ = xs.shape
        device = xs.device
        n_prompt = self.n_prompt
        n_respond = self.n_respond
        UNIT_LEN = 163

        # 🆕 使用 163-token 统一协议构建序列
        full_sequence, prefix_len = self._prepare_163_sequence(xs, ys, device)  # [B, n_prompt*163 + 163]
        input_ids = full_sequence.clone()

        # 🆕 初始化 Answer 部分为 MASK
        target_board_start = n_prompt * UNIT_LEN
        answer_start = target_board_start + 82
        answer_end = target_board_start + UNIT_LEN
        input_ids[:, answer_start:answer_end] = 11  # 🆕 MASK token (ID 11)

        # 初始化 mask 状态
        masked_indices = torch.ones(B, 81, device=device, dtype=torch.bool)
        initial_mask = masked_indices.clone()

        # 使用 scheduler 计算每步 unmask 的数量
        num_transfer_tokens = get_num_transfer_tokens(
            mask_index=masked_indices,
            steps=self.inference_steps,
            scheduler=self.inference_scheduler,
            stochastic=False,
        )
        effective_steps = num_transfer_tokens.size(1)

        final_logits = None

        # 迭代去噪
        for step in range(effective_steps):
            if masked_indices.sum() == 0:
                break

            # Forward pass
            embeds = self._read_in(input_ids)  # [B, seq_len, n_embd]

            # 🆕 坐标编码：注入到 Quiz 和 Answer 位置（使用 Mixin 方法）
            embeds = self._inject_sudoku_coords(embeds, n_points, device)

            # Time conditioning
            t_scalar = torch.ones(B, device=device)
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, input_ids.shape[1], 1))
            embeds = embeds + time_emb

            # 🆕 Block attention bias
            if self.use_block_diffusion:
                attention_bias = self._create_block_causal_attention_bias(
                    total_points=n_points,
                    n_prompt=n_prompt,
                    block_size=self.block_size,
                    device=device,
                    dtype=embeds.dtype
                )
                attention_bias = attention_bias.expand(B, -1, -1, -1)
            else:
                attention_bias = self._backbone.get_bidirectional_attention_bias(
                    seq_len=input_ids.shape[1],
                    device=device
                )

            # Backbone
            dummy_input_ids = torch.zeros(B, input_ids.shape[1], dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                output_hidden_states=True,
                attention_bias=attention_bias,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)  # [B, seq_len, 12]

            # 🆕 提取 Answer 部分的 logits 并转换为 digit logits
            sol_vocab_logits = vocab_logits[:, answer_start:answer_end, :]  # [B, 81, 12]
            logits = self._vocab_logits_to_digit_logits(sol_vocab_logits)  # [B, 1, 81, 10]
            logits = logits.squeeze(1)  # [B, 81, 10]
            final_logits = logits.unsqueeze(1)  # [B, 1, 81, 10] for return format

            # 计算置信度
            if self.inference_confidence_alg == "entropy":
                probs = F.softmax(logits, dim=-1)
                confidence = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [B, 81]
            else:
                probs = F.softmax(logits, dim=-1)
                confidence = -probs.max(dim=-1)[0]  # [B, 81]

            confidence = confidence.masked_fill(~masked_indices, float("inf"))

            # 根据 scheduler 决定本步 unmask 多少个位置
            for b in range(B):
                k = num_transfer_tokens[b, step].item()
                if k == 0:
                    continue

                unfilled = masked_indices[b].nonzero(as_tuple=True)[0]
                if unfilled.numel() == 0:
                    continue

                k = min(k, unfilled.numel())
                unfilled_confidence = confidence[b, unfilled]
                _, topk = torch.topk(unfilled_confidence, k=k, largest=False)
                cells = unfilled[topk]
                pred_digits = torch.argmax(logits[b, cells, :], dim=-1)
                # 🆕 转换为 Nebula tokens 并更新序列
                pred_tokens = self._digits_to_nebula_tokens(pred_digits.unsqueeze(0)).squeeze(0)
                input_ids[b, answer_start + cells] = pred_tokens
                masked_indices[b, cells] = False

        if final_logits is None:
            # Final forward pass
            embeds = self._read_in(input_ids)
            # 🆕 坐标编码：注入到 Quiz 和 Answer 位置（使用 Mixin 方法）
            embeds = self._inject_sudoku_coords(embeds, n_points, device)
            t_scalar = torch.ones(B, device=device)
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, input_ids.shape[1], 1))
            embeds = embeds + time_emb

            if self.use_block_diffusion:
                attention_bias = self._create_block_causal_attention_bias(
                    total_points=n_points,
                    n_prompt=n_prompt,
                    block_size=self.block_size,
                    device=device,
                    dtype=embeds.dtype
                )
                attention_bias = attention_bias.expand(B, -1, -1, -1)
            else:
                attention_bias = self._backbone.get_bidirectional_attention_bias(
                    seq_len=input_ids.shape[1],
                    device=device
                )

            dummy_input_ids = torch.zeros(B, input_ids.shape[1], dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                output_hidden_states=True,
                attention_bias=attention_bias,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)
            sol_vocab_logits = vocab_logits[:, answer_start:answer_end, :]
            logits = self._vocab_logits_to_digit_logits(sol_vocab_logits)
            final_logits = logits  # [B, 1, 81, 10]

        return final_logits

    @torch.no_grad()
    def generate(self, prefix, max_new_tokens=81, **kwargs):
        if isinstance(prefix, dict):
            xs = prefix["xs"]
            ys = prefix["ys"]
            respond_position_mask = prefix.get("respond_position_mask", None)
        elif isinstance(prefix, tuple):
            xs, ys = prefix[0], prefix[1]
            respond_position_mask = prefix[2] if len(prefix) > 2 else None
        else:
            raise TypeError("SudokuLLaDABlock.generate expects prefix as dict or tuple (xs,ys[,mask]).")

        out = self.forward(xs, ys, train_mode=False, respond_position_mask=respond_position_mask)
        pred_logits = out[0] if isinstance(out, tuple) else out
        pred_digits = torch.argmax(pred_logits[:, -1, :, :], dim=-1)
        return pred_digits


class SudokuBOPAR(BOPARPromptRespond, SudokuMixin):
    """
    数独专用 BOP-AR (ScatDiff) 模型（基于 BOPARPromptRespond + SudokuMixin）

    特性：
    1. Offset-Causal Attention（层级并行生成）
    2. 离散 Embedding 输入（Nebula vocab: 0-11）
    3. 12-class 分类输出（Nebula vocab）
    4. 坐标编码
    5. Focal Loss + 时间重加权
    6. 🆕 Sudoku-Aware Block Partitioning：基于 163-token 协议（81 Q + 1 '=' + 81 A）
    7. 🆕 Unified 163-Token Protocol：与 SudokuDream 对齐
    8. 🆕 Nebula Tokenizer：统一词表映射
    """

    def _compute_sudoku_block_ids(self, seq_len: int, block_size: int, device: torch.device) -> torch.Tensor:
        """
        计算 Sudoku-Aware Block IDs（基于 163-token 协议）
        
        序列结构：(Q1 '=' A1) (Q2 '=' A2) ...
        每个 board = 163 tokens: 81 Q + 1 '=' + 81 A
        """
        UNIT_LEN = 163
        pos = torch.arange(seq_len, device=device)
        board_idx = pos // UNIT_LEN
        pos_in_board = pos % UNIT_LEN
        
        is_quiz = (pos_in_board < 81)
        is_eq = (pos_in_board == 81)
        is_ans = (pos_in_board > 81)
        
        internal_block_id = torch.zeros_like(pos)
        
        if block_size == 9:
            internal_block_id[is_quiz] = pos_in_board[is_quiz] // 9
            internal_block_id[is_eq] = 9
            internal_block_id[is_ans] = 10 + (pos_in_board[is_ans] - 82) // 9
            num_blocks_per_unit = 19
        elif block_size == 81:
            internal_block_id[is_quiz] = 0
            internal_block_id[is_eq] = 1
            internal_block_id[is_ans] = 2
            num_blocks_per_unit = 3
        else:
            quiz_blocks = (81 + block_size - 1) // block_size
            internal_block_id[is_quiz] = pos_in_board[is_quiz] // block_size
            internal_block_id[is_eq] = quiz_blocks
            ans_pos_in_board = pos_in_board[is_ans] - 82
            ans_blocks = (81 + block_size - 1) // block_size
            internal_block_id[is_ans] = quiz_blocks + 1 + (ans_pos_in_board // block_size)
            num_blocks_per_unit = quiz_blocks + 1 + ans_blocks
        
        return board_idx * num_blocks_per_unit + internal_block_id

    def _create_scatdiff_attention_bias(self, total_points, n_prompt, block_size, device, dtype):
        """
        创建 Sudoku-Aware ScatDiff (BOP-AR) attention bias（基于 163-token 协议）
        
        Args:
            total_points: 总点数（每个点包含 163 tokens）
            n_prompt: prompt 点数（每个点包含 163 tokens）
            block_size: 块大小（9=行，81=整盘）
            device: torch device
            dtype: dtype for the bias
            
        Returns:
            attention_bias: [1, 1, seq_len, seq_len] with 0=attend, -inf=mask
        """
        UNIT_LEN = 163
        seq_len = total_points * UNIT_LEN  # 总 token 数
        prompt_len = n_prompt * UNIT_LEN   # Prompt 部分的 token 数
        
        # 初始化 bias 为 -inf
        bias = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        
        # 计算每个 token 的 Sudoku block ID
        block_ids = self._compute_sudoku_block_ids(seq_len, block_size, device)
        
        # 🆕 在每个 block 内计算 offset（基于 163-token 协议）
        # offset = position within the block
        UNIT_LEN = 163
        pos = torch.arange(seq_len, device=device)
        board_idx = pos // UNIT_LEN
        pos_in_board = pos % UNIT_LEN
        
        is_quiz = (pos_in_board < 81)
        is_eq = (pos_in_board == 81)
        is_ans = (pos_in_board > 81)
        
        offsets = torch.zeros(seq_len, dtype=torch.long, device=device)
        
        if block_size == 81:
            # 整盘模式：offset = 0（整个单元是一个块）
            offsets[:] = 0
        elif block_size == 9:
            # 行模式：offset = 在行内的位置（0-8）
            offsets[is_quiz] = pos_in_board[is_quiz] % 9
            offsets[is_eq] = 0  # '=' token 的 offset
            offsets[is_ans] = (pos_in_board[is_ans] - 82) % 9
        else:
            # 其他 block_size：offset = 在块内的位置
            offsets[is_quiz] = pos_in_board[is_quiz] % block_size
            offsets[is_eq] = 0
            offsets[is_ans] = (pos_in_board[is_ans] - 82) % block_size
        
        # 标记 prompt 区域
        is_prompt = torch.arange(seq_len, device=device) < prompt_len
        
        # 向量化 attention mask 构造
        off_i = offsets.view(-1, 1)  # Query offsets [seq_len, 1]
        off_j = offsets.view(1, -1)   # Key offsets [1, seq_len]
        block_i = block_ids.view(-1, 1)  # Query block IDs [seq_len, 1]
        block_j = block_ids.view(1, -1)  # Key block IDs [1, seq_len]
        
        # 核心 ScatDiff 逻辑（Sudoku-aware）：
        # Key j 可见，如果：
        # 1. j 在 Prompt 区域，或
        # 2. j 和 i 在同一块内，且 offset_j <= offset_i（块内因果），或
        # 3. j 的块在 i 的块之前（块间因果）
        mask = (
            is_prompt.view(1, -1).expand(seq_len, -1) |  # Prompt 区域
            ((block_j == block_i) & (off_j <= off_i)) |  # 同一块内，offset 因果
            (block_j < block_i)  # 不同块，块间因果
        )
        
        # 应用 mask
        bias[0, 0, mask] = 0
        
        return bias

    def __init__(
        self,
        n_dims=81,
        n_positions=1000,
        n_embd=256,
        n_layer=12,
        n_head=8,
        n_prompt=5,
        n_respond=1,
        *,
        mlp_ratio=4.0,
        block_group_size=1,
        block_size=1,  # BOP-AR block_size（控制层数）
        # 数独参数
        num_timesteps=20,
        alpha=0.25,
        gamma=1.0,
        # 🆕 Mask controls
        mask_prob_override=None,
        mask_prob_min=0.0,
        mask_prob_max=1.0,
        t_sampling_power=1.0,
        loss_mode="composite",
        **extra,
    ):
        # 🏗️ 使用统一的参数过滤函数
        from model_utils import extract_sudoku_config
        sudoku_cfg, clean_extra = extract_sudoku_config(extra)
        
        # 使用 sudoku_cfg 中的值（如果存在），否则使用显式参数
        final_use_multistep = sudoku_cfg.get('use_multistep_inference', False)
        final_inference_steps = sudoku_cfg.get('inference_steps', 10)
        final_inference_k_per_step = sudoku_cfg.get('inference_k_per_step', 4)
        final_inference_scheduler = sudoku_cfg.get('inference_scheduler', None)
        final_inference_confidence_alg = sudoku_cfg.get('inference_confidence_alg', 'entropy')
        final_loss_mode = sudoku_cfg.get('loss_mode', loss_mode)
        final_alpha = sudoku_cfg.get('alpha', alpha)
        final_gamma = sudoku_cfg.get('gamma', gamma)
        final_num_timesteps = sudoku_cfg.get('num_timesteps', num_timesteps)
        
        # 使用 clean_extra 中的 training_strategy（如果存在），否则使用硬编码的值
        training_strategy_extra = clean_extra.pop('training_strategy', None)
        if training_strategy_extra is not None:
            final_training_strategy = training_strategy_extra
        else:
            final_training_strategy = {
                'mask_mode': 'timestep',
                'num_timesteps': final_num_timesteps,
                'loss_reweighting': {
                    'enable_token_reweight': True,
                    'time_weight_mode': 'linear',
                }
            }
        
        # 调用父类初始化
        super().__init__(
            n_dims=81,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_prompt=n_prompt,
            n_respond=n_respond,
            mlp_ratio=mlp_ratio,
            block_group_size=block_group_size,
            block_size=block_size,
            mask_epsilon=1e-3,
            loss_weight_type="1/t",
            train_mask_ratio=0.5,
            eval_mask_ratio=1.0,
            eval_mask_mode="fixed",
            use_prompt_context=True,
            use_multistep_inference=final_use_multistep,
            inference_steps=final_inference_steps,
            inference_confidence_alg=final_inference_confidence_alg,
            training_strategy=final_training_strategy,
            **clean_extra,
        )

        self.name = "sudoku_bopar"
        self.num_timesteps = final_num_timesteps
        self.alpha = final_alpha
        self.gamma = final_gamma

        self.mask_prob_override = sudoku_cfg.get('mask_prob_override', mask_prob_override)
        self.mask_prob_min = float(sudoku_cfg.get('mask_prob_min', mask_prob_min))
        self.mask_prob_max = float(sudoku_cfg.get('mask_prob_max', mask_prob_max))
        self.t_sampling_power = float(sudoku_cfg.get('t_sampling_power', t_sampling_power))
        self.loss_mode = str(final_loss_mode)
        assert self.loss_mode in {"composite", "ce", "ce_ignore_index"}, \
            f"Unknown loss_mode={self.loss_mode}. Use 'composite', 'ce', or 'ce_ignore_index'."
        
        # 🆕 设置多步推理参数（父类已设置 inference_steps 和 inference_confidence_alg）
        self.inference_k_per_step = int(final_inference_k_per_step)

        # 🆕 动态 scheduler 支持（向后兼容）
        if final_inference_scheduler is not None:
            # 模式1: 使用动态 scheduler（类似 ICL）
            from dllm.core.schedulers import LinearAlphaScheduler
            if isinstance(final_inference_scheduler, str):
                if final_inference_scheduler == "linear":
                    self.inference_scheduler = LinearAlphaScheduler()
                else:
                    raise ValueError(f"Unknown scheduler: {final_inference_scheduler}")
            else:
                self.inference_scheduler = final_inference_scheduler
            self.use_dynamic_scheduler = True
            self.inference_k_per_step = None  # 不使用固定 k
            print(f"  [Inference] Using dynamic scheduler: {type(self.inference_scheduler).__name__}")
        else:
            # 模式2: 使用固定 k_per_step（当前默认行为）
            self.inference_scheduler = None
            self.use_dynamic_scheduler = False
            self.inference_k_per_step = int(final_inference_k_per_step)
            print(f"  [Inference] Using fixed k_per_step: {self.inference_k_per_step}")

        # 🏗️ 使用 Mixin 设置数独协议
        use_coord_emb = sudoku_cfg.get('use_coordinate_embedding', True)  # 从 sudoku_cfg 读取
        self.setup_sudoku_protocol(n_embd, use_coordinate_embedding=use_coord_emb)

        print(f"[SudokuBOPAR] Initialized:")
        print(f"  Coordinate Embedding: {'enabled' if self.use_coordinate_embedding else 'disabled'}")
        print(f"  block_size: {block_size}")
        print(f"  num_timesteps: {num_timesteps}")
        if self.mask_prob_override is not None:
            print(f"  Mask prob override: {self.mask_prob_override}")
        else:
            print(f"  Mask prob range: [{self.mask_prob_min}, {self.mask_prob_max}], t_sampling_power={self.t_sampling_power}")
        print(f"  Loss mode: {self.loss_mode}")

    def _compute_mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        if self.mask_prob_override is not None:
            mask_prob = torch.full_like(t, float(self.mask_prob_override), dtype=torch.float32)
        else:
            base = (t.float() + 1.0) / float(self.num_timesteps)
            mask_prob = self.mask_prob_min + (self.mask_prob_max - self.mask_prob_min) * base
        return torch.clamp(mask_prob, 0.0, 1.0)

    def _sample_timesteps(self, B: int, device) -> torch.Tensor:
        if self.num_timesteps <= 1:
            return torch.zeros(B, device=device, dtype=torch.long)
        if self.t_sampling_power == 1.0:
            return torch.randint(0, self.num_timesteps, (B,), device=device)
        u = torch.rand(B, device=device)
        u = torch.clamp(u, 1e-8, 1.0)
        u = u ** self.t_sampling_power
        t = torch.floor(u * self.num_timesteps).long()
        return torch.clamp(t, 0, self.num_timesteps - 1)

    def forward(self, xs, ys, train_mode=True, respond_position_mask=None):
        """
        Forward pass - 使用 163-token 统一协议
        
        序列结构：(Q1 '=' A1) (Q2 '=' A2) ... (Q_prompt '=' A_prompt) (Q_target '=' A_target)
        """
        B, n_points, d = xs.shape
        device = xs.device

        assert d == 81, f"数独任务要求 d=81，实际为 {d}"
        assert n_points == self.n_prompt + self.n_respond

        # 🆕 使用 163-token 统一协议构建序列（使用 Mixin 方法）
        full_sequence, prefix_len = self._prepare_163_sequence(xs, ys, device)
        B, seq_len = full_sequence.shape
        UNIT_LEN = 163

        n_prompt = self.n_prompt
        n_respond = self.n_respond

        if train_mode:
            t = self._sample_timesteps(B, device)
            mask_prob = self._compute_mask_prob(t)

            # 🆕 在 163-token 协议下，只对 Answer 部分应用 mask
            target_part_mask = torch.zeros_like(full_sequence, dtype=torch.bool)
            for board_idx in range(n_prompt, n_points):
                board_start = board_idx * UNIT_LEN
                answer_start = board_start + 82
                answer_end = board_start + UNIT_LEN
                target_part_mask[:, answer_start:answer_end] = True

            rand = torch.rand_like(full_sequence, dtype=torch.float32)
            masked_indices = (rand < mask_prob.reshape(B, 1)) & target_part_mask
            input_ids = full_sequence.clone()
            input_ids[masked_indices] = 11  # 🆕 MASK token (ID 11)

            embeds = self._read_in(input_ids)

            # 🆕 坐标编码：注入到 Quiz 和 Answer 位置（使用 Mixin 方法）
            embeds = self._inject_sudoku_coords(embeds, n_points, device)

            t_scalar = t.float() / self.num_timesteps
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, seq_len, 1))
            embeds = embeds + time_emb

            # 🆕 创建 Sudoku-Aware BOP-AR Attention Bias
            attention_bias = self._create_scatdiff_attention_bias(
                total_points=n_points,
                n_prompt=n_prompt,
                block_size=self.block_size,
                device=device,
                dtype=embeds.dtype
            )
            attention_bias = attention_bias.expand(B, -1, -1, -1)
            
            dummy_input_ids = torch.zeros(B, seq_len, dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                output_hidden_states=True,
                attention_bias=attention_bias,
            )
            h = out.hidden_states[-1]

            logits = self._read_out(h)  # [B, seq_len, 12] (Nebula vocab)

            # 🆕 提取 Answer 部分的 logits
            target_board_start = n_prompt * UNIT_LEN
            answer_start = target_board_start + 82
            answer_end = target_board_start + UNIT_LEN
            answer_vocab_logits = logits[:, answer_start:answer_end, :]  # [B, 81, 12]

            # 计算 Loss（使用 Nebula vocab）
            target_vocab_ids = self._digits_to_nebula_tokens(ys[:, -1, :].long())  # [B, 81]
            answer_mask = masked_indices[:, answer_start:answer_end]

            ce_loss = F.cross_entropy(
                answer_vocab_logits.reshape(-1, 12),
                target_vocab_ids.reshape(-1),
                reduction='none'
            )
            ce_loss = ce_loss.reshape(B, 81)
            
            if self.loss_mode == "ce":
                final_loss = (ce_loss * answer_mask.float()).sum() / (answer_mask.sum() + 1e-8)
            else:
                with torch.no_grad():
                    probs = F.softmax(answer_vocab_logits, dim=-1)
                    target_probs = probs.gather(2, target_vocab_ids.unsqueeze(-1)).squeeze(-1)
                    focal_weight = self.alpha * (1 - target_probs) ** self.gamma
                time_weight = (self.num_timesteps - t.float()).reshape(B, 1)
                final_loss = (ce_loss * focal_weight * time_weight * answer_mask.float()).sum() / (answer_mask.sum() + 1e-8)

            # 🆕 转换为 digit logits（用于统一接口）
            sol_logits = self._vocab_logits_to_digit_logits(answer_vocab_logits)  # [B, 1, 81, 10]
            mask_in_sol = answer_mask.unsqueeze(1)
            return final_loss, sol_logits, t, mask_in_sol
        else:
            # 🆕 推理模式（基于 163-token 协议）
            # 🆕 多步推理：与 mask-去噪训练对齐（默认关闭，保持向后兼容）
            if getattr(self, "use_multistep_inference", False):
                final_logits = self._multistep_inference(xs=xs, ys=ys)
                final_mask = torch.zeros(B, 1, 81, device=device, dtype=torch.bool)
                return final_logits, final_mask

            # 单步推理
            # 初始化 Answer 部分为 MASK
            input_ids = full_sequence.clone()
            target_board_start = n_prompt * UNIT_LEN
            answer_start = target_board_start + 82
            answer_end = target_board_start + UNIT_LEN
            input_ids[:, answer_start:answer_end] = 11  # 🆕 MASK token (ID 11)  # MASK token

            # Embedding
            embeds = self._read_in(input_ids)

            # 🆕 坐标编码：注入到 Quiz 和 Answer 位置（使用 Mixin 方法）
            embeds = self._inject_sudoku_coords(embeds, n_points, device)

            # Time conditioning
            t_scalar = torch.ones(B, device=device)
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, seq_len, 1))
            embeds = embeds + time_emb

            # Attention bias
            attention_bias = self._create_scatdiff_attention_bias(
                total_points=n_points,
                n_prompt=n_prompt,
                block_size=self.block_size,
                device=device,
                dtype=embeds.dtype
            )
            attention_bias = attention_bias.expand(B, -1, -1, -1)

            dummy_input_ids = torch.zeros(B, seq_len, dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                output_hidden_states=True,
                attention_bias=attention_bias,
            )
            h = out.hidden_states[-1]

            logits = self._read_out(h)  # [B, seq_len, 12] (Nebula vocab)

            # 提取 Answer 部分的 logits
            answer_vocab_logits = logits[:, answer_start:answer_end, :]  # [B, 81, 12]
            sol_logits = self._vocab_logits_to_digit_logits(answer_vocab_logits)  # [B, 1, 81, 10]

            mask = torch.ones(B, 1, 81, device=device, dtype=torch.bool)
            return sol_logits, mask

    @torch.no_grad()
    def _multistep_inference_discrete(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """
        Multi-step denoising inference for Sudoku BOP-AR (163-token protocol).
        Iteratively fills MASK positions by selecting low-entropy cells (BPD-style).

        Returns:
            final_logits: [B, 1, 81, 10]
        """
        B, n_points, _ = xs.shape
        device = xs.device
        n_prompt = self.n_prompt
        n_respond = self.n_respond
        UNIT_LEN = 163

        # 🆕 使用 163-token 统一协议构建序列
        full_sequence, prefix_len = self._prepare_163_sequence(xs, ys, device)  # [B, n_prompt*163 + 163]
        input_ids = full_sequence.clone()

        # 🆕 初始化 Answer 部分为 MASK
        target_board_start = n_prompt * UNIT_LEN
        answer_start = target_board_start + 82
        answer_end = target_board_start + UNIT_LEN
        input_ids[:, answer_start:answer_end] = 11  # 🆕 MASK token (ID 11)

        filled = torch.zeros(B, 81, dtype=torch.bool, device=device)
        final_logits = None

        steps = max(1, int(getattr(self, 'inference_steps', 10)))
        k_per_step = max(1, int(getattr(self, 'inference_k_per_step', 4)))
        
        for _ in range(steps):
            if filled.all():
                break

            # Forward pass
            embeds = self._read_in(input_ids)  # [B, seq_len, n_embd]

            # 🆕 坐标编码：注入到 Quiz 和 Answer 位置（使用 Mixin 方法）
            embeds = self._inject_sudoku_coords(embeds, n_points, device)

            # Time conditioning
            t_scalar = torch.ones(B, device=device)
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, input_ids.shape[1], 1))
            embeds = embeds + time_emb

            # 🆕 ScatDiff attention bias
            attention_bias = self._create_scatdiff_attention_bias(
                total_points=n_points,
                n_prompt=n_prompt,
                block_size=self.block_size,
                device=device,
                dtype=embeds.dtype
            )
            attention_bias = attention_bias.expand(B, -1, -1, -1)

            # Backbone
            dummy_input_ids = torch.zeros(B, input_ids.shape[1], dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                output_hidden_states=True,
                attention_bias=attention_bias,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)  # [B, seq_len, 12]

            # 🆕 提取 Answer 部分的 logits 并转换为 digit logits
            sol_vocab_logits = vocab_logits[:, answer_start:answer_end, :]  # [B, 81, 12]
            logits = self._vocab_logits_to_digit_logits(sol_vocab_logits)  # [B, 1, 81, 10]
            logits = logits.squeeze(1)  # [B, 81, 10]
            final_logits = logits.unsqueeze(1)  # [B, 1, 81, 10] for return format

            # Entropy over 10 classes
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [B, 81]
            entropy = entropy.masked_fill(filled, float("inf"))

            # Fill top-k lowest entropy per sample
            for b in range(B):
                unfilled = (~filled[b]).nonzero(as_tuple=True)[0]
                if unfilled.numel() == 0:
                    continue
                k = min(k_per_step, unfilled.numel())
                unfilled_entropy = entropy[b, unfilled]
                _, topk = torch.topk(unfilled_entropy, k=k, largest=False)
                cells = unfilled[topk]
                pred_digits = torch.argmax(logits[b, cells, :], dim=-1)
                # 🆕 转换为 Nebula tokens 并更新序列
                pred_tokens = self._digits_to_nebula_tokens(pred_digits.unsqueeze(0)).squeeze(0)
                input_ids[b, answer_start + cells] = pred_tokens
                filled[b, cells] = True

        if final_logits is None:
            # Final forward pass
            embeds = self._read_in(input_ids)
            # 🆕 坐标编码：注入到 Quiz 和 Answer 位置（使用 Mixin 方法）
            embeds = self._inject_sudoku_coords(embeds, n_points, device)
            t_scalar = torch.ones(B, device=device)
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, input_ids.shape[1], 1))
            embeds = embeds + time_emb
            
            attention_bias = self._create_scatdiff_attention_bias(
                total_points=n_points,
                n_prompt=n_prompt,
                block_size=self.block_size,
                device=device,
                dtype=embeds.dtype
            )
            attention_bias = attention_bias.expand(B, -1, -1, -1)
            
            dummy_input_ids = torch.zeros(B, input_ids.shape[1], dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                output_hidden_states=True,
                attention_bias=attention_bias,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)
            sol_vocab_logits = vocab_logits[:, answer_start:answer_end, :]
            logits = self._vocab_logits_to_digit_logits(sol_vocab_logits)
            final_logits = logits  # [B, 1, 81, 10]

        return final_logits

    def _multistep_inference(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """多步去噪推理（路由方法）"""
        if self.use_dynamic_scheduler:
            return self._multistep_inference_dynamic(xs, ys)
        else:
            return self._multistep_inference_discrete(xs, ys)

    def _multistep_inference_dynamic(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """
        Multi-step denoising inference with dynamic scheduler for Sudoku BOP-AR (163-token protocol).
        Uses get_num_transfer_tokens to dynamically determine unmask count per step.

        Returns:
            final_logits: [B, 1, 81, 10]
        """
        from dllm.utils.generation_utils import get_num_transfer_tokens

        B, n_points, _ = xs.shape
        device = xs.device
        n_prompt = self.n_prompt
        n_respond = self.n_respond
        UNIT_LEN = 163

        # 🆕 使用 163-token 统一协议构建序列
        full_sequence, prefix_len = self._prepare_163_sequence(xs, ys, device)
        input_ids = full_sequence.clone()

        # 🆕 初始化 Answer 部分为 MASK
        target_board_start = n_prompt * UNIT_LEN
        answer_start = target_board_start + 82
        answer_end = target_board_start + UNIT_LEN
        input_ids[:, answer_start:answer_end] = 11  # 🆕 MASK token (ID 11)

        # 初始化 mask 状态
        masked_indices = torch.ones(B, 81, device=device, dtype=torch.bool)
        initial_mask = masked_indices.clone()

        # 使用 scheduler 计算每步 unmask 的数量
        num_transfer_tokens = get_num_transfer_tokens(
            mask_index=masked_indices,
            steps=self.inference_steps,
            scheduler=self.inference_scheduler,
            stochastic=False,
        )
        effective_steps = num_transfer_tokens.size(1)

        final_logits = None

        # 迭代去噪
        for step in range(effective_steps):
            if masked_indices.sum() == 0:
                break

            # Forward pass
            embeds = self._read_in(input_ids)

            # 🆕 坐标编码：注入到 Quiz 和 Answer 位置（使用 Mixin 方法）
            embeds = self._inject_sudoku_coords(embeds, n_points, device)

            # Time conditioning
            t_scalar = torch.ones(B, device=device)
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, input_ids.shape[1], 1))
            embeds = embeds + time_emb

            # 🆕 ScatDiff attention bias
            attention_bias = self._create_scatdiff_attention_bias(
                total_points=n_points,
                n_prompt=n_prompt,
                block_size=self.block_size,
                device=device,
                dtype=embeds.dtype
            )
            attention_bias = attention_bias.expand(B, -1, -1, -1)

            # Backbone
            dummy_input_ids = torch.zeros(B, input_ids.shape[1], dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                output_hidden_states=True,
                attention_bias=attention_bias,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)

            # 🆕 提取 Answer 部分的 logits 并转换为 digit logits
            sol_vocab_logits = vocab_logits[:, answer_start:answer_end, :]
            logits = self._vocab_logits_to_digit_logits(sol_vocab_logits)
            logits = logits.squeeze(1)  # [B, 81, 10]
            final_logits = logits.unsqueeze(1)  # [B, 1, 81, 10]

            # 计算置信度
            if self.inference_confidence_alg == "entropy":
                probs = F.softmax(logits, dim=-1)
                confidence = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [B, 81]
            else:
                probs = F.softmax(logits, dim=-1)
                confidence = -probs.max(dim=-1)[0]  # [B, 81]

            confidence = confidence.masked_fill(~masked_indices, float("inf"))

            # 根据 scheduler 决定本步 unmask 多少个位置
            for b in range(B):
                k = num_transfer_tokens[b, step].item()
                if k == 0:
                    continue

                unfilled = masked_indices[b].nonzero(as_tuple=True)[0]
                if unfilled.numel() == 0:
                    continue

                k = min(k, unfilled.numel())
                unfilled_confidence = confidence[b, unfilled]
                _, topk = torch.topk(unfilled_confidence, k=k, largest=False)
                cells = unfilled[topk]
                pred_digits = torch.argmax(logits[b, cells, :], dim=-1)
                # 🆕 转换为 Nebula tokens 并更新序列
                pred_tokens = self._digits_to_nebula_tokens(pred_digits.unsqueeze(0)).squeeze(0)
                input_ids[b, answer_start + cells] = pred_tokens
                masked_indices[b, cells] = False

        if final_logits is None:
            # Final forward pass
            embeds = self._read_in(input_ids)
            embeds = self._inject_sudoku_coords(embeds, n_points, device)
            t_scalar = torch.ones(B, device=device)
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, input_ids.shape[1], 1))
            embeds = embeds + time_emb

            attention_bias = self._create_scatdiff_attention_bias(
                total_points=n_points,
                n_prompt=n_prompt,
                block_size=self.block_size,
                device=device,
                dtype=embeds.dtype
            )
            attention_bias = attention_bias.expand(B, -1, -1, -1)

            dummy_input_ids = torch.zeros(B, input_ids.shape[1], dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                output_hidden_states=True,
                attention_bias=attention_bias,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)
            sol_vocab_logits = vocab_logits[:, answer_start:answer_end, :]
            logits = self._vocab_logits_to_digit_logits(sol_vocab_logits)
            final_logits = logits  # [B, 1, 81, 10]

        return final_logits

    @torch.no_grad()
    def generate(self, prefix, max_new_tokens=81, **kwargs):
        if isinstance(prefix, dict):
            xs = prefix["xs"]
            ys = prefix["ys"]
            respond_position_mask = prefix.get("respond_position_mask", None)
        elif isinstance(prefix, tuple):
            xs, ys = prefix[0], prefix[1]
            respond_position_mask = prefix[2] if len(prefix) > 2 else None
        else:
            raise TypeError("SudokuBOPAR.generate expects prefix as dict or tuple (xs,ys[,mask]).")

        out = self.forward(xs, ys, train_mode=False, respond_position_mask=respond_position_mask)
        pred_logits = out[0] if isinstance(out, tuple) else out
        pred_digits = torch.argmax(pred_logits[:, -1, :, :], dim=-1)
        return pred_digits


class SudokuRBOAR(RBOARPromptRespond, SudokuMixin):
    """
    数独专用 RBO-AR 模型（支持 BPD 推理，基于 RBOARPromptRespond + SudokuMixin）

    特性：
    1. Random Priority Attention（随机顺序生成）
    2. 离散 Embedding 输入（Nebula vocab: 0-11）
    3. 12-class 分类输出（Nebula vocab）
    4. 坐标编码
    5. Focal Loss + 时间重加权
    6. 🆕 BPD 推理（基于熵的优先级生成）
    7. 🆕 Sudoku-Aware Block Partitioning：基于 163-token 协议（81 Q + 1 '=' + 81 A）
    8. 🆕 Unified 163-Token Protocol：与 SudokuDream 对齐
    9. 🆕 Nebula Tokenizer：统一词表映射
    """

    def _compute_sudoku_block_ids(self, seq_len: int, block_size: int, device: torch.device) -> torch.Tensor:
        """
        计算 Sudoku-Aware Block IDs（基于 163-token 协议）
        
        序列结构：(Q1 '=' A1) (Q2 '=' A2) ...
        每个 board = 163 tokens: 81 Q + 1 '=' + 81 A
        """
        UNIT_LEN = 163
        pos = torch.arange(seq_len, device=device)
        board_idx = pos // UNIT_LEN
        pos_in_board = pos % UNIT_LEN
        
        is_quiz = (pos_in_board < 81)
        is_eq = (pos_in_board == 81)
        is_ans = (pos_in_board > 81)
        
        internal_block_id = torch.zeros_like(pos)
        
        if block_size == 9:
            internal_block_id[is_quiz] = pos_in_board[is_quiz] // 9
            internal_block_id[is_eq] = 9
            internal_block_id[is_ans] = 10 + (pos_in_board[is_ans] - 82) // 9
            num_blocks_per_unit = 19
        elif block_size == 81:
            internal_block_id[is_quiz] = 0
            internal_block_id[is_eq] = 1
            internal_block_id[is_ans] = 2
            num_blocks_per_unit = 3
        else:
            quiz_blocks = (81 + block_size - 1) // block_size
            internal_block_id[is_quiz] = pos_in_board[is_quiz] // block_size
            internal_block_id[is_eq] = quiz_blocks
            ans_pos_in_board = pos_in_board[is_ans] - 82
            ans_blocks = (81 + block_size - 1) // block_size
            internal_block_id[is_ans] = quiz_blocks + 1 + (ans_pos_in_board // block_size)
            num_blocks_per_unit = quiz_blocks + 1 + ans_blocks
        
        return board_idx * num_blocks_per_unit + internal_block_id

    def _create_rbo_attention_bias(self, total_points, n_prompt, block_size, device, dtype, external_priorities=None):
        """
        创建 Sudoku-Aware RBO-AR (Random Block-Order) attention bias（基于 163-token 协议）
        
        Args:
            total_points: 总点数（每个点包含 163 tokens）
            n_prompt: prompt 点数（每个点包含 163 tokens）
            block_size: 块大小（9=行，81=整盘）
            device: torch device
            dtype: dtype for the bias
            external_priorities: Optional tensor of shape [num_blocks] with block priorities
            
        Returns:
            attention_bias: [1, 1, seq_len, seq_len] with 0=attend, -inf=mask
        """
        UNIT_LEN = 163
        seq_len = total_points * UNIT_LEN  # 总 token 数
        prompt_len = n_prompt * UNIT_LEN   # Prompt 部分的 token 数
        respond_len = seq_len - prompt_len
        
        # 初始化 bias 为 -inf
        bias = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        
        # 计算每个 token 的 Sudoku block ID
        block_ids = self._compute_sudoku_block_ids(seq_len, block_size, device)
        
        # === Step 1: 为每个块分配优先级 ===
        pos_priority = torch.full((seq_len,), -1, device=device, dtype=torch.long)
        
        if respond_len > 0:
            # 获取 Respond 区域的所有唯一 block ID
            respond_block_ids = block_ids[prompt_len:]
            unique_blocks = torch.unique(respond_block_ids)
            num_blocks = len(unique_blocks)
            
            # 优先级分配
            if external_priorities is not None:
                priorities = external_priorities.to(device)
                assert len(priorities) == num_blocks, \
                    f"external_priorities length {len(priorities)} != num_blocks {num_blocks}"
            else:
                if self.random_order:
                    if self.priority_seed is not None:
                        generator = torch.Generator(device=device).manual_seed(self.priority_seed)
                        priorities = torch.randperm(num_blocks, device=device, generator=generator)
                    else:
                        priorities = torch.randperm(num_blocks, device=device)
                else:
                    priorities = torch.arange(num_blocks, device=device)
            
            # 创建 block_id -> priority 映射
            block_to_priority = {}
            for i, block_id in enumerate(unique_blocks):
                block_to_priority[block_id.item()] = priorities[i].item()
            
            # 为每个位置分配优先级
            for pos in range(prompt_len, seq_len):
                block_id = block_ids[pos].item()
                if block_id in block_to_priority:
                    pos_priority[pos] = block_to_priority[block_id]
        
        # === Step 2: 构造 attention mask ===
        idx_i = torch.arange(seq_len, device=device).view(-1, 1)
        idx_j = torch.arange(seq_len, device=device).view(1, -1)
        priority_i = pos_priority.view(-1, 1)
        priority_j = pos_priority.view(1, -1)
        
        # 核心因果规则
        mask_prompt = (priority_j == -1)  # Prompt 区域始终可见
        mask_inter_block = (priority_j < priority_i) & (priority_j != -1)  # 块间因果
        mask_intra_block = (priority_j == priority_i) & (idx_j <= idx_i)  # 块内因果
        
        mask = mask_prompt | mask_inter_block | mask_intra_block
        bias[0, 0, mask] = 0
        
        return bias

    def __init__(
        self,
        n_dims=81,
        n_positions=1000,
        n_embd=256,
        n_layer=12,
        n_head=8,
        n_prompt=5,
        n_respond=1,
        *,
        mlp_ratio=4.0,
        block_group_size=1,
        block_size=4,  # RBO-AR block_size
        random_order=True,
        priority_seed=None,
        # 数独参数
        num_timesteps=20,
        alpha=0.25,
        gamma=1.0,
        # 🆕 Mask controls
        mask_prob_override=None,
        mask_prob_min=0.0,
        mask_prob_max=1.0,
        t_sampling_power=1.0,
        loss_mode="composite",
        **extra,
    ):
        # 🏗️ 使用统一的参数过滤函数
        from model_utils import extract_sudoku_config
        sudoku_cfg, clean_extra = extract_sudoku_config(extra)
        
        # 使用 sudoku_cfg 中的值（如果存在），否则使用显式参数
        final_use_multistep = sudoku_cfg.get('use_multistep_inference', False)
        final_inference_steps = sudoku_cfg.get('inference_steps', 10)
        final_inference_confidence_alg = sudoku_cfg.get('inference_confidence_alg', 'entropy')
        final_loss_mode = sudoku_cfg.get('loss_mode', loss_mode)
        final_alpha = sudoku_cfg.get('alpha', alpha)
        final_gamma = sudoku_cfg.get('gamma', gamma)
        final_num_timesteps = sudoku_cfg.get('num_timesteps', num_timesteps)
        
        # 使用 clean_extra 中的 training_strategy（如果存在），否则使用硬编码的值
        training_strategy_extra = clean_extra.pop('training_strategy', None)
        if training_strategy_extra is not None:
            final_training_strategy = training_strategy_extra
        else:
            final_training_strategy = {
                'mask_mode': 'timestep',
                'num_timesteps': final_num_timesteps,
                'loss_reweighting': {
                    'enable_token_reweight': True,
                    'time_weight_mode': 'linear',
                }
            }
        
        # 调用父类初始化
        super().__init__(
            n_dims=81,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_prompt=n_prompt,
            n_respond=n_respond,
            mlp_ratio=mlp_ratio,
            block_group_size=block_group_size,
            block_size=block_size,
            random_order=random_order,
            priority_seed=priority_seed,
            mask_epsilon=1e-3,
            loss_weight_type="1/t",
            train_mask_ratio=0.5,
            eval_mask_ratio=1.0,
            eval_mask_mode="fixed",
            use_prompt_context=True,
            use_multistep_inference=final_use_multistep,
            inference_steps=final_inference_steps,
            inference_confidence_alg=final_inference_confidence_alg,
            training_strategy=final_training_strategy,
            **clean_extra,
        )

        self.name = "sudoku_rboar"
        self.num_timesteps = final_num_timesteps
        self.alpha = final_alpha
        self.gamma = final_gamma

        self.mask_prob_override = sudoku_cfg.get('mask_prob_override', mask_prob_override)
        self.mask_prob_min = float(sudoku_cfg.get('mask_prob_min', mask_prob_min))
        self.mask_prob_max = float(sudoku_cfg.get('mask_prob_max', mask_prob_max))
        self.t_sampling_power = float(sudoku_cfg.get('t_sampling_power', t_sampling_power))
        self.loss_mode = str(final_loss_mode)
        assert self.loss_mode in {"composite", "ce", "ce_ignore_index"}, \
            f"Unknown loss_mode={self.loss_mode}. Use 'composite', 'ce', or 'ce_ignore_index'."

        # 🏗️ 使用 Mixin 设置数独协议
        use_coord_emb = sudoku_cfg.get('use_coordinate_embedding', True)  # 从 sudoku_cfg 读取
        self.setup_sudoku_protocol(n_embd, use_coordinate_embedding=use_coord_emb)

        print(f"[SudokuRBOAR] Initialized:")
        print(f"  Coordinate Embedding: {'enabled' if self.use_coordinate_embedding else 'disabled'}")
        print(f"  block_size: {block_size}")
        print(f"  random_order: {random_order}")
        print(f"  num_timesteps: {num_timesteps}")
        if self.mask_prob_override is not None:
            print(f"  Mask prob override: {self.mask_prob_override}")
        else:
            print(f"  Mask prob range: [{self.mask_prob_min}, {self.mask_prob_max}], t_sampling_power={self.t_sampling_power}")
        print(f"  Loss mode: {self.loss_mode}")

    def _compute_mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        if self.mask_prob_override is not None:
            mask_prob = torch.full_like(t, float(self.mask_prob_override), dtype=torch.float32)
        else:
            base = (t.float() + 1.0) / float(self.num_timesteps)
            mask_prob = self.mask_prob_min + (self.mask_prob_max - self.mask_prob_min) * base
        return torch.clamp(mask_prob, 0.0, 1.0)

    def _sample_timesteps(self, B: int, device) -> torch.Tensor:
        if self.num_timesteps <= 1:
            return torch.zeros(B, device=device, dtype=torch.long)
        if self.t_sampling_power == 1.0:
            return torch.randint(0, self.num_timesteps, (B,), device=device)
        u = torch.rand(B, device=device)
        u = torch.clamp(u, 1e-8, 1.0)
        u = u ** self.t_sampling_power
        t = torch.floor(u * self.num_timesteps).long()
        return torch.clamp(t, 0, self.num_timesteps - 1)

    def forward(self, xs, ys, train_mode=True, respond_position_mask=None):
        """
        Forward pass - 使用 163-token 统一协议
        
        序列结构：(Q1 '=' A1) (Q2 '=' A2) ... (Q_prompt '=' A_prompt) (Q_target '=' A_target)
        """
        B, n_points, d = xs.shape
        device = xs.device

        assert d == 81, f"数独任务要求 d=81，实际为 {d}"
        assert n_points == self.n_prompt + self.n_respond

        # 🆕 使用 163-token 统一协议构建序列（使用 Mixin 方法）
        full_sequence, prefix_len = self._prepare_163_sequence(xs, ys, device)
        B, seq_len = full_sequence.shape
        UNIT_LEN = 163

        n_prompt = self.n_prompt
        n_respond = self.n_respond

        if train_mode:
            t = self._sample_timesteps(B, device)
            mask_prob = self._compute_mask_prob(t)

            # 🆕 在 163-token 协议下，只对 Answer 部分应用 mask
            target_part_mask = torch.zeros_like(full_sequence, dtype=torch.bool)
            for board_idx in range(n_prompt, n_points):
                board_start = board_idx * UNIT_LEN
                answer_start = board_start + 82
                answer_end = board_start + UNIT_LEN
                target_part_mask[:, answer_start:answer_end] = True

            rand = torch.rand_like(full_sequence, dtype=torch.float32)
            masked_indices = (rand < mask_prob.reshape(B, 1)) & target_part_mask
            input_ids = full_sequence.clone()
            input_ids[masked_indices] = 11  # 🆕 MASK token (ID 11)

            embeds = self._read_in(input_ids)

            # 🆕 坐标编码：注入到 Quiz 和 Answer 位置（使用 Mixin 方法）
            embeds = self._inject_sudoku_coords(embeds, n_points, device)

            t_scalar = t.float() / self.num_timesteps
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, seq_len, 1))
            embeds = embeds + time_emb

            # 🆕 创建 Sudoku-Aware RBO-AR Attention Bias
            attention_bias = self._create_rbo_attention_bias(
                total_points=n_points,
                n_prompt=n_prompt,
                block_size=self.block_size,
                device=device,
                dtype=embeds.dtype,
                external_priorities=None
            )
            attention_bias = attention_bias.expand(B, -1, -1, -1)
            
            dummy_input_ids = torch.zeros(B, seq_len, dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                output_hidden_states=True,
                attention_bias=attention_bias,
            )
            h = out.hidden_states[-1]

            logits = self._read_out(h)  # [B, seq_len, 12] (Nebula vocab)

            # 🆕 提取 Answer 部分的 logits
            target_board_start = n_prompt * UNIT_LEN
            answer_start = target_board_start + 82
            answer_end = target_board_start + UNIT_LEN
            answer_vocab_logits = logits[:, answer_start:answer_end, :]  # [B, 81, 12]

            # 计算 Loss（使用 Nebula vocab）
            target_vocab_ids = self._digits_to_nebula_tokens(ys[:, -1, :].long())  # [B, 81]
            answer_mask = masked_indices[:, answer_start:answer_end]

            ce_loss = F.cross_entropy(
                answer_vocab_logits.reshape(-1, 12),
                target_vocab_ids.reshape(-1),
                reduction='none'
            )
            ce_loss = ce_loss.reshape(B, 81)
            
            if self.loss_mode == "ce":
                final_loss = (ce_loss * answer_mask.float()).sum() / (answer_mask.sum() + 1e-8)
            else:
                with torch.no_grad():
                    probs = F.softmax(answer_vocab_logits, dim=-1)
                    target_probs = probs.gather(2, target_vocab_ids.unsqueeze(-1)).squeeze(-1)
                    focal_weight = self.alpha * (1 - target_probs) ** self.gamma
                time_weight = (self.num_timesteps - t.float()).reshape(B, 1)
                final_loss = (ce_loss * focal_weight * time_weight * answer_mask.float()).sum() / (answer_mask.sum() + 1e-8)

            # 🆕 转换为 digit logits（用于统一接口）
            sol_logits = self._vocab_logits_to_digit_logits(answer_vocab_logits)  # [B, 1, 81, 10]
            mask_in_sol = answer_mask.unsqueeze(1)
            return final_loss, sol_logits, t, mask_in_sol
        else:
            # 🆕 推理模式（基于 163-token 协议）
            # 初始化 Answer 部分为 MASK
            input_ids = full_sequence.clone()
            target_board_start = n_prompt * UNIT_LEN
            answer_start = target_board_start + 82
            answer_end = target_board_start + UNIT_LEN
            input_ids[:, answer_start:answer_end] = 11  # 🆕 MASK token (ID 11)  # MASK token

            # Embedding
            embeds = self._read_in(input_ids)

            # 🆕 坐标编码：注入到 Quiz 和 Answer 位置（使用 Mixin 方法）
            embeds = self._inject_sudoku_coords(embeds, n_points, device)

            # Time conditioning
            t_scalar = torch.ones(B, device=device)
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, seq_len, 1))
            embeds = embeds + time_emb

            # Attention bias
            attention_bias = self._create_rbo_attention_bias(
                total_points=n_points,
                n_prompt=n_prompt,
                block_size=self.block_size,
                device=device,
                dtype=embeds.dtype,
                external_priorities=None
            )
            attention_bias = attention_bias.expand(B, -1, -1, -1)

            dummy_input_ids = torch.zeros(B, seq_len, dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                output_hidden_states=True,
                attention_bias=attention_bias,
            )
            h = out.hidden_states[-1]

            logits = self._read_out(h)  # [B, seq_len, 12] (Nebula vocab)

            # 提取 Answer 部分的 logits
            answer_vocab_logits = logits[:, answer_start:answer_end, :]  # [B, 81, 12]
            sol_logits = self._vocab_logits_to_digit_logits(answer_vocab_logits)  # [B, 1, 81, 10]

            mask = torch.ones(B, 1, 81, device=device, dtype=torch.bool)
            return sol_logits, mask

    def generate_bpd(self, xs, ys, n_prompt, n_respond, device):
        """
        BPD 推理：基于熵的自适应生成（基于 163-token 协议）

        策略：
        1. 初始化 Answer 部分为 MASK
        2. 循环：
           a. Probe - Forward 计算所有格子的预测分布
           b. 计算熵 - 低熵 = 高置信度
           c. 填充 Top-K 最确定的格子
           d. 重复直到所有格子都被填充
        """
        B, n_points, d = xs.shape
        UNIT_LEN = 163

        # 🆕 使用 163-token 协议构建序列
        full_sequence, prefix_len = self._build_full_sequence(xs, ys)
        B, seq_len = full_sequence.shape

        # 初始化 Answer 部分为 MASK
        input_ids = full_sequence.clone()
        target_board_start = n_prompt * UNIT_LEN
        answer_start = target_board_start + 82
        answer_end = target_board_start + UNIT_LEN
        input_ids[:, answer_start:answer_end] = 11  # 🆕 MASK token (ID 11)

        # 记录哪些格子已被填充
        filled = torch.zeros(B, 81, device=device, dtype=torch.bool)

        # BPD 循环（最多 81 步，每步填充一些格子）
        max_iterations = 81
        k_per_step = 5

        for iteration in range(max_iterations):
            if filled.all():
                break

            # Probe: Forward 计算预测
            embeds = self._read_in(input_ids)

            # 🆕 坐标编码：注入到 Quiz 和 Answer 位置（使用 Mixin 方法）
            embeds = self._inject_sudoku_coords(embeds, n_points, device)

            # Time conditioning
            t_scalar = torch.ones(B, device=device)
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, seq_len, 1))
            embeds = embeds + time_emb

            # Attention bias
            attention_bias = self._create_rbo_attention_bias(
                total_points=n_points,
                n_prompt=n_prompt,
                block_size=self.block_size,
                device=device,
                dtype=embeds.dtype,
                external_priorities=None
            )
            attention_bias = attention_bias.expand(B, -1, -1, -1)

            dummy_input_ids = torch.zeros(B, seq_len, dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                output_hidden_states=True,
                attention_bias=attention_bias,
            )
            h = out.hidden_states[-1]

            logits = self._read_out(h)  # [B, seq_len, 12] (Nebula vocab)

            # 🆕 提取 Answer 部分的 logits
            answer_vocab_logits = logits[:, answer_start:answer_end, :]  # [B, 81, 12]

            # 计算熵（只对未填充的格子，使用 Nebula vocab）
            probs = F.softmax(answer_vocab_logits, dim=-1)  # [B, 81, 12]
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [B, 81]

            # 将已填充的格子的熵设为无穷大
            entropy = entropy.masked_fill(filled, float('inf'))

            # 选择 Top-K 最确定的格子（熵最小）
            topk_values, topk_indices = torch.topk(entropy, k=min(k_per_step, (~filled).sum().item()), largest=False)

            # 填充这些格子
            for b in range(B):
                for k_idx in range(len(topk_indices[b])):
                    cell_idx = topk_indices[b, k_idx].item()
                    if topk_values[b, k_idx] == float('inf'):
                        continue

                    # 获取预测值（Nebula vocab ID）
                    pred_vocab_id = answer_vocab_logits[b, cell_idx, :].argmax().item()

                    # 🆕 填充（基于 163-token 协议，使用 Nebula vocab）
                    input_ids[b, answer_start + cell_idx] = pred_vocab_id
                    filled[b, cell_idx] = True

        # 🆕 最终 forward 获取完整预测（基于 163-token 协议）
        embeds = self._read_in(input_ids)

        # 🆕 坐标编码：注入到 Quiz 和 Answer 位置（使用 Mixin 方法）
        embeds = self._inject_sudoku_coords(embeds, n_points, device)

        # Time conditioning
        t_scalar = torch.ones(B, device=device)
        time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, seq_len, 1))
        embeds = embeds + time_emb

        # Attention bias
        attention_bias = self._create_rbo_attention_bias(
            total_points=n_points,
            n_prompt=n_prompt,
            block_size=self.block_size,
            device=device,
            dtype=embeds.dtype,
            external_priorities=None
        )
        attention_bias = attention_bias.expand(B, -1, -1, -1)

        dummy_input_ids = torch.zeros(B, seq_len, dtype=torch.long, device=device)
        out = self._backbone(
            input_ids=dummy_input_ids,
                input_embeddings=embeds,
            output_hidden_states=True,
            attention_bias=attention_bias,
        )
        h = out.hidden_states[-1]

        logits = self._read_out(h)  # [B, seq_len, 12] (Nebula vocab)

        # 🆕 提取 Answer 部分的 logits
        answer_vocab_logits = logits[:, answer_start:answer_end, :]  # [B, 81, 12]
        sol_logits = self._vocab_logits_to_digit_logits(answer_vocab_logits)  # [B, 1, 81, 10]

        mask = torch.ones(B, 1, 81, device=device, dtype=torch.bool)
        return sol_logits, mask

    @torch.no_grad()
    def generate(self, prefix, max_new_tokens=81, **kwargs):
        if isinstance(prefix, dict):
            xs = prefix["xs"]
            ys = prefix["ys"]
            respond_position_mask = prefix.get("respond_position_mask", None)
        elif isinstance(prefix, tuple):
            xs, ys = prefix[0], prefix[1]
            respond_position_mask = prefix[2] if len(prefix) > 2 else None
        else:
            raise TypeError("SudokuRBOAR.generate expects prefix as dict or tuple (xs,ys[,mask]).")

        out = self.forward(xs, ys, train_mode=False, respond_position_mask=respond_position_mask)
        pred_logits = out[0] if isinstance(out, tuple) else out
        pred_digits = torch.argmax(pred_logits[:, -1, :, :], dim=-1)
        return pred_digits


# ============================================================
# SudokuBADAR: Block-level Autoregressive Diffusion for Sudoku
# ============================================================
class SudokuBADAR(BADARPromptRespond, SudokuMixin):
    """
    数独专用 BAD-AR 模型 - 高性能向量化版本（基于 BADARPromptRespond + SudokuMixin）
    
    特性：
    1. Block-level Diffusion (Inter-block): 块间扩散逻辑，随机 Mask 若干块
    2. Intra-block AR: 块内严格因果顺序
    3. 离散 Embedding 输入（Nebula vocab: 0-11）
    4. 12-class 分类输出（Nebula vocab）
    5. 坐标编码
    6. Focal Loss + 时间重加权
    7. 🆕 Sudoku-Aware Block Partitioning：基于 163-token 协议（81 Q + 1 '=' + 81 A）
    8. 🆕 Unified 163-Token Protocol：与 SudokuDream 对齐
    9. 🆕 Nebula Tokenizer：统一词表映射
    10. ⚡ 全向量化实现（Zero-For-Loop）：使用索引映射技巧，消除所有循环
    
    核心逻辑：
    - 只有 Answer 部分（索引 82-162）参与 Block 划分和 Mask
    - Block 划分：按 block_size（通常为 9，即一行）划分 Answer 部分
    - 块间 Diffusion：不同被 Mask 的块之间互不可见
    - 块内 AR：每个 Mask 块内部严格因果顺序
    - 全局可见性：Prompt（Quiz + '='）和可见 Answer 块对所有位置可见
    """
    
    def _compute_sudoku_block_ids(self, seq_len: int, block_size: int, device: torch.device) -> torch.Tensor:
        """
        计算全局唯一的 Block IDs（Quiz/Eq 为 -1, Answer 按块分配全局唯一 ID）
        
        序列结构：(Q1 '=' A1) (Q2 '=' A2) ...
        每个 board = 163 tokens: 81 Q + 1 '=' + 81 A
        
        对于 BAD-AR：
        - Quiz 部分（0-80）：标记为 -1（背景，永远可见）
        - '=' 部分（81）：标记为 -1（背景，永远可见）
        - Answer 部分（82-162）：按 block_size 分配全局唯一正数 ID
        
        Args:
            seq_len: 总 token 数
            block_size: 块大小（通常为 9，即一行）
            device: torch device
            
        Returns:
            block_ids: [seq_len] tensor，每个 token 的全局唯一 block ID
        """
        UNIT_LEN = 163
        pos = torch.arange(seq_len, device=device)
        board_idx = pos // UNIT_LEN
        pos_in_board = pos % UNIT_LEN
        
        is_ans = (pos_in_board >= 82)
        block_ids = torch.full((seq_len,), -1, device=device, dtype=torch.long)
        
        if is_ans.any():
            ans_relative_pos = pos_in_board[is_ans] - 82  # Answer 部分从 0 开始
            inner_block_id = ans_relative_pos // block_size
            num_blocks_per_ans = (81 + block_size - 1) // block_size
            # 全局唯一 ID：board_idx * num_blocks_per_ans + inner_block_id
            block_ids[is_ans] = board_idx[is_ans] * num_blocks_per_ans + inner_block_id
        
        return block_ids
    
    def _create_bad_ar_attention_bias(self, b, total_points, n_prompt, block_size, 
                                     masked_block_indices, device, dtype, respond_indices_batch=None):
        """
        创建 Sudoku-Aware BAD-AR attention bias（全向量化版本，Zero-For-Loop）
        
        核心逻辑：
        1. 只有 Answer 部分参与 Block 划分
        2. Quiz 和 '=' 部分永远可见（block_id = -1）
        3. 未被 Mask 的 Answer 块永远可见
        4. 被 Mask 的 Answer 块：块间互不可见，块内 AR
        
        性能优化：
        - 使用索引映射技巧：pad_lookup = torch.cat([pad, masked_block_indices], dim=1)
        - 通过 token_mask = pad_lookup[:, block_ids + 1] 一次性获取全序列掩码
        
        Args:
            b: batch size
            total_points: 总点数（每个点包含 163 tokens）
            n_prompt: prompt 点数
            block_size: 块大小（通常为 9，即一行）
            masked_block_indices: [b, num_blocks] bool tensor，True 表示该块被 mask
            device: torch device
            dtype: torch dtype
            respond_indices_batch: 对于数独任务，通常为 None（sequential 模式）
            
        Returns:
            attention_bias: [b, n_head, seq_len, seq_len] additive attention bias
        """
        seq_len = total_points * 163
        idx_range = torch.arange(seq_len, device=device)
        
        # === 1. 获取全局 Block IDs [seq_len] ===
        block_ids = self._compute_sudoku_block_ids(seq_len, block_size, device)
        
        # === 2. ⚡ 索引映射技巧：建立位置到 Masked 状态的映射 [b, seq_len] ===
        # 巧妙利用 block_ids 作为索引从 masked_block_indices 中取值
        # 我们把 block_id = -1 的位置对应到 masked_block_indices 的一个 padding 位(False)
        pad = torch.zeros((b, 1), dtype=torch.bool, device=device)
        full_mask_lookup = torch.cat([pad, masked_block_indices], dim=1)  # [b, num_blocks + 1]
        
        # pos_is_masked[b, seq_len]: 通过 block_ids+1 映射（block_id=-1 -> index 0, block_id=0 -> index 1, ...）
        pos_is_masked = full_mask_lookup[:, block_ids + 1]
        
        # === 3. 向量化构建规则 [b, seq, seq] ===
        q_idx = idx_range.view(1, -1, 1).expand(b, -1, -1)  # [b, seq_len, 1]
        k_idx = idx_range.view(1, 1, -1).expand(b, -1, -1)  # [b, 1, seq_len]
        q_block = block_ids.view(1, -1, 1).expand(b, -1, -1)  # [b, seq_len, 1]
        k_block = block_ids.view(1, 1, -1).expand(b, -1, -1)  # [b, 1, seq_len]
        
        k_is_masked = pos_is_masked.unsqueeze(1)  # [b, 1, seq_len]
        q_is_masked = pos_is_masked.unsqueeze(2)  # [b, seq_len, 1]
        
        # 可见性判断（完全向量化）
        is_context_k = (k_block == -1).expand(-1, seq_len, -1)  # [b, seq_len, seq_len] - Quiz 和 '=' 永远可见
        is_visible_ans_k = ((~k_is_masked) & (k_block != -1)).expand(-1, seq_len, -1)  # [b, seq_len, seq_len] - 未被 mask 的 Answer 块
        is_intra_ar = (q_block == k_block) & (k_block != -1) & q_is_masked.expand(-1, -1, seq_len) & (k_idx <= q_idx)  # [b, seq_len, seq_len] - 同一块内 AR
        
        is_visible = is_context_k | is_visible_ans_k | is_intra_ar  # [b, seq_len, seq_len]
        
        # === 4. 构造 Bias 并显式包含 n_head 维度 ===
        bias = torch.where(
            is_visible.unsqueeze(1).expand(-1, self.n_head, -1, -1),  # [b, n_head, seq_len, seq_len]
            torch.zeros((b, self.n_head, seq_len, seq_len), device=device, dtype=dtype),
            torch.full((b, self.n_head, seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        )
        
        return bias
    
    def __init__(
        self,
        n_dims=81,
        n_positions=1000,
        n_embd=256,
        n_layer=12,
        n_head=8,
        n_prompt=5,
        n_respond=1,
        *,
        mlp_ratio=4.0,
        block_group_size=1,
        block_size=9,  # BAD-AR block_size（通常为 9，即一行）
        **extra,
    ):
        # 1. 提取配置
        from model_utils import extract_sudoku_config
        sudoku_cfg, clean_extra = extract_sudoku_config(extra)
        
        # 2. 初始化父类
        super().__init__(
            n_dims=81,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_prompt=n_prompt,
            n_respond=n_respond,
            mlp_ratio=mlp_ratio,
            block_group_size=block_group_size,
            block_size=block_size,
            **clean_extra,
        )

        self.name = "sudoku_bad_ar"
        self.num_timesteps = sudoku_cfg.get('num_timesteps', 20)
        self.alpha = sudoku_cfg.get('alpha', 0.25)
        self.gamma = sudoku_cfg.get('gamma', 1.0)
        self.loss_mode = sudoku_cfg.get('loss_mode', 'composite')

        # 🆕 多步推理参数
        final_use_multistep = sudoku_cfg.get('use_multistep_inference', False)
        final_inference_steps = sudoku_cfg.get('inference_steps', 10)
        final_inference_k_per_step = sudoku_cfg.get('inference_k_per_step', 4)
        final_inference_scheduler = sudoku_cfg.get('inference_scheduler', None)
        final_inference_confidence_alg = sudoku_cfg.get('inference_confidence_alg', 'entropy')

        self.use_multistep_inference = bool(final_use_multistep)
        self.inference_steps = int(final_inference_steps)
        self.inference_confidence_alg = str(final_inference_confidence_alg)

        # 🆕 动态 scheduler 支持（向后兼容）
        if final_inference_scheduler is not None:
            # 模式1: 使用动态 scheduler（类似 ICL）
            from dllm.core.schedulers import LinearAlphaScheduler
            if isinstance(final_inference_scheduler, str):
                if final_inference_scheduler == "linear":
                    self.inference_scheduler = LinearAlphaScheduler()
                else:
                    raise ValueError(f"Unknown scheduler: {final_inference_scheduler}")
            else:
                self.inference_scheduler = final_inference_scheduler
            self.use_dynamic_scheduler = True
            self.inference_k_per_step = None  # 不使用固定 k
            print(f"  [Inference] Using dynamic scheduler: {type(self.inference_scheduler).__name__}")
        else:
            # 模式2: 使用固定 k_per_step（当前默认行为）
            self.inference_scheduler = None
            self.use_dynamic_scheduler = False
            self.inference_k_per_step = int(final_inference_k_per_step)
            print(f"  [Inference] Using fixed k_per_step: {self.inference_k_per_step}")

        # 3. 设置数独协议
        use_coord_emb = sudoku_cfg.get('use_coordinate_embedding', True)
        self.setup_sudoku_protocol(n_embd, use_coordinate_embedding=use_coord_emb)
        coord_status = "with" if use_coord_emb else "without"
        print(f"[{self.name}] Fast Vectorized Version Initialized (block_size={block_size}, {coord_status} coordinate embedding)")
    
    def forward(self, xs, ys, train_mode=True, respond_position_mask=None):
        """
        Forward pass - 全向量化实现（Zero-For-Loop）
        
        序列结构：(Q1 '=' A1) (Q2 '=' A2) ... (Q_prompt '=' A_prompt) (Q_target '=' A_target)
        
        核心逻辑：
        1. 只有 Answer 部分（索引 82-162）参与 Block 划分和 Mask
        2. Block-level Masking：使用阈值向量化生成 masked_block_indices
        3. Token-level Masking：使用索引映射技巧一次性获取全序列掩码
        4. 不同被 Mask 的块之间互不可见，每个被 Mask 的块内部严格因果顺序
        """
        from training_strategy_utils import sample_timestep_with_strategy
        
        B, n_points, d = xs.shape
        device = xs.device
        UNIT_LEN = 163

        assert d == 81, f"数独任务要求 d=81，实际为 {d}"
        assert n_points == self.n_prompt + self.n_respond

        # 🆕 使用 163-token 统一协议构建序列（使用 Mixin 方法）
        full_sequence, _ = self._prepare_163_sequence(xs, ys, device)
        seq_len = full_sequence.shape[1]
        
        # 计算块数
        num_blocks_per_ans = (81 + self.block_size - 1) // self.block_size
        total_ans_blocks = self.n_respond * num_blocks_per_ans

        # 采样（使用统一的训练策略函数）
        t_scalar, t_int = sample_timestep_with_strategy(
            B, device, train_mode, self.training_strategy,
            self.num_timesteps, 1e-3, 0.5, 1.0, "fixed"
        )
        
        # ⚡ 彻底消除采样循环：向量化生成 masked_block_indices
        if train_mode:
            threshold = torch.rand((B, total_ans_blocks), device=device)
            masked_block_indices = (threshold < t_scalar.view(B, 1))
            # 至少 mask 一个块兜底（确保每个 batch 至少有一个块被 mask）
            masked_block_indices[:, 0] = True
        else:
            # 推理模式：所有 Answer 块都被 mask
            masked_block_indices = torch.ones((B, total_ans_blocks), dtype=torch.bool, device=device)

        # ⚡ 彻底消除 Token Mask 循环：使用索引映射技巧
        block_ids = self._compute_sudoku_block_ids(seq_len, self.block_size, device)
        pad = torch.zeros((B, 1), dtype=torch.bool, device=device)
        full_m_lookup = torch.cat([pad, masked_block_indices], dim=1)  # [B, num_blocks + 1]
        token_masked = full_m_lookup[:, block_ids + 1]  # [B, seq_len]

        # 应用 Mask Token
        input_ids = full_sequence.clone()
        input_ids[token_masked] = self.mask_token_id  # ID 11 为 MASK

        # Embedding & Coords
        embeds = self._read_in(input_ids)
        embeds = self._inject_sudoku_coords(embeds, n_points, device)
        embeds = embeds + self._time_mlp(t_scalar.view(B, 1, 1).expand(B, seq_len, 1))

        # Bias
        attention_bias = self._create_bad_ar_attention_bias(
            B, n_points, self.n_prompt, self.block_size, masked_block_indices, device, embeds.dtype
        )

        # Forward
        dummy = torch.zeros(B, seq_len, dtype=torch.long, device=device)
        h = self._backbone(
            input_ids=dummy,
            input_embeddings=embeds,  # 修复：使用正确的参数名 input_embeddings
            attention_bias=attention_bias,
            output_hidden_states=True
        ).hidden_states[-1]
        logits = self._read_out(h)

        # 提取目标 (取最后一个 board 的 Answer)
        target_start = (n_points - 1) * UNIT_LEN + 82
        ans_logits = logits[:, target_start : target_start+81, :]
        target_digits = ys[:, -1, :].long()
        ans_mask = token_masked[:, target_start : target_start+81]

        if train_mode:
            loss = self._compute_sudoku_loss(ans_logits, target_digits, ans_mask, t_int, mode=self.loss_mode)
            return loss, self._vocab_logits_to_digit_logits(ans_logits), t_scalar, ans_mask.unsqueeze(1)
        else:
            # 推理模式
            if self.use_multistep_inference:
                # 多步推理
                return self._multistep_inference(xs, ys)

            # 单步推理：所有 Answer 位置都被 mask
            mask = torch.ones(B, 1, 81, device=device, dtype=torch.bool)
            return self._vocab_logits_to_digit_logits(ans_logits), mask
    
    @torch.no_grad()
    def _multistep_inference(self, xs, ys):
        """
        多步推理路由方法
        根据 use_dynamic_scheduler 决定使用哪种推理模式
        """
        if self.use_dynamic_scheduler:
            return self._multistep_inference_dynamic(xs, ys)
        else:
            return self._multistep_inference_discrete(xs, ys)

    @torch.no_grad()
    def _multistep_inference_discrete(self, xs, ys):
        """
        固定 k_per_step 的多步推理（基于 163-token 协议，与 forward 方法对齐）
        每步 unmask 固定数量的位置
        """
        B, n_points, d = xs.shape
        device = xs.device
        UNIT_LEN = 163

        # 🆕 使用 163-token 统一协议构建序列（使用 Mixin 方法）
        full_sequence, _ = self._prepare_163_sequence(xs, ys, device)
        seq_len = full_sequence.shape[1]
        input_ids = full_sequence.clone()

        # 初始化：所有 Answer 位置都是 MASK (ID 11)
        # 与 forward 方法对齐：取最后一个 board 的 Answer
        target_start = (n_points - 1) * UNIT_LEN + 82
        answer_start = target_start
        answer_end = target_start + 81
        input_ids[:, answer_start:answer_end] = 11  # MASK token (ID 11)

        # 记录哪些位置已被填充
        filled = torch.zeros(B, 81, device=device, dtype=torch.bool)
        final_logits = None

        steps = max(1, int(self.inference_steps))
        k_per_step = max(1, int(self.inference_k_per_step))

        for step in range(steps):
            if filled.all():
                break

            # Forward pass（与 forward 方法对齐）
            embeds = self._read_in(input_ids)
            embeds = self._inject_sudoku_coords(embeds, n_points, device)
            embeds = embeds + self._time_mlp(torch.ones(B, device=device).view(B, 1, 1).expand(B, seq_len, 1))

            # 计算块数
            num_blocks_per_ans = (81 + self.block_size - 1) // self.block_size
            total_ans_blocks = self.n_respond * num_blocks_per_ans

            # 🎯 动态解除遮蔽：根据已填充的 cells 计算 masked_block_indices
            # 初始化为全部 unmasked（False）
            masked_block_indices = torch.zeros((B, total_ans_blocks), dtype=torch.bool, device=device)

            # 对于每个 block，如果有任何 cell 未填充，则标记该 block 为 masked
            for b in range(B):
                for blk_idx in range(num_blocks_per_ans):
                    # 计算该 block 对应的 cell 范围
                    start_cell = blk_idx * self.block_size
                    end_cell = min(start_cell + self.block_size, 81)
                    # 如果该 block 中有任何 cell 未填充，标记为 masked
                    if not filled[b, start_cell:end_cell].all():
                        masked_block_indices[b, blk_idx] = True

            # Attention bias
            attention_bias = self._create_bad_ar_attention_bias(
                B, n_points, self.n_prompt, self.block_size, masked_block_indices, device, embeds.dtype
            )

            # Backbone forward
            dummy = torch.zeros(B, seq_len, dtype=torch.long, device=device)
            h = self._backbone(
                input_ids=dummy,
                input_embeddings=embeds,
                attention_bias=attention_bias,
                output_hidden_states=True
            ).hidden_states[-1]
            vocab_logits = self._read_out(h)

            # 提取 Answer 部分的 logits
            ans_logits = vocab_logits[:, answer_start:answer_end, :]  # [B, 81, 12]
            digit_logits = self._vocab_logits_to_digit_logits(ans_logits)  # [B, 1, 81, 10]
            digit_logits = digit_logits.squeeze(1)  # [B, 81, 10]
            final_logits = digit_logits.unsqueeze(1)  # [B, 1, 81, 10]

            # 计算置信度
            probs = torch.softmax(digit_logits, dim=-1)
            if self.inference_confidence_alg == "entropy":
                confidence = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [B, 81]
            else:
                confidence = -probs.max(dim=-1)[0]  # [B, 81]

            # 只考虑仍被 mask 的位置
            confidence = confidence.masked_fill(filled, float('inf'))

            # 选择置信度最低的 k 个位置进行 unmask
            for b in range(B):
                unfilled = (~filled[b]).nonzero(as_tuple=True)[0]
                if unfilled.numel() == 0:
                    continue
                k = min(k_per_step, unfilled.numel())
                unfilled_confidence = confidence[b, unfilled]
                _, topk = torch.topk(unfilled_confidence, k=k, largest=False)
                cells = unfilled[topk]
                pred_digits = torch.argmax(digit_logits[b, cells, :], dim=-1)
                # 🆕 转换为 Nebula tokens 并更新序列
                pred_tokens = self._digits_to_nebula_tokens(pred_digits.unsqueeze(0)).squeeze(0)
                input_ids[b, answer_start + cells] = pred_tokens
                filled[b, cells] = True

        if final_logits is None:
            # Final forward pass
            embeds = self._read_in(input_ids)
            embeds = self._inject_sudoku_coords(embeds, n_points, device)
            embeds = embeds + self._time_mlp(torch.ones(B, device=device).view(B, 1, 1).expand(B, seq_len, 1))

            num_blocks_per_ans = (81 + self.block_size - 1) // self.block_size
            total_ans_blocks = self.n_respond * num_blocks_per_ans

            # 🎯 动态解除遮蔽：根据已填充的 cells 计算 masked_block_indices
            masked_block_indices = torch.zeros((B, total_ans_blocks), dtype=torch.bool, device=device)
            for b in range(B):
                for blk_idx in range(num_blocks_per_ans):
                    start_cell = blk_idx * self.block_size
                    end_cell = min(start_cell + self.block_size, 81)
                    if not filled[b, start_cell:end_cell].all():
                        masked_block_indices[b, blk_idx] = True

            attention_bias = self._create_bad_ar_attention_bias(
                B, n_points, self.n_prompt, self.block_size, masked_block_indices, device, embeds.dtype
            )

            dummy = torch.zeros(B, seq_len, dtype=torch.long, device=device)
            h = self._backbone(
                input_ids=dummy,
                input_embeddings=embeds,
                attention_bias=attention_bias,
                output_hidden_states=True
            ).hidden_states[-1]
            vocab_logits = self._read_out(h)
            ans_logits = vocab_logits[:, answer_start:answer_end, :]
            final_logits = self._vocab_logits_to_digit_logits(ans_logits)  # [B, 1, 81, 10]

        mask = torch.zeros(B, 1, 81, device=device, dtype=torch.bool)
        return final_logits, mask

    @torch.no_grad()
    def _multistep_inference_dynamic(self, xs, ys):
        """
        动态 scheduler 的多步推理（基于 163-token 协议，与 forward 方法对齐）
        使用 LinearAlphaScheduler 动态决定每步 unmask 的数量
        """
        from dllm.utils.generation_utils import get_num_transfer_tokens

        B, n_points, d = xs.shape
        device = xs.device
        UNIT_LEN = 163

        # 🆕 使用 163-token 统一协议构建序列（使用 Mixin 方法）
        full_sequence, _ = self._prepare_163_sequence(xs, ys, device)
        seq_len = full_sequence.shape[1]
        input_ids = full_sequence.clone()

        # 初始化：所有 Answer 位置都是 MASK (ID 11)
        # 与 forward 方法对齐：取最后一个 board 的 Answer
        target_start = (n_points - 1) * UNIT_LEN + 82
        answer_start = target_start
        answer_end = target_start + 81
        input_ids[:, answer_start:answer_end] = 11  # MASK token (ID 11)

        # 初始化 mask 状态
        masked_indices = torch.ones(B, 81, device=device, dtype=torch.bool)
        initial_mask = masked_indices.clone()

        # 使用 scheduler 计算每步 unmask 的数量
        num_transfer_tokens = get_num_transfer_tokens(
            mask_index=masked_indices,
            steps=self.inference_steps,
            scheduler=self.inference_scheduler,
            stochastic=False,
        )
        effective_steps = num_transfer_tokens.size(1)

        final_logits = None

        # 迭代去噪
        for step in range(effective_steps):
            if masked_indices.sum() == 0:
                break

            # Forward pass（与 forward 方法对齐）
            embeds = self._read_in(input_ids)
            embeds = self._inject_sudoku_coords(embeds, n_points, device)
            embeds = embeds + self._time_mlp(torch.ones(B, device=device).view(B, 1, 1).expand(B, seq_len, 1))

            # 计算块数
            num_blocks_per_ans = (81 + self.block_size - 1) // self.block_size
            total_ans_blocks = self.n_respond * num_blocks_per_ans

            # 🎯 动态解除遮蔽：根据 masked_indices 计算 masked_block_indices
            # 初始化为全部 unmasked（False）
            masked_block_indices = torch.zeros((B, total_ans_blocks), dtype=torch.bool, device=device)

            # 对于每个 block，如果有任何 cell 仍被 mask，则标记该 block 为 masked
            for b in range(B):
                for blk_idx in range(num_blocks_per_ans):
                    # 计算该 block 对应的 cell 范围
                    start_cell = blk_idx * self.block_size
                    end_cell = min(start_cell + self.block_size, 81)
                    # 如果该 block 中有任何 cell 仍被 mask，标记为 masked
                    if masked_indices[b, start_cell:end_cell].any():
                        masked_block_indices[b, blk_idx] = True

            # Attention bias
            attention_bias = self._create_bad_ar_attention_bias(
                B, n_points, self.n_prompt, self.block_size, masked_block_indices, device, embeds.dtype
            )

            # Backbone forward
            dummy = torch.zeros(B, seq_len, dtype=torch.long, device=device)
            h = self._backbone(
                input_ids=dummy,
                input_embeddings=embeds,
                attention_bias=attention_bias,
                output_hidden_states=True
            ).hidden_states[-1]
            vocab_logits = self._read_out(h)

            # 提取 Answer 部分的 logits
            ans_logits = vocab_logits[:, answer_start:answer_end, :]  # [B, 81, 12]
            digit_logits = self._vocab_logits_to_digit_logits(ans_logits)  # [B, 1, 81, 10]
            digit_logits = digit_logits.squeeze(1)  # [B, 81, 10]
            final_logits = digit_logits.unsqueeze(1)  # [B, 1, 81, 10]

            # 计算置信度
            probs = torch.softmax(digit_logits, dim=-1)
            if self.inference_confidence_alg == "entropy":
                confidence = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [B, 81]
            else:
                confidence = -probs.max(dim=-1)[0]  # [B, 81]

            confidence = confidence.masked_fill(~masked_indices, float("inf"))

            # 根据 scheduler 决定本步 unmask 多少个位置
            for b in range(B):
                k = num_transfer_tokens[b, step].item()
                if k == 0:
                    continue

                unfilled = masked_indices[b].nonzero(as_tuple=True)[0]
                if unfilled.numel() == 0:
                    continue

                k = min(k, unfilled.numel())
                unfilled_confidence = confidence[b, unfilled]
                _, topk = torch.topk(unfilled_confidence, k=k, largest=False)
                cells = unfilled[topk]
                pred_digits = torch.argmax(digit_logits[b, cells, :], dim=-1)
                # 🆕 转换为 Nebula tokens 并更新序列
                pred_tokens = self._digits_to_nebula_tokens(pred_digits.unsqueeze(0)).squeeze(0)
                input_ids[b, answer_start + cells] = pred_tokens
                masked_indices[b, cells] = False

        if final_logits is None:
            # Final forward pass
            embeds = self._read_in(input_ids)
            embeds = self._inject_sudoku_coords(embeds, n_points, device)
            embeds = embeds + self._time_mlp(torch.ones(B, device=device).view(B, 1, 1).expand(B, seq_len, 1))

            num_blocks_per_ans = (81 + self.block_size - 1) // self.block_size
            total_ans_blocks = self.n_respond * num_blocks_per_ans

            # 🎯 动态解除遮蔽：根据 masked_indices 计算 masked_block_indices
            masked_block_indices = torch.zeros((B, total_ans_blocks), dtype=torch.bool, device=device)
            for b in range(B):
                for blk_idx in range(num_blocks_per_ans):
                    start_cell = blk_idx * self.block_size
                    end_cell = min(start_cell + self.block_size, 81)
                    if masked_indices[b, start_cell:end_cell].any():
                        masked_block_indices[b, blk_idx] = True

            attention_bias = self._create_bad_ar_attention_bias(
                B, n_points, self.n_prompt, self.block_size, masked_block_indices, device, embeds.dtype
            )

            dummy = torch.zeros(B, seq_len, dtype=torch.long, device=device)
            h = self._backbone(
                input_ids=dummy,
                input_embeddings=embeds,
                attention_bias=attention_bias,
                output_hidden_states=True
            ).hidden_states[-1]
            vocab_logits = self._read_out(h)
            ans_logits = vocab_logits[:, answer_start:answer_end, :]
            final_logits = self._vocab_logits_to_digit_logits(ans_logits)  # [B, 1, 81, 10]

        mask = torch.zeros(B, 1, 81, device=device, dtype=torch.bool)
        return final_logits, mask
    @torch.no_grad()
    def generate(self, prefix, max_new_tokens=81, **kwargs):
        """
        Generate Sudoku solution using BAD-AR inference
        """
        if isinstance(prefix, dict):
            xs = prefix["xs"]
            ys = prefix["ys"]
            respond_position_mask = prefix.get("respond_position_mask", None)
        elif isinstance(prefix, tuple):
            xs, ys = prefix[0], prefix[1]
            respond_position_mask = prefix[2] if len(prefix) > 2 else None
        else:
            raise TypeError("SudokuBADAR.generate expects prefix as dict or tuple (xs,ys[,mask]).")
        
        out = self.forward(xs, ys, train_mode=False, respond_position_mask=respond_position_mask)
        pred_logits = out[0] if isinstance(out, tuple) else out
        pred_digits = torch.argmax(pred_logits[:, -1, :, :], dim=-1)
        return pred_digits


class SudokuDream(nn.Module):
    """
    Sudoku Dream (MDM) model aligned with dllm-pathfinding/core-nebula DreamDlmModel logic.

    Key alignment points:
    - Tokenization compatible with core-nebula SudokuTokenizer:
        tokens: '1'..'9', '$', '=' plus an extra [MASK] token.
      We map input digits:
        - quiz: 0 -> '$', 1..9 -> '1'..'9'
        - solution: 1..9 -> '1'..'9'
    - Training (diffusion):
        build full_sequence = prefix(q+'=') + target(solution)
        sample continuous t, compute p_mask from LinearAlphaScheduler
        mask only target part
        CE loss with ignore_index=-100 (same masking convention)
    - Inference:
        use DreamSampler.sample(..., steps=...) for multi-step sampling (core-nebula style).
        return 10-class logits over digits 0..9 (0 is never valid for solution; given -inf).
    """

    def __init__(
        self,
        n_positions=256,
        n_embd=256,
        n_layer=12,
        n_head=8,
        n_prompt=0,
        n_respond=1,
        *,
        # Dream sampling / inference
        use_multistep_inference=True,
        inference_steps=50,
        inference_alg="entropy",
        inference_temperature=0.0,
        inference_top_k=50,
        inference_top_p=1.0,
        # diffusion masking
        time_epsilon=1e-3,
        # 🏗️ Logits Shift 控制（对齐 v3 分支）
        apply_logits_shift=True,  # 是否应用 Logits Shift（默认 True，对齐 v3）
        verify_shift_alignment=True,  # 是否验证 Shift 对齐（开发阶段启用）
        **extra,
    ):
        super().__init__()
        if DreamConfig is None or _DreamBase is None or LinearAlphaScheduler is None or DreamSampler is None:
            raise ImportError("Dream dependencies not available (dllm Dream). Please ensure dllm is importable.")

        self.family = "sudoku_dream"
        self.name = "sudoku_dream"
        self.n_positions = n_positions
        self.n_prompt = n_prompt
        self.n_respond = n_respond

        if self.n_respond != 1:
            raise NotImplementedError("SudokuDream currently supports n_respond=1 only.")

        # ICL context length check:
        # Each full example contributes 163 tokens: 81(Q)+1('=')+81(A)
        # Target example contributes 82 prefix (Q+'=') + 81 answer tokens.
        required_len = int(self.n_prompt) * 163 + 163
        if int(self.n_positions) < required_len:
            # Keep backward compatible by auto-expanding positions for Dream backbone.
            self.n_positions = required_len

        # Core-nebula SudokuTokenizer vocab: 11 tokens ('1'..'9','$','=')
        # We add one extra [MASK] token at the end (like core-nebula DreamDlmModel).
        self.base_vocab_size = 11
        self.vocab_size = self.base_vocab_size + 1
        self.mask_token_id = self.vocab_size - 1  # last
        # In core-nebula they set pad_token_id=vocab_size-2 (works for their setup); keep aligned.
        self.pad_token_id = self.vocab_size - 2

        self.time_epsilon = float(time_epsilon)
        self.scheduler = LinearAlphaScheduler()

        # 🏗️ Logits Shift 控制参数
        self.apply_logits_shift = bool(apply_logits_shift)
        self.verify_shift_alignment = bool(verify_shift_alignment)
        
        if not self.apply_logits_shift:
            import warnings
            warnings.warn(
                "apply_logits_shift=False: Logits shift is disabled. "
                "This may cause misalignment with v3 branch and degrade performance. "
                "Only disable for debugging purposes."
            )

        self.use_multistep_inference = bool(use_multistep_inference)
        self.inference_steps = int(inference_steps)
        self.inference_alg = str(inference_alg)
        self.inference_temperature = float(inference_temperature)
        self.inference_top_k = int(inference_top_k)
        self.inference_top_p = float(inference_top_p)

        cfg = DreamConfig(
            vocab_size=self.vocab_size,
            max_position_embeddings=int(self.n_positions),
            hidden_size=int(n_embd),
            intermediate_size=int(4 * n_embd),
            num_hidden_layers=int(n_layer),
            num_attention_heads=int(n_head),
            num_key_value_heads=int(n_head),
            use_cache=False,
            pad_token_id=self.pad_token_id,
            mask_token_id=self.mask_token_id,
            # eos_token_id is required by some generation utils; keep default-ish
            eos_token_id=0,
        )
        self._backbone = _DreamBase(cfg)

    @staticmethod
    def _digits_to_nebula_tokens(digits: torch.Tensor) -> torch.Tensor:
        """
        Map digits 0..9 to core-nebula sudoku tokens:
          1..9 -> 0..8
          0 -> 9 ('$')
        
        This mapping is equivalent to SudokuTokenizer.encode() when:
        - Input digits are converted to strings with '0' -> '$'
        - Then encoded using SudokuTokenizer
        
        Args:
            digits: Tensor of shape [..., 81] with values in [0, 9]
        
        Returns:
            tokens: Tensor of same shape with values in [0, 10]
                - 0..8: digits 1..9
                - 9: digit 0 (blank, represented as '$')
                - 10: '=' (added separately in _build_full_sequence)
        """
        # digits: [..., 81]
        # 🏗️ 类型检查：确保输入是整数类型
        if digits.dtype not in (torch.long, torch.int32, torch.int64):
            digits = digits.long()
        
        out = digits.clone()
        # 🏗️ 修复：使用 mask 区分原始的 0 和 1..9，避免冲突
        # 先标记原始的 0
        mask_zero = (out == 0)
        # 1..9 -> 0..8 (对非 0 的值减 1)
        mask_nonzero = ~mask_zero
        out[mask_nonzero] = out[mask_nonzero] - 1
        # 0 -> '$' (id 9) (原始的 0 转换为 9)
        out[mask_zero] = 9
        return out.long()

    def _build_full_sequence(self, xs: torch.Tensor, ys: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Build ICL-style core-nebula token sequence:
          (Q1 '=' A1) (Q2 '=' A2) ... (Q_prompt '=' A_prompt) (Q_target '=') (A_target)

        - Each complete example length: 163
        - Target prefix length: 82

        Returns:
          full_sequence: [B, n_prompt*163 + 163]
          prefix_len: n_prompt*163 + 82
        """
        B, n_points, _ = xs.shape
        assert n_points == self.n_prompt + self.n_respond, "SudokuDream expects n_points == n_prompt + n_respond"

        device = xs.device
        eq = torch.full((B, 1), 10, dtype=torch.long, device=device)  # '=' token id in base vocab

        parts = []
        # prompt examples: full (Q '=' A)
        for i in range(n_points - 1):
            q_i = self._digits_to_nebula_tokens(xs[:, i, :].long())
            a_i = self._digits_to_nebula_tokens(ys[:, i, :].long())
            parts.extend([q_i, eq, a_i])

        # target example: (Q '=') + A
        q_t = self._digits_to_nebula_tokens(xs[:, -1, :].long())
        a_t = self._digits_to_nebula_tokens(ys[:, -1, :].long())

        prefix_tokens = torch.cat(parts + [q_t, eq], dim=1)
        prefix_len = int(prefix_tokens.shape[1])
        full = torch.cat([prefix_tokens, a_t], dim=1)
        return full, prefix_len

    @torch.no_grad()
    def _dream_sample(self, prefix: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Call DreamSampler.sample like core-nebula DreamDlmModel.generate.
        """
        class _TokenizerWrapper:
            def __init__(self, mask_token_id, eos_token_id):
                self.mask_token_id = mask_token_id
                self.eos_token_id = eos_token_id

        tok = _TokenizerWrapper(mask_token_id=self.mask_token_id, eos_token_id=self._backbone.config.eos_token_id)
        sampler = DreamSampler(model=self._backbone, tokenizer=tok)
        prompts = [p for p in prefix]
        # IMPORTANT: DreamSampler currently assumes fp32 confidence tensors internally.
        # When running under bf16 autocast (e.g. accelerate), dtype mismatches can occur.
        # Force disable autocast here to keep sampler internals consistent.
        with torch.autocast(device_type=prefix.device.type, enabled=False):
            out = sampler.sample(
                inputs=prompts,
                steps=self.inference_steps,
                alg=self.inference_alg,
                temperature=self.inference_temperature,
                top_k=self.inference_top_k,
                top_p=self.inference_top_p,
                max_new_tokens=max_new_tokens,
                return_dict=False,
            )
        return out

    def generate(self, prefix, **kwargs):
        """
        Core-nebula compatible generate() API.
        Uses DreamSampler multi-step sampling to generate 81 target tokens.
        """
        max_new_tokens = kwargs.get("max_new_tokens", 81)
        # prefix: [B, prefix_len]
        return self._dream_sample(prefix, max_new_tokens=max_new_tokens)

    def forward(self, xs, ys=None, train_mode=True, respond_position_mask=None, task_type=None, **kwargs):
        """
        Dual-interface forward for compatibility:

        1) ICL-style (this repo):
           forward(xs=[B,n_points,81], ys=[B,n_points,81], train_mode=..., respond_position_mask=None)
           Returns sudoku-style logits [B,1,81,10] (+mask) for eval, and (loss, logits, _, mask) for train.

        2) core-nebula DreamDlmModel-style:
           forward(input_ids=[B,L], task_type='diffusion') -> logits [B,L,vocab]
           Used by core_nebula_dream_trainer training loop.
        """
        # Interface (2): core-nebula style
        if ys is None and task_type is not None:
            # xs is actually input_ids
            input_ids = xs
            return self._backbone(input_ids=input_ids).logits

        # Interface (1): ICL-style
        # respond_position_mask not supported in this core-nebula aligned mode
        if respond_position_mask is not None:
            raise NotImplementedError("SudokuDream does not support respond_position_mask (core-nebula aligned).")

        device = xs.device
        full_sequence, prefix_len = self._build_full_sequence(xs, ys)  # [B,163], 82
        B, seq_len = full_sequence.shape

        if train_mode:
            # diffusion masking on token sequence (core-nebula style)
            t = self.time_epsilon + (1.0 - self.time_epsilon) * torch.rand(B, device=device)
            p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(B, seq_len)

            target_part_mask = torch.zeros_like(full_sequence, dtype=torch.bool)
            # 🏗️ 对齐 v3 分支：使用 prefix_len 而不是硬编码 -81
            # 对于 n_prompt=0: prefix_len=82, 所以 prefix_len: = [-81:] (等价)
            # 但使用 prefix_len 更通用，支持 n_prompt>0 的情况
            target_part_mask[:, prefix_len:] = True

            masked_indices = (torch.rand_like(full_sequence, dtype=torch.float32) < p_mask) & target_part_mask
            noised_input_ids = full_sequence.clone()
            noised_input_ids[masked_indices] = self.mask_token_id

            diffusion_ys = full_sequence.clone()
            diffusion_ys[~masked_indices] = -100

            raw_logits = self._backbone(input_ids=noised_input_ids).logits  # [B, L, vocab]
            
            # 🏗️ Logits Shift: 对齐 Transformer 的 Off-by-one 特性
            # 在标准 Transformer 中，位置 i 的输出（Logits）是用来预测位置 i+1 的 Token 的
            # Shift 操作将 Logits 右移一位，使得位置 i 的预测能与标签序列中的位置 i 正确对齐
            # 这与 v3 分支的实现完全一致：torch.cat([output[:, :1], output[:, :-1]], dim=1)
            if self.apply_logits_shift:
                # v3 对齐的 Shift 操作
                shifted_logits = torch.cat([raw_logits[:, :1], raw_logits[:, :-1]], dim=1)  # [B,L,vocab]
                
                # 🏗️ 验证 Shift 对齐（开发阶段）
                if self.verify_shift_alignment:
                    # 验证 Shift 操作的正确性
                    # 1. 形状验证：shifted_logits 应该与 full_sequence 长度一致
                    assert shifted_logits.shape[1] == seq_len, \
                        f"Shifted logits length {shifted_logits.shape[1]} != sequence length {seq_len}"
                    assert shifted_logits.shape[0] == B, \
                        f"Shifted logits batch size {shifted_logits.shape[0]} != batch size {B}"
                    # 2. Shift 逻辑验证：shifted_logits[i] 应该预测 full_sequence[i]
                    #    - shifted_logits[0] = raw_logits[0] (第一个位置保持不变)
                    #    - shifted_logits[i] = raw_logits[i-1] for i > 0 (右移一位)
                    #    - 这确保了位置 i 的 logits 预测位置 i 的 token（而不是 i+1）
                    assert torch.equal(shifted_logits[:, 0, :], raw_logits[:, 0, :]), \
                        "First position of shifted_logits should equal raw_logits[0]"
                    if seq_len > 1:
                        assert torch.equal(shifted_logits[:, 1:, :], raw_logits[:, :-1, :]), \
                            "Shifted logits[1:] should equal raw_logits[:-1] (right shift by 1)"
            else:
                # 不使用 Shift（仅用于调试，不推荐）
                # ⚠️ 警告：禁用 Shift 会导致预测和标签错位，严重影响性能
                shifted_logits = raw_logits
            
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(shifted_logits.reshape(-1, shifted_logits.size(-1)), diffusion_ys.reshape(-1))

            # Return sudoku-style logits over the 81 solution tokens for monitoring/training loop compatibility.
            # Always the last 81 tokens correspond to target answer (ICL compatible).
            # 🏗️ 对齐 v3 分支：使用 prefix_len 而不是硬编码 -81
            sol_vocab_logits = shifted_logits[:, prefix_len:, :]  # [B,81,vocab]
            sol_logits = self._vocab_logits_to_digit_logits(sol_vocab_logits)  # [B,1,81,10]

            # Mask over which solution cells were actually masked (for fair accuracy computation if desired)
            # 🏗️ 对齐 v3 分支：使用 prefix_len 而不是硬编码 -81
            sol_mask = masked_indices[:, prefix_len:]  # [B,81]
            sol_mask = sol_mask.unsqueeze(1)  # [B,1,81]
            return loss, sol_logits, None, sol_mask

        # Inference
        prefix = full_sequence[:, :prefix_len]  # [B, n_prompt*163 + 82]
        if self.use_multistep_inference:
            generated = self._dream_sample(prefix, max_new_tokens=81)  # expects full sequence returned
            # DreamSampler may return right-aligned canvas; keep last full sequence length
            if generated.shape[1] >= full_sequence.shape[1]:
                generated = generated[:, -full_sequence.shape[1] :]
            target_tokens = generated[:, -81:]
            pred_logits = self._tokens_to_digit_logits(target_tokens)
            mask = torch.zeros(B, 1, 81, device=device, dtype=torch.bool)
            return pred_logits, mask

        # single-step: run model once with masked target and return logits (not recommended)
        noised = full_sequence.clone()
        noised[:, -81:] = self.mask_token_id
        logits = self._backbone(input_ids=noised).logits  # [B,seq_len,vocab]
        # take target token logits positions (predict next token at each position)
        target_pos = torch.arange(seq_len - 81, seq_len, device=device)
        # prediction for token at position i comes from logits at i-1 due to LM shift
        pred_token_logits = logits[:, target_pos - 1, :]  # [B,81,vocab]
        pred_logits = self._vocab_logits_to_digit_logits(pred_token_logits)
        mask = torch.ones(B, 1, 81, device=device, dtype=torch.bool)
        return pred_logits, mask

    def _tokens_to_digit_logits(self, tokens_81: torch.Tensor) -> torch.Tensor:
        """
        Convert predicted token IDs (core-nebula vocab) to 10-class logits over digits 0..9.
        - digit 0 is invalid for solutions -> very negative logits
        - digits 1..9 map from token ids 0..8
        """
        B = tokens_81.shape[0]
        logits = torch.full((B, 1, 81, 10), -1e9, device=tokens_81.device, dtype=torch.float32)
        # token 0..8 => digit 1..9
        digit = tokens_81.long() + 1
        # '$'(9) or '='(10) or mask(11) are invalid -> keep as -inf
        valid = (tokens_81 >= 0) & (tokens_81 <= 8)
        idx_b = torch.arange(B, device=tokens_81.device)[:, None]
        idx_c = torch.arange(81, device=tokens_81.device)[None, :]
        # set logits high for valid digits
        logits[idx_b, 0, idx_c, digit.clamp(0, 9)] = torch.where(valid, torch.tensor(1e3, device=tokens_81.device), torch.tensor(-1e9, device=tokens_81.device))
        # ensure digit 0 stays invalid
        logits[:, :, :, 0] = -1e9
        return logits

    def _vocab_logits_to_digit_logits(self, vocab_logits: torch.Tensor) -> torch.Tensor:
        """
        Convert vocab logits [B,81,vocab] to digit logits [B,1,81,10].
        """
        B = vocab_logits.shape[0]
        out = torch.full((B, 1, 81, 10), -1e9, device=vocab_logits.device, dtype=vocab_logits.dtype)
        # digit 1..9 from token 0..8
        out[:, 0, :, 1:10] = vocab_logits[:, :, 0:9]
        out[:, 0, :, 0] = -1e9
        return out

def build_sudoku_model(conf):
    """
    构建数独模型

    Args:
        conf: 模型配置字典

    Returns:
        数独模型实例
    """
    family = conf.get('family', '')

    if family in ['llada', 'sudoku_llada']:
        return SudokuLLaDA(**conf)
    elif family == 'sudoku_dream':
        return SudokuDream(**conf)
    elif family == 'sudoku_ar':
        return SudokuAR(**conf)
    elif family == 'sudoku_llada_block':
        return SudokuLLaDABlock(**conf)
    elif family == 'sudoku_bopar':
        return SudokuBOPAR(**conf)
    elif family == 'sudoku_rboar':
        return SudokuRBOAR(**conf)
    elif family == 'sudoku_badar':
        return SudokuBADAR(**conf)
    else:
        raise NotImplementedError(f"不支持的模型类型: {family}")
