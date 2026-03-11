"""
路径规划专用模型（基于 Sudoku 风格）
==========================================

核心特性：
1. Role Embedding：区分边列表、查询、路径三种角色
2. 离散时间步训练：采样 t ∈ [0, 19]，使用 Focal Loss 重加权
3. 分类任务：输出节点 ID 的 logits
4. 支持 BPD-AR 推理（熵引导的自适应推理）

协议结构：
- Edge List: (l-1)*d*2 tokens (所有边平铺，l个节点需要(l-1)*d条边)
- Sep token '/': 1 token
- Query: 2 tokens (起点和终点)
- Eq token '=': 1 token
- Path: l tokens (目标路径序列，l个节点)

UNIT_LEN = (l-1)*d*2 + 1 + 2 + 1 + l

词表：
- 节点 ID: 0 到 N-1
- Sep '/': N
- Eq '=': N+1
- MASK: N+2
- 总 vocab_size: N+3
"""

import torch
from torch import nn
import torch.nn.functional as F
import math


def _filter_parent_kwargs(extra_kwargs):
    """
    过滤掉父类不支持的参数

    父类（TransformerModelPromptRespond, LLaDAPromptRespond等）不接受以下参数：
    - family: 路径规划模型的family标识
    - num_nodes, degree, path_len: 路径规划特定参数
    - use_role_embedding, loss_mode, alpha, gamma, num_timesteps: 路径规划特定参数
    - block_size: 某些父类不接受（会在子类中单独处理）
    - use_bpd_inference, bpd_steps, bpd_k_per_step: BPD推理参数

    Args:
        extra_kwargs: 额外的关键字参数字典

    Returns:
        过滤后的参数字典
    """
    # 需要过滤的参数列表
    pathfinding_specific_params = {
        'family', 'num_nodes', 'degree', 'path_len',
        'use_role_embedding', 'loss_mode', 'alpha', 'gamma', 'num_timesteps',
        'use_bpd_inference', 'bpd_steps', 'bpd_k_per_step'
    }

    # 过滤参数
    filtered = {k: v for k, v in extra_kwargs.items() if k not in pathfinding_specific_params}
    return filtered


from models_prompt_respond import (
    LLaDAPromptRespond,
    TransformerModelPromptRespond,
    LLaDABlockDiffusion,
    BOPARPromptRespond,
    RBOARPromptRespond,
    BADARPromptRespond,
)


class PathfindingRoleEmbedding(nn.Module):
    """
    路径规划 Role 编码器

    为序列中的不同部分注入角色信息：
    - Role 0: 边列表 (Edges) - 无序/乱序
    - Role 1: 查询 (Query) - 包含起点/终点
    - Role 2: 目标路径 (Path) - 严格有序

    仅为 Role 2 (Path) 提供位置编码，因为路径是有序的。
    """

    def __init__(self, n_embd, max_path_len):
        super().__init__()
        # 三种角色：Edges, Query, Path
        self.role_emb = nn.Embedding(3, n_embd)
        # 仅为 Path 提供位置编码
        self.path_pos_emb = nn.Embedding(max_path_len, n_embd)
        self.max_path_len = max_path_len

    def forward(self, d, l, device):
        """
        计算单个 unit 的 role embedding

        Args:
            d: 分支数 (degree)
            l: 路径长度
            device: torch device

        Returns:
            role_emb: [UNIT_LEN, n_embd] role embeddings
            pos_emb: [UNIT_LEN, n_embd] position embeddings (仅 Path 部分非零)
        """
        # 计算 UNIT_LEN
        edge_len = (l - 1) * d * 2  # 修正：l 表示路径节点数，边数为 (l-1) * d
        UNIT_LEN = edge_len + 1 + 2 + 1 + l  # 修正：路径有 l 个节点

        # 1. 计算 Role IDs
        role_ids = torch.zeros(UNIT_LEN, device=device, dtype=torch.long)
        edge_end = edge_len
        query_start = edge_end + 1  # 跳过 '/'
        query_end = query_start + 2
        path_start = query_end + 1  # 跳过 '='

        # Role 0: Edges
        role_ids[:edge_end] = 0
        # Role 1: Query (包括 '/' 和起点终点)
        role_ids[edge_end:query_end] = 1
        # Role 2: Path (包括 '=' 和路径节点)
        role_ids[query_end:] = 2

        # 2. 计算 Path Position IDs (仅 Path 部分)
        pos_ids = torch.zeros(UNIT_LEN, device=device, dtype=torch.long)
        path_len = l  # 修正：l 表示路径节点数
        pos_ids[path_start:path_start + path_len] = torch.arange(path_len, device=device)

        # 3. 获取 embeddings
        role_emb = self.role_emb(role_ids)  # [UNIT_LEN, n_embd]
        pos_emb = self.path_pos_emb(pos_ids)  # [UNIT_LEN, n_embd]

        return role_emb, pos_emb


# ============================================================
# PathfindingMixin: 封装路径规划特有的逻辑
# ============================================================

class PathfindingMixin:
    """
    路径规划模型 Mixin 类，封装所有共有逻辑

    提供：
    - 统一的词表设置 (vocab_size = N + 3)
    - 序列构建
    - Role 编码注入
    - Loss 计算
    """

    def setup_pathfinding_protocol(
        self,
        n_embd: int,
        num_nodes: int,
        max_path_len: int,
        use_role_embedding: bool = True
    ):
        """
        设置路径规划协议相关的组件

        Args:
            n_embd: embedding 维度
            num_nodes: 图中节点总数 N
            max_path_len: 最大路径长度
            use_role_embedding: 是否使用 role embedding
        """
        self.num_nodes = num_nodes
        self.vocab_size = num_nodes + 3  # 节点 + '/' + '=' + MASK
        self.sep_token_id = num_nodes      # '/'
        self.eq_token_id = num_nodes + 1   # '='
        self.mask_token_id = num_nodes + 2 # MASK

        # 输入输出层
        self._read_in = nn.Embedding(self.vocab_size, n_embd)
        self._read_out = nn.Linear(n_embd, self.vocab_size)

        # Role 编码（可选）
        self.use_role_embedding = use_role_embedding
        if self.use_role_embedding:
            self.role_emb = PathfindingRoleEmbedding(n_embd, max_path_len)
        else:
            self.role_emb = None

        # MASK token embedding
        self.mask_embedding = nn.Parameter(torch.randn(1, n_embd))

    def _compute_unit_len(self, d: int, l: int) -> int:
        """
        计算单个 unit 的长度

        Args:
            d: 分支数
            l: 路径长度（节点数）

        Returns:
            UNIT_LEN: 单个 unit 的 token 数量
        """
        # 修正：l 表示路径节点数，不是边数
        # edges: (l-1) * d * 2, sep: 1, query: 2, eq: 1, path: l
        return (l - 1) * d * 2 + 1 + 2 + 1 + l

    def _build_pathfinding_sequence(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        d: int,
        l: int,
        device: torch.device
    ) -> tuple[torch.Tensor, int]:
        """
        构建路径规划序列

        Args:
            xs: [B, n_points, edge_len + 2] 边列表 + 起点终点
            ys: [B, n_points, l] 目标路径（l个节点）
            d: 分支数
            l: 路径长度（节点数）
            device: torch device

        Returns:
            (full_sequence, prefix_len)
            full_sequence: [B, n_prompt*UNIT_LEN + UNIT_LEN]
            prefix_len: prompt 部分 + 当前 unit 的 edges/query/eq 的长度
        """
        B = xs.shape[0]
        n_prompt = self.n_prompt
        n_respond = self.n_respond

        edge_len = (l - 1) * d * 2  # 修正：l 表示路径节点数，边数为 (l-1) * d
        UNIT_LEN = self._compute_unit_len(d, l)

        # 分离 xs 为 edges 和 query
        edges = xs[:, :, :edge_len]  # [B, n_points, edge_len]
        query = xs[:, :, edge_len:]  # [B, n_points, 2]

        # 构建完整序列
        units = []
        for i in range(n_prompt + n_respond):
            # Edges
            unit = [edges[:, i, :]]  # [B, edge_len]
            # Sep '/'
            unit.append(torch.full((B, 1), self.sep_token_id, device=device, dtype=torch.long))
            # Query (start, goal)
            unit.append(query[:, i, :])  # [B, 2]
            # Eq '='
            unit.append(torch.full((B, 1), self.eq_token_id, device=device, dtype=torch.long))
            # Path (l个节点)
            unit.append(ys[:, i, :])  # [B, l]

            units.append(torch.cat(unit, dim=1))  # [B, UNIT_LEN]

        full_sequence = torch.cat(units, dim=1)  # [B, (n_prompt+n_respond)*UNIT_LEN]

        # prefix_len: prompt 部分 + 当前 respond unit 的 edges/query/eq
        prefix_len = n_prompt * UNIT_LEN + edge_len + 1 + 2 + 1

        return full_sequence, prefix_len

    def _inject_pathfinding_roles(
        self,
        embeds: torch.Tensor,
        d: int,
        l: int,
        total_points: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        将 role embedding 注入到序列中

        Args:
            embeds: [B, seq_len, n_embd]
            d: 分支数
            l: 路径长度
            total_points: 总点数
            device: torch device

        Returns:
            embeds: [B, seq_len, n_embd] with role embeddings injected
        """
        if not self.use_role_embedding or self.role_emb is None:
            return embeds

        UNIT_LEN = self._compute_unit_len(d, l)

        # 获取单个 unit 的 role embedding
        role_emb, pos_emb = self.role_emb(d, l, device)  # [UNIT_LEN, n_embd]

        # 重复到所有 units
        full_role_emb = role_emb.unsqueeze(0).repeat(total_points, 1, 1)  # [total_points, UNIT_LEN, n_embd]
        full_role_emb = full_role_emb.reshape(1, -1, role_emb.shape[-1])  # [1, total_points*UNIT_LEN, n_embd]

        full_pos_emb = pos_emb.unsqueeze(0).repeat(total_points, 1, 1)
        full_pos_emb = full_pos_emb.reshape(1, -1, pos_emb.shape[-1])

        # 注入到 embeddings
        embeds = embeds + full_role_emb + full_pos_emb

        return embeds

    def _compute_pathfinding_loss(
        self,
        logits: torch.Tensor,
        target_path: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor = None,
        mode: str = "ce",
        alpha: float = 0.25,
        gamma: float = 1.0,
        num_timesteps: int = 20,
    ) -> torch.Tensor:
        """
        计算路径规划 Loss

        Args:
            logits: [B, path_len, vocab_size] 预测 logits
            target_path: [B, path_len] 目标路径节点 ID
            mask: [B, path_len] boolean mask
            t: [B] timestep tensor (for composite loss)
            mode: "ce" or "composite"
            alpha: Focal Loss 参数
            gamma: Focal Loss 参数
            num_timesteps: 时间步数

        Returns:
            loss: scalar tensor
        """
        B, path_len, vocab_size = logits.shape

        # Flatten
        logits_flat = logits.reshape(-1, vocab_size)  # [B*path_len, vocab_size]
        target_flat = target_path.reshape(-1)  # [B*path_len]
        mask_flat = mask.reshape(-1)  # [B*path_len]

        if mode == "ce":
            # 标准 Cross-Entropy
            ce_loss = F.cross_entropy(logits_flat, target_flat, reduction='none')
            ce_loss = ce_loss * mask_flat
            loss = ce_loss.sum() / (mask_flat.sum() + 1e-8)
        else:
            # Composite loss: CE + Focal + Time weighting
            ce_loss = F.cross_entropy(logits_flat, target_flat, reduction='none')

            # Focal Loss weighting
            probs = F.softmax(logits_flat, dim=-1)
            target_probs = probs[torch.arange(len(target_flat)), target_flat]
            focal_weight = (1 - target_probs) ** gamma

            # Time weighting: w(t) = (T - t) / T
            if t is not None:
                time_weight = (num_timesteps - t.float()) / num_timesteps
                time_weight = time_weight.unsqueeze(1).expand(B, path_len).reshape(-1)
            else:
                time_weight = torch.ones_like(mask_flat)

            # Combined loss
            weighted_loss = alpha * focal_weight * time_weight * ce_loss * mask_flat
            loss = weighted_loss.sum() / (mask_flat.sum() + 1e-8)

        return loss


# ============================================================
# PathfindingAR: 自回归模型
# ============================================================

class PathfindingAR(TransformerModelPromptRespond, PathfindingMixin):
    """
    路径规划专用 AR 模型（基于 TransformerModelPromptRespond + PathfindingMixin）

    特点：
    1. 因果注意力（causal attention）
    2. 输入：节点 ID token embeddings
    3. 输出：节点 ID 的 logits
    4. Loss：仅计算 Path 部分的 Cross-Entropy
    """

    def __init__(
        self,
        n_dims,  # 路径规划：edge_len + 2 (edges + query)
        n_positions=1000,
        n_embd=256,
        n_layer=12,
        n_head=8,
        n_prompt=5,
        n_respond=1,
        *,
        mlp_ratio=4.0,
        type="llama",  # 默认使用 llama backbone
        # 路径规划特定参数
        num_nodes=100,  # 图中节点总数
        degree=5,  # 分支数 d
        path_len=10,  # 路径长度 l
        use_role_embedding=True,
        loss_mode="ce",  # "ce" or "composite"
        alpha=0.25,
        gamma=1.0,
        num_timesteps=20,
        **extra,
    ):
        # 过滤掉父类不支持的参数
        parent_kwargs = _filter_parent_kwargs(extra)

        # 调用父类初始化（显式传递 type 参数）
        super().__init__(
            n_dims=n_dims,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_prompt=n_prompt,
            n_respond=n_respond,
            mlp_ratio=mlp_ratio,
            type=type,  # 显式传递 type 参数
            **parent_kwargs,
        )

        self.name = "pathfinding_ar"
        self.family = "pathfinding_ar"  # 用于评估代码识别任务类型
        self.degree = degree
        self.path_len = path_len
        self.loss_mode = loss_mode
        self.alpha = alpha
        self.gamma = gamma
        self.num_timesteps = num_timesteps

        # 使用 Mixin 设置路径规划协议
        self.setup_pathfinding_protocol(
            n_embd=n_embd,
            num_nodes=num_nodes,
            max_path_len=path_len,  # 修正：path_len 表示节点数
            use_role_embedding=use_role_embedding
        )

        print(f"[PathfindingAR] Initialized:")
        print(f"  vocab_size: {self.vocab_size} (0-{num_nodes-1}: nodes, {num_nodes}: '/', {num_nodes+1}: '=', {num_nodes+2}: MASK)")
        print(f"  degree: {degree}, path_len: {path_len}")
        print(f"  Role Embedding: {'enabled' if use_role_embedding else 'disabled'}")
        print(f"  Loss mode: {loss_mode}")

    def forward(self, xs, ys, train_mode=True, respond_position_mask=None):
        """
        前向传播

        Args:
            xs: [B, n_points, edge_len + 2] 边列表 + 起点终点
            ys: [B, n_points, path_len] 目标路径
            train_mode: 训练模式（保持接口一致）
            respond_position_mask: respond位置mask（保持接口一致）

        Returns:
            logits: [B, n_respond, path_len, vocab_size]
            loss: scalar tensor
        """
        B = xs.shape[0]
        device = xs.device
        d = self.degree
        l = self.path_len

        # 构建完整序列
        full_sequence, prefix_len = self._build_pathfinding_sequence(xs, ys, d, l, device)

        # Token embeddings
        embeds = self._read_in(full_sequence)  # [B, seq_len, n_embd]

        # 注入 role embeddings
        total_points = self.n_prompt + self.n_respond
        embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)

        # Backbone forward
        out = self._backbone(
            inputs_embeds=embeds,  # PathfindingAR 使用 Transformer backbone，保持 inputs_embeds
            output_hidden_states=True,
        )
        h = out.hidden_states[-1]  # [B, seq_len, n_embd]

        # 输出 logits
        vocab_logits = self._read_out(h)  # [B, seq_len, vocab_size]

        # 提取 Path 部分的 logits
        UNIT_LEN = self._compute_unit_len(d, l)
        path_logits_list = []
        for i in range(self.n_respond):
            unit_start = (self.n_prompt + i) * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1  # edges + '/' + query + '='
            path_end = path_start + l
            path_logits = vocab_logits[:, path_start:path_end, :]  # [B, l, vocab_size]
            path_logits_list.append(path_logits)

        logits = torch.stack(path_logits_list, dim=1)  # [B, n_respond, l, vocab_size]

        if train_mode:
            # 训练模式：计算 loss
            target_path = ys[:, self.n_prompt:, :]  # [B, n_respond, l]
            mask = torch.ones_like(target_path, dtype=torch.float32)  # 全部计算 loss

            # Flatten for loss computation
            logits_flat = logits.reshape(B * self.n_respond, l, self.vocab_size)
            target_flat = target_path.reshape(B * self.n_respond, l)
            mask_flat = mask.reshape(B * self.n_respond, l)

            loss = self._compute_pathfinding_loss(
                logits_flat, target_flat, mask_flat,
                mode=self.loss_mode,
                alpha=self.alpha,
                gamma=self.gamma,
                num_timesteps=self.num_timesteps,
            )

            return loss, logits
        else:
            # 推理模式：需要 mask path 部分，避免数据泄漏
            # 🔧 修复：在推理时将 path 部分 mask 掉，避免看到目标答案
            UNIT_LEN = self._compute_unit_len(d, l)
            for i in range(self.n_respond):
                unit_start = (self.n_prompt + i) * UNIT_LEN
                path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
                path_end = path_start + l
                full_sequence[:, path_start:path_end] = self.mask_token_id
            
            # 重新计算 embeddings（因为序列已改变）
            embeds = self._read_in(full_sequence)
            embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)
            
            # 重新 forward（因为序列已改变）
            out = self._backbone(
                inputs_embeds=embeds,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)
            
            # 重新提取 Path logits
            path_logits_list = []
            for i in range(self.n_respond):
                unit_start = (self.n_prompt + i) * UNIT_LEN
                path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
                path_end = path_start + l
                path_logits = vocab_logits[:, path_start:path_end, :]
                path_logits_list.append(path_logits)
            
            logits = torch.stack(path_logits_list, dim=1)
            mask = torch.ones(B, self.n_respond, l, device=device, dtype=torch.bool)
            return logits, mask


# ============================================================
# PathfindingLLaDA: Masked Diffusion 模型
# ============================================================

class PathfindingLLaDA(LLaDAPromptRespond, PathfindingMixin):
    """
    路径规划专用 LLaDA 模型（基于 LLaDAPromptRespond + PathfindingMixin）

    特点：
    1. 双向注意力（bidirectional attention）
    2. Masked diffusion training：随机 mask 路径节点
    3. 离散时间步训练：t ∈ [0, T-1]
    4. Loss：Composite loss (CE + Focal + Time weighting)
    """

    def __init__(
        self,
        n_dims,  # 路径规划：edge_len + 2 (edges + query)
        n_positions=1000,
        n_embd=256,
        n_layer=12,
        n_head=8,
        n_prompt=5,
        n_respond=1,
        *,
        mlp_ratio=4.0,
        block_group_size=1,
        # 路径规划特定参数
        num_nodes=100,
        degree=5,
        path_len=10,
        use_role_embedding=True,
        # Diffusion 参数
        num_timesteps=20,
        alpha=0.25,
        gamma=1.0,
        loss_mode="composite",
        mask_prob_override=None,
        mask_prob_min=0.0,
        mask_prob_max=1.0,
        t_sampling_power=1.0,
        # Inference 参数
        use_multistep_inference=False,
        inference_steps=20,
        inference_k_per_step=4,
        inference_scheduler=None,  # 🆕 动态 scheduler（可选）
        inference_confidence_alg="entropy",  # 🆕 置信度算法
        **extra,
    ):
        # 过滤掉父类不支持的参数
        parent_kwargs = _filter_parent_kwargs(extra)

        # 调用父类初始化
        super().__init__(
            n_dims=n_dims,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_prompt=n_prompt,
            n_respond=n_respond,
            mlp_ratio=mlp_ratio,
            block_group_size=block_group_size,
            **parent_kwargs,
        )

        self.name = "pathfinding_llada"
        self.family = "pathfinding_llada"  # 用于评估代码识别任务类型
        self.degree = degree
        self.path_len = path_len
        self.num_timesteps = num_timesteps
        self.alpha = alpha
        self.gamma = gamma
        self.loss_mode = loss_mode

        # Mask ratio controls
        self.mask_prob_override = mask_prob_override
        self.mask_prob_min = float(mask_prob_min)
        self.mask_prob_max = float(mask_prob_max)
        self.t_sampling_power = float(t_sampling_power)

        # Inference controls - 支持两种模式
        self.use_multistep_inference = bool(use_multistep_inference)
        self.inference_steps = int(inference_steps)
        self.inference_confidence_alg = inference_confidence_alg

        # 🆕 动态 scheduler 支持（向后兼容）
        if inference_scheduler is not None:
            # 模式1: 使用动态 scheduler（类似 ICL）
            from dllm.core.schedulers import LinearAlphaScheduler
            if isinstance(inference_scheduler, str):
                if inference_scheduler == "linear":
                    self.inference_scheduler = LinearAlphaScheduler()
                else:
                    raise ValueError(f"Unknown scheduler: {inference_scheduler}")
            else:
                self.inference_scheduler = inference_scheduler
            self.use_dynamic_scheduler = True
            self.inference_k_per_step = None  # 不使用固定 k
            print(f"  [Inference] Using dynamic scheduler: {type(self.inference_scheduler).__name__}")
        else:
            # 模式2: 使用固定 k_per_step（当前默认行为）
            self.inference_scheduler = None
            self.use_dynamic_scheduler = False
            self.inference_k_per_step = int(inference_k_per_step)
            print(f"  [Inference] Using fixed k_per_step: {self.inference_k_per_step}")

        # 使用 Mixin 设置路径规划协议
        self.setup_pathfinding_protocol(
            n_embd=n_embd,
            num_nodes=num_nodes,
            max_path_len=path_len,  # 修正：path_len 表示节点数
            use_role_embedding=use_role_embedding
        )

        print(f"[PathfindingLLaDA] Initialized:")
        print(f"  vocab_size: {self.vocab_size}")
        print(f"  degree: {degree}, path_len: {path_len}")
        print(f"  num_timesteps: {num_timesteps}")
        print(f"  Focal Loss: alpha={alpha}, gamma={gamma}")
        print(f"  Role Embedding: {'enabled' if use_role_embedding else 'disabled'}")
        print(f"  Loss mode: {loss_mode}")
        if mask_prob_override is not None:
            print(f"  Mask prob override: {mask_prob_override}")
        else:
            print(f"  Mask prob range: [{mask_prob_min}, {mask_prob_max}]")
        if use_multistep_inference:
            print(f"  Multi-step inference: enabled (steps={inference_steps}, k={inference_k_per_step})")

    def _compute_mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        """计算 mask 概率"""
        if self.mask_prob_override is not None:
            return torch.full_like(t, float(self.mask_prob_override), dtype=torch.float32)
        else:
            base = (t.float() + 1.0) / float(self.num_timesteps)
            base = base ** self.t_sampling_power
            return self.mask_prob_min + (self.mask_prob_max - self.mask_prob_min) * base

    def forward(self, xs, ys, train_mode=True, respond_position_mask=None):
        """
        前向传播

        Args:
            xs: [B, n_points, edge_len + 2]
            ys: [B, n_points, path_len]
            train_mode: 训练模式（保持接口一致）
            respond_position_mask: respond位置mask（保持接口一致）

        Returns:
            logits: [B, n_respond, path_len, vocab_size]
            loss: scalar tensor
        """
        B = xs.shape[0]
        device = xs.device
        d = self.degree
        l = self.path_len

        if self.training:
            # 训练模式：采样时间步并 mask
            t = torch.randint(0, self.num_timesteps, (B,), device=device)
            mask_prob = self._compute_mask_prob(t)

            # 构建序列并 mask Path 部分
            full_sequence, prefix_len = self._build_pathfinding_sequence(xs, ys, d, l, device)

            # Mask respond units 的 Path 部分
            UNIT_LEN = self._compute_unit_len(d, l)
            for i in range(self.n_respond):
                unit_start = (self.n_prompt + i) * UNIT_LEN
                path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
                path_end = path_start + l

                # 随机 mask
                mask_matrix = torch.rand(B, l, device=device) < mask_prob.unsqueeze(1)
                full_sequence[:, path_start:path_end][mask_matrix] = self.mask_token_id

            # Token embeddings
            embeds = self._read_in(full_sequence)

            # 注入 role embeddings
            total_points = self.n_prompt + self.n_respond
            embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)

            # Time conditioning
            t_scalar = t.float() / self.num_timesteps
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, full_sequence.shape[1], 1))
            embeds = embeds + time_emb

            # Backbone
            dummy_input_ids = torch.zeros(B, full_sequence.shape[1], dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)

            # 提取 Path logits
            path_logits_list = []
            for i in range(self.n_respond):
                unit_start = (self.n_prompt + i) * UNIT_LEN
                path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
                path_end = path_start + l
                path_logits = vocab_logits[:, path_start:path_end, :]
                path_logits_list.append(path_logits)

            logits = torch.stack(path_logits_list, dim=1)  # [B, n_respond, l, vocab_size]

            # 计算 loss
            target_path = ys[:, self.n_prompt:, :]
            mask = torch.ones_like(target_path, dtype=torch.float32)

            logits_flat = logits.reshape(B * self.n_respond, l, self.vocab_size)
            target_flat = target_path.reshape(B * self.n_respond, l)
            mask_flat = mask.reshape(B * self.n_respond, l)

            # 扩展 t 到 respond units
            t_expanded = t.unsqueeze(1).expand(B, self.n_respond).reshape(B * self.n_respond)

            loss = self._compute_pathfinding_loss(
                logits_flat, target_flat, mask_flat,
                t=t_expanded,
                mode=self.loss_mode,
                alpha=self.alpha,
                gamma=self.gamma,
                num_timesteps=self.num_timesteps,
            )

            return loss, logits
        else:
            # 推理模式
            if self.use_multistep_inference:
                return self._multistep_inference(xs, ys)
            else:
                return self._single_step_inference(xs, ys)

    def _single_step_inference(self, xs, ys):
        """单步推理：全部 mask 然后预测"""
        B = xs.shape[0]
        device = xs.device
        d = self.degree
        l = self.path_len

        # 构建序列，Path 部分全部 mask
        full_sequence, prefix_len = self._build_pathfinding_sequence(xs, ys, d, l, device)

        UNIT_LEN = self._compute_unit_len(d, l)
        for i in range(self.n_respond):
            unit_start = (self.n_prompt + i) * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            full_sequence[:, path_start:path_end] = self.mask_token_id

        # Forward
        embeds = self._read_in(full_sequence)
        total_points = self.n_prompt + self.n_respond
        embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)

        t_scalar = torch.ones(B, device=device)
        time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, full_sequence.shape[1], 1))
        embeds = embeds + time_emb

        dummy_input_ids = torch.zeros(B, full_sequence.shape[1], dtype=torch.long, device=device)


        out = self._backbone(


            input_ids=dummy_input_ids,


            input_embeddings=embeds,
            output_hidden_states=True,
        )
        h = out.hidden_states[-1]
        vocab_logits = self._read_out(h)

        # 提取 logits
        path_logits_list = []
        for i in range(self.n_respond):
            unit_start = (self.n_prompt + i) * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            path_logits = vocab_logits[:, path_start:path_end, :]
            path_logits_list.append(path_logits)

        logits = torch.stack(path_logits_list, dim=1)  # [B, n_respond, l, vocab_size]
        mask = torch.ones(B, self.n_respond, l, device=device, dtype=torch.bool)

        return logits, mask

    @torch.no_grad()
    def _multistep_inference(self, xs, ys):
        """多步去噪推理 - 支持固定 k 和动态 scheduler 两种模式"""
        if self.use_dynamic_scheduler:
            return self._multistep_inference_dynamic(xs, ys)
        else:
            return self._multistep_inference_fixed_k(xs, ys)

    def _multistep_inference_fixed_k(self, xs, ys):
        """多步去噪推理（固定 k_per_step 模式）"""
        B = xs.shape[0]
        device = xs.device
        d = self.degree
        l = self.path_len

        # 初始化：全部 mask
        full_sequence, prefix_len = self._build_pathfinding_sequence(xs, ys, d, l, device)
        input_ids = full_sequence.clone()

        UNIT_LEN = self._compute_unit_len(d, l)
        for i in range(self.n_respond):
            unit_start = (self.n_prompt + i) * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            input_ids[:, path_start:path_end] = self.mask_token_id

        filled = torch.zeros(B, l, dtype=torch.bool, device=device)
        final_logits = None

        steps = max(1, self.inference_steps)
        k_per_step = max(1, self.inference_k_per_step)

        for _ in range(steps):
            if filled.all():
                break

            # Forward
            embeds = self._read_in(input_ids)
            total_points = self.n_prompt + self.n_respond
            embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)

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

            # 提取 Path logits（仅第一个 respond unit）
            unit_start = self.n_prompt * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            logits = vocab_logits[:, path_start:path_end, :]  # [B, l, vocab_size]
            final_logits = logits.unsqueeze(1)  # [B, 1, l, vocab_size]

            # 计算熵
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [B, l]
            entropy = entropy.masked_fill(filled, float("inf"))

            # Fill top-k lowest entropy
            for b in range(B):
                unfilled = (~filled[b]).nonzero(as_tuple=True)[0]
                if unfilled.numel() == 0:
                    continue
                k = min(k_per_step, unfilled.numel())
                unfilled_entropy = entropy[b, unfilled]
                _, topk = torch.topk(unfilled_entropy, k=k, largest=False)
                cells = unfilled[topk]
                pred_nodes = torch.argmax(logits[b, cells, :], dim=-1)
                input_ids[b, path_start + cells] = pred_nodes
                filled[b, cells] = True

        if final_logits is None:
            # Final forward
            embeds = self._read_in(input_ids)
            total_points = self.n_prompt + self.n_respond
            embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)
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
            unit_start = self.n_prompt * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            logits = vocab_logits[:, path_start:path_end, :]
            final_logits = logits.unsqueeze(1)

        mask = torch.ones(B, self.n_respond, l, device=device, dtype=torch.bool)
        return final_logits, mask

    def _multistep_inference_dynamic(self, xs, ys):
        """多步去噪推理（动态 scheduler 模式，类似 ICL）"""
        from dllm.utils.generation_utils import get_num_transfer_tokens

        B = xs.shape[0]
        device = xs.device
        d = self.degree
        l = self.path_len

        # 初始化：全部 mask
        full_sequence, prefix_len = self._build_pathfinding_sequence(xs, ys, d, l, device)
        input_ids = full_sequence.clone()

        UNIT_LEN = self._compute_unit_len(d, l)
        for i in range(self.n_respond):
            unit_start = (self.n_prompt + i) * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            input_ids[:, path_start:path_end] = self.mask_token_id

        # 初始化 mask 状态
        masked_indices = torch.ones(B, l, device=device, dtype=torch.bool)
        initial_mask = masked_indices.clone()

        # 🆕 使用 scheduler 计算每步 unmask 的数量
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

            # Forward
            embeds = self._read_in(input_ids)
            total_points = self.n_prompt + self.n_respond
            embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)

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

            # 提取 Path logits（仅第一个 respond unit）
            unit_start = self.n_prompt * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            logits = vocab_logits[:, path_start:path_end, :]  # [B, l, vocab_size]
            final_logits = logits.unsqueeze(1)  # [B, 1, l, vocab_size]

            # 计算置信度（熵或其他算法）
            if self.inference_confidence_alg == "entropy":
                probs = F.softmax(logits, dim=-1)
                confidence = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [B, l] - 熵越低越confident
            else:
                # 可以添加其他置信度算法
                probs = F.softmax(logits, dim=-1)
                confidence = -probs.max(dim=-1)[0]  # [B, l] - 最大概率的负值

            confidence = confidence.masked_fill(~masked_indices, float("inf"))

            # 🆕 根据 scheduler 决定本步 unmask 多少个位置
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
                pred_nodes = torch.argmax(logits[b, cells, :], dim=-1)
                input_ids[b, path_start + cells] = pred_nodes
                masked_indices[b, cells] = False

        if final_logits is None:
            # Final forward
            embeds = self._read_in(input_ids)
            total_points = self.n_prompt + self.n_respond
            embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)
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
            unit_start = self.n_prompt * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            logits = vocab_logits[:, path_start:path_end, :]
            final_logits = logits.unsqueeze(1)

        mask = torch.ones(B, self.n_respond, l, device=device, dtype=torch.bool)
        return final_logits, mask


# ============================================================
# PathfindingLLaDABlock: Block Diffusion 模型
# ============================================================

class PathfindingLLaDABlock(LLaDABlockDiffusion, PathfindingMixin):
    """
    路径规划专用 LLaDA Block Diffusion 模型

    特点：
    1. Block-causal attention：块间因果，块内双向
    2. Masked diffusion training
    3. 适用于路径规划任务的 block 划分
    """

    def __init__(
        self,
        n_dims,
        n_positions=1000,
        n_embd=256,
        n_layer=12,
        n_head=8,
        n_prompt=5,
        n_respond=1,
        *,
        mlp_ratio=4.0,
        block_group_size=1,
        # 路径规划特定参数
        num_nodes=100,
        degree=5,
        path_len=10,
        use_role_embedding=True,
        # Block diffusion 参数
        use_block_diffusion=True,
        block_size=1,  # 每个 block 包含多少个路径节点
        # Diffusion 参数
        num_timesteps=20,
        alpha=0.25,
        gamma=1.0,
        loss_mode="composite",
        mask_prob_override=None,
        mask_prob_min=0.0,
        mask_prob_max=1.0,
        t_sampling_power=1.0,
        # Inference 参数
        use_multistep_inference=False,
        inference_steps=20,
        inference_k_per_step=4,
        inference_scheduler=None,  # 🆕 动态 scheduler（可选）
        inference_confidence_alg="entropy",  # 🆕 置信度算法
        **extra,
    ):
        # 过滤掉父类不支持的参数
        parent_kwargs = _filter_parent_kwargs(extra)

        # 调用父类初始化
        super().__init__(
            n_dims=n_dims,
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
            **parent_kwargs,
        )

        self.name = "pathfinding_llada_block"
        self.family = "pathfinding_llada_block"  # 用于评估代码识别任务类型
        self.degree = degree
        self.path_len = path_len
        self.num_timesteps = num_timesteps
        self.alpha = alpha
        self.gamma = gamma
        self.loss_mode = loss_mode
        self.block_size = block_size

        # Mask ratio controls
        self.mask_prob_override = mask_prob_override
        self.mask_prob_min = float(mask_prob_min)
        self.mask_prob_max = float(mask_prob_max)
        self.t_sampling_power = float(t_sampling_power)

        # Inference controls - 支持两种模式
        self.use_multistep_inference = bool(use_multistep_inference)
        self.inference_steps = int(inference_steps)
        self.inference_confidence_alg = inference_confidence_alg

        # 🆕 动态 scheduler 支持（向后兼容）
        if inference_scheduler is not None:
            # 模式1: 使用动态 scheduler（类似 ICL）
            from dllm.core.schedulers import LinearAlphaScheduler
            if isinstance(inference_scheduler, str):
                if inference_scheduler == "linear":
                    self.inference_scheduler = LinearAlphaScheduler()
                else:
                    raise ValueError(f"Unknown scheduler: {inference_scheduler}")
            else:
                self.inference_scheduler = inference_scheduler
            self.use_dynamic_scheduler = True
            self.inference_k_per_step = None  # 不使用固定 k
            print(f"  [Inference] Using dynamic scheduler: {type(self.inference_scheduler).__name__}")
        else:
            # 模式2: 使用固定 k_per_step（当前默认行为）
            self.inference_scheduler = None
            self.use_dynamic_scheduler = False
            self.inference_k_per_step = int(inference_k_per_step)
            print(f"  [Inference] Using fixed k_per_step: {self.inference_k_per_step}")

        # 使用 Mixin 设置路径规划协议
        self.setup_pathfinding_protocol(
            n_embd=n_embd,
            num_nodes=num_nodes,
            max_path_len=path_len,  # 修正：path_len 表示节点数
            use_role_embedding=use_role_embedding
        )

        print(f"[PathfindingLLaDABlock] Initialized:")
        print(f"  vocab_size: {self.vocab_size}")
        print(f"  degree: {degree}, path_len: {path_len}")
        print(f"  Block diffusion: {'enabled' if use_block_diffusion else 'disabled'}")
        print(f"  block_size: {block_size}")
        print(f"  num_timesteps: {num_timesteps}")
        print(f"  Loss mode: {loss_mode}")

    def _compute_mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        """计算 mask 概率"""
        if self.mask_prob_override is not None:
            return torch.full_like(t, float(self.mask_prob_override), dtype=torch.float32)
        else:
            base = (t.float() + 1.0) / float(self.num_timesteps)
            base = base ** self.t_sampling_power
            return self.mask_prob_min + (self.mask_prob_max - self.mask_prob_min) * base

    def _compute_block_ids(self, d: int, l: int, device: torch.device) -> torch.Tensor:
        """
        计算路径规划序列的 block IDs

        Block 划分策略：
        - Edges: 每个 edge 作为一个 block（或按 block_size 分组）
        - Query: 独立 block
        - Path: 按 block_size 划分
        """
        UNIT_LEN = self._compute_unit_len(d, l)
        edge_len = (l - 1) * d * 2  # 修正：l 表示路径节点数，边数为 (l-1) * d

        block_ids = torch.zeros(UNIT_LEN, dtype=torch.long, device=device)

        # Edges: 按 block_size 分组
        num_edge_blocks = (edge_len + self.block_size - 1) // self.block_size
        for i in range(edge_len):
            block_ids[i] = i // self.block_size

        # Sep '/' + Query: 独立 block
        query_block_id = num_edge_blocks
        block_ids[edge_len:edge_len + 3] = query_block_id  # '/' + start + goal

        # Eq '=' + Path: 按 block_size 划分
        path_start_idx = edge_len + 3
        eq_and_path_block_start = query_block_id + 1
        block_ids[path_start_idx] = eq_and_path_block_start  # '=' 独立

        # Path 节点按 block_size 分组
        for i in range(l):
            block_ids[path_start_idx + 1 + i] = eq_and_path_block_start + 1 + (i // self.block_size)

        return block_ids

    def forward(self, xs, ys, train_mode=True, respond_position_mask=None):
        """
        前向传播

        Args:
            xs: [B, n_points, edge_len + 2]
            ys: [B, n_points, path_len]
            train_mode: 训练模式（保持接口一致）
            respond_position_mask: respond位置mask（保持接口一致）

        Returns:
            logits: [B, n_respond, path_len, vocab_size]
            loss: scalar tensor
        """
        B = xs.shape[0]
        device = xs.device
        d = self.degree
        l = self.path_len

        if self.training:
            # 训练模式
            t = torch.randint(0, self.num_timesteps, (B,), device=device)
            mask_prob = self._compute_mask_prob(t)

            # 构建序列并 mask
            full_sequence, prefix_len = self._build_pathfinding_sequence(xs, ys, d, l, device)

            UNIT_LEN = self._compute_unit_len(d, l)
            for i in range(self.n_respond):
                unit_start = (self.n_prompt + i) * UNIT_LEN
                path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
                path_end = path_start + l
                mask_matrix = torch.rand(B, l, device=device) < mask_prob.unsqueeze(1)
                full_sequence[:, path_start:path_end][mask_matrix] = self.mask_token_id

            # Token embeddings
            embeds = self._read_in(full_sequence)

            # 注入 role embeddings
            total_points = self.n_prompt + self.n_respond
            embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)

            # Time conditioning
            t_scalar = t.float() / self.num_timesteps
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, full_sequence.shape[1], 1))
            embeds = embeds + time_emb

            # 计算 block IDs
            if self.use_block_diffusion:
                unit_block_ids = self._compute_block_ids(d, l, device)
                # 重复到所有 units
                block_ids = unit_block_ids.unsqueeze(0).repeat(total_points, 1).reshape(-1)
                # 添加 unit offset
                for i in range(total_points):
                    max_block_in_unit = unit_block_ids.max().item() + 1
                    block_ids[i * UNIT_LEN:(i + 1) * UNIT_LEN] += i * max_block_in_unit
                block_ids = block_ids.unsqueeze(0).expand(B, -1)  # [B, seq_len]
            else:
                block_ids = None

            # Backbone
            dummy_input_ids = torch.zeros(B, embeds.shape[1], dtype=torch.long, device=device)

            out = self._backbone(

                input_ids=dummy_input_ids,

                input_embeddings=embeds,
                block_ids=block_ids,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)

            # 提取 Path logits
            path_logits_list = []
            for i in range(self.n_respond):
                unit_start = (self.n_prompt + i) * UNIT_LEN
                path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
                path_end = path_start + l
                path_logits = vocab_logits[:, path_start:path_end, :]
                path_logits_list.append(path_logits)

            logits = torch.stack(path_logits_list, dim=1)

            # 计算 loss
            target_path = ys[:, self.n_prompt:, :]
            mask = torch.ones_like(target_path, dtype=torch.float32)

            logits_flat = logits.reshape(B * self.n_respond, l, self.vocab_size)
            target_flat = target_path.reshape(B * self.n_respond, l)
            mask_flat = mask.reshape(B * self.n_respond, l)

            t_expanded = t.unsqueeze(1).expand(B, self.n_respond).reshape(B * self.n_respond)

            loss = self._compute_pathfinding_loss(
                logits_flat, target_flat, mask_flat,
                t=t_expanded,
                mode=self.loss_mode,
                alpha=self.alpha,
                gamma=self.gamma,
                num_timesteps=self.num_timesteps,
            )

            return loss, logits
        else:
            # 推理模式
            if self.use_multistep_inference:
                # 多步推理
                return self._multistep_inference(xs, ys)

            # 单步推理：简化版，全部 mask 然后预测
            full_sequence, prefix_len = self._build_pathfinding_sequence(xs, ys, d, l, device)

            UNIT_LEN = self._compute_unit_len(d, l)
            for i in range(self.n_respond):
                unit_start = (self.n_prompt + i) * UNIT_LEN
                path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
                path_end = path_start + l
                full_sequence[:, path_start:path_end] = self.mask_token_id

            embeds = self._read_in(full_sequence)
            total_points = self.n_prompt + self.n_respond
            embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)

            t_scalar = torch.ones(B, device=device)
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, full_sequence.shape[1], 1))
            embeds = embeds + time_emb

            if self.use_block_diffusion:
                unit_block_ids = self._compute_block_ids(d, l, device)
                block_ids = unit_block_ids.unsqueeze(0).repeat(total_points, 1).reshape(-1)
                for i in range(total_points):
                    max_block_in_unit = unit_block_ids.max().item() + 1
                    block_ids[i * UNIT_LEN:(i + 1) * UNIT_LEN] += i * max_block_in_unit
                block_ids = block_ids.unsqueeze(0).expand(B, -1)
            else:
                block_ids = None

            dummy_input_ids = torch.zeros(B, embeds.shape[1], dtype=torch.long, device=device)


            out = self._backbone(


                input_ids=dummy_input_ids,


                input_embeddings=embeds,
                block_ids=block_ids,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)

            path_logits_list = []
            for i in range(self.n_respond):
                unit_start = (self.n_prompt + i) * UNIT_LEN
                path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
                path_end = path_start + l
                path_logits = vocab_logits[:, path_start:path_end, :]
                path_logits_list.append(path_logits)

            logits = torch.stack(path_logits_list, dim=1)  # [B, n_respond, l, vocab_size]
            mask = torch.ones(B, self.n_respond, l, device=device, dtype=torch.bool)

            return logits, mask

    @torch.no_grad()
    def _multistep_inference(self, xs, ys):
        """多步去噪推理 - 支持固定 k 和动态 scheduler 两种模式"""
        if self.use_dynamic_scheduler:
            return self._multistep_inference_dynamic(xs, ys)
        else:
            return self._multistep_inference_fixed_k(xs, ys)
    def _multistep_inference_fixed_k(self, xs, ys):
        """多步去噪推理（固定 k_per_step 模式）"""
        B = xs.shape[0]
        device = xs.device
        d = self.degree
        l = self.path_len

        # 初始化：全部 mask
        full_sequence, prefix_len = self._build_pathfinding_sequence(xs, ys, d, l, device)
        input_ids = full_sequence.clone()

        UNIT_LEN = self._compute_unit_len(d, l)
        for i in range(self.n_respond):
            unit_start = (self.n_prompt + i) * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            input_ids[:, path_start:path_end] = self.mask_token_id

        filled = torch.zeros(B, l, dtype=torch.bool, device=device)
        final_logits = None

        steps = max(1, self.inference_steps)
        k_per_step = max(1, self.inference_k_per_step)

        for _ in range(steps):
            if filled.all():
                break

            # Forward
            embeds = self._read_in(input_ids)
            total_points = self.n_prompt + self.n_respond
            embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)

            t_scalar = torch.ones(B, device=device)
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, input_ids.shape[1], 1))
            embeds = embeds + time_emb

            # 计算 block IDs
            if self.use_block_diffusion:
                unit_block_ids = self._compute_block_ids(d, l, device)
                block_ids = unit_block_ids.unsqueeze(0).repeat(total_points, 1).reshape(-1)
                for i in range(total_points):
                    max_block_in_unit = unit_block_ids.max().item() + 1
                    block_ids[i * UNIT_LEN:(i + 1) * UNIT_LEN] += i * max_block_in_unit
                block_ids = block_ids.unsqueeze(0).expand(B, -1)
            else:
                block_ids = None

            dummy_input_ids = torch.zeros(B, input_ids.shape[1], dtype=torch.long, device=device)

            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                block_ids=block_ids,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)

            # 提取 Path logits（仅第一个 respond unit）
            unit_start = self.n_prompt * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            logits = vocab_logits[:, path_start:path_end, :]  # [B, l, vocab_size]
            final_logits = logits.unsqueeze(1)  # [B, 1, l, vocab_size]

            # 计算熵
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [B, l]
            entropy = entropy.masked_fill(filled, float("inf"))

            # Fill top-k lowest entropy
            for b in range(B):
                unfilled = (~filled[b]).nonzero(as_tuple=True)[0]
                if unfilled.numel() == 0:
                    continue
                k = min(k_per_step, unfilled.numel())
                unfilled_entropy = entropy[b, unfilled]
                _, topk = torch.topk(unfilled_entropy, k=k, largest=False)
                cells = unfilled[topk]
                pred_nodes = torch.argmax(logits[b, cells, :], dim=-1)
                input_ids[b, path_start + cells] = pred_nodes
                filled[b, cells] = True

        if final_logits is None:
            # Final forward
            embeds = self._read_in(input_ids)
            total_points = self.n_prompt + self.n_respond
            embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)
            t_scalar = torch.ones(B, device=device)
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, input_ids.shape[1], 1))
            embeds = embeds + time_emb

            if self.use_block_diffusion:
                unit_block_ids = self._compute_block_ids(d, l, device)
                block_ids = unit_block_ids.unsqueeze(0).repeat(total_points, 1).reshape(-1)
                for i in range(total_points):
                    max_block_in_unit = unit_block_ids.max().item() + 1
                    block_ids[i * UNIT_LEN:(i + 1) * UNIT_LEN] += i * max_block_in_unit
                block_ids = block_ids.unsqueeze(0).expand(B, -1)
            else:
                block_ids = None

            dummy_input_ids = torch.zeros(B, input_ids.shape[1], dtype=torch.long, device=device)

            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                block_ids=block_ids,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)
            unit_start = self.n_prompt * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            logits = vocab_logits[:, path_start:path_end, :]
            final_logits = logits.unsqueeze(1)

        mask = torch.ones(B, self.n_respond, l, device=device, dtype=torch.bool)
        return final_logits, mask

    def _multistep_inference_dynamic(self, xs, ys):
        """多步去噪推理（动态 scheduler 模式，类似 ICL）"""
        from dllm.utils.generation_utils import get_num_transfer_tokens

        B = xs.shape[0]
        device = xs.device
        d = self.degree
        l = self.path_len

        # 初始化：全部 mask
        full_sequence, prefix_len = self._build_pathfinding_sequence(xs, ys, d, l, device)
        input_ids = full_sequence.clone()

        UNIT_LEN = self._compute_unit_len(d, l)
        for i in range(self.n_respond):
            unit_start = (self.n_prompt + i) * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            input_ids[:, path_start:path_end] = self.mask_token_id

        # 初始化 mask 状态
        masked_indices = torch.ones(B, l, device=device, dtype=torch.bool)
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

            # Forward
            embeds = self._read_in(input_ids)
            total_points = self.n_prompt + self.n_respond
            embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)

            t_scalar = torch.ones(B, device=device)
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, input_ids.shape[1], 1))
            embeds = embeds + time_emb

            # 计算 block IDs
            if self.use_block_diffusion:
                unit_block_ids = self._compute_block_ids(d, l, device)
                block_ids = unit_block_ids.unsqueeze(0).repeat(total_points, 1).reshape(-1)
                for i in range(total_points):
                    max_block_in_unit = unit_block_ids.max().item() + 1
                    block_ids[i * UNIT_LEN:(i + 1) * UNIT_LEN] += i * max_block_in_unit
                block_ids = block_ids.unsqueeze(0).expand(B, -1)
            else:
                block_ids = None

            dummy_input_ids = torch.zeros(B, input_ids.shape[1], dtype=torch.long, device=device)

            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                block_ids=block_ids,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)

            # 提取 Path logits（仅第一个 respond unit）
            unit_start = self.n_prompt * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            logits = vocab_logits[:, path_start:path_end, :]  # [B, l, vocab_size]
            final_logits = logits.unsqueeze(1)  # [B, 1, l, vocab_size]

            # 计算置信度
            if self.inference_confidence_alg == "entropy":
                probs = F.softmax(logits, dim=-1)
                confidence = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [B, l]
            else:
                probs = F.softmax(logits, dim=-1)
                confidence = -probs.max(dim=-1)[0]  # [B, l]

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
                pred_nodes = torch.argmax(logits[b, cells, :], dim=-1)
                input_ids[b, path_start + cells] = pred_nodes
                masked_indices[b, cells] = False

        if final_logits is None:
            # Final forward
            embeds = self._read_in(input_ids)
            total_points = self.n_prompt + self.n_respond
            embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)
            t_scalar = torch.ones(B, device=device)
            time_emb = self._time_mlp(t_scalar.reshape(B, 1, 1).expand(B, input_ids.shape[1], 1))
            embeds = embeds + time_emb

            if self.use_block_diffusion:
                unit_block_ids = self._compute_block_ids(d, l, device)
                block_ids = unit_block_ids.unsqueeze(0).repeat(total_points, 1).reshape(-1)
                for i in range(total_points):
                    max_block_in_unit = unit_block_ids.max().item() + 1
                    block_ids[i * UNIT_LEN:(i + 1) * UNIT_LEN] += i * max_block_in_unit
                block_ids = block_ids.unsqueeze(0).expand(B, -1)
            else:
                block_ids = None

            dummy_input_ids = torch.zeros(B, input_ids.shape[1], dtype=torch.long, device=device)

            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                block_ids=block_ids,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)
            unit_start = self.n_prompt * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            logits = vocab_logits[:, path_start:path_end, :]
            final_logits = logits.unsqueeze(1)

        mask = torch.ones(B, self.n_respond, l, device=device, dtype=torch.bool)
        return final_logits, mask


# ============================================================
# PathfindingBOPAR: Block-Offset Parallel AR 模型
# ============================================================

class PathfindingBOPAR(BOPARPromptRespond, PathfindingMixin):
    """
    路径规划专用 BOP-AR 模型（Block-Offset Parallel Autoregressive）

    特点：
    1. ScatDiff: Scatter Diffusion with offset autoregressive
    2. Block-causal attention
    3. 层级化生成路径
    """

    def __init__(
        self,
        n_dims,
        n_positions=1000,
        n_embd=256,
        n_layer=12,
        n_head=8,
        n_prompt=5,
        n_respond=1,
        *,
        mlp_ratio=4.0,
        block_group_size=1,
        # 路径规划特定参数
        num_nodes=100,
        degree=5,
        path_len=10,
        use_role_embedding=True,
        # BOP-AR 参数
        block_size=1,
        # Loss 参数
        loss_mode="ce",
        alpha=0.25,
        gamma=1.0,
        num_timesteps=20,
        # Inference 参数
        use_multistep_inference=False,
        inference_steps=20,
        inference_k_per_step=4,
        inference_scheduler=None,
        inference_confidence_alg="entropy",
        **extra,
    ):
        # 过滤掉父类不支持的参数
        parent_kwargs = _filter_parent_kwargs(extra)

        # 调用父类初始化
        super().__init__(
            n_dims=n_dims,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_prompt=n_prompt,
            n_respond=n_respond,
            mlp_ratio=mlp_ratio,
            block_group_size=block_group_size,
            block_size=block_size,
            **parent_kwargs,
        )

        self.name = "pathfinding_bopar"
        self.family = "pathfinding_bopar"  # 用于评估代码识别任务类型
        self.degree = degree
        self.path_len = path_len
        self.block_size = block_size
        self.loss_mode = loss_mode
        self.alpha = alpha
        self.gamma = gamma
        self.num_timesteps = num_timesteps

        # 使用 Mixin 设置路径规划协议
        self.setup_pathfinding_protocol(
            n_embd=n_embd,
            num_nodes=num_nodes,
            max_path_len=path_len,  # 修正：path_len 表示节点数
            use_role_embedding=use_role_embedding
        )

        print(f"[PathfindingBOPAR] Initialized:")
        print(f"  vocab_size: {self.vocab_size}")
        print(f"  degree: {degree}, path_len: {path_len}")
        print(f"  block_size: {block_size}")
        print(f"  Loss mode: {loss_mode}")

        # 🆕 多步推理支持
        self.use_multistep_inference = use_multistep_inference
        self.inference_steps = int(inference_steps)
        self.inference_confidence_alg = inference_confidence_alg

        # 🆕 动态 scheduler 支持（向后兼容）
        if inference_scheduler is not None:
            # 模式1: 使用动态 scheduler（类似 ICL）
            from dllm.core.schedulers import LinearAlphaScheduler
            if isinstance(inference_scheduler, str):
                if inference_scheduler == "linear":
                    self.inference_scheduler = LinearAlphaScheduler()
                else:
                    raise ValueError(f"Unknown scheduler: {inference_scheduler}")
            else:
                self.inference_scheduler = inference_scheduler
            self.use_dynamic_scheduler = True
            self.inference_k_per_step = None  # 不使用固定 k
            print(f"  [Inference] Using dynamic scheduler: {type(self.inference_scheduler).__name__}")
        else:
            # 模式2: 使用固定 k_per_step（当前默认行为）
            self.inference_scheduler = None
            self.use_dynamic_scheduler = False
            self.inference_k_per_step = int(inference_k_per_step)
            print(f"  [Inference] Using fixed k_per_step: {self.inference_k_per_step}")

    def _compute_block_ids(self, d: int, l: int, device: torch.device) -> torch.Tensor:
        """计算 block IDs（与 PathfindingLLaDABlock 相同）"""
        UNIT_LEN = self._compute_unit_len(d, l)
        edge_len = (l - 1) * d * 2  # 修正：l 表示路径节点数，边数为 (l-1) * d

        block_ids = torch.zeros(UNIT_LEN, dtype=torch.long, device=device)

        # Edges: 按 block_size 分组
        num_edge_blocks = (edge_len + self.block_size - 1) // self.block_size
        for i in range(edge_len):
            block_ids[i] = i // self.block_size

        # Sep '/' + Query: 独立 block
        query_block_id = num_edge_blocks
        block_ids[edge_len:edge_len + 3] = query_block_id

        # Eq '=' + Path: 按 block_size 划分
        path_start_idx = edge_len + 3
        eq_and_path_block_start = query_block_id + 1
        block_ids[path_start_idx] = eq_and_path_block_start

        # Path 节点按 block_size 分组
        for i in range(l):
            block_ids[path_start_idx + 1 + i] = eq_and_path_block_start + 1 + (i // self.block_size)

        return block_ids

    def forward(self, xs, ys, train_mode=True, respond_position_mask=None):
        """
        前向传播

        Args:
            xs: [B, n_points, edge_len + 2]
            ys: [B, n_points, path_len]
            train_mode: 训练模式（保持接口一致）
            respond_position_mask: respond位置mask（保持接口一致）

        Returns:
            logits: [B, n_respond, path_len, vocab_size]
            loss: scalar tensor
        """
        B = xs.shape[0]
        device = xs.device
        d = self.degree
        l = self.path_len

        # 构建序列
        full_sequence, prefix_len = self._build_pathfinding_sequence(xs, ys, d, l, device)

        # Token embeddings
        embeds = self._read_in(full_sequence)

        # 注入 role embeddings
        total_points = self.n_prompt + self.n_respond
        embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)

        # 🆕 创建 ScatDiff (BOP-AR) attention bias
        UNIT_LEN = self._compute_unit_len(d, l)
        seq_len = total_points * UNIT_LEN
        attention_bias = self._create_scatdiff_attention_bias(
            total_points=total_points,
            n_prompt=self.n_prompt,
            block_size=self.block_size,
            device=device,
            dtype=embeds.dtype
        )

        # Backbone
        dummy_input_ids = torch.zeros(B, embeds.shape[1], dtype=torch.long, device=device)

        out = self._backbone(
            input_ids=dummy_input_ids,
            input_embeddings=embeds,
            attention_bias=attention_bias,  # 🆕 使用 attention_bias 而不是 block_ids
            output_hidden_states=True,
        )
        h = out.hidden_states[-1]
        vocab_logits = self._read_out(h)

        # 提取 Path logits
        path_logits_list = []
        for i in range(self.n_respond):
            unit_start = (self.n_prompt + i) * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            path_logits = vocab_logits[:, path_start:path_end, :]
            path_logits_list.append(path_logits)

        logits = torch.stack(path_logits_list, dim=1)

        # 计算 loss
        if self.training:
            target_path = ys[:, self.n_prompt:, :]
            mask = torch.ones_like(target_path, dtype=torch.float32)

            logits_flat = logits.reshape(B * self.n_respond, l, self.vocab_size)
            target_flat = target_path.reshape(B * self.n_respond, l)
            mask_flat = mask.reshape(B * self.n_respond, l)

            loss = self._compute_pathfinding_loss(
                logits_flat, target_flat, mask_flat,
                mode=self.loss_mode,
                alpha=self.alpha,
                gamma=self.gamma,
                num_timesteps=self.num_timesteps,
            )
            return loss, logits
        else:
            # 推理模式
            if self.use_multistep_inference:
                # 多步推理
                return self._multistep_inference(xs, ys)

            # 单步推理：需要 mask path 部分，避免数据泄漏
            # 🔧 修复：在推理时将 path 部分 mask 掉，避免看到目标答案
            UNIT_LEN = self._compute_unit_len(d, l)
            for i in range(self.n_respond):
                unit_start = (self.n_prompt + i) * UNIT_LEN
                path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
                path_end = path_start + l
                full_sequence[:, path_start:path_end] = self.mask_token_id
            
            # 重新计算 embeddings（因为序列已改变）
            embeds = self._read_in(full_sequence)
            embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)
            
            # 重新计算 block IDs
            unit_block_ids = self._compute_block_ids(d, l, device)
            block_ids = unit_block_ids.unsqueeze(0).repeat(total_points, 1).reshape(-1)
            for i in range(total_points):
                max_block_in_unit = unit_block_ids.max().item() + 1
                block_ids[i * UNIT_LEN:(i + 1) * UNIT_LEN] += i * max_block_in_unit
            block_ids = block_ids.unsqueeze(0).expand(B, -1)
            
            # 重新 forward（因为序列已改变）
            dummy_input_ids = torch.zeros(B, embeds.shape[1], dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                block_ids=block_ids,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)
            
            # 重新提取 Path logits
            path_logits_list = []
            for i in range(self.n_respond):
                unit_start = (self.n_prompt + i) * UNIT_LEN
                path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
                path_end = path_start + l
                path_logits = vocab_logits[:, path_start:path_end, :]
                path_logits_list.append(path_logits)
            
            logits = torch.stack(path_logits_list, dim=1)
            mask = torch.ones(B, self.n_respond, l, device=device, dtype=torch.bool)
            return logits, mask

    def _multistep_inference(self, xs, ys):
        """多步去噪推理（路由方法）"""
        if self.use_dynamic_scheduler:
            return self._multistep_inference_dynamic(xs, ys)
        else:
            return self._multistep_inference_fixed_k(xs, ys)

    def _multistep_inference_fixed_k(self, xs, ys):
        """多步去噪推理（固定 k 模式）"""
        B = xs.shape[0]
        device = xs.device
        d = self.degree
        l = self.path_len

        # 初始化：全部 mask
        full_sequence, prefix_len = self._build_pathfinding_sequence(xs, ys, d, l, device)
        input_ids = full_sequence.clone()

        UNIT_LEN = self._compute_unit_len(d, l)
        for i in range(self.n_respond):
            unit_start = (self.n_prompt + i) * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            input_ids[:, path_start:path_end] = self.mask_token_id

        # 初始化 mask 状态
        masked_indices = torch.ones(B, l, device=device, dtype=torch.bool)

        # 迭代去噪
        for step in range(self.inference_steps):
            if masked_indices.sum() == 0:
                break

            # Forward
            embeds = self._read_in(input_ids)
            total_points = self.n_prompt + self.n_respond
            embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)

            # 计算 block IDs
            unit_block_ids = self._compute_block_ids(d, l, device)
            block_ids = unit_block_ids.unsqueeze(0).repeat(total_points, 1).reshape(-1)
            for i in range(total_points):
                max_block_in_unit = unit_block_ids.max().item() + 1
                block_ids[i * UNIT_LEN:(i + 1) * UNIT_LEN] += i * max_block_in_unit
            block_ids = block_ids.unsqueeze(0).expand(B, -1)

            dummy_input_ids = torch.zeros(B, input_ids.shape[1], dtype=torch.long, device=device)

            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                block_ids=block_ids,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)

            # 提取 Path logits（仅第一个 respond unit）
            unit_start = self.n_prompt * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            logits = vocab_logits[:, path_start:path_end, :]  # [B, l, vocab_size]

            # 计算置信度
            probs = F.softmax(logits, dim=-1)
            confidence = -probs.max(dim=-1)[0]  # [B, l]
            confidence = confidence.masked_fill(~masked_indices, float("inf"))

            # 选择 top-k 最不确定的位置
            for b in range(B):
                unfilled = masked_indices[b].nonzero(as_tuple=True)[0]
                if unfilled.numel() == 0:
                    continue

                k = min(self.inference_k_per_step, unfilled.numel())
                unfilled_confidence = confidence[b, unfilled]
                _, topk = torch.topk(unfilled_confidence, k=k, largest=False)
                cells = unfilled[topk]
                pred_nodes = torch.argmax(logits[b, cells, :], dim=-1)
                input_ids[b, path_start + cells] = pred_nodes
                masked_indices[b, cells] = False

        # Final forward
        embeds = self._read_in(input_ids)
        total_points = self.n_prompt + self.n_respond
        embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)

        unit_block_ids = self._compute_block_ids(d, l, device)
        block_ids = unit_block_ids.unsqueeze(0).repeat(total_points, 1).reshape(-1)
        for i in range(total_points):
            max_block_in_unit = unit_block_ids.max().item() + 1
            block_ids[i * UNIT_LEN:(i + 1) * UNIT_LEN] += i * max_block_in_unit
        block_ids = block_ids.unsqueeze(0).expand(B, -1)

        dummy_input_ids = torch.zeros(B, input_ids.shape[1], dtype=torch.long, device=device)

        out = self._backbone(
            input_ids=dummy_input_ids,
            input_embeddings=embeds,
            block_ids=block_ids,
            output_hidden_states=True,
        )
        h = out.hidden_states[-1]
        vocab_logits = self._read_out(h)

        unit_start = self.n_prompt * UNIT_LEN
        path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
        path_end = path_start + l
        logits = vocab_logits[:, path_start:path_end, :]
        final_logits = logits.unsqueeze(1)  # [B, 1, l, vocab_size]

        mask = torch.ones(B, self.n_respond, l, device=device, dtype=torch.bool)
        return final_logits, mask

    def _multistep_inference_dynamic(self, xs, ys):
        """多步去噪推理（动态 scheduler 模式，类似 ICL）"""
        from dllm.utils.generation_utils import get_num_transfer_tokens

        B = xs.shape[0]
        device = xs.device
        d = self.degree
        l = self.path_len

        # 初始化：全部 mask
        full_sequence, prefix_len = self._build_pathfinding_sequence(xs, ys, d, l, device)
        input_ids = full_sequence.clone()

        UNIT_LEN = self._compute_unit_len(d, l)
        for i in range(self.n_respond):
            unit_start = (self.n_prompt + i) * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            input_ids[:, path_start:path_end] = self.mask_token_id

        # 初始化 mask 状态
        masked_indices = torch.ones(B, l, device=device, dtype=torch.bool)
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

            # Forward
            embeds = self._read_in(input_ids)
            total_points = self.n_prompt + self.n_respond
            embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)

            # 计算 block IDs
            unit_block_ids = self._compute_block_ids(d, l, device)
            block_ids = unit_block_ids.unsqueeze(0).repeat(total_points, 1).reshape(-1)
            for i in range(total_points):
                max_block_in_unit = unit_block_ids.max().item() + 1
                block_ids[i * UNIT_LEN:(i + 1) * UNIT_LEN] += i * max_block_in_unit
            block_ids = block_ids.unsqueeze(0).expand(B, -1)

            dummy_input_ids = torch.zeros(B, input_ids.shape[1], dtype=torch.long, device=device)

            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                block_ids=block_ids,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)

            # 提取 Path logits（仅第一个 respond unit）
            unit_start = self.n_prompt * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            logits = vocab_logits[:, path_start:path_end, :]  # [B, l, vocab_size]
            final_logits = logits.unsqueeze(1)  # [B, 1, l, vocab_size]

            # 计算置信度
            if self.inference_confidence_alg == "entropy":
                probs = F.softmax(logits, dim=-1)
                confidence = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [B, l]
            else:
                probs = F.softmax(logits, dim=-1)
                confidence = -probs.max(dim=-1)[0]  # [B, l]

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
                pred_nodes = torch.argmax(logits[b, cells, :], dim=-1)
                input_ids[b, path_start + cells] = pred_nodes
                masked_indices[b, cells] = False

        if final_logits is None:
            # Final forward
            embeds = self._read_in(input_ids)
            total_points = self.n_prompt + self.n_respond
            embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)

            unit_block_ids = self._compute_block_ids(d, l, device)
            block_ids = unit_block_ids.unsqueeze(0).repeat(total_points, 1).reshape(-1)
            for i in range(total_points):
                max_block_in_unit = unit_block_ids.max().item() + 1
                block_ids[i * UNIT_LEN:(i + 1) * UNIT_LEN] += i * max_block_in_unit
            block_ids = block_ids.unsqueeze(0).expand(B, -1)

            dummy_input_ids = torch.zeros(B, input_ids.shape[1], dtype=torch.long, device=device)

            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                block_ids=block_ids,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)
            unit_start = self.n_prompt * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            logits = vocab_logits[:, path_start:path_end, :]
            final_logits = logits.unsqueeze(1)

        mask = torch.ones(B, self.n_respond, l, device=device, dtype=torch.bool)
        return final_logits, mask


# ============================================================
# PathfindingBADAR: Block Autoregressive Diffusion 模型
# ============================================================

class PathfindingBADAR(BADARPromptRespond, PathfindingMixin):
    """
    路径规划专用 BAD-AR 模型（Block Autoregressive Diffusion）

    特性：
    1. Block-level Diffusion (Inter-block): 块间扩散逻辑，随机 Mask 若干块
    2. Intra-block AR: 块内严格因果顺序
    3. 高性能向量化实现

    核心逻辑：
    - 只有 Path 部分参与 Block 划分和 Mask
    - Block 划分：按 block_size 划分 Path 部分
    - 块间 Diffusion：不同被 Mask 的块之间互不可见
    - 块内 AR：每个 Mask 块内部严格因果顺序
    - 全局可见性：Edges + Query + Eq 和可见 Path 块对所有位置可见
    """

    def __init__(
        self,
        n_dims,
        n_positions=1000,
        n_embd=256,
        n_layer=12,
        n_head=8,
        n_prompt=5,
        n_respond=1,
        *,
        mlp_ratio=4.0,
        block_group_size=1,
        # 路径规划特定参数
        num_nodes=100,
        degree=5,
        path_len=10,
        use_role_embedding=True,
        # BAD-AR 参数
        block_size=1,  # Path 块大小
        num_timesteps=20,
        alpha=0.25,
        gamma=1.0,
        loss_mode="composite",
        # Inference 参数
        use_multistep_inference=False,
        inference_steps=20,
        inference_k_per_step=4,
        inference_scheduler=None,
        inference_confidence_alg="entropy",
        **extra,
    ):
        # 过滤掉父类不支持的参数
        parent_kwargs = _filter_parent_kwargs(extra)

        # 调用父类初始化
        super().__init__(
            n_dims=n_dims,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_prompt=n_prompt,
            n_respond=n_respond,
            mlp_ratio=mlp_ratio,
            block_group_size=block_group_size,
            block_size=block_size,
            **parent_kwargs,
        )

        self.name = "pathfinding_badar"
        self.family = "pathfinding_badar"  # 用于评估代码识别任务类型
        self.degree = degree
        self.path_len = path_len
        self.block_size = block_size
        self.num_timesteps = num_timesteps
        self.alpha = alpha
        self.gamma = gamma
        self.loss_mode = loss_mode

        # 使用 Mixin 设置路径规划协议
        self.setup_pathfinding_protocol(
            n_embd=n_embd,
            num_nodes=num_nodes,
            max_path_len=path_len,  # 修正：path_len 表示节点数
            use_role_embedding=use_role_embedding
        )

        print(f"[PathfindingBADAR] Initialized:")
        print(f"  vocab_size: {self.vocab_size}")
        print(f"  degree: {degree}, path_len: {path_len}")
        print(f"  block_size: {block_size}")
        print(f"  num_timesteps: {num_timesteps}")
        print(f"  Loss mode: {loss_mode}")

        # 🆕 多步推理支持
        self.use_multistep_inference = use_multistep_inference
        self.inference_steps = int(inference_steps)
        self.inference_confidence_alg = inference_confidence_alg

        # 🆕 动态 scheduler 支持（向后兼容）
        if inference_scheduler is not None:
            # 模式1: 使用动态 scheduler（类似 ICL）
            from dllm.core.schedulers import LinearAlphaScheduler
            if isinstance(inference_scheduler, str):
                if inference_scheduler == "linear":
                    self.inference_scheduler = LinearAlphaScheduler()
                else:
                    raise ValueError(f"Unknown scheduler: {inference_scheduler}")
            else:
                self.inference_scheduler = inference_scheduler
            self.use_dynamic_scheduler = True
            self.inference_k_per_step = None  # 不使用固定 k
            print(f"  [Inference] Using dynamic scheduler: {type(self.inference_scheduler).__name__}")
        else:
            # 模式2: 使用固定 k_per_step（当前默认行为）
            self.inference_scheduler = None
            self.use_dynamic_scheduler = False
            self.inference_k_per_step = int(inference_k_per_step)
            print(f"  [Inference] Using fixed k_per_step: {self.inference_k_per_step}")

    def _compute_pathfinding_block_ids(self, seq_len: int, d: int, l: int, block_size: int, device: torch.device) -> torch.Tensor:
        """
        计算全局唯一的 Block IDs（Edges/Query/Eq 为 -1, Path 按块分配全局唯一 ID）

        序列结构：(Edges '/' Query '=' Path) per unit
        UNIT_LEN = (l-1)*d*2 + 1 + 2 + 1 + l

        对于 BAD-AR：
        - Edges 部分：标记为 -1（背景，永远可见）
        - '/' 部分：标记为 -1（背景，永远可见）
        - Query 部分：标记为 -1（背景，永远可见）
        - '=' 部分：标记为 -1（背景，永远可见）
        - Path 部分：按 block_size 分配全局唯一正数 ID

        Args:
            seq_len: 总 token 数
            d: 分支数
            l: 路径长度
            block_size: 块大小
            device: torch device

        Returns:
            block_ids: [seq_len] tensor，每个 token 的全局唯一 block ID
        """
        UNIT_LEN = (l - 1) * d * 2 + 1 + 2 + 1 + l
        edge_len = (l - 1) * d * 2  # 修正：l 表示路径节点数，边数为 (l-1) * d
        path_start_in_unit = edge_len + 1 + 2 + 1  # edges + '/' + query + '='

        pos = torch.arange(seq_len, device=device)
        unit_idx = pos // UNIT_LEN
        pos_in_unit = pos % UNIT_LEN

        is_path = (pos_in_unit >= path_start_in_unit)
        block_ids = torch.full((seq_len,), -1, device=device, dtype=torch.long)

        if is_path.any():
            path_relative_pos = pos_in_unit[is_path] - path_start_in_unit  # Path 部分从 0 开始
            inner_block_id = path_relative_pos // block_size
            num_blocks_per_path = (l + block_size - 1) // block_size
            # 全局唯一 ID：unit_idx * num_blocks_per_path + inner_block_id
            block_ids[is_path] = unit_idx[is_path] * num_blocks_per_path + inner_block_id

        return block_ids

    def _create_bad_ar_attention_bias(self, b, total_points, n_prompt, d, l, block_size,
                                     masked_block_indices, device, dtype, respond_indices_batch=None):
        """
        创建 Pathfinding-Aware BAD-AR attention bias（全向量化版本）

        核心逻辑：
        1. 只有 Path 部分参与 Block 划分
        2. Edges/Query/Eq 部分永远可见（block_id = -1）
        3. 未被 Mask 的 Path 块永远可见
        4. 被 Mask 的 Path 块：块间互不可见，块内 AR

        Args:
            b: batch size
            total_points: 总点数
            n_prompt: prompt 点数
            d: 分支数
            l: 路径长度
            block_size: 块大小
            masked_block_indices: [b, num_blocks] bool tensor
            device: torch device
            dtype: torch dtype
            respond_indices_batch: 未使用

        Returns:
            attention_bias: [b, n_head, seq_len, seq_len]
        """
        UNIT_LEN = (l - 1) * d * 2 + 1 + 2 + 1 + l
        seq_len = total_points * UNIT_LEN
        idx_range = torch.arange(seq_len, device=device)

        # 1. 获取全局 Block IDs [seq_len]
        block_ids = self._compute_pathfinding_block_ids(seq_len, d, l, block_size, device)

        # 2. 索引映射技巧：建立位置到 Masked 状态的映射 [b, seq_len]
        pad = torch.zeros((b, 1), dtype=torch.bool, device=device)
        full_mask_lookup = torch.cat([pad, masked_block_indices], dim=1)  # [b, num_blocks + 1]

        # pos_is_masked[b, seq_len]: 通过 block_ids+1 映射
        pos_is_masked = full_mask_lookup[:, block_ids + 1]

        # 3. 向量化构建规则 [b, seq, seq]
        q_idx = idx_range.view(1, -1, 1).expand(b, -1, -1)  # [b, seq_len, 1]
        k_idx = idx_range.view(1, 1, -1).expand(b, -1, -1)  # [b, 1, seq_len]
        q_block = block_ids.view(1, -1, 1).expand(b, -1, -1)  # [b, seq_len, 1]
        k_block = block_ids.view(1, 1, -1).expand(b, -1, -1)  # [b, 1, seq_len]

        k_is_masked = pos_is_masked.unsqueeze(1)  # [b, 1, seq_len]
        q_is_masked = pos_is_masked.unsqueeze(2)  # [b, seq_len, 1]

        # 可见性判断（完全向量化）
        is_context_k = (k_block == -1).expand(-1, seq_len, -1)  # Edges/Query/Eq 永远可见
        is_visible_path_k = ((~k_is_masked) & (k_block != -1)).expand(-1, seq_len, -1)  # 未被 mask 的 Path 块
        is_intra_ar = (q_block == k_block) & (k_block != -1) & q_is_masked.expand(-1, -1, seq_len) & (k_idx <= q_idx)  # 同一块内 AR

        is_visible = is_context_k | is_visible_path_k | is_intra_ar  # [b, seq_len, seq_len]

        # 4. 构造 Bias 并显式包含 n_head 维度
        bias = torch.where(
            is_visible.unsqueeze(1).expand(-1, self.n_head, -1, -1),  # [b, n_head, seq_len, seq_len]
            torch.zeros((b, self.n_head, seq_len, seq_len), device=device, dtype=dtype),
            torch.full((b, self.n_head, seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        )

        return bias

    def forward(self, xs, ys, train_mode=True, respond_position_mask=None):
        """
        前向传播

        Args:
            xs: [B, n_points, edge_len + 2]
            ys: [B, n_points, path_len]
            train_mode: 训练模式（保持接口一致）
            respond_position_mask: respond位置mask（保持接口一致）

        Returns:
            logits: [B, n_respond, path_len, vocab_size]
            loss: scalar tensor
        """
        B = xs.shape[0]
        device = xs.device
        d = self.degree
        l = self.path_len

        # 构建序列
        full_sequence, prefix_len = self._build_pathfinding_sequence(xs, ys, d, l, device)

        # Token embeddings
        embeds = self._read_in(full_sequence)

        # 注入 role embeddings
        total_points = self.n_prompt + self.n_respond
        embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)

        if self.training:
            # 训练模式：采样时间步并 mask blocks
            t = torch.randint(0, self.num_timesteps, (B,), device=device)

            # 计算每个 respond unit 的 Path 块数
            num_blocks_per_path = (l + self.block_size - 1) // self.block_size
            total_respond_blocks = self.n_respond * num_blocks_per_path

            # 计算 mask 概率
            mask_prob = (t.float() + 1.0) / float(self.num_timesteps)

            # 随机 mask blocks
            masked_block_indices = torch.rand(B, total_respond_blocks, device=device) < mask_prob.unsqueeze(1)

            # 创建 attention bias
            attention_bias = self._create_bad_ar_attention_bias(
                B, total_points, self.n_prompt, d, l, self.block_size,
                masked_block_indices, device, embeds.dtype, None
            )

            # Backbone
            dummy_input_ids = torch.zeros(B, embeds.shape[1], dtype=torch.long, device=device)

            out = self._backbone(

                input_ids=dummy_input_ids,

                input_embeddings=embeds,
                attention_bias=attention_bias,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)

            # 提取 Path logits
            UNIT_LEN = self._compute_unit_len(d, l)
            path_logits_list = []
            for i in range(self.n_respond):
                unit_start = (self.n_prompt + i) * UNIT_LEN
                path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
                path_end = path_start + l
                path_logits = vocab_logits[:, path_start:path_end, :]
                path_logits_list.append(path_logits)

            logits = torch.stack(path_logits_list, dim=1)

            # 计算 loss
            target_path = ys[:, self.n_prompt:, :]
            mask = torch.ones_like(target_path, dtype=torch.float32)

            logits_flat = logits.reshape(B * self.n_respond, l, self.vocab_size)
            target_flat = target_path.reshape(B * self.n_respond, l)
            mask_flat = mask.reshape(B * self.n_respond, l)

            t_expanded = t.unsqueeze(1).expand(B, self.n_respond).reshape(B * self.n_respond)

            loss = self._compute_pathfinding_loss(
                logits_flat, target_flat, mask_flat,
                t=t_expanded,
                mode=self.loss_mode,
                alpha=self.alpha,
                gamma=self.gamma,
                num_timesteps=self.num_timesteps,
            )

            return loss, logits
        else:
            # 推理模式
            if self.use_multistep_inference:
                # 多步推理
                return self._multistep_inference(xs, ys)

            # 单步推理：需要 mask path 部分，避免数据泄漏
            # 🔧 修复：在推理时将 path 部分 mask 掉，避免看到目标答案
            UNIT_LEN = self._compute_unit_len(d, l)
            for i in range(self.n_respond):
                unit_start = (self.n_prompt + i) * UNIT_LEN
                path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
                path_end = path_start + l
                full_sequence[:, path_start:path_end] = self.mask_token_id
            
            # 重新计算 embeddings（因为序列已改变）
            embeds = self._read_in(full_sequence)
            embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)
            
            # 创建 attention bias（全部可见，因为已经 mask 了 path）
            # 计算每个 respond unit 的 Path 块数
            num_blocks_per_path = (l + self.block_size - 1) // self.block_size
            total_respond_blocks = self.n_respond * num_blocks_per_path
            
            # 全部 mask（推理时所有 block 都被 mask）
            masked_block_indices = torch.ones(B, total_respond_blocks, device=device, dtype=torch.bool)
            
            # 创建 attention bias
            attention_bias = self._create_bad_ar_attention_bias(
                B, total_points, self.n_prompt, d, l, self.block_size,
                masked_block_indices, device, embeds.dtype, None
            )
            
            # 重新 forward（因为序列已改变）
            dummy_input_ids = torch.zeros(B, embeds.shape[1], dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                attention_bias=attention_bias,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)

            # 重新提取 Path logits
            path_logits_list = []
            for i in range(self.n_respond):
                unit_start = (self.n_prompt + i) * UNIT_LEN
                path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
                path_end = path_start + l
                path_logits = vocab_logits[:, path_start:path_end, :]
                path_logits_list.append(path_logits)

    def _multistep_inference(self, xs, ys):
        """多步去噪推理（路由方法）"""
        if self.use_dynamic_scheduler:
            return self._multistep_inference_dynamic(xs, ys)
        else:
            return self._multistep_inference_fixed_k(xs, ys)

    def _multistep_inference_fixed_k(self, xs, ys):
        """多步去噪推理（固定 k 模式）"""
        B = xs.shape[0]
        device = xs.device
        d = self.degree
        l = self.path_len

        # 初始化：全部 mask
        full_sequence, prefix_len = self._build_pathfinding_sequence(xs, ys, d, l, device)
        input_ids = full_sequence.clone()

        UNIT_LEN = self._compute_unit_len(d, l)
        for i in range(self.n_respond):
            unit_start = (self.n_prompt + i) * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            input_ids[:, path_start:path_end] = self.mask_token_id

        # 初始化 mask 状态
        masked_indices = torch.ones(B, l, device=device, dtype=torch.bool)

        # 迭代去噪
        for step in range(self.inference_steps):
            if masked_indices.sum() == 0:
                break

            # Forward
            embeds = self._read_in(input_ids)
            total_points = self.n_prompt + self.n_respond
            embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)

            # 创建 attention bias（全部 mask）
            num_blocks_per_path = (l + self.block_size - 1) // self.block_size
            total_respond_blocks = self.n_respond * num_blocks_per_path
            masked_block_indices = torch.ones(B, total_respond_blocks, device=device, dtype=torch.bool)

            attention_bias = self._create_bad_ar_attention_bias(
                B, total_points, self.n_prompt, d, l, self.block_size,
                masked_block_indices, device, embeds.dtype, None
            )

            dummy_input_ids = torch.zeros(B, input_ids.shape[1], dtype=torch.long, device=device)

            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                attention_bias=attention_bias,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)

            # 提取 Path logits（仅第一个 respond unit）
            unit_start = self.n_prompt * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            logits = vocab_logits[:, path_start:path_end, :]  # [B, l, vocab_size]

            # 计算置信度
            probs = F.softmax(logits, dim=-1)
            confidence = -probs.max(dim=-1)[0]  # [B, l]
            confidence = confidence.masked_fill(~masked_indices, float("inf"))

            # 选择 top-k 最不确定的位置
            for b in range(B):
                unfilled = masked_indices[b].nonzero(as_tuple=True)[0]
                if unfilled.numel() == 0:
                    continue

                k = min(self.inference_k_per_step, unfilled.numel())
                unfilled_confidence = confidence[b, unfilled]
                _, topk = torch.topk(unfilled_confidence, k=k, largest=False)
                cells = unfilled[topk]
                pred_nodes = torch.argmax(logits[b, cells, :], dim=-1)
                input_ids[b, path_start + cells] = pred_nodes
                masked_indices[b, cells] = False

        # Final forward
        embeds = self._read_in(input_ids)
        total_points = self.n_prompt + self.n_respond
        embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)

        num_blocks_per_path = (l + self.block_size - 1) // self.block_size
        total_respond_blocks = self.n_respond * num_blocks_per_path
        masked_block_indices = torch.ones(B, total_respond_blocks, device=device, dtype=torch.bool)

        attention_bias = self._create_bad_ar_attention_bias(
            B, total_points, self.n_prompt, d, l, self.block_size,
            masked_block_indices, device, embeds.dtype, None
        )

        dummy_input_ids = torch.zeros(B, input_ids.shape[1], dtype=torch.long, device=device)

        out = self._backbone(
            input_ids=dummy_input_ids,
            input_embeddings=embeds,
            attention_bias=attention_bias,
            output_hidden_states=True,
        )
        h = out.hidden_states[-1]
        vocab_logits = self._read_out(h)

        unit_start = self.n_prompt * UNIT_LEN
        path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
        path_end = path_start + l
        logits = vocab_logits[:, path_start:path_end, :]
        final_logits = logits.unsqueeze(1)  # [B, 1, l, vocab_size]

        mask = torch.ones(B, self.n_respond, l, device=device, dtype=torch.bool)
        return final_logits, mask

    def _multistep_inference_dynamic(self, xs, ys):
        """多步去噪推理（动态 scheduler 模式，类似 ICL）"""
        from dllm.utils.generation_utils import get_num_transfer_tokens

        B = xs.shape[0]
        device = xs.device
        d = self.degree
        l = self.path_len

        # 初始化：全部 mask
        full_sequence, prefix_len = self._build_pathfinding_sequence(xs, ys, d, l, device)
        input_ids = full_sequence.clone()

        UNIT_LEN = self._compute_unit_len(d, l)
        for i in range(self.n_respond):
            unit_start = (self.n_prompt + i) * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            input_ids[:, path_start:path_end] = self.mask_token_id

        # 初始化 mask 状态
        masked_indices = torch.ones(B, l, device=device, dtype=torch.bool)
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

            # Forward
            embeds = self._read_in(input_ids)
            total_points = self.n_prompt + self.n_respond
            embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)

            # 创建 attention bias（全部 mask）
            num_blocks_per_path = (l + self.block_size - 1) // self.block_size
            total_respond_blocks = self.n_respond * num_blocks_per_path
            masked_block_indices = torch.ones(B, total_respond_blocks, device=device, dtype=torch.bool)

            attention_bias = self._create_bad_ar_attention_bias(
                B, total_points, self.n_prompt, d, l, self.block_size,
                masked_block_indices, device, embeds.dtype, None
            )

            dummy_input_ids = torch.zeros(B, input_ids.shape[1], dtype=torch.long, device=device)

            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                attention_bias=attention_bias,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)

            # 提取 Path logits（仅第一个 respond unit）
            unit_start = self.n_prompt * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            logits = vocab_logits[:, path_start:path_end, :]  # [B, l, vocab_size]
            final_logits = logits.unsqueeze(1)  # [B, 1, l, vocab_size]

            # 计算置信度
            if self.inference_confidence_alg == "entropy":
                probs = F.softmax(logits, dim=-1)
                confidence = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [B, l]
            else:
                probs = F.softmax(logits, dim=-1)
                confidence = -probs.max(dim=-1)[0]  # [B, l]

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
                pred_nodes = torch.argmax(logits[b, cells, :], dim=-1)
                input_ids[b, path_start + cells] = pred_nodes
                masked_indices[b, cells] = False

        if final_logits is None:
            # Final forward
            embeds = self._read_in(input_ids)
            total_points = self.n_prompt + self.n_respond
            embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)

            num_blocks_per_path = (l + self.block_size - 1) // self.block_size
            total_respond_blocks = self.n_respond * num_blocks_per_path
            masked_block_indices = torch.ones(B, total_respond_blocks, device=device, dtype=torch.bool)

            attention_bias = self._create_bad_ar_attention_bias(
                B, total_points, self.n_prompt, d, l, self.block_size,
                masked_block_indices, device, embeds.dtype, None
            )

            dummy_input_ids = torch.zeros(B, input_ids.shape[1], dtype=torch.long, device=device)

            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                attention_bias=attention_bias,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)
            unit_start = self.n_prompt * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            logits = vocab_logits[:, path_start:path_end, :]
            final_logits = logits.unsqueeze(1)

        mask = torch.ones(B, self.n_respond, l, device=device, dtype=torch.bool)
        return final_logits, mask


# ============================================================
# PathfindingRBOAR: Random Order Generation 模型
# ============================================================

class PathfindingRBOAR(RBOARPromptRespond, PathfindingMixin):
    """
    路径规划专用 RBO-AR 模型（Random Block Order Autoregressive）

    特性：
    1. 随机顺序生成：不按固定顺序生成路径节点
    2. BPD 推理：基于熵引导的自适应推理
    3. 块级随机顺序
    """

    def __init__(
        self,
        n_dims,
        n_positions=1000,
        n_embd=256,
        n_layer=12,
        n_head=8,
        n_prompt=5,
        n_respond=1,
        *,
        mlp_ratio=4.0,
        block_group_size=1,
        # 路径规划特定参数
        num_nodes=100,
        degree=5,
        path_len=10,
        use_role_embedding=True,
        # RBO-AR 参数
        block_size=1,
        loss_mode="ce",
        alpha=0.25,
        gamma=1.0,
        num_timesteps=20,
        # BPD 推理参数
        use_bpd_inference=True,
        bpd_steps=20,
        bpd_k_per_step=1,
        **extra,
    ):
        # 过滤掉父类不支持的参数
        parent_kwargs = _filter_parent_kwargs(extra)

        # 调用父类初始化
        super().__init__(
            n_dims=n_dims,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_prompt=n_prompt,
            n_respond=n_respond,
            mlp_ratio=mlp_ratio,
            block_group_size=block_group_size,
            block_size=block_size,
            **parent_kwargs,
        )

        self.name = "pathfinding_rboar"
        self.family = "pathfinding_rboar"  # 用于评估代码识别任务类型
        self.degree = degree
        self.path_len = path_len
        self.block_size = block_size
        self.loss_mode = loss_mode
        self.alpha = alpha
        self.gamma = gamma
        self.num_timesteps = num_timesteps

        # BPD 推理参数
        self.use_bpd_inference = use_bpd_inference
        self.bpd_steps = bpd_steps
        self.bpd_k_per_step = bpd_k_per_step

        # 使用 Mixin 设置路径规划协议
        self.setup_pathfinding_protocol(
            n_embd=n_embd,
            num_nodes=num_nodes,
            max_path_len=path_len,  # 修正：path_len 表示节点数
            use_role_embedding=use_role_embedding
        )

        print(f"[PathfindingRBOAR] Initialized:")
        print(f"  vocab_size: {self.vocab_size}")
        print(f"  degree: {degree}, path_len: {path_len}")
        print(f"  block_size: {block_size}")
        print(f"  Loss mode: {loss_mode}")
        if use_bpd_inference:
            print(f"  BPD inference: enabled (steps={bpd_steps}, k={bpd_k_per_step})")





    def _compute_pathfinding_block_ids(self, seq_len: int, d: int, l: int, block_size: int, device: torch.device) -> torch.Tensor:
        """计算 block IDs（与 BADAR 相同）"""
        UNIT_LEN = (l - 1) * d * 2 + 1 + 2 + 1 + l
        edge_len = (l - 1) * d * 2  # 修正：l 表示路径节点数，边数为 (l-1) * d
        path_start_in_unit = edge_len + 1 + 2 + 1

        pos = torch.arange(seq_len, device=device)
        unit_idx = pos // UNIT_LEN
        pos_in_unit = pos % UNIT_LEN

        is_path = (pos_in_unit >= path_start_in_unit)
        block_ids = torch.full((seq_len,), -1, device=device, dtype=torch.long)

        if is_path.any():
            path_relative_pos = pos_in_unit[is_path] - path_start_in_unit
            inner_block_id = path_relative_pos // block_size
            num_blocks_per_path = (l + block_size - 1) // block_size
            block_ids[is_path] = unit_idx[is_path] * num_blocks_per_path + inner_block_id

        return block_ids

    def forward(self, xs, ys, train_mode=True, respond_position_mask=None):
        """
        前向传播

        Args:
            xs: [B, n_points, edge_len + 2]
            ys: [B, n_points, path_len]
            train_mode: 训练模式（保持接口一致）
            respond_position_mask: respond位置mask（保持接口一致）

        Returns:
            logits: [B, n_respond, path_len, vocab_size]
            loss: scalar tensor
        """
        B = xs.shape[0]
        device = xs.device
        d = self.degree
        l = self.path_len

        # 构建序列
        full_sequence, prefix_len = self._build_pathfinding_sequence(xs, ys, d, l, device)

        # Token embeddings
        embeds = self._read_in(full_sequence)

        # 注入 role embeddings
        total_points = self.n_prompt + self.n_respond
        embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)

        if self.training:
            # 训练模式：随机顺序生成
            # 生成随机 block 顺序
            num_blocks_per_path = (l + self.block_size - 1) // self.block_size
            total_respond_blocks = self.n_respond * num_blocks_per_path

            # 为每个 batch 生成随机顺序
            respond_indices_batch = []
            for _ in range(B):
                perm = torch.randperm(total_respond_blocks, device=device)
                respond_indices_batch.append(perm)
            respond_indices_batch = torch.stack(respond_indices_batch, dim=0)  # [B, total_respond_blocks]

            # 创建 attention bias（使用 RBO-AR 的逻辑）
            attention_bias = self._create_rbo_ar_attention_bias(
                B, total_points, self.n_prompt, d, l, self.block_size,
                respond_indices_batch, device, embeds.dtype
            )

            # Backbone
            dummy_input_ids = torch.zeros(B, embeds.shape[1], dtype=torch.long, device=device)

            out = self._backbone(

                input_ids=dummy_input_ids,

                input_embeddings=embeds,
                attention_bias=attention_bias,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)

            # 提取 Path logits
            UNIT_LEN = self._compute_unit_len(d, l)
            path_logits_list = []
            for i in range(self.n_respond):
                unit_start = (self.n_prompt + i) * UNIT_LEN
                path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
                path_end = path_start + l
                path_logits = vocab_logits[:, path_start:path_end, :]
                path_logits_list.append(path_logits)

            logits = torch.stack(path_logits_list, dim=1)

            # 计算 loss
            target_path = ys[:, self.n_prompt:, :]
            mask = torch.ones_like(target_path, dtype=torch.float32)

            logits_flat = logits.reshape(B * self.n_respond, l, self.vocab_size)
            target_flat = target_path.reshape(B * self.n_respond, l)
            mask_flat = mask.reshape(B * self.n_respond, l)

            loss = self._compute_pathfinding_loss(
                logits_flat, target_flat, mask_flat,
                mode=self.loss_mode,
                alpha=self.alpha,
                gamma=self.gamma,
                num_timesteps=self.num_timesteps,
            )

            return loss, logits
        else:
            # 推理模式
            if self.use_bpd_inference:
                return self.generate_bpd(xs, ys)
            else:
                # 简单推理：需要 mask path 部分，避免数据泄漏
                # 🔧 修复：在推理时将 path 部分 mask 掉，避免看到目标答案
                UNIT_LEN = self._compute_unit_len(d, l)
                for i in range(self.n_respond):
                    unit_start = (self.n_prompt + i) * UNIT_LEN
                    path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
                    path_end = path_start + l
                    full_sequence[:, path_start:path_end] = self.mask_token_id
                
                # 重新计算 embeddings（因为序列已改变）
                embeds = self._read_in(full_sequence)
                embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)
                
                # 创建 attention bias（使用顺序生成，因为已经 mask 了 path）
                # 计算每个 respond unit 的 Path 块数
                num_blocks_per_path = (l + self.block_size - 1) // self.block_size
                total_respond_blocks = self.n_respond * num_blocks_per_path
                
                # 顺序生成（sequential order）
                respond_indices_batch = []
                for _ in range(B):
                    perm = torch.arange(total_respond_blocks, device=device)
                    respond_indices_batch.append(perm)
                respond_indices_batch = torch.stack(respond_indices_batch, dim=0)  # [B, total_respond_blocks]
                
                # 创建 attention bias
                attention_bias = self._create_rbo_ar_attention_bias(
                    B, total_points, self.n_prompt, d, l, self.block_size,
                    respond_indices_batch, device, embeds.dtype
                )
                
                # 重新 forward（因为序列已改变）
                dummy_input_ids = torch.zeros(B, embeds.shape[1], dtype=torch.long, device=device)
                out = self._backbone(
                    input_ids=dummy_input_ids,
                    input_embeddings=embeds,
                    attention_bias=attention_bias,
                    output_hidden_states=True,
                )
                h = out.hidden_states[-1]
                vocab_logits = self._read_out(h)

                # 重新提取 Path logits
                path_logits_list = []
                for i in range(self.n_respond):
                    unit_start = (self.n_prompt + i) * UNIT_LEN
                    path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
                    path_end = path_start + l
                    path_logits = vocab_logits[:, path_start:path_end, :]
                    path_logits_list.append(path_logits)

                logits = torch.stack(path_logits_list, dim=1)  # [B, n_respond, l, vocab_size]
                mask = torch.ones(B, self.n_respond, l, device=device, dtype=torch.bool)

                return logits, mask

    def _create_rbo_ar_attention_bias(self, b, total_points, n_prompt, d, l, block_size,
                                     respond_indices_batch, device, dtype):
        """
        创建 RBO-AR attention bias（随机块顺序）

        Args:
            b: batch size
            total_points: 总点数
            n_prompt: prompt 点数
            d: 分支数
            l: 路径长度
            block_size: 块大小
            respond_indices_batch: [b, num_blocks] 每个 batch 的块顺序
            device: torch device
            dtype: torch dtype

        Returns:
            attention_bias: [b, n_head, seq_len, seq_len]
        """
        UNIT_LEN = (l - 1) * d * 2 + 1 + 2 + 1 + l
        seq_len = total_points * UNIT_LEN
        idx_range = torch.arange(seq_len, device=device)

        # 获取 block IDs
        block_ids = self._compute_pathfinding_block_ids(seq_len, d, l, block_size, device)

        # 计算每个块的生成顺序
        num_blocks_per_path = (l + block_size - 1) // block_size
        prompt_blocks = n_prompt * num_blocks_per_path

        # 为每个 block 分配顺序索引
        block_order = torch.full((b, seq_len), -1, dtype=torch.long, device=device)
        for batch_idx in range(b):
            for pos_idx in range(seq_len):
                bid = block_ids[pos_idx].item()
                if bid >= prompt_blocks:  # respond block
                    respond_bid = bid - prompt_blocks
                    order = (respond_indices_batch[batch_idx] == respond_bid).nonzero(as_tuple=True)[0]
                    if len(order) > 0:
                        block_order[batch_idx, pos_idx] = order[0].item()

        # 构建 attention mask
        q_idx = idx_range.view(1, -1, 1).expand(b, -1, -1)
        k_idx = idx_range.view(1, 1, -1).expand(b, -1, -1)
        q_block = block_ids.view(1, -1, 1).expand(b, -1, -1)
        k_block = block_ids.view(1, 1, -1).expand(b, -1, -1)
        q_order = block_order.unsqueeze(2).expand(-1, -1, seq_len)
        k_order = block_order.unsqueeze(1).expand(-1, seq_len, -1)

        # 可见性规则
        is_context_k = (k_block == -1).expand(-1, seq_len, -1)  # Edges/Query/Eq 永远可见
        is_earlier_block = (k_order < q_order) & (k_order >= 0) & (q_order >= 0)  # 更早生成的块可见
        is_same_block_ar = (q_block == k_block) & (k_block >= prompt_blocks) & (k_idx <= q_idx)  # 同块内 AR

        is_visible = is_context_k | is_earlier_block | is_same_block_ar

        # 构造 bias
        bias = torch.where(
            is_visible.unsqueeze(1).expand(-1, self.n_head, -1, -1),
            torch.zeros((b, self.n_head, seq_len, seq_len), device=device, dtype=dtype),
            torch.full((b, self.n_head, seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        )

        return bias

    @torch.no_grad()
    def generate_bpd(self, xs, ys):
        """
        BPD 推理：基于熵引导的自适应推理

        在每一步中：
        1. 计算所有未填充位置的熵
        2. 选择熵最低的 k 个位置
        3. 填充这些位置
        4. 重复直到所有位置都被填充

        Args:
            xs: [B, n_points, edge_len + 2]
            ys: [B, n_points, path_len + 1]

        Returns:
            logits: [B, n_respond, path_len, vocab_size]
            loss: scalar tensor (0.0)
        """
        B = xs.shape[0]
        device = xs.device
        d = self.degree
        l = self.path_len

        # 初始化序列
        full_sequence, prefix_len = self._build_pathfinding_sequence(xs, ys, d, l, device)
        input_ids = full_sequence.clone()

        # 初始化 Path 部分为 MASK（仅 respond units）
        UNIT_LEN = self._compute_unit_len(d, l)
        for i in range(self.n_respond):
            unit_start = (self.n_prompt + i) * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            input_ids[:, path_start:path_end] = self.mask_token_id

        # 跟踪已填充的位置
        filled = torch.zeros(B, l, dtype=torch.bool, device=device)
        final_logits = None

        steps = max(1, self.bpd_steps)
        k_per_step = max(1, self.bpd_k_per_step)

        for step in range(steps):
            if filled.all():
                break

            # Forward pass
            embeds = self._read_in(input_ids)
            total_points = self.n_prompt + self.n_respond
            embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)

            dummy_input_ids = torch.zeros(B, embeds.shape[1], dtype=torch.long, device=device)


            out = self._backbone(


                input_ids=dummy_input_ids,


                input_embeddings=embeds,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)

            # 提取 Path logits（仅第一个 respond unit）
            unit_start = self.n_prompt * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            logits = vocab_logits[:, path_start:path_end, :]  # [B, l, vocab_size]
            final_logits = logits.unsqueeze(1)  # [B, 1, l, vocab_size]

            # 计算熵
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [B, l]
            entropy = entropy.masked_fill(filled, float("inf"))

            # Fill top-k lowest entropy
            for b in range(B):
                unfilled = (~filled[b]).nonzero(as_tuple=True)[0]
                if unfilled.numel() == 0:
                    continue
                k = min(k_per_step, unfilled.numel())
                unfilled_entropy = entropy[b, unfilled]
                _, topk = torch.topk(unfilled_entropy, k=k, largest=False)
                cells = unfilled[topk]
                pred_nodes = torch.argmax(logits[b, cells, :], dim=-1)
                input_ids[b, path_start + cells] = pred_nodes
                filled[b, cells] = True

        if final_logits is None:
            # Final forward pass
            embeds = self._read_in(input_ids)
            total_points = self.n_prompt + self.n_respond
            embeds = self._inject_pathfinding_roles(embeds, d, l, total_points, device)
            dummy_input_ids = torch.zeros(B, embeds.shape[1], dtype=torch.long, device=device)

            out = self._backbone(

                input_ids=dummy_input_ids,

                input_embeddings=embeds,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            vocab_logits = self._read_out(h)
            unit_start = self.n_prompt * UNIT_LEN
            path_start = unit_start + (l - 1) * d * 2 + 1 + 2 + 1
            path_end = path_start + l
            logits = vocab_logits[:, path_start:path_end, :]
            final_logits = logits.unsqueeze(1)

        mask = torch.ones(B, 1, l, device=device, dtype=torch.bool)
        return final_logits, mask


def build_pathfinding_model(conf):
    """
    构建路径查找模型

    Args:
        conf: 模型配置字典

    Returns:
        路径查找模型实例
    """
    family = conf.get('family', '')

    if family in ['pathfinding_ar', 'ar']:
        return PathfindingAR(**conf)
    elif family == 'pathfinding_llada':
        return PathfindingLLaDA(**conf)
    elif family == 'pathfinding_llada_block':
        return PathfindingLLaDABlock(**conf)
    elif family == 'pathfinding_bopar':
        return PathfindingBOPAR(**conf)
    elif family == 'pathfinding_badar':
        return PathfindingBADAR(**conf)
    elif family == 'pathfinding_rboar':
        return PathfindingRBOAR(**conf)
    else:
        raise NotImplementedError(f"不支持的路径查找模型类型: {family}")
