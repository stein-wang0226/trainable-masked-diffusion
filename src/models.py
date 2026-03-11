import torch
from torch import nn
# 确保能 import 到仓库根下的 dllm 包
import os, sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ✅ 确保能 import 到仓库根下的 dllm 包（向后兼容：只在路径不存在时添加）
# 使用规范化路径比较，避免重复添加
repo_root_norm = os.path.normpath(repo_root)
dllm_path = os.path.join(repo_root, "dllm")
dllm_path_norm = os.path.normpath(dllm_path)

# 检查路径是否已存在（使用规范化路径比较，兼容不同路径表示）
if not any(os.path.normpath(p) == repo_root_norm for p in sys.path):
    sys.path.insert(0, repo_root)
if not any(os.path.normpath(p) == dllm_path_norm for p in sys.path):
    sys.path.insert(0, dllm_path)

import math

from base_models import NeuralNetwork, ParallelNetworks
from dllm.pipelines.dream.models.configuration_dream import DreamConfig
from dllm.pipelines.dream.models.modeling_dream import DreamBaseModel, DreamModel

from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
import warnings
from sklearn import tree

import xgboost as xgb
from transformers import (
    AutoConfig, AutoModel,
    GPT2Config, GPT2Model,
    GPTJConfig, GPTJModel,
)
# ===== HuggingFace Transformers =====
from transformers import (
    AutoConfig, AutoModel,
    GPT2Config, GPT2Model,
    GPTJConfig, GPTJModel,
)
try:
    from transformers import Qwen2Config, Qwen2Model
except ImportError:
    Qwen2Config, Qwen2Model = None, None

# ===== LLaMA (Llama2 / Llama3 / Llama3.1 均兼容) =====
try:
    from transformers import LlamaConfig, LlamaModel
except ImportError:
    raise ImportError("❌ 请安装 transformers>=4.40 以支持 LLaMA 模型。")
try:
    from transformers import Qwen2Config, Qwen2Model
except ImportError:
    Qwen2Config, Qwen2Model = None, None

from base_models import NeuralNetwork, ParallelNetworks
# >>> 关键：导入 LLaDA 基础模型与配置（不是 LM 包装）
from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig
from dllm.pipelines.llada.models.modeling_llada import LLaDAModel as _LLaDABase
from dllm.core.schedulers import BaseAlphaScheduler, LinearAlphaScheduler


def _combine_xs_ys(xs_b, ys_b):
    """
    Interleave (x_i, y_i) -> zs, 并把 y 扩成最后一维第一个槽位。
    xs_b: [B, T, D]
    ys_b: [B, T]
    return: zs [B, 2T, D]
    """
    bsize, points, dim = xs_b.shape
    ys_b_wide = torch.cat(
        (ys_b.view(bsize, points, 1),
         torch.zeros(bsize, points, dim - 1, device=ys_b.device, dtype=xs_b.dtype)),
        dim=2,
    )
    zs = torch.stack((xs_b, ys_b_wide), dim=2).view(bsize, 2 * points, dim)
    return zs

import torch
import torch.nn as nn
from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig
from dllm.pipelines.llada.models.modeling_llada import LLaDAModel as _LLaDABase


import logging
# 配置日志
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128,
                    n_layer=12, n_head=4, type="gpt2", mlp_ratio=4.0,
                pretrained=False, model_name_or_path=None):
        super().__init__()
        self.family = type.lower()  # ✅ 新增
        self.mlp_ratio = mlp_ratio
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        head_dim = n_embd // n_head  # 每个注意力头的维度
        # ===== GPT2 =====
        if type == "gpt2":
            configuration = GPT2Config(
                n_positions=2 * n_positions,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
                use_cache=False,
            )
            self._backbone = GPT2Model(configuration)
        # ===== GPTJ =====
        elif type == "gptJ":
            configuration = GPTJConfig(
                n_positions=2 * n_positions,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                rotary_dim=head_dim,       # ✅ 修复核心：rotary_dim 与 head_dim 对齐
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
                use_cache=False,
            )
            self._backbone = GPTJModel(configuration)
        # ===== LLaMA 家族（支持参数自定义或预训练加载）=====
        elif self.family in ["llama", "llama2", "llama3"]:
            try:
                from transformers import LlamaConfig, LlamaModel
            except ImportError:
                raise ImportError("请安装 `transformers>=4.40` 以支持 LLaMA 模型。")

            if pretrained:
                # ✅ 直接加载预训练权重（大模型用）
                model_id = model_name_or_path or {
                    "llama3": "meta-llama/Meta-Llama-3-8B",
                    "llama2": "meta-llama/Llama-2-7b-hf",
                }.get(self.family, None)

                if model_id is None:
                    raise ValueError(f"Please provide model_name_or_path for pretrained {self.family}.")

                print(f"[Loading pretrained {self.family.upper()} from {model_id}]")
                self._backbone = AutoModel.from_pretrained(model_id)
                n_embd = self._backbone.config.hidden_size

            else:
                # ✅ 自定义配置（实验版）
                if LlamaConfig is None:
                    raise ImportError("Please install transformers>=4.40 for LLaMA.")
                print(f"[Building custom {self.family.upper()} config: "
                      f"d_model={n_embd}, n_layer={n_layer}, n_head={n_head}]")
                configuration = LlamaConfig(
                    hidden_size=n_embd,
                    num_hidden_layers=n_layer,
                    num_attention_heads=n_head,
                    intermediate_size=int(n_embd * mlp_ratio),
                    max_position_embeddings=2 * n_positions,
                    use_cache=False,
                )
                self._backbone = LlamaModel(configuration)


        # ===== Qwen 系列 =====
        elif self.family in ["qwen", "qwen2", "qwen2.5"]:
            # === 合并 QwenModel ===
            if pretrained:
                model_id = model_name_or_path or {
                    "qwen": "Qwen/Qwen-7B",
                    "qwen2": "Qwen/Qwen2-7B-Instruct",
                    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
                }.get(self.family)
                print(f"[Loading pretrained {self.family.upper()} from {model_id}]")
                self._backbone = AutoModel.from_pretrained(model_id)
                n_embd = self._backbone.config.hidden_size
            else:
                if Qwen2Config is None or Qwen2Model is None:
                    raise ImportError("请安装 transformers>=4.40 以支持 Qwen2 系列。")
                config = Qwen2Config(
                    max_position_embeddings=2 * n_positions,
                    hidden_size=n_embd,
                    intermediate_size=int(4 * n_embd),
                    num_hidden_layers=n_layer,
                    num_attention_heads=n_head,
                    num_key_value_heads=n_head,
                    use_cache=False,
                )
                self._backbone = Qwen2Model(config)
        
        else:
            raise ValueError(f"Unsupported model type: {type}")


        self.name = f"{type}_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_positions = n_positions
        self.n_dims = n_dims
        self.hide_last_target = True   # ✅ 新增修复，用于评估时隐藏最后目标 训练有因果注意力不用隐藏

        # ===== 输入输出层 =====
        self._read_in = nn.Linear(n_dims, n_embd)
        self._read_out = nn.Linear(n_embd, 1)

        # ===== 安全维度对齐层 =====
        hidden_size = self._backbone.config.hidden_size
        self._align_proj = nn.Linear(n_embd, hidden_size) if n_embd != hidden_size else nn.Identity()

        print(f"[{type.upper()} Wrapper] n_embd={n_embd}, n_head={n_head}, "
              f"head_dim={head_dim}, rotary_dim={head_dim}, hidden_size={hidden_size}")


    def forward(self, xs, ys, inds=None):
        """
        支持 hide_last_target=True 的公平评估模式。
        xs: [B, T, D]
        ys: [B, T]
        """
        # ✅ 自动同步设备
        device = next(self.parameters()).device
        xs, ys = xs.to(device), ys.to(device)
        if inds is None:
            inds = torch.arange(ys.shape[1], device=device)
        else:
            inds = torch.as_tensor(inds, device=device)
        # ✅ 新增，eval时若开启 hide_last_target 且当前在 eval 模式下，隐藏最后标签 （其实几乎没影响 ）
        if self.hide_last_target and not self.training:
            ys = ys.clone()
            ys[:,-1:] = 0.0 # # 模型在输入中看不到真实答案
        zs = _combine_xs_ys(xs, ys)
        zs = zs.to(device)
        embeds = self._read_in(zs)
        embeds = self._align_proj(embeds)  # 对齐 hidden_size
        ### 手动添加attention_mask，因为是传入 embed
        if not self.training and self.hide_last_target:
            B, T = embeds.size(0), embeds.size(1)
            ## 下三角因果掩码，确保 token_i 只能看到 <= i 的信息
            attention_mask = torch.ones((B, T), device=device, dtype=torch.float32)
            attention_mask[:, -1:] = 0  # 屏蔽最后1个token pos
        else:
            attention_mask = torch.ones((embeds.size(0), embeds.size(1)), device=device)

        try:
            outputs = self._backbone(inputs_embeds=embeds, attention_mask=attention_mask)
        except TypeError:
            # GPT2 / GPTJ 可能不接受 attention_mask=None
            print("err:不接受 attention_mask=None")
            outputs = self._backbone(inputs_embeds=embeds)
        
        
        # === 读出 ===
        h = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]   # [B, 2T, H]
        pred_all = self._read_out(h)[..., 0]   # [B, 2T]

        return pred_all[:, ::2][:, inds]             # [B, |inds|]


## ============================================================
## LLaDA Masked ICL Wrapper V2 - 统一版本（支持单步/多步推理）
## ============================================================
class LLaDAMaskedICLWrapperV2(nn.Module):
    """
    LLaDA Masked Diffusion ICL Wrapper V2 - Unified Version
    -------------------------------------------------------
    ✅ 统一版本：已替代 V1，所有功能都集成在此版本中
    
    核心特性：
    1. 🎯 多步反向扩散推理：r_{t_1} → r_{t_2} → ... → r_0（可选）
    2. 🔄 每步包含：predict → partial unmask → remask → reduce t
    3. 🔧 单步推理：use_multistep_inference=False（与 V1 行为一致）
    4. 🎓 两种训练/推理模式：
       - "curriculum": Curriculum Learning + 完整 mask（训练 0.5→1.0，推理 1.0）
       - "fixed": 固定 mask ratio（训练和推理相同，避免 train-test mismatch）
    
    理论依据：
    - 标准 LLaDA 推理是迭代的反向扩散过程
    - 每步按比例 s/t 随机 remask 部分 token
    - 逐步收敛到最终输出
    
    向后兼容：
    - family="llada" 或 "llada_masked" 会自动使用此版本（默认单步推理）
    - family="llada_v2" 或 "llada_masked_v2" 也使用此版本（默认多步推理）
    """
    
    def __init__(
        self,
        n_dims,
        n_positions,
        n_embd=256,
        n_layer=12,
        n_head=8,
        *,
        mlp_ratio=4.0,
        block_group_size=1,
        scheduler=None,
        mask_epsilon=1e-3,
        loss_weight_type="1/t",
        curriculum_schedule="linear",    # 🎯 新增：从配置文件读取
        train_eval_mask_mode="curriculum",  # 🎯 新增：训练/推理 mask 模式
        fixed_mask_ratio=0.5,           # 🎯 新增：固定 mask ratio 模式的 mask 率
        use_multistep_inference=True,  # 🎯 新增：是否使用多步推理
        inference_steps=10,             # 🎯 新增：推理步数
        inference_step_size=0.1,        # 🎯 新增：每步时间步长
        **extra,
    ):
        super().__init__()
        self.name = "llada_masked_v2_multistep"
        self.n_positions = n_positions
        self.n_dims = n_dims
        self.mask_epsilon = mask_epsilon
        self.loss_weight_type = loss_weight_type
        self.train_eval_mask_mode = train_eval_mask_mode  # "curriculum" 或 "fixed"
        self.fixed_mask_ratio = fixed_mask_ratio  # 固定 mask ratio（仅在 fixed 模式使用）

        self.d_model = int(n_embd)

        # === Backbone Config ===
        cfg = LLaDAConfig(
            n_heads=int(n_head),
            n_layers=int(n_layer),
            kv_heads=int(n_head),
            max_sequence_length=int(2 * n_positions),
            rope=True,
            alibi=False,
            use_cache=False,
            weight_tying=False,
            block_group_size=int(block_group_size),
        )
        cfg.d_model = self.d_model
        cfg.mlp_hidden_size = int(cfg.d_model * mlp_ratio)
        if not hasattr(cfg, "effective_n_kv_heads"):
            cfg.effective_n_kv_heads = getattr(cfg, "kv_heads", cfg.n_heads)
        if not hasattr(cfg, "n_kv_heads"):
            cfg.n_kv_heads = cfg.kv_heads
        self._backbone = _LLaDABase(cfg, init_params=True)

        # === Time Embedding MLP ===
        self._time_mlp = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model)
        )

        self._read_in = nn.Linear(n_dims, cfg.d_model)
        self._read_out = nn.Linear(cfg.d_model, 1)

        # 添加学习掩码 token
        self.mask_embedding = nn.Parameter(torch.randn(1, 1, n_dims))

        # 🎓 Curriculum Learning: 逐渐增加 mask 难度
        self.training_progress = 0.0
        self.curriculum_schedule = curriculum_schedule  # 从参数读取，可选: "linear", "cosine", "exponential"

        # 🎯 多步反向扩散推理参数
        self.use_multistep_inference = use_multistep_inference
        self.inference_steps = inference_steps
        self.inference_step_size = inference_step_size

        mode_info = f"Mode={self.train_eval_mask_mode}"
        if self.train_eval_mask_mode == "fixed":
            mode_info += f", FixedMaskRatio={self.fixed_mask_ratio}"
        print(f"[LLaDA V2] d_model={self.d_model}, Multistep={use_multistep_inference}, Steps={inference_steps}, Curriculum={self.curriculum_schedule}, {mode_info}")

    def _compute_max_t(self):
        """
        计算当前训练进度对应的 max_t（curriculum 的 mask ratio）
        
        Returns:
            max_t: 当前的最大 mask ratio (0.5 → 1.0)
        """
        if self.curriculum_schedule == "linear":
            max_t = 0.5 + 0.5 * self.training_progress
        elif self.curriculum_schedule == "cosine":
            max_t = 0.5 + 0.5 * (1 - math.cos(self.training_progress * math.pi)) / 2
        else:  # exponential
            max_t = 0.5 + 0.5 * (self.training_progress ** 2)
        return max_t

    def _combine(self, xs_b, ys_b_wide):
        """Interleave [x1, y1, x2, y2, ...]"""
        bsize, points, dim = xs_b.shape
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def _forward_single_step(self, xs, ys_pred, t_scalar, masked_indices, device):
        """
        单步前向：用于多步迭代推理
        
        Args:
            xs: [B, N, D]
            ys_pred: [B, N] - 当前预测的 y 值
            t_scalar: [B] - 当前时间步
            masked_indices: [B, N] - 当前哪些位置被 mask
            device: 设备
        
        Returns:
            pred_y: [B, N] - 模型预测
        """
        b, n_points, d = xs.shape
        
        # 构建序列：用预测值替换被 mask 的位置
        ys_input = ys_pred.clone().float().unsqueeze(-1)  # [B, N, 1]
        ys_wide = torch.cat([ys_input, torch.zeros(b, n_points, d - 1, device=device)], dim=2)
        zs = torch.stack((xs, ys_wide), dim=2).view(b, 2 * n_points, d)
        
        # Embedding + Time conditioning
        embeds = self._read_in(zs)
        time_emb = self._time_mlp(t_scalar.view(b, 1, 1).expand(b, 2 * n_points, 1))
        embeds = embeds + time_emb
        
        # 应用 mask
        mask_embed = self._read_in(self.mask_embedding.squeeze(0))
        for i in range(b):
            if masked_indices[i].any():
                y_positions = masked_indices[i].nonzero(as_tuple=False).squeeze(-1)
                full_idx = y_positions * 2 + 1
                embeds[i, full_idx, :] = mask_embed
        
        # Backbone forward
        dummy_input_ids = torch.zeros(b, 2 * n_points, dtype=torch.long, device=device)
        out = self._backbone(
            input_ids=dummy_input_ids,
            input_embeddings=embeds,
            output_hidden_states=True,
        )
        h = out.hidden_states[-1]
        pred_y = self._read_out(h)[:, 0::2, 0]  # [B, N]
        
        return pred_y


    # multistep_inference （drop）
    def _multistep_inference(self, xs, ys, device):
        """
        🎯 多步反向扩散推理：r_{t_1} → r_{t_2} → ... → r_0
        
        理论过程：
        - 初始化：t=1.0，全部 mask
        - 每一步：
          1. 用模型预测所有被 mask 的 token
          2. 部分 unmask：用预测值更新
          3. 按比例 s/t 随机 remask 一部分
          4. 减小 t，继续迭代
        - 最终：收敛到 t≈0，所有 token 都 unmask
        
        Args:
            xs: [B, N, D]
            ys: [B, N, 1] or [B, N] - 仅用于形状，实际不使用真实值
            device: 设备
        
        Returns:
            pred_y: [B, N] - 最终预测
        """
        b, n_points, d = xs.shape
        
        # 初始化：t=1.0，全部 mask
        masked_indices = torch.ones(b, n_points, device=device, dtype=torch.bool)
        ys_pred = torch.zeros(b, n_points, device=device)
        
        step_size = self.inference_step_size
        num_steps = self.inference_steps
        
        for step_idx in range(num_steps):
            # 当前时间步 t
            t_curr = torch.ones(b, device=device) * max(1.0 - step_idx * step_size, self.mask_epsilon)
            
            # Step 1: 预测所有被 mask 的 token
            pred_y = self._forward_single_step(xs, ys_pred, t_curr, masked_indices, device)
            
            # Step 2: 部分 unmask：用预测值更新被 mask 的位置
            ys_pred = torch.where(masked_indices, pred_y, ys_pred)
            
            # Step 3: Remask：按比例 s/t 随机 remask 一部分
            if step_idx < num_steps - 1:  # 最后一步不需要 remask
                # 计算 remask 比例：s/t
                t_next = torch.ones(b, device=device) * max(1.0 - (step_idx + 1) * step_size, self.mask_epsilon)
                remask_ratio = t_next / (t_curr + 1e-8)  # [B]
                remask_ratio = torch.clamp(remask_ratio, min=0.0, max=1.0)
                
                # 重新生成 masked_indices：按照 t_next 的比例随机 mask
                new_masked_indices = torch.rand(b, n_points, device=device) < remask_ratio[:, None]
                masked_indices = new_masked_indices
        
        return ys_pred
    # eval时用了eval中的复写safe_forward，因为要支持两种模式
    def forward(self, xs, ys, train_mode=True):
        """
        LLaDA V2 Forward with Multi-step Inference
        
        xs: [B, N, D]
        ys: [B, N, 1] or [B, N]
        train_mode=True → compute (loss, pred, t)
        train_mode=False → multi-step inference or single-step (based on flag)
        """
        b, n_points, d = xs.shape
        device = xs.device

        # ===== Step 1️⃣: Sample timestep t ∈ [ε,1] =====
        if self.train_eval_mask_mode == "curriculum":
            # 🎓 模式1：Curriculum Learning + 完整 mask
            # 训练时：逐渐增加 mask ratio (0.5 → 1.0)
            # 推理时：两类mode（见eval.py safe_forward）：1. ratio 2. full
            if train_mode:
                # 🎓 Curriculum Learning
                max_t = self._compute_max_t()  # 使用辅助方法计算
                t_scalar = self.mask_epsilon + (max_t - self.mask_epsilon) * torch.rand(b, device=device)
            else:
                # 推理：使用完整 mask (t=1.0) ratio-infer mode在eval中调用fixed mode实现
                t_scalar = torch.ones(b, device=device)
        
        elif self.train_eval_mask_mode == "fixed":
            # 🔧 模式2：固定 mask ratio（训练和推理相同）
            # 训练和推理都使用相同的 mask ratio，避免 train-test mismatch
            if train_mode:
                # 训练时：使用固定 mask ratio（可配置，默认 0.5）
                t_scalar = self.mask_epsilon + (self.fixed_mask_ratio - self.mask_epsilon) * torch.rand(b, device=device)
            else:
                # 推理时：使用相同的固定 mask ratio
                t_scalar = torch.ones(b, device=device) * self.fixed_mask_ratio
        
        else:
            raise ValueError(f"Unknown train_eval_mask_mode: {self.train_eval_mask_mode}. "
                           f"Must be 'curriculum' or 'fixed'")

        # ===== Step 2️⃣: 确定掩码位置 =====
        masked_indices = torch.rand(b, n_points, device=device) < t_scalar[:, None]

        # ===== Step 3️⃣: 构建完整序列（使用真实 y 值）=====
        ys_input = ys.clone().float()
        if ys_input.dim() == 2:
            ys_input = ys_input.unsqueeze(-1)
        
        ys_wide = torch.cat([ys_input, torch.zeros(b, n_points, d - 1, device=device)], dim=2)
        zs = torch.stack((xs, ys_wide), dim=2).view(b, 2 * n_points, d)

        # ===== Step 4️⃣: Embedding + Time conditioning =====
        embeds = self._read_in(zs)
        time_emb = self._time_mlp(t_scalar.view(b, 1, 1).expand(b, 2 * n_points, 1))
        embeds = embeds + time_emb

        # ===== Step 5️⃣: 在 embedding 层应用掩码 =====
        mask_embed = self._read_in(self.mask_embedding.squeeze(0))
        for i in range(b):
            if masked_indices[i].any():
                y_positions = masked_indices[i].nonzero(as_tuple=False).squeeze(-1)
                full_idx = y_positions * 2 + 1
                embeds[i, full_idx, :] = mask_embed

        # ===== Step 6️⃣: Backbone forward =====
        dummy_input_ids = torch.zeros(b, 2 * n_points, dtype=torch.long, device=device)
        out = self._backbone(
            input_ids=dummy_input_ids,
            input_embeddings=embeds,
            output_hidden_states=True,
        )
        h = out.hidden_states[-1]
        pred_y = self._read_out(h)[:, 0::2, 0]  # [B, N]

        # ===== Step 7️⃣: Inference =====
        if not train_mode:
            if self.use_multistep_inference:
                # 🎯 多步反向扩散推理
                return self._multistep_inference(xs, ys, device)
            else:
                # 🔧 单步推理（fallback / fix 版本）
                return pred_y
    
        # ===== Step 8️⃣: Training Loss 计算 =====
        target = ys.squeeze(-1) if ys.dim() == 3 else ys
        diff = pred_y - target
        mask = masked_indices.float()
        
        if self.loss_weight_type == "1/t":
            weight = (1.0 / (t_scalar + 1e-8)).unsqueeze(1)
            weight = torch.clamp(weight, min=0.1, max=10.0)
        elif self.loss_weight_type == "ones":
            weight = torch.ones_like(t_scalar).unsqueeze(1)
        else:
            raise ValueError(f"Unknown loss_weight_type: {self.loss_weight_type}")
        
        per_sample_loss = (diff.square() * mask * weight).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        weighted_loss = per_sample_loss.mean()
        
        return weighted_loss, pred_y, t_scalar


## dream v2 optimized version
class DreamDlmModel(nn.Module):
    """
    Dream v2 - Optimized Version (基于 LLaDA v4 的修复)
    ====================================================
    应用了与 LLaDA v4 相同的优化：
    1. ✅ 降低掩码比例（平均 25%）
    2. ✅ 限制权重范围 [0.1, 10]
    3. ✅ 修复推理时的掩码率
    4. ✅ 改进 Loss 归一化（每样本独立）
    5. ✅ 支持配置参数
    
    - Non-autoregressive (no causal mask)
    - Loss: (1/t) * MSE computed only on masked tokens
    """

    def __init__(
        self,
        n_dims,
        n_positions,
        n_embd=128,
        n_layer=12,
        n_head=4,
        *,
        loss_weight_type="1/t",
        mask_epsilon=1e-3,
        curriculum_schedule="linear",    # 🎯 新增：从配置文件读取
        train_eval_mask_mode="curriculum",  # 🎯 新增：训练/推理 mask 模式
        fixed_mask_ratio=0.5,           # 🎯 新增：固定 mask ratio 模式的 mask 率
    ):
        super(DreamDlmModel, self).__init__()
        self.family = "dream_v2"
        self.name = f"dream_v2_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_positions = n_positions
        self.n_dims = n_dims
        self.loss_weight_type = loss_weight_type
        self.mask_epsilon = mask_epsilon
        self.train_eval_mask_mode = train_eval_mask_mode  # "curriculum" 或 "fixed"
        self.fixed_mask_ratio = fixed_mask_ratio  # 固定 mask ratio（仅在 fixed 模式使用）

        configuration = DreamConfig(
            max_position_embeddings=2 * n_positions,
            hidden_size=n_embd,
            intermediate_size=4 * n_embd,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            num_key_value_heads=n_head,
            use_cache=False,
        )

        # === Model backbone ===
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = DreamModel(configuration)
        self._read_out = nn.Linear(n_embd, 1)

        # === Learnable mask token ===
        self.mask_embedding = nn.Parameter(torch.randn(1, 1, n_dims))

        # 🎓 Curriculum Learning: 逐渐增加 mask 难度
        self.training_progress = 0.0  # 0.0 到 1.0，由外部训练循环更新
        self.curriculum_schedule = curriculum_schedule  # 从参数读取，可选: "linear", "cosine", "exponential"

        mode_info = f"Mode={self.train_eval_mask_mode}"
        if self.train_eval_mask_mode == "fixed":
            mode_info += f", FixedMaskRatio={self.fixed_mask_ratio}"
        print(f"[Dream v2] Optimized with: loss_weight={loss_weight_type}, mask_epsilon={mask_epsilon}, Curriculum={self.curriculum_schedule}, {mode_info}")

    def _compute_max_t(self):
        """
        计算当前训练进度对应的 max_t（curriculum 的 mask ratio）
        
        Returns:
            max_t: 当前的最大 mask ratio (0.5 → 1.0)
        """
        if self.curriculum_schedule == "linear":
            max_t = 0.5 + 0.5 * self.training_progress
        elif self.curriculum_schedule == "cosine":
            max_t = 0.5 + 0.5 * (1 - math.cos(self.training_progress * math.pi)) / 2
        else:  # exponential
            max_t = 0.5 + 0.5 * (self.training_progress ** 2)
        return max_t

    
    def _combine(self, xs_b, ys_b):
        """Interleave [x1,y1,x2,y2,...]"""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def _combine_infilling(self, xs_b):
        """Replace all y's with mask embedding."""
        bsize, points, dim = xs_b.shape
        mask_embeds = self.mask_embedding.expand(bsize, points, dim)
        zs = torch.stack((xs_b, mask_embeds), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None, task_type="diffusion_autoregressive", train_mode=True):
        """
        Dream v2 Optimized Forward (应用 LLaDA v4 的修复)
        
        xs: [B, N, D]
        ys: [B, N, 1] or [B, N]
        train_mode=True  -> return (loss, pred_y, t)
        train_mode=False -> return pred_y only
        
        优化点：
        1. ✅ 降低掩码比例到 25%
        2. ✅ 限制权重范围 [0.1, 10]
        3. ✅ 推理时 t=0.5（而非随机或固定 t=1）
        4. ✅ 每样本独立归一化
        """
        if inds is None:
            inds = torch.arange(ys.shape[1], device=xs.device)
        else:
            inds = torch.tensor(inds, device=xs.device)

        b_size, n_points, n_dims = xs.shape

        if task_type == "diffusion_autoregressive":
            # ===== Step 1️⃣: Sample timestep t ∈ [ε,1] =====
            if self.train_eval_mask_mode == "curriculum":
                # 🎓 模式1：Curriculum Learning + 完整 mask
                # 训练时：逐渐增加 mask ratio (0.5 → 1.0)
                # 推理时：使用完整 mask (t=1.0)
                if train_mode:
                    # 🎓 Curriculum Learning: 逐渐增加最大 mask 率
                    max_t = self._compute_max_t()  # 使用辅助方法计算
                    # t ∈ [ε, max_t]，随训练进度逐渐增加上界
                    t_scalar = self.mask_epsilon + (max_t - self.mask_epsilon) * torch.rand((b_size,), device=xs.device)
                else:
                    # 推理：使用完整 mask (t=1.0) 进行真正的 ICL 测试
                    t_scalar = torch.ones((b_size,), device=xs.device)
            
            elif self.train_eval_mask_mode == "fixed":
                # 🔧 模式2：固定 mask ratio（训练和推理相同）
                # 训练和推理都使用相同的 mask ratio，避免 train-test mismatch
                if train_mode:
                    # 训练时：使用固定 mask ratio（可配置，默认 0.5）
                    t_scalar = self.mask_epsilon + (self.fixed_mask_ratio - self.mask_epsilon) * torch.rand((b_size,), device=xs.device)
                else:
                    # 推理时：使用相同的固定 mask ratio
                    t_scalar = torch.ones((b_size,), device=xs.device) * self.fixed_mask_ratio
            
            else:
                raise ValueError(f"Unknown train_eval_mask_mode: {self.train_eval_mask_mode}. "
                               f"Must be 'curriculum' or 'fixed'") 
            
            # ===== Step 2️⃣: Combine [x, y] sequence =====
            zs = self._combine(xs, ys)
            embeds = self._read_in(zs)

            # ===== Step 3️⃣: 关键修复！直接使用 t 作为掩码概率（与 LLaDA 对齐）=====
            # 原来：使用 cosine schedule，导致掩码率过高（70%-98%）
            # 现在：直接使用 t 作为掩码概率，掩码率 = 10%-50%（与 LLaDA 一致）
            mask_embedding = self._read_in(self.mask_embedding.squeeze(0))
            masked_indices = torch.rand(b_size, n_points, device=xs.device) < t_scalar[:, None]
            
            # 应用掩码到 embedding
            for i in range(b_size):
                if masked_indices[i].any():
                    y_positions = masked_indices[i].nonzero(as_tuple=False).squeeze(-1)
                    full_seq_indices = y_positions * 2 + 1
                    embeds[i, full_seq_indices, :] = mask_embedding

            # ===== Step 4️⃣: Model forward =====
            output = self._backbone.model(inputs_embeds=embeds).last_hidden_state
            pred_y = self._read_out(output)[:, 0::2, 0][:, inds]  # [B, N]

            # ===== Step 5️⃣: Inference =====
            if not train_mode:
                return pred_y
                
            # ===== Step 6️⃣: Loss 计算（优化版）=====
            target = ys.squeeze(-1) if ys.dim() == 3 else ys
            diff = pred_y - target
            mask = masked_indices[:, inds].float()

            # ✅ 修复：限制权重范围
            if self.loss_weight_type == "1/t":
                weight = (1.0 / (t_scalar + 1e-8)).unsqueeze(1)
                weight = torch.clamp(weight, min=0.1, max=10.0)  # 限制在 [0.1, 10]
            elif self.loss_weight_type == "ones":
                weight = torch.ones_like(t_scalar).unsqueeze(1)
            else:
                raise ValueError(f"Unknown loss_weight_type: {self.loss_weight_type}")

            # ✅ 修复：每样本独立归一化
            per_sample_loss = (diff.square() * mask * weight).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            weighted_loss = per_sample_loss.mean()

            return weighted_loss, pred_y, t_scalar

        # === Infilling mode ===
        elif task_type == "infilling":
            zs = self._combine_infilling(xs)
            embeds = self._read_in(zs)
            output = self._backbone.model(inputs_embeds=embeds).last_hidden_state
            pred_y = self._read_out(output)[:, 1::2, 0][:, inds]
            return pred_y

        else:
            raise ValueError(f"Unknown task_type: {task_type}")






# ========== 构建函数 ========== #
def build_model(conf):
    family = conf["family"]

    # 🧭 兼容不同命名 （s）
    n_layer = conf.get("n_layer", conf.get("n_layers", 6))
    n_head = conf.get("n_head", conf.get("n_heads", 8))
    n_embd = conf.get("n_embd", conf.get("d_model", 256))


    if family in ["gpt2", "gptJ", "gptj", "qwen", "qwen2", "qwen2.5", "llama", "llama2", "llama3"]:
        model = TransformerModel(
            n_dims=conf["n_dims"],
            n_positions=conf["n_positions"],
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            type=family,
            pretrained=conf.get("pretrained", False),
            model_name_or_path=conf.get("model_name_or_path", None),
        )
        model.hide_last_target = conf.get("hide_last_target", False)
        return model
    

    elif family == "llada" or family == "llada_masked":
        # ✅ 统一使用 V2，默认单步推理（与 V1 行为一致，保持向后兼容）
        return LLaDAMaskedICLWrapperV2(
            n_dims=conf["n_dims"],
            n_positions=conf["n_positions"],
            n_embd=conf["n_embd"],
            n_layer=conf["n_layers"],
            n_head=conf["n_heads"],
            mlp_ratio=conf.get("mlp_ratio", 4.0),
            block_group_size=conf.get("block_group_size", 1),
            scheduler=LinearAlphaScheduler(),
            loss_weight_type=conf.get("loss_weight_type", "1/t"),
            mask_epsilon=conf.get("mask_epsilon", 1e-3),
            curriculum_schedule=conf.get("curriculum_schedule", "linear"),
            train_eval_mask_mode=conf.get("train_eval_mask_mode", "curriculum"),
            fixed_mask_ratio=conf.get("fixed_mask_ratio", 0.5),
            use_multistep_inference=conf.get("use_multistep_inference", False),  # 默认单步，与 V1 一致
            inference_steps=conf.get("inference_steps", 10),
            inference_step_size=conf.get("inference_step_size", 0.1),
        )

    elif family == "llada_v2" or family == "llada_masked_v2":
        # 🎯 V2: 支持多步反向扩散推理
        return LLaDAMaskedICLWrapperV2(
            n_dims=conf["n_dims"],
            n_positions=conf["n_positions"],
            n_embd=conf["n_embd"],
            n_layer=conf["n_layers"],
            n_head=conf["n_heads"],
            mlp_ratio=conf.get("mlp_ratio", 4.0),
            block_group_size=conf.get("block_group_size", 1),
            scheduler=LinearAlphaScheduler(),
            loss_weight_type=conf.get("loss_weight_type", "1/t"),
            mask_epsilon=conf.get("mask_epsilon", 1e-3),
            curriculum_schedule=conf.get("curriculum_schedule", "linear"),  # 🎯 新增：从配置文件读取
            train_eval_mask_mode=conf.get("train_eval_mask_mode", "curriculum"),  # 🎯 新增：训练/推理 mask 模式
            fixed_mask_ratio=conf.get("fixed_mask_ratio", 0.5),  # 🎯 新增：固定 mask ratio
            use_multistep_inference=conf.get("use_multistep_inference", True),  # 🎯 新增
            inference_steps=conf.get("inference_steps", 10),  # 🎯 新增
            inference_step_size=conf.get("inference_step_size", 0.1),  # 🎯 新增
        )

    elif family == "dream":
        model = DreamDlmModel(
            n_dims=conf["n_dims"],        # ✅ 用字典取值，而非 conf.n_dims
            n_positions=conf["n_positions"],
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            loss_weight_type=conf.get("loss_weight_type", "1/t"),  # ✅ 支持配置
            mask_epsilon=conf.get("mask_epsilon", 1e-3),  # ✅ 支持配置
            curriculum_schedule=conf.get("curriculum_schedule", "linear"),  # 🎯 新增：从配置文件读取
            train_eval_mask_mode=conf.get("train_eval_mask_mode", "curriculum"),  # 🎯 新增：训练/推理 mask 模式
            fixed_mask_ratio=conf.get("fixed_mask_ratio", 0.5),  # 🎯 新增：固定 mask ratio
        )
        return model

    else:
        raise NotImplementedError(f"Unsupported model family: {family}")
    return model



def get_relevant_baselines(task_name):
    # 将任务名称映射到对应baseline模型列表
    task_to_baselines = {
        "linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "linear_classification": [
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "sparse_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
        "relu_2nn_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": NeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),
        ],
        "decision_tree": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (DecisionTreeModel, {"max_depth": 4}),
            (DecisionTreeModel, {"max_depth": None}),
            (XGBoostModel, {}),
            (AveragingModel, {}),
        ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models


class NNModel:
    def __init__(self, n_neighbors, weights="uniform"):
        # should we be picking k optimally
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"NN_n={n_neighbors}_{weights}"

    def __call__(self, xs, ys, inds=None):# xs：[batch_size, n_points, n_dims]  , ys:[batch_size, n_points]
        # 返回形状为 [batch_size, len(inds)] 的张量，包含指定位置的预测结果。
        if inds is None:
            inds = range(ys.shape[1]) # 默认 [0,n]
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point # 第一个点的预测值为 0 ,没有可供参考的历史点，预测值直接设为 0
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i] # train_xs 和 train_ys：从输入中提取历史点的特征 , 标签
            test_x = xs[:, i : i + 1] # 当前测试i点的特征
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()# 当前测试点与所有历史点之间的欧几里得距离

            if self.weights == "uniform":  # 权重相同
                weights = torch.ones_like(dist) # 权重与距离成反比
            else:
                weights = 1.0 / dist #
                inf_mask = torch.isinf(weights).float()  # deal with exact match # 处理距离为零的情况（feature is same）
                inf_row = torch.any(inf_mask, axis=1)
                weights[inf_row] = inf_mask[inf_row] # 1

            pred = []
            k = min(i, self.n_neighbors)
            ranks = dist.argsort()[:, :k]  # 选择 topk
            for y, w, n in zip(train_ys, weights, ranks): # n:topk的切片索引
                y, w = y[n], w[n]
                pred.append((w * y).sum() / w.sum()) # topk 求加权平均
            preds.append(torch.stack(pred))
        # 将所有位置的预测结果拼接成一个张量，形状为 [batch_size, len(inds)]
        return torch.stack(preds, dim=1) # 讲inds维拼接


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank
# due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver # torch.linalg.lstsq 中的求解器（driver）
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu() # 数据移动到 CPU
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self.driver
            ) # 根据train x,y 求解得到 线形的w matrix
            pred = test_x @ ws # 利用w得到预测值 [batch_size, 1, 1]
            preds.append(pred[:, 0, 0]) # [bs]

        return torch.stack(preds, dim=1)


class AveragingModel:
    def __init__(self):
        self.name = "averaging"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            train_zs = train_xs * train_ys.unsqueeze(dim=-1)
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1) # 直接计算 w
            pred = test_x @ w_p # pred
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class LassoModel:
    def __init__(self, alpha, max_iter=100000):
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]): # 每个prompt分别计算预测值
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(
                        alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1) #

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)




# Gradient Descent and variants.
# Example usage: gd_model = GDModel(NeuralNetwork, {'in_size': 50, 'hidden_size':400, 'out_size' :1}, opt_alg = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
class GDModel:
    def __init__(
        self,
        model_class,
        model_class_args,
        opt_alg="sgd",
        batch_size=1,
        num_steps=1000,
        lr=1e-3,
        loss_name="squared",
    ):
        # model_class: torch.nn model class
        # model_class_args: a dict containing arguments for model_class
        # opt_alg can be 'sgd' or 'adam'
        # verbose: whether to print the progress or not
        # batch_size: batch size for sgd
        self.model_class = model_class
        self.model_class_args = model_class_args
        self.opt_alg = opt_alg
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.loss_name = loss_name

        self.name = f"gd_model_class={model_class}_model_class_args={model_class_args}_opt_alg={opt_alg}_lr={lr}_batch_size={batch_size}_num_steps={num_steps}_loss_name={loss_name}"

    def __call__(self, xs, ys, inds=None, verbose=False, print_step=100):
        # inds is a list containing indices where we want the prediction.
        # prediction made at all indices by default.
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        xs, ys = xs.cuda(), ys.cuda()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            model = ParallelNetworks(
                ys.shape[0], self.model_class, **self.model_class_args
            )
            model.cuda()
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])

                train_xs, train_ys = xs[:, :i], ys[:, :i] #
                test_xs, test_ys = xs[:, i : i + 1], ys[:, i : i + 1]

                if self.opt_alg == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
                elif self.opt_alg == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                else:
                    raise NotImplementedError(f"{self.opt_alg} not implemented.")

                if self.loss_name == "squared":
                    loss_criterion = nn.MSELoss()
                else:
                    raise NotImplementedError(f"{self.loss_name} not implemented.")

                # Training loop
                for j in range(self.num_steps):

                    # Prepare batch
                    mask = torch.zeros(i).bool()
                    perm = torch.randperm(i)
                    mask[perm[: self.batch_size]] = True
                    train_xs_cur, train_ys_cur = train_xs[:, mask, :], train_ys[:, mask]

                    if verbose and j % print_step == 0:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(train_xs_cur)
                            loss = loss_criterion(
                                outputs[:, :, 0], train_ys_cur
                            ).detach()
                            outputs_test = model(test_xs)
                            test_loss = loss_criterion(
                                outputs_test[:, :, 0], test_ys
                            ).detach()
                            print(
                                f"ind:{i},step:{j}, train_loss:{loss.item()}, test_loss:{test_loss.item()}"
                            )

                    optimizer.zero_grad()

                    model.train()
                    outputs = model(train_xs_cur)
                    loss = loss_criterion(outputs[:, :, 0], train_ys_cur)
                    loss.backward()
                    optimizer.step()

                model.eval()
                pred = model(test_xs).detach()

                assert pred.shape[1] == 1 and pred.shape[2] == 1
                pred = pred[:, 0, 0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class DecisionTreeModel:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.name = f"decision_tree_max_depth={max_depth}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class XGBoostModel:
    def __init__(self):
        self.name = "xgboost"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = xgb.XGBRegressor()

                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0].item()

            preds.append(pred)

        return torch.stack(preds, dim=1)
