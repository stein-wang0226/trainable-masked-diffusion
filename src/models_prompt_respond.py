"""
Prompt-Respond Models for In-Context Learning
==============================================

新实验设置：
- Prompt: 前 n_prompt 个 (x, y) 对，作为上下文
- Respond: 后 k 个 (x, y) 对，作为预测目标

模型类型：
1. TransformerModelPromptRespond: AR model，自回归预测 respond 部分
2. LLaDAPromptRespond: Masked Diffusion model，只对 respond 部分 mask
3. DreamPromptRespond: Dream model，只对 respond 部分 mask
4. SDARPromptRespond: SDAR Masked Diffusion model，暂时注释掉，因为导入失败
"""

import torch
from torch import nn
import os
import sys
import math

# 🚀 强制路径注入逻辑
# 获取当前文件的绝对路径 (src/models_prompt_respond.py)
_current_file = os.path.abspath(__file__)
# 获取 src 目录的路径
_src_dir = os.path.dirname(_current_file)
# 获取项目根目录 (train_package/)
_root_dir = os.path.dirname(_src_dir)

# 将根目录插入到 sys.path 的最前面，确保 dllm_rl 被发现
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

# 打印调试信息，方便在日志中确认
print(f"--- [DEBUG] Distributed Python Path Fix ---")
print(f"File position: {_current_file}")
print(f"Injecting root path: {_root_dir}")
print(f"Directory contains dllm_rl: {os.path.exists(os.path.join(_root_dir, 'dllm_rl'))}")
print(f"--- ---------------------------------- ---")

# ✅ 确保能 import 到仓库根下的 dllm 包
# 将 dllm 目录添加到 sys.path（向后兼容：只在路径不存在时添加）
dllm_path = os.path.join(_root_dir, "dllm")
dllm_path_norm = os.path.normpath(dllm_path)
if not any(os.path.normpath(p) == dllm_path_norm for p in sys.path):
    sys.path.insert(0, dllm_path)

# 🔧 TypedDict 安全导入（兼容不同 Python 版本）
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

from transformers import (
    GPT2Config, GPT2Model,
    GPTJConfig, GPTJModel,
    LlamaConfig, LlamaModel,
)

try:
    from transformers import Qwen2Config, Qwen2Model
except ImportError:
    Qwen2Config, Qwen2Model = None, None

from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig
from dllm.pipelines.llada.models.modeling_llada import LLaDAModel as _LLaDABase
from dllm.pipelines.dream.models.configuration_dream import DreamConfig
from dllm.pipelines.dream.models.modeling_dream import DreamModel
from dllm.core.schedulers import LinearAlphaScheduler
from dllm.core.samplers.utils import get_num_transfer_tokens

# ===== SDAR 相关代码已暂时注释（效果不佳，改用 LLaDABlock） =====
# # 🆕 Import SDAR model from dllm_rl
# SDAR_AVAILABLE = False
# _SDARBase = None
# SDARConfig = None
SDAR_AVAILABLE = False  # 保留变量避免其他代码报错

# # 🔧 增强路径设置：支持本地开发和分布式环境
# def _setup_dllm_rl_import_path():
#     """
#     设置 dllm_rl 的导入路径
#     支持多种环境：本地开发、分布式训练等
# 
#     Returns:
#         str: dllm_rl 目录的绝对路径，如果未找到返回 None
#     """
#     # 候选路径（按优先级排序）
#     candidate_base_dirs = [
#         # 1. 当前工作目录（分布式环境优先，boot_ddp.py 设置的）
#         os.getcwd(),
#         # 2. 相对于当前文件的 repo_root（本地开发）
#         os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
#         # 3. PYTHONPATH 中的路径（boot_ddp.py 可能已经设置）
#         *[p for p in os.environ.get('PYTHONPATH', '').split(':') if p],
#     ]
# 
#     # 尝试查找 dllm_rl 目录
#     found_path = None
#     for base_dir in candidate_base_dirs:
#         if not base_dir:
#             continue
#         dllm_rl_candidate = os.path.join(base_dir, 'dllm_rl')
#         if os.path.exists(dllm_rl_candidate) and os.path.isdir(dllm_rl_candidate):
#             # 找到了，将父目录添加到 sys.path
#             base_dir_norm = os.path.normpath(base_dir)
#             if not any(os.path.normpath(p) == base_dir_norm for p in sys.path):
#                 sys.path.insert(0, base_dir)
#                 print(f"🔧 [SDAR导入] 添加路径到 sys.path: {base_dir}")
#             found_path = dllm_rl_candidate
#             print(f"✅ [SDAR导入] 找到 dllm_rl 目录: {dllm_rl_candidate}")
#             break
# 
#     if not found_path:
#         print(f"⚠️ [SDAR导入] 在以下位置未找到 dllm_rl 目录:")
#         for base_dir in candidate_base_dirs[:3]:  # 只显示前3个
#             if base_dir:
#                 print(f"   - {os.path.join(base_dir, 'dllm_rl')}")
# 
#     return found_path
# 
# # 尝试设置 dllm_rl 路径
# _dllm_rl_path = _setup_dllm_rl_import_path()
# 
# # 尝试导入 SDAR 模型
# try:
#     # 🔧 修复 LossKwargs 占位符（兼容旧版本 transformers）
#     import transformers.utils as transformers_utils
#     if not hasattr(transformers_utils, 'LossKwargs'):
#             LossKwargs = TypedDict('LossKwargs', {}, total=False)
#             transformers_utils.LossKwargs = LossKwargs
# 
#     # 🔧 SDAR 代码已经兼容旧版本 transformers（通过可选导入 use_kernel_forward_from_hub）
#     # 理论上可以在旧版本上运行，但建议使用 transformers>=4.48.0 以获得最佳兼容性
# 
#     # 🚀 直接导入
#     from dllm_rl.models.sdar.modeling_sdar import SDARModel as _SDARBase
#     from dllm_rl.models.sdar.configuration_sdar import SDARConfig
# 
#     SDAR_AVAILABLE = True
#     print(f"✅ [SDAR导入] 模型导入成功")
# except ImportError as e:
#     # 🔧 详细的错误诊断
#     error_msg = str(e)
#     print(f"❌ [SDAR导入] 导入失败 (ImportError): {error_msg}")
# 
#     # 检查是否是 transformers 版本问题
#     if "use_kernel_forward_from_hub" in error_msg or "cannot import name" in error_msg:
#         print(f"   检测到可能的 transformers 版本相关问题")
#         try:
#             import transformers
#             print(f"   当前 transformers 版本: {transformers.__version__}")
#             print(f"   建议版本: transformers>=4.48.0 (旧版本也应该可以工作)")
#             print(f"   如果问题持续，请运行: pip install --upgrade 'transformers>=4.48.0'")
#         except:
#             pass
# 
#     # 诊断信息
#     print(f"\n📍 [SDAR导入] 诊断信息:")
#     print(f"   当前工作目录: {os.getcwd()}")
#     print(f"   当前文件路径: {os.path.abspath(__file__)}")
#     print(f"   dllm_rl 查找结果: {_dllm_rl_path or '未找到'}")
#     print(f"   PYTHONPATH: {os.environ.get('PYTHONPATH', '未设置')}")
#     print(f"\n   sys.path (前5个):")
#     for i, p in enumerate(sys.path[:5]):
#         print(f"     [{i}] {p}")
# 
#     # 如果找到了 dllm_rl 但仍然导入失败，尝试重新导入
#     if _dllm_rl_path:
#         print(f"\n   🔧 dllm_rl 目录存在但导入失败，检查目录内容:")
#         try:
#             import os as os_check
#             sdar_model_path = os_check.path.join(_dllm_rl_path, 'models', 'sdar')
#             if os_check.path.exists(sdar_model_path):
#                 print(f"   ✅ SDAR模型目录存在: {sdar_model_path}")
#                 model_files = os_check.listdir(sdar_model_path)
#                 print(f"   包含文件: {model_files}")
#             else:
#                 print(f"   ❌ SDAR模型目录不存在: {sdar_model_path}")
#         except Exception as check_err:
#             print(f"   检查失败: {check_err}")
# 
#     import traceback
#     traceback.print_exc()
#     SDAR_AVAILABLE = False
# except Exception as e:
#     # 🔧 其他错误
#     print(f"❌ [SDAR导入] 导入失败 (其他错误): {e}")
#     print(f"   当前工作目录: {os.getcwd()}")
#     print(f"   dllm_rl 路径: {_dllm_rl_path or '未找到'}")
# 
#     import traceback
#     traceback.print_exc()
#     SDAR_AVAILABLE = False
# ===== SDAR 导入代码结束 =====


# 🆕 Import utilities from separate module
from model_utils import (
    cart_weight,
    combine_xs_ys,
    create_prefix_lm_mask,
    create_inference_scheduler,
    compute_actual_n_respond,
    sample_timestep_for_mdm,
    generate_masked_indices_for_mdm,
    extract_respond_predictions,
    extract_respond_masked_indices,
    extract_respond_targets,
    compute_multistep_confidence,
)
from training_strategy_utils import sample_timestep_with_strategy, compute_loss_weight_with_strategy

import logging
import re
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ============================================================
# AR Transformer Model - Prompt-Respond Version
# ============================================================
class TransformerModelPromptRespond(nn.Module):
    """
    AR Transformer for Prompt-Respond Setting
    
    Training/Eval:
    - Input: [prompt_pairs (n_prompt), respond_pairs (k)]
    - Predict: respond 部分的 y 值（自回归）
    - Loss: 只在 respond 部分计算
    """
    
    def __init__(self, n_dims, n_positions, n_embd=128,
                 n_layer=12, n_head=4, type="gpt2", mlp_ratio=4.0,
                 n_prompt=20, n_respond=5,
                 attention_mode="causal",  # 🆕 新增：注意力模式 （"causal" 或 "prefix_lm"）
                 pretrained=False, model_name_or_path=None):
        super().__init__()
        self.family = type.lower()
        self.mlp_ratio = mlp_ratio
        self.n_prompt = n_prompt  # prompt 长度
        self.n_respond = n_respond  # respond 长度
        self.attention_mode = attention_mode  # 🆕 注意力模式
        
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        head_dim = n_embd // n_head
        
        # 🆕 验证 attention_mode 参数
        valid_modes = ["causal", "prefix_lm"]
        assert attention_mode in valid_modes, \
            f"attention_mode must be one of {valid_modes}, got '{attention_mode}'"
        
        # ===== 构建 backbone =====
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
            
        elif type == "gptJ":
            configuration = GPTJConfig(
                n_positions=2 * n_positions,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                rotary_dim=head_dim,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
                use_cache=False,
            )
            self._backbone = GPTJModel(configuration)
            
        elif self.family in ["llama", "llama2", "llama3"]:
            if pretrained:
                model_id = model_name_or_path or {
                    "llama3": "meta-llama/Meta-Llama-3-8B",
                    "llama2": "meta-llama/Llama-2-7b-hf",
                }.get(self.family, None)
                print(f"[Loading pretrained {self.family.upper()} from {model_id}]")
                from transformers import AutoModel
                self._backbone = AutoModel.from_pretrained(model_id)
                n_embd = self._backbone.config.hidden_size
            else:
                configuration = LlamaConfig(
                    hidden_size=n_embd,
                    num_hidden_layers=n_layer,
                    num_attention_heads=n_head,
                    intermediate_size=int(n_embd * mlp_ratio),
                    max_position_embeddings=2 * n_positions,
                    use_cache=False,
                )
                self._backbone = LlamaModel(configuration)
                
        elif self.family in ["qwen", "qwen2", "qwen2.5"]:
            if pretrained:
                model_id = model_name_or_path or {
                    "qwen": "Qwen/Qwen-7B",
                    "qwen2": "Qwen/Qwen2-7B-Instruct",
                    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
                }.get(self.family)
                print(f"[Loading pretrained {self.family.upper()} from {model_id}]")
                from transformers import AutoModel
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
        
        self.name = f"{type}_prompt_respond_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_positions = n_positions
        self.n_dims = n_dims
        
        # ===== 输入输出层 =====
        self._read_in = nn.Linear(n_dims, n_embd)
        self._read_out = nn.Linear(n_embd, 1)
        
        # ===== 维度对齐层 =====
        hidden_size = self._backbone.config.hidden_size
        self._align_proj = nn.Linear(n_embd, hidden_size) if n_embd != hidden_size else nn.Identity()
        
        print(f"[{type.upper()} Prompt-Respond] n_embd={n_embd}, n_prompt={n_prompt}, n_respond={n_respond}, "
              f"attention_mode={attention_mode}")
    
    def forward(self, xs, ys, inds=None, use_autoregressive_eval=False, respond_position_mask=None):
        """
        Forward pass for Prompt-Respond setting
        
        Args:
            xs: [B, n_prompt + n_respond, D]
            ys: [B, n_prompt + n_respond]
            inds: 要预测的位置索引（默认为 respond 部分）
            use_autoregressive_eval: 如果为 True，在评估时使用自回归预测（避免 label 泄漏）
                                    如果为 False，使用真实标签（原方案，适合用于训练）
            respond_position_mask: [B, total_points] boolean tensor 标记 respond 位置
                                  - None: Sequential 模式（respond 在末尾）
                                  - 提供值: Non-Sequential 模式（respond 位置由 mask 指定）
        
        Returns:
            pred: [B, len(inds)] - 预测的 y 值
            loss_mask: [B, len(inds)] - 哪些位置计算 loss（respond 部分）
        """
        device = next(self.parameters()).device
        xs, ys = xs.to(device), ys.to(device)
        
        B, total_points, D = xs.shape
        # 支持动态 n_respond：根据实际输入的数据长度计算
        actual_n_respond = total_points - self.n_prompt
        assert total_points >= self.n_prompt, \
            f"Expected at least {self.n_prompt} points (prompt), got {total_points}"
        assert actual_n_respond <= self.n_respond, \
            f"Actual respond count {actual_n_respond} exceeds model max {self.n_respond}"
        
        if inds is None:
            # 默认只预测 respond 部分（根据实际数据长度）
            respond_start = self.n_prompt
            inds = torch.arange(respond_start, total_points, device=device)
        else:
            inds = torch.as_tensor(inds, device=device)
        
        # ===== 自回归评估模式 =====
        if use_autoregressive_eval and not self.training:
            # 初始化：prompt 部分使用真实值，respond 部分初始化为 0
            ys_input = ys.clone()
            
            if respond_position_mask is not None:
                # 🆕 Non-Sequential 模式：按照 respond 在序列中的顺序生成
                # 为每个 batch 找到 respond 位置并排序
                pred_respond_list = []
                
                for b in range(B):
                    # 找到这个 batch 的所有 respond 位置
                    respond_positions = respond_position_mask[b].nonzero(as_tuple=True)[0]  # [actual_n_respond]
                    respond_positions_sorted = torch.sort(respond_positions)[0]  # 按序列位置排序
                    
                    # 初始化这个 batch 的输入
                    ys_input_b = ys_input[b].clone()
                    ys_input_b[respond_positions_sorted] = 0.0  # respond 位置初始化为 0
                    
                    # 按照序列顺序逐个生成
                    pred_respond_b = []
                    for pos in respond_positions_sorted:
                        # 构建当前输入序列（只对这个 batch）
                        xs_b = xs[b:b+1]  # [1, total_points, D]
                        ys_input_b_expanded = ys_input_b.unsqueeze(0)  # [1, total_points]
                        
                        zs = combine_xs_ys(xs_b, ys_input_b_expanded)
                        embeds = self._read_in(zs)
                        embeds = self._align_proj(embeds)
                        
                        # 创建注意力 mask（根据 attention_mode）
                        if self.attention_mode == "causal":
                            attention_mask = torch.ones((1, 2 * total_points), device=device)
                        elif self.attention_mode == "prefix_lm":
                            # 使用 Prefix-LM mask
                            attention_mask = create_prefix_lm_mask(
                                1, total_points, self.n_prompt, respond_position_mask[b:b+1], device
                            )
                        else:
                            attention_mask = torch.ones((1, 2 * total_points), device=device)
                        
                        # 前向传播
                        outputs = self._backbone(inputs_embeds=embeds, attention_mask=attention_mask)
                        h = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
                        pred_all = self._read_out(h)[..., 0]  # [1, 2T]
                        
                        # 获取当前位置的预测
                        pred_i = pred_all[0, pos * 2]  # x 位置的预测对应 y 值
                        pred_respond_b.append(pred_i)
                        
                        # 更新输入：使用预测值替换当前位置的 y
                        ys_input_b[pos] = pred_i
                    
                    pred_respond_list.append(torch.stack(pred_respond_b))
                
                # 组合所有 batch 的预测
                pred = torch.stack(pred_respond_list, dim=0)  # [B, actual_n_respond]
                loss_mask = torch.ones(B, len(inds), device=device)
            
            else:
                # Sequential 模式：标准自回归生成
                respond_start = self.n_prompt
                actual_n_respond = total_points - self.n_prompt
                
                # 初始化：prompt 部分使用真实值，respond 部分初始化为 0
                ys_input[:, respond_start:] = 0.0
                
                # 逐位置自回归预测
                pred_respond = []
                for i in range(actual_n_respond):
                    # 构建当前输入序列
                    zs = combine_xs_ys(xs, ys_input)
                    embeds = self._read_in(zs)
                    embeds = self._align_proj(embeds)
                    
                    # 因果注意力 mask
                    attention_mask = torch.ones((B, 2 * total_points), device=device)
                    
                    # 前向传播
                    outputs = self._backbone(inputs_embeds=embeds, attention_mask=attention_mask)
                    h = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
                    pred_all = self._read_out(h)[..., 0]  # [B, 2T]
                    
                    # 获取当前位置的预测（respond 部分的第 i 个位置）
                    # 在 AR 模型中，pred_all[:, 2*i] 是位置 i 的 x 对应的 y 预测
                    current_pos = respond_start + i
                    pred_i = pred_all[:, current_pos * 2]  # x 位置的预测对应 y 值
                    pred_respond.append(pred_i)
                    
                    # 更新输入：使用预测值替换当前位置的 y
                    ys_input[:, current_pos] = pred_i
                
                # 组合所有预测
                pred = torch.stack(pred_respond, dim=1)  # [B, actual_n_respond]
                loss_mask = torch.ones(B, len(inds), device=device)
            
        else:
            # ===== 原始方案：使用真实标签（训练模式或非自回归评估）=====
            # 构建输入序列 [x1, y1, x2, y2, ..., xn, yn]
            # 注意：prompt 部分的 y 值作为上下文，respond 部分的 y 值用于计算 loss
            zs = combine_xs_ys(xs, ys)
            embeds = self._read_in(zs)
            embeds = self._align_proj(embeds)
            
            # ===== 创建注意力 mask（根据 attention_mode）=====
            if self.attention_mode == "causal":
                # 标准因果 mask（backbone 自动应用）
                attention_mask = torch.ones((B, 2 * total_points), device=device)
            
            elif self.attention_mode == "prefix_lm":
                # Prefix-LM: Prompt 双向，Respond 因果
                attention_mask = create_prefix_lm_mask(
                    B, total_points, self.n_prompt, respond_position_mask, device
                )
            
            else:
                raise ValueError(f"Unknown attention_mode: {self.attention_mode}")
            
            outputs = self._backbone(inputs_embeds=embeds, attention_mask=attention_mask)
            
            # ===== 读出预测 =====
            h = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            pred_all = self._read_out(h)[..., 0]  # [B, 2T]
            
            # ===== 提取 respond 部分的预测 =====
            if respond_position_mask is not None:
                # Non-Sequential: 使用 mask 提取 respond 位置的预测
                pred = torch.stack([
                    pred_all[i, ::2][respond_position_mask[i]]
                    for i in range(B)
                ])  # [B, actual_n_respond]
            else:
                # Sequential: 直接切片
                pred = pred_all[:, ::2][:, inds]
            
            # ===== 创建 loss mask：所有 respond 位置都计算 loss =====
            loss_mask = torch.ones(B, len(inds), device=device)  # respond 部分全部计算 loss
        
        return pred, loss_mask


# ============================================================
# LLaDA Masked ICL - Prompt-Respond Version
# ============================================================
class LLaDAPromptRespond(nn.Module):
    """
    LLaDA Masked Diffusion for Prompt-Respond Setting
    
    Training/Eval:
    - Prompt 部分：始终可见，不 mask
    - Respond 部分：随机 mask，用于训练和预测
    - Loss: 只在 respond 的 masked 位置计算
    """
    
    def __init__(
        self,
        n_dims,
        n_positions,
        n_embd=256,
        n_layer=12,
        n_head=8,
        n_prompt=20,
        n_respond=5,
        *,
        mlp_ratio=4.0,
        block_group_size=1,
        mask_epsilon=1e-3,
        loss_weight_type="1/t",
        train_mask_ratio=0.5,
        eval_mask_ratio=1.0,
        eval_mask_mode="fixed",  # "fixed" (固定mask ratio，通常为全mask) 或 "sample" (随机采样mask，与训练一致)
        use_prompt_context=True,  # True: prompt 作为上下文; False: prompt 也被 mask（无 context 模式）
        # 多步推理优化选项（可选，默认关闭以保持向后兼容）
        use_multistep_inference=False,
        inference_steps=10,
        inference_scheduler=None,
        inference_confidence_alg="entropy",
        # 🆕 可选的训练策略配置（用于支持diffusion-vs-ar风格的训练）
        training_strategy=None,
        **extra,
    ):
        super().__init__()
        self.name = "llada_prompt_respond"
        self.n_positions = n_positions
        self.n_dims = n_dims
        self.n_prompt = n_prompt
        self.n_respond = n_respond
        self.mask_epsilon = mask_epsilon
        self.loss_weight_type = loss_weight_type
        self.train_mask_ratio = train_mask_ratio
        self.eval_mask_ratio = eval_mask_ratio
        self.eval_mask_mode = eval_mask_mode
        self.use_prompt_context = use_prompt_context
        self.d_model = int(n_embd)

        # === 多步推理优化选项（可选）===
        self.use_multistep_inference = use_multistep_inference
        self.inference_steps = inference_steps
        self.inference_scheduler = inference_scheduler or LinearAlphaScheduler()
        self.inference_confidence_alg = inference_confidence_alg

        # === 🆕 训练策略配置（向后兼容）===
        if training_strategy is None:
            # 默认：使用原有的连续mask模式
            self.training_strategy = {
                'mask_mode': 'ratio',  # 'ratio' 或 'timestep'
                'loss_reweighting': {
                    'enable_token_reweight': False,
                    'time_weight_mode': self.loss_weight_type  # 使用原有参数
                }
            }
        else:
            self.training_strategy = training_strategy
            # 验证配置
            assert 'mask_mode' in training_strategy, "training_strategy must contain 'mask_mode'"
            assert training_strategy['mask_mode'] in ['ratio', 'timestep'], \
                f"mask_mode must be 'ratio' or 'timestep', got {training_strategy['mask_mode']}"

        # 如果使用timestep模式，保存num_timesteps
        if self.training_strategy['mask_mode'] == 'timestep':
            self.num_timesteps = self.training_strategy.get('num_timesteps', 20)
            print(f"[LLaDA] Training strategy: timestep mode (T={self.num_timesteps})")
        else:
            self.num_timesteps = 20  # 默认值，用于ratio模式的伪时间步计算
            print(f"[LLaDA] Training strategy: ratio mode (original)")
        
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
        
        # 初始化 read_out 层，使用较小的初始化范围，避免初始输出过大
        nn.init.normal_(self._read_out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self._read_out.bias)
        
        # === Learnable mask token ===
        self.mask_embedding = nn.Parameter(torch.randn(1, 1, n_dims))
        
        print(f"[LLaDA Prompt-Respond] d_model={self.d_model}, "
              f"n_prompt={n_prompt}, n_respond={n_respond}, "
              f"train_mask_ratio={self.train_mask_ratio}, eval_mask_ratio={self.eval_mask_ratio}, "
              f"eval_mask_mode={self.eval_mask_mode}, use_prompt_context={self.use_prompt_context}")
        if self.use_multistep_inference:
            print(f"  [Multi-step Inference] Enabled: steps={self.inference_steps}, "
                  f"confidence_alg={self.inference_confidence_alg}")
    
    def forward(self, xs, ys, train_mode=True, respond_position_mask=None):
        """
        Forward pass for Prompt-Respond setting
        
        Args:
            xs: [B, n_prompt + n_respond, D]
            ys: [B, n_prompt + n_respond]
            train_mode: True for training, False for inference
            respond_position_mask: [B, total_points] boolean tensor marking respond positions.
                                   If None, assumes respond positions are at the end (sequential mode).
                                   If provided, uses it to determine which positions are respond pairs
                                   (non-sequential mode).
        
        Returns:
            If train_mode: (loss, pred_y, t_scalar)
            Else: pred_y
        """
        b, total_points, d = xs.shape
        device = xs.device
        
        # 🔧 使用辅助函数：计算实际 n_respond
        actual_n_respond = compute_actual_n_respond(total_points, self.n_prompt, respond_position_mask)
        if respond_position_mask is not None:
            assert respond_position_mask.shape == (b, total_points), \
                f"respond_position_mask shape mismatch: expected {(b, total_points)}, got {respond_position_mask.shape}"
        assert total_points >= self.n_prompt, \
            f"Expected at least {self.n_prompt} points (prompt), got {total_points}"
        assert actual_n_respond <= self.n_respond, \
            f"Actual respond count {actual_n_respond} exceeds model max {self.n_respond}"
        
        # 🔧 采样 timestep 或 mask ratio（支持两种训练策略）
        t_scalar, t = sample_timestep_with_strategy(
            b, device, train_mode, self.training_strategy,
            self.num_timesteps, self.mask_epsilon, self.train_mask_ratio,
            self.eval_mask_ratio, self.eval_mask_mode
        )
        
        # 🔧 使用辅助函数：生成 masked_indices
        masked_indices = generate_masked_indices_for_mdm(
            b, total_points, self.n_prompt, actual_n_respond,
            t_scalar, device, self.use_prompt_context, respond_position_mask
        )
        
        # ===== Step 3: 构建序列 =====
        ys_input = ys.clone().float()
        if ys_input.dim() == 2:
            ys_input = ys_input.unsqueeze(-1)
        
        ys_wide = torch.cat([ys_input, torch.zeros(b, total_points, d - 1, device=device)], dim=2)
        zs = torch.stack((xs, ys_wide), dim=2).view(b, 2 * total_points, d)
        
        # ===== Step 4: Embedding + Time conditioning =====
        embeds = self._read_in(zs)
        time_emb = self._time_mlp(t_scalar.view(b, 1, 1).expand(b, 2 * total_points, 1))
        embeds = embeds + time_emb
        
        # ===== Step 5: 应用 mask =====
        mask_embed = self._read_in(self.mask_embedding.squeeze(0))
        for i in range(b):
            if masked_indices[i].any():
                # 找到所有被 mask 的位置（包括 prompt 和 respond）
                masked_positions = masked_indices[i].nonzero(as_tuple=False).squeeze(-1)
                # 转换为序列中的 y 位置索引（y 位置在序列中是奇数索引：1, 3, 5, ...）
                full_idx = masked_positions * 2 + 1
                embeds[i, full_idx, :] = mask_embed
        
        # ===== Step 6: Backbone forward =====
        dummy_input_ids = torch.zeros(b, 2 * total_points, dtype=torch.long, device=device)
        out = self._backbone(
            input_ids=dummy_input_ids,
            input_embeddings=embeds,
            output_hidden_states=True,
        )
        h = out.hidden_states[-1]
        pred_y_all = self._read_out(h)[:, 0::2, 0]  # [B, total_points]
        
        # 🔧 使用辅助函数：提取 respond 部分的预测
        pred_y = extract_respond_predictions(pred_y_all, self.n_prompt, actual_n_respond, respond_position_mask)
        
        if not train_mode:
            # 推理模式：支持单步或多步推理
            if self.use_multistep_inference:
                return self._multistep_inference(xs, ys, device, respond_position_mask=respond_position_mask)
            else:
                # 🔧 使用辅助函数：提取 respond 部分的 masked_indices
                respond_masked_indices = extract_respond_masked_indices(
                    masked_indices, self.n_prompt, actual_n_respond, respond_position_mask
                )
                return pred_y, respond_masked_indices  # 返回 (pred_y, mask)
        
        # ===== Step 8: Training Loss（只在 respond 的 masked 位置）=====
        target = ys.squeeze(-1) if ys.dim() == 3 else ys
        
        # 🔧 使用辅助函数：提取 respond target 和 masked_indices
        respond_target = extract_respond_targets(target, self.n_prompt, actual_n_respond, respond_position_mask)
        respond_masked_indices = extract_respond_masked_indices(masked_indices, self.n_prompt, actual_n_respond, respond_position_mask)
        
        diff = pred_y - respond_target
        
        # 只在 respond 部分的 masked 位置计算 loss
        mask = respond_masked_indices.float()
        per_sample_loss = (diff.square() * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        # 🔧 使用训练策略计算loss权重
        weighted_loss = compute_loss_weight_with_strategy(
            per_sample_loss, t, t_scalar, self.training_strategy, self.num_timesteps
        )

        # 🔧 返回 respond_masked_indices 用于计算 train MSE
        return weighted_loss, pred_y, t_scalar, respond_masked_indices
    
    @torch.no_grad()
    def _multistep_inference(self, xs, ys, device, respond_position_mask=None):
        """
        多步迭代推理（参考 LLaDA/Dream generate.py 逻辑）
        
        流程：
        1. 初始：mask 所有 respond 位置
        2. 使用 scheduler 预先计算每步 unmask 的数量
        3. 迭代：每步 forward → 计算 confidence → 选择 top-k 位置 unmask
        4. 逐步 refine 直到所有位置都被 unmask
        
        Args:
            xs: [B, n_prompt + n_respond, D]
            ys: [B, n_prompt + n_respond] (用于构建初始序列，但 respond 部分会被 mask)
            device: device
            respond_position_mask: [B, total_points] boolean tensor marking respond positions.
                                   Currently not fully supported for multistep inference - will use
                                   sequential mode (respond at the end) if None.
        
        Returns:
            (pred_y, initial_mask): 
                - pred_y: [B, actual_n_respond] - 最终预测的 respond 部分
                - initial_mask: [B, actual_n_respond] - 初始 mask（所有位置都为 True，表示初始时所有位置都被 mask）
        """
        # TODO: Full support for non-sequential mode in multistep inference
        # For now, multistep inference only works in sequential mode
        if respond_position_mask is not None:
            raise NotImplementedError("Non-sequential mode is not yet supported for multistep inference")
        from dllm.utils.generation_utils import get_num_transfer_tokens
        
        b, total_points, d = xs.shape
        actual_n_respond = total_points - self.n_prompt
        respond_start = self.n_prompt
        
        # 初始化：所有 respond 位置都 mask
        ys_pred = torch.zeros(b, actual_n_respond, device=device)
        masked_indices = torch.ones(b, actual_n_respond, device=device, dtype=torch.bool)
        
        # 保存初始 mask（用于评估时只对初始被 mask 的位置计算 MSE）
        initial_mask = masked_indices.clone()
        
        # 预先计算所有步的 unmask 数量（使用 scheduler，与源码逻辑一致）
        num_transfer_tokens = get_num_transfer_tokens(
            mask_index=masked_indices,
            steps=self.inference_steps,
            scheduler=self.inference_scheduler,
            stochastic=False,
        )
        effective_steps = num_transfer_tokens.size(1)
        
        # 预定义时间步序列（从 1.0 到 epsilon，对应 scheduler 的时间步语义）
        # 注意：时间步是递减的（从高 mask ratio 到低 mask ratio）
        time_steps = torch.linspace(1.0, self.mask_epsilon, effective_steps + 1, device=device)
        
        # 迭代去噪
        for step in range(effective_steps):
            # 使用预定义的时间步（而不是动态 mask ratio）
            t_curr = time_steps[step]  # 当前时间步
            
            # 构建当前状态：prompt + 部分预测的 respond
            ys_current = ys.clone()
            ys_current[:, respond_start:] = ys_pred
            
            # 构建序列
            ys_input = ys_current.clone().float()
            if ys_input.dim() == 2:
                ys_input = ys_input.unsqueeze(-1)
            ys_wide = torch.cat([ys_input, torch.zeros(b, total_points, d - 1, device=device)], dim=2)
            zs = torch.stack((xs, ys_wide), dim=2).view(b, 2 * total_points, d)
            
            # Embedding + Time conditioning（使用预定义的时间步）
            embeds = self._read_in(zs)
            time_emb = self._time_mlp(t_curr.view(1, 1, 1).expand(b, 2 * total_points, 1))
            embeds = embeds + time_emb
            
            # 应用 mask（只 mask 还未预测的位置）
            mask_embed = self._read_in(self.mask_embedding.squeeze(0))
            for i in range(b):
                if masked_indices[i].any():
                    y_positions = masked_indices[i].nonzero(as_tuple=False).squeeze(-1)
                    full_idx = (respond_start + y_positions) * 2 + 1
                    embeds[i, full_idx, :] = mask_embed
            
            # Forward pass
            dummy_input_ids = torch.zeros(b, 2 * total_points, dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            pred_y_all = self._read_out(h)[:, 0::2, 0]  # [B, total_points]
            pred_y_respond = pred_y_all[:, respond_start:respond_start+actual_n_respond]  # [B, actual_n_respond]
            
            # 🔧 使用辅助函数：计算 confidence
            confidence = compute_multistep_confidence(
                pred_y_respond, ys_pred, step, 
                self.inference_confidence_alg, device
            )
            
            # 只考虑当前 masked 位置的 confidence
            confidence = torch.where(
                masked_indices, 
                confidence, 
                torch.tensor(-float('inf'), device=device)
            )
            
            # 根据 scheduler 决定 unmask 的数量和位置
            for j in range(b):
                num_unmask = int(num_transfer_tokens[j, step].item())
                if num_unmask > 0 and masked_indices[j].any():
                    # 选择 confidence 最高的 num_unmask 个位置
                    available_mask_count = masked_indices[j].sum().item()
                    k = min(num_unmask, available_mask_count)
                    if k > 0:
                        _, top_indices = torch.topk(confidence[j], k=k)
                        
                        # Unmask 这些位置：使用预测值
                        ys_pred[j, top_indices] = pred_y_respond[j, top_indices]
                        masked_indices[j, top_indices] = False
        
        # 返回预测值和初始 mask（用于评估时只对初始被 mask 的位置计算 MSE）
        return ys_pred, initial_mask


# ============================================================
# Dream Model - Prompt-Respond Version
# ============================================================
class DreamPromptRespond(nn.Module):
    """
    Dream Diffusion Model for Prompt-Respond Setting
    
    与 LLaDA 类似，只在 respond 部分 mask 和计算 loss
    """
    
    def __init__(
        self,
        n_dims,
        n_positions,
        n_embd=128,
        n_layer=12,
        n_head=4,
        n_prompt=20,
        n_respond=5,
        *,
        loss_weight_type="1/t",
        mask_epsilon=1e-3,
        train_mask_ratio=0.5,
        eval_mask_ratio=1.0,
        eval_mask_mode="fixed",  # "fixed" (固定mask ratio，通常为全mask) 或 "sample" (随机采样mask，与训练一致)
        use_prompt_context=True,  # True: prompt 作为上下文; False: prompt 也被 mask（无 context 模式）
        # 多步推理优化选项（可选，默认关闭以保持向后兼容）
        use_multistep_inference=False,
        use_dllm_generate=False,  # 🎯 新增：是否使用 dllm 的 generate 函数
        inference_steps=10,
        inference_scheduler=None,
        inference_confidence_alg="entropy",
        inference_temperature=1.0,  # 🎯 新增：dllm generate 的温度参数
        inference_top_p=1.0,         # 🎯 新增：dllm generate 的 top_p 参数
        inference_top_k=50,          # 🎯 新增：dllm generate 的 top_k 参数
        **extra,
    ):
        super(DreamPromptRespond, self).__init__()
        self.family = "dream_prompt_respond"
        self.name = f"dream_prompt_respond_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_positions = n_positions
        self.n_dims = n_dims
        self.n_prompt = n_prompt
        self.n_respond = n_respond
        self.loss_weight_type = loss_weight_type
        self.mask_epsilon = mask_epsilon
        self.train_mask_ratio = train_mask_ratio
        self.eval_mask_ratio = eval_mask_ratio
        self.eval_mask_mode = eval_mask_mode
        self.use_prompt_context = use_prompt_context
        
        # === 🆕 训练策略配置（与 LLaDA 对齐）===
        # 默认：使用原有的连续mask模式
        self.training_strategy = {
            'mask_mode': 'ratio',  # 'ratio' 或 'timestep'
            'loss_reweighting': {
                'enable_token_reweight': False,
                'time_weight_mode': self.loss_weight_type  # 使用原有参数
            }
        }
        self.num_timesteps = 20  # 默认值，用于ratio模式的伪时间步计算
        
        # === 多步推理优化选项（可选）===
        self.use_multistep_inference = use_multistep_inference
        self.use_dllm_generate = use_dllm_generate  # 🎯 新增
        self.inference_steps = inference_steps
        self.inference_scheduler = inference_scheduler or LinearAlphaScheduler()
        self.inference_confidence_alg = inference_confidence_alg
        self.inference_temperature = inference_temperature  # 🎯 新增
        self.inference_top_p = inference_top_p  # 🎯 新增
        self.inference_top_k = inference_top_k  # 🎯 新增
        
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
        
        # === Time Embedding MLP ===
        # 🔧 修复：添加时间嵌入，与 LLaDA 保持一致
        self._time_mlp = nn.Sequential(
            nn.Linear(1, n_embd),
            nn.SiLU(),
            nn.Linear(n_embd, n_embd)
        )
        
        # === 初始化 read_out 层，使用较小的初始化范围，避免初始输出过大 ===
        # 🔧 修复：与 LLaDA 保持一致的特殊初始化
        nn.init.normal_(self._read_out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self._read_out.bias)
        
        # === Learnable mask token ===
        self.mask_embedding = nn.Parameter(torch.randn(1, 1, n_dims))
        
        print(f"[Dream Prompt-Respond] n_prompt={n_prompt}, n_respond={n_respond}, "
              f"loss_weight={loss_weight_type}, train_mask_ratio={self.train_mask_ratio}, eval_mask_ratio={self.eval_mask_ratio}, "
              f"eval_mask_mode={self.eval_mask_mode}, use_prompt_context={self.use_prompt_context}")
        if self.use_multistep_inference:
            mode_str = "dllm.generate" if self.use_dllm_generate else "custom"
            print(f"  [Multi-step Inference] Enabled: mode={mode_str}, steps={self.inference_steps}, "
                  f"confidence_alg={self.inference_confidence_alg}")
            if self.use_dllm_generate:
                print(f"    [DLLM Generate] temperature={self.inference_temperature}, "
                      f"top_p={self.inference_top_p}, top_k={self.inference_top_k}")
    
    def forward(self, xs, ys, train_mode=True, respond_position_mask=None):
        """
        Forward pass for Prompt-Respond setting
        
        Args:
            xs: [B, n_prompt + n_respond, D]
            ys: [B, n_prompt + n_respond]
            train_mode: True for training, False for inference
            respond_position_mask: [B, total_points] boolean tensor marking respond positions.
                                   If None, assumes respond positions are at the end (sequential mode).
                                   If provided, uses it to determine which positions are respond pairs
                                   (non-sequential mode).
        
        Returns:
            If train_mode: (loss, pred_y, t_scalar)
            Else: pred_y
        """
        b_size, total_points, n_dims = xs.shape
        device = xs.device
        
        # 🔧 使用辅助函数：计算实际 n_respond
        actual_n_respond = compute_actual_n_respond(total_points, self.n_prompt, respond_position_mask)
        if respond_position_mask is not None:
            assert respond_position_mask.shape == (b_size, total_points), \
                f"respond_position_mask shape mismatch: expected {(b_size, total_points)}, got {respond_position_mask.shape}"
        assert total_points >= self.n_prompt, \
            f"Expected at least {self.n_prompt} points (prompt), got {total_points}"
        assert actual_n_respond <= self.n_respond, \
            f"Actual respond count {actual_n_respond} exceeds model max {self.n_respond}"
        
        # 🔧 修复：使用与 LLaDA 相同的训练策略采样 timestep
        t_scalar, t = sample_timestep_with_strategy(
            b_size, device, train_mode, self.training_strategy,
            self.num_timesteps, self.mask_epsilon, self.train_mask_ratio,
            self.eval_mask_ratio, self.eval_mask_mode
        )
        
        # ===== Step 2: Combine [x, y] sequence =====
        zs = combine_xs_ys(xs, ys)
        embeds = self._read_in(zs)
        
        # 🔧 修复：添加时间嵌入（与 LLaDA 保持一致）
        time_emb = self._time_mlp(t_scalar.view(b_size, 1, 1).expand(b_size, 2 * total_points, 1))
        embeds = embeds + time_emb
        
        # 🔧 使用辅助函数：生成 masked_indices
        masked_indices = generate_masked_indices_for_mdm(
            b_size, total_points, self.n_prompt, actual_n_respond,
            t_scalar, device, self.use_prompt_context, respond_position_mask
        )
        
        # ===== Step 4: 应用掩码到 embedding =====
        mask_embedding = self._read_in(self.mask_embedding.squeeze(0))
        for i in range(b_size):
            if masked_indices[i].any():
                # 找到所有被 mask 的位置（包括 prompt 和 respond）
                masked_positions = masked_indices[i].nonzero(as_tuple=False).squeeze(-1)
                # 转换为序列中的 y 位置索引（y 位置在序列中是奇数索引：1, 3, 5, ...）
                full_seq_indices = masked_positions * 2 + 1
                embeds[i, full_seq_indices, :] = mask_embedding
        
        # ===== Step 5: Model forward =====
        output = self._backbone.model(inputs_embeds=embeds).last_hidden_state
        pred_y_all = self._read_out(output)[:, 0::2, 0]  # [B, total_points]
        
        # 🔧 使用辅助函数：提取 respond 部分的预测
        pred_y = extract_respond_predictions(pred_y_all, self.n_prompt, actual_n_respond, respond_position_mask)
        
        if not train_mode:
            # 推理模式：支持单步或多步推理
            if self.use_multistep_inference:
                if self.use_dllm_generate:
                    # 🎯 使用 dllm 的 generate 函数进行迭代式推理
                    return self._dllm_generate_inference(xs, ys, device, respond_position_mask=respond_position_mask)
                else:
                    # 使用自定义的多步推理
                    if respond_position_mask is not None:
                        raise NotImplementedError("Non-sequential mode is not yet supported for multistep inference")
                    return self._multistep_inference(xs, ys, device)
            else:
                # 🔧 使用辅助函数：提取 respond 部分的 masked_indices
                respond_masked_indices = extract_respond_masked_indices(
                    masked_indices, self.n_prompt, actual_n_respond, respond_position_mask
                )
                return pred_y, respond_masked_indices  # 返回 (pred_y, mask)
        
        # ===== Step 7: Loss 计算（只在 respond 的 masked 位置）=====
        target = ys.squeeze(-1) if ys.dim() == 3 else ys
        
        # 🔧 使用辅助函数：提取 respond target 和 masked_indices
        respond_target = extract_respond_targets(target, self.n_prompt, actual_n_respond, respond_position_mask)
        respond_masked_indices = extract_respond_masked_indices(masked_indices, self.n_prompt, actual_n_respond, respond_position_mask)
        
        diff = pred_y - respond_target
        
        # 只在 respond 部分的 masked 位置计算 loss
        mask = respond_masked_indices.float()
        
        # 🔧 修复：与 LLaDA 对齐 - 先计算 per_sample_loss（不带权重），然后使用训练策略应用权重
        per_sample_loss = (diff.square() * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        # 🔧 修复：使用与 LLaDA 相同的训练策略计算loss权重
        weighted_loss = compute_loss_weight_with_strategy(
            per_sample_loss, t, t_scalar, self.training_strategy, self.num_timesteps
        )
        
        # 🔧 返回 respond_masked_indices 用于计算 train MSE
        # ✅ 统一返回格式：与其他 MDM 模型一致（4 个值）
        return weighted_loss, pred_y, t_scalar, respond_masked_indices
    
    @torch.no_grad()
    def _multistep_inference(self, xs, ys, device):
        """
        多步迭代推理（参考 Dream generate.py 逻辑）
        
        流程：
        1. 初始：mask 所有 respond 位置
        2. 使用 scheduler 预先计算每步 unmask 的数量
        3. 迭代：每步 forward → 计算 confidence → 选择 top-k 位置 unmask
        4. 逐步 refine 直到所有位置都被 unmask
        
        Args:
            xs: [B, n_prompt + n_respond, D]
            ys: [B, n_prompt + n_respond] (用于构建初始序列，但 respond 部分会被 mask)
            device: device
        
        Returns:
            (pred_y, initial_mask): 
                - pred_y: [B, actual_n_respond] - 最终预测的 respond 部分
                - initial_mask: [B, actual_n_respond] - 初始 mask（所有位置都为 True，表示初始时所有位置都被 mask）
        """
        from dllm.utils.generation_utils import get_num_transfer_tokens
        
        b_size, total_points, n_dims = xs.shape
        actual_n_respond = total_points - self.n_prompt
        respond_start = self.n_prompt
        
        # 初始化：所有 respond 位置都 mask
        ys_pred = torch.zeros(b_size, actual_n_respond, device=device)
        masked_indices = torch.ones(b_size, actual_n_respond, device=device, dtype=torch.bool)
        
        # 保存初始 mask（用于评估时只对初始被 mask 的位置计算 MSE）
        initial_mask = masked_indices.clone()
        
        # 预先计算所有步的 unmask 数量（使用 scheduler，与源码逻辑一致）
        num_transfer_tokens = get_num_transfer_tokens(
            mask_index=masked_indices,
            steps=self.inference_steps,
            scheduler=self.inference_scheduler,
            stochastic=False,
        )
        effective_steps = num_transfer_tokens.size(1)
        
        # 🔧 修复：预定义时间步序列（从 1.0 到 epsilon，对应 scheduler 的时间步语义）
        # 注意：时间步是递减的（从高 mask ratio 到低 mask ratio）
        time_steps = torch.linspace(1.0, self.mask_epsilon, effective_steps + 1, device=device)
        
        # 迭代去噪
        for step in range(effective_steps):
            # 使用预定义的时间步（而不是动态 mask ratio）
            t_curr = time_steps[step]  # 当前时间步
            
            # 构建当前状态：prompt + 部分预测的 respond
            ys_current = ys.clone()
            ys_current[:, respond_start:] = ys_pred
            
            # 构建序列
            zs = combine_xs_ys(xs, ys_current)
            embeds = self._read_in(zs)
            
            # 🔧 修复：添加时间嵌入
            time_emb = self._time_mlp(t_curr.view(1, 1, 1).expand(b_size, 2 * total_points, 1))
            embeds = embeds + time_emb
            
            # 应用 mask（只 mask 还未预测的位置）
            mask_embedding = self._read_in(self.mask_embedding.squeeze(0))
            for i in range(b_size):
                if masked_indices[i].any():
                    y_positions = masked_indices[i].nonzero(as_tuple=False).squeeze(-1)
                    full_seq_indices = (respond_start + y_positions) * 2 + 1
                    embeds[i, full_seq_indices, :] = mask_embedding
            
            # Forward pass
            output = self._backbone.model(inputs_embeds=embeds).last_hidden_state
            pred_y_all = self._read_out(output)[:, 0::2, 0]  # [B, total_points]
            pred_y_respond = pred_y_all[:, respond_start:respond_start+actual_n_respond]  # [B, actual_n_respond]
            
            # 🔧 使用辅助函数：计算 confidence
            confidence = compute_multistep_confidence(
                pred_y_respond, ys_pred, step, 
                self.inference_confidence_alg, device
            )
            
            # 只考虑当前 masked 位置的 confidence
            confidence = torch.where(
                masked_indices,
                confidence,
                torch.tensor(-float('inf'), device=device)
            )
            
            # 根据 scheduler 决定 unmask 的数量和位置
            for j in range(b_size):
                num_unmask = int(num_transfer_tokens[j, step].item())
                if num_unmask > 0 and masked_indices[j].any():
                    # 选择 confidence 最高的 num_unmask 个位置
                    available_mask_count = masked_indices[j].sum().item()
                    k = min(num_unmask, available_mask_count)
                    if k > 0:
                        _, top_indices = torch.topk(confidence[j], k=k)
                        
                        # Unmask 这些位置：使用预测值
                        ys_pred[j, top_indices] = pred_y_respond[j, top_indices]
                        masked_indices[j, top_indices] = False
        
        # 返回预测值和初始 mask（用于评估时只对初始被 mask 的位置计算 MSE）
        return ys_pred, initial_mask
    
    @torch.no_grad()
    def _dllm_generate_inference(self, xs, ys, device, respond_position_mask=None):
        """
        使用 dllm.pipelines.dream.infilling 进行迭代式推理
        
        这个方法包装了 dllm 的标准 infilling 函数，将其适配到连续值回归任务。
        
        核心思想：
        1. 将整个序列（prompt + respond）编码为 embeddings
        2. 在 respond 部分使用 mask embedding 替换
        3. 使用 dllm 的迭代式 diffusion 逻辑逐步 unmask
        
        Args:
            xs: [B, n_prompt + n_respond, D]
            ys: [B, n_prompt + n_respond]
            device: device
            respond_position_mask: [B, total_points] boolean tensor marking respond positions.
                                  If None, assumes respond positions are at the end (sequential mode).
                                  If provided, uses it to determine which positions are respond pairs
                                  (non-sequential mode, e.g., fixed/random permutation).
        
        Returns:
            (pred_y, initial_mask):
                - pred_y: [B, actual_n_respond] - 最终预测的 respond 部分
                - initial_mask: [B, actual_n_respond] - 初始 mask（所有位置都为 True）
        """
        from dllm.pipelines.dream import infilling
        from dllm.pipelines.dream.generate import sample_tokens
        import torch.nn.functional as F
        from dllm.utils.generation_utils import get_num_transfer_tokens
        
        b_size, total_points, n_dims = xs.shape
        
        # 🔧 使用辅助函数：计算实际 n_respond 和 respond 位置
        actual_n_respond = compute_actual_n_respond(total_points, self.n_prompt, respond_position_mask)
        
        # 🔧 为每个 batch 确定 respond 位置的序列索引（用于伪 token 序列）
        # 存储每个 batch 的 respond 位置的序列索引（y 位置是奇数索引）
        respond_seq_indices_per_batch = []
        if respond_position_mask is not None:
            for b in range(b_size):
                respond_positions = respond_position_mask[b].nonzero(as_tuple=True)[0]  # [actual_n_respond]
                # 转换为序列索引（y 位置是奇数索引：1, 3, 5, ...）
                respond_seq_indices = (respond_positions * 2 + 1).tolist()
                respond_seq_indices_per_batch.append(respond_seq_indices)
        else:
            # Sequential mode: respond 在末尾
            respond_start = self.n_prompt
            respond_seq_indices = [(respond_start + j) * 2 + 1 for j in range(actual_n_respond)]
            respond_seq_indices_per_batch = [respond_seq_indices] * b_size
        
        # 创建一个简单的 tokenizer wrapper（dllm 函数需要）
        class TokenizerWrapper:
            def __init__(self, mask_token_id=0, eos_token_id=1):
                self.mask_token_id = mask_token_id
                self.eos_token_id = eos_token_id
        
        tokenizer = TokenizerWrapper(mask_token_id=0, eos_token_id=1)
        
        # 创建一个模型包装器，将连续值任务转换为 dllm 期望的格式
        class ModelWrapper(nn.Module):
            def __init__(self, parent_model):
                super().__init__()
                self.parent = parent_model
                self.device = device
            
            def __call__(self, x, attention_mask=None, pos_id=None):
                """
                包装 forward，使其兼容 dllm 的 infilling 函数
                
                Args:
                    x: [B, 2*T] - 伪 token ids（实际上会被忽略，我们用 embeddings）
                    attention_mask: attention mask
                    pos_id: position ids
                
                Returns:
                    outputs with .logits attribute [B, 2*T, D]
                """
                # x 的形状是 [B, 2*T]，包含 mask_token_id 的位置
                b, seq_len = x.shape
                n_points = seq_len // 2
                
                # 构建 embeddings：从 xs, ys 重建序列
                # 这里我们需要访问外部的 xs, ys_pred
                # 使用闭包来访问
                xs_current = self.parent._current_xs
                ys_current = self.parent._current_ys
                
                # 构建序列
                zs = combine_xs_ys(xs_current, ys_current)
                embeds = self.parent._read_in(zs)
                
                # 应用 mask：将 x 中的 mask_token_id 位置替换为 mask_embedding
                mask_embedding = self.parent._read_in(self.parent.mask_embedding.squeeze(0))
                mask_positions = (x == tokenizer.mask_token_id)
                
                for i in range(b):
                    if mask_positions[i].any():
                        masked_seq_indices = mask_positions[i].nonzero(as_tuple=False).squeeze(-1)
                        embeds[i, masked_seq_indices, :] = mask_embedding
                
                # Backbone forward
                output = self.parent._backbone.model(inputs_embeds=embeds).last_hidden_state
                
                # 将输出转换为 logits 格式（虽然这里是回归任务，但我们模拟 logits）
                # 输出形状: [B, 2*T, hidden_size]
                # 我们需要将其转换为 [B, 2*T, vocab_size] 的形式
                # 但由于是回归任务，我们只需要保持维度一致即可
                
                class OutputWrapper:
                    def __init__(self, hidden_states):
                        # 对于回归任务，我们将 hidden_states 作为 "logits"
                        # 形状保持 [B, 2*T, hidden_size]
                        self.logits = hidden_states
                
                return OutputWrapper(output)
        
        # 包装模型
        model_wrapper = ModelWrapper(self)
        
        # 初始化：prompt 部分用真实值，respond 部分全部 mask
        # 构建初始序列（伪 token ids，实际会在 wrapper 中被替换）
        initial_mask = torch.ones(b_size, actual_n_respond, device=device, dtype=torch.bool)
        
        # 存储当前状态（供 wrapper 访问）
        self._current_xs = xs
        self._current_ys = ys.clone()
        
        # 🔧 初始化 respond 部分为 0（根据 respond_position_mask）
        if respond_position_mask is not None:
            for b in range(b_size):
                respond_positions = respond_position_mask[b].nonzero(as_tuple=True)[0]
                self._current_ys[b, respond_positions] = 0.0
        else:
            respond_start = self.n_prompt
            self._current_ys[:, respond_start:] = 0.0
        
        # 🔧 构建伪 token 序列：[B, 2*T]
        # prompt 部分：非 mask（用任意非 mask_token_id 的值）
        # respond 部分：mask（用 mask_token_id）
        pseudo_tokens = torch.zeros(b_size, 2 * total_points, dtype=torch.long, device=device)
        for i in range(b_size):
            # respond 部分的 y 位置标记为 mask（根据 respond_seq_indices_per_batch）
            for y_idx in respond_seq_indices_per_batch[i]:
                pseudo_tokens[i, y_idx] = tokenizer.mask_token_id
        
        # 预计算 transfer tokens（unmask 的 schedule）
        mask_index = pseudo_tokens == tokenizer.mask_token_id
        num_transfer_tokens_list = get_num_transfer_tokens(
            mask_index=mask_index,
            steps=self.inference_steps,
            scheduler=self.inference_scheduler,
            stochastic=False,
        )
        effective_steps = num_transfer_tokens_list.size(1)
        
        # 迭代式 diffusion（参考 dllm 的 infilling 逻辑）
        x = pseudo_tokens.clone()
        for step_i in range(effective_steps):
            mask_index = x == tokenizer.mask_token_id
            
            # Forward pass
            logits = model_wrapper(x, attention_mask=None, pos_id=None).logits
            # AR-shift（与 dllm 保持一致）
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            
            # 只考虑 masked 位置的 logits
            mask_logits = logits[mask_index]  # [num_masked, hidden_size]
            
            # 从 logits 中提取预测值（通过 read_out）
            pred_values = self._read_out(mask_logits).squeeze(-1)  # [num_masked]
            
            # 计算 confidence（使用指定的算法）
            if self.inference_confidence_alg == "entropy":
                # 对于回归任务，使用预测值的绝对值作为 confidence（越大越有信心）
                confidence = -torch.abs(pred_values)
            else:
                # 其他算法（如 random）
                confidence = torch.rand_like(pred_values)
            
            # Scatter confidence back to full tensor
            full_confidence = torch.full_like(
                x, -torch.inf, device=device, dtype=logits.dtype
            )
            full_confidence[mask_index] = confidence
            
            # 根据 schedule unmask 位置
            for j in range(b_size):
                number_transfer_tokens = num_transfer_tokens_list[j, step_i]
                if number_transfer_tokens > 0:
                    # Top-k selection
                    _, transfer_index = torch.topk(
                        full_confidence[j], number_transfer_tokens
                    )
                    
                    # Unmask：将选中的位置标记为非 mask
                    x[j, transfer_index] = 1  # 用任意非 mask_token_id 的值
                    
                    # 🔧 更新 ys_current（用预测值更新）
                    # 需要找到这些位置对应的 y 索引
                    for idx in transfer_index:
                        if idx % 2 == 1:  # 只处理 y 位置（奇数索引）
                            y_pos = idx // 2
                            
                            # 🔧 检查这个位置是否是 respond 位置
                            is_respond = False
                            if respond_position_mask is not None:
                                is_respond = respond_position_mask[j, y_pos].item()
                            else:
                                is_respond = (y_pos >= self.n_prompt)
                            
                            if is_respond:
                                # 重新 forward 获取当前预测
                                output_full = self._backbone.model(
                                    inputs_embeds=self._read_in(
                                        combine_xs_ys(self._current_xs, self._current_ys)
                                    )
                                ).last_hidden_state
                                pred_y_full = self._read_out(output_full)[:, 0::2, 0]
                                self._current_ys[j, y_pos] = pred_y_full[j, y_pos]
        
        # 🔧 最终预测：重新 forward 获取最终输出
        zs_final = combine_xs_ys(self._current_xs, self._current_ys)
        embeds_final = self._read_in(zs_final)
        output_final = self._backbone.model(inputs_embeds=embeds_final).last_hidden_state
        pred_y_all = self._read_out(output_final)[:, 0::2, 0]  # [B, total_points]
        
        # 🔧 使用辅助函数提取 respond 部分的预测
        pred_y_respond = extract_respond_predictions(
            pred_y_all, self.n_prompt, actual_n_respond, respond_position_mask
        )
        
        # 清理临时变量
        del self._current_xs
        del self._current_ys
        
        return pred_y_respond, initial_mask

# 
# # ============================================================
# # SDAR Masked ICL - Prompt-Respond Version
# # ============================================================
# class SDARPromptRespond(nn.Module):
#     """
#     SDAR Block Diffusion for Prompt-Respond Setting
#     
#     Training/Eval:
#     - Prompt 部分：始终可见，不 mask（全局双向attention）
#     - Respond 部分：随机 mask，用于训练和预测（可选block-causal attention）
#     - Loss: 只在 respond 的 masked 位置计算
#     
#     Block Diffusion特性：
#     - use_block_diffusion=False: 标准Masked Diffusion（全局双向attention）
#     - use_block_diffusion=True: Block Diffusion（块间因果，块内双向）
#       * Prompt区域：全部可见（双向attention）
#       * Respond区域：Block-causal attention
#         - 块间（Inter-block）：因果（第i块只能看到前i-1块）
#         - 块内（Intra-block）：双向（同一块内所有位置互相可见）
#     """
#     
#     def __init__(
#         self,
#         n_dims,
#         n_positions,
#         n_embd=256,
#         n_layer=12,
#         n_head=8,
#         n_prompt=20,
#         n_respond=5,
#         *,
#         mlp_ratio=4.0,
#         mask_epsilon=1e-3,
#         loss_weight_type="1/t",
#         train_mask_ratio=0.5,
#         eval_mask_ratio=1.0,
#         eval_mask_mode="fixed",  # "fixed" (固定mask ratio，通常为全mask) 或 "sample" (随机采样mask，与训练一致)
#         use_prompt_context=True,  # True: prompt 作为上下文; False: prompt 也被 mask（无 context 模式）
#         # 🆕 Block Diffusion 参数
#         use_block_diffusion=False,  # 是否启用Block Diffusion
#         block_size=4,               # 块大小（以点为单位，每个点包含x和y两个position）
#         # 多步推理优化选项（可选，默认关闭以保持向后兼容）
#         use_multistep_inference=False,
#         inference_steps=10,
#         inference_scheduler=None,
#         inference_confidence_alg="entropy",
#         # 🆕 Block-by-block inference 参数
#         use_block_by_block_inference=None,  # None: 自动根据 use_block_diffusion 决定
#         inference_steps_per_block=None,  # None: 自动从 inference_steps 计算
#         **extra,
#     ):
#         super().__init__()
#         if not SDAR_AVAILABLE:
#             raise ImportError("SDAR model is not available. Please check dLLM-RL/models/sdar is accessible.")
#         
#         self.name = "sdar_block_diffusion" if use_block_diffusion else "sdar_prompt_respond"
#         self.n_positions = n_positions
#         self.n_dims = n_dims
#         self.n_prompt = n_prompt
#         self.n_respond = n_respond
#         self.mask_epsilon = mask_epsilon
#         self.loss_weight_type = loss_weight_type
#         self.train_mask_ratio = train_mask_ratio
#         self.eval_mask_ratio = eval_mask_ratio
#         self.eval_mask_mode = eval_mask_mode
#         self.use_prompt_context = use_prompt_context
#         self.d_model = int(n_embd)
#         
#         # === Block Diffusion 参数 ===
#         self.use_block_diffusion = use_block_diffusion
#         self.block_size = block_size
#         
#         # === 多步推理优化选项（可选）===
#         self.use_multistep_inference = use_multistep_inference
#         self.inference_steps = inference_steps
#         self.inference_scheduler = inference_scheduler or LinearAlphaScheduler()
#         self.inference_confidence_alg = inference_confidence_alg
#         
#         # === Block-by-block inference 参数 ===
#         # 如果 use_block_by_block_inference 为 None，则根据 use_block_diffusion 自动决定
#         if use_block_by_block_inference is None:
#             self.use_block_by_block_inference = use_block_diffusion
#         else:
#             self.use_block_by_block_inference = use_block_by_block_inference
#         self.inference_steps_per_block = inference_steps_per_block
#         
#         # === Backbone Config ===
#         cfg = SDARConfig(
#             hidden_size=int(n_embd),
#             num_hidden_layers=int(n_layer),
#             num_attention_heads=int(n_head),
#             num_key_value_heads=int(n_head),
#             intermediate_size=int(n_embd * mlp_ratio),
#             max_position_embeddings=int(2 * n_positions),
#             vocab_size=1,  # 占位符，因为我们使用连续值输入
#             use_cache=True,  # 启用 cache 以支持 block-by-block inference
#         )
#         self._backbone = _SDARBase(cfg)
#         
#         # === Time Embedding MLP ===
#         self._time_mlp = nn.Sequential(
#             nn.Linear(1, self.d_model),
#             nn.SiLU(),
#             nn.Linear(self.d_model, self.d_model)
#         )
#         
#         self._read_in = nn.Linear(n_dims, cfg.hidden_size)
#         self._read_out = nn.Linear(cfg.hidden_size, 1)
#         
#         # 初始化 read_out 层，使用较小的初始化范围，避免初始输出过大
#         nn.init.normal_(self._read_out.weight, mean=0.0, std=0.02)
#         nn.init.zeros_(self._read_out.bias)
#         
#         # === Learnable mask token ===
#         self.mask_embedding = nn.Parameter(torch.randn(1, 1, n_dims))
#         
#         model_type = "Block Diffusion" if self.use_block_diffusion else "Masked Diffusion"
#         print(f"[SDAR {model_type}] d_model={self.d_model}, "
#               f"n_prompt={n_prompt}, n_respond={n_respond}, "
#               f"train_mask_ratio={self.train_mask_ratio}, eval_mask_ratio={self.eval_mask_ratio}, "
#               f"eval_mask_mode={self.eval_mask_mode}, use_prompt_context={self.use_prompt_context}")
#         if self.use_block_diffusion:
#             print(f"  [Block Diffusion] Enabled: block_size={self.block_size}")
#             print(f"    - Prompt: Full bidirectional attention")
#             print(f"    - Respond: Block-causal attention (inter-block causal, intra-block bidirectional)")
#         if self.use_multistep_inference:
#             print(f"  [Multi-step Inference] Enabled: steps={self.inference_steps}, "
#                   f"confidence_alg={self.inference_confidence_alg}")
#     
#     def _create_block_causal_attention_mask(
#         self, 
#         total_points: int, 
#         n_prompt: int, 
#         block_size: int, 
#         device: torch.device,
#         dtype: torch.dtype
#     ) -> torch.Tensor:
#         """
#         创建Block-Causal Attention Mask for Prompt-Respond ICL任务
#         
#         序列结构：[x1, y1, x2, y2, ..., x_prompt, y_prompt | x_respond, y_respond, ...]
#         
#         Attention规则：
#         1. Prompt区域（前n_prompt个点，即前2*n_prompt个position）：
#            - 完全双向attention（所有位置互相可见）
#         2. Respond区域（后面的点）：
#            - Block-Causal Attention：
#              * 块间（Inter-block）：第i块只能看到第0~i-1块（因果）
#              * 块内（Intra-block）：同一块内所有位置互相可见（双向）
#            - 所有respond块都可以看到完整的prompt区域
#         
#         Args:
#             total_points: 总点数（n_prompt + n_respond）
#             n_prompt: prompt点数
#             block_size: 块大小（以点为单位，每个点包含x和y两个position）
#             device: torch device
#             dtype: dtype for the mask
#         
#         Returns:
#             attention_mask: [1, 1, seq_len, seq_len] 的attention mask
#                            0 = can attend, -inf = cannot attend (additive mask)
#         """
#         seq_len = 2 * total_points  # 每个点包含x和y
#         prompt_len = 2 * n_prompt    # prompt部分的序列长度
#         respond_len = seq_len - prompt_len  # respond部分的序列长度
# 
#         # 🔧 修复问题1：初始化mask为全1（默认不能attend），然后显式设置0（可以attend）
#         # 这样可以确保只有被显式允许的位置才能attend，避免信息泄露
#         mask = torch.ones(1, 1, seq_len, seq_len, device=device, dtype=dtype)
# 
#         # === Step 1: Prompt区域（完全双向attention）===
#         # Prompt区域内所有位置互相可见
#         mask[:, :, :prompt_len, :prompt_len] = 0  # 0表示可以attend
# 
#         # === Step 2: Respond区域（Block-Causal Attention）===
#         if respond_len > 0:
#             # 计算respond部分的块数（block_size是以点为单位，需要转换为position）
#             block_size_pos = 2 * block_size  # 每个块包含block_size个点，即2*block_size个position
#             num_respond_blocks = (respond_len + block_size_pos - 1) // block_size_pos
# 
#             for block_i in range(num_respond_blocks):
#                 # 当前块的position范围
#                 block_start = prompt_len + block_i * block_size_pos
#                 block_end = min(prompt_len + (block_i + 1) * block_size_pos, seq_len)
# 
#                 # 规则1：当前块可以看到整个Prompt区域
#                 mask[:, :, block_start:block_end, :prompt_len] = 0
# 
#                 # 规则2：当前块可以看到之前的所有Respond块（因果）
#                 if block_i > 0:
#                     prev_respond_end = prompt_len + block_i * block_size_pos
#                     mask[:, :, block_start:block_end, prompt_len:prev_respond_end] = 0
# 
#                 # 规则3：当前块内部
#                 # 🔧 修复问题2：当 block_size=1 时，块内应该是因果的（等价于 AR）
#                 if block_size == 1:
#                     # block_size=1：块内因果（等价于 AR）
#                     # 对于 [x_i, y_i]：
#                     # - x_i 可以看到自己
#                     # - y_i 可以看到 x_i 和自己
#                     # - x_i 不能看到 y_i（因果约束）
#                     if block_start + 1 < block_end:
#                         # x_i 看自己
#                         mask[:, :, block_start:block_start + 1, block_start:block_start + 1] = 0
#                         # y_i 看 x_i 和自己
#                         mask[:, :, block_start + 1:block_end, block_start:block_end] = 0
#                         # x_i 不能看到 y_i（因果约束，保持为1）
#                         # 注意：由于初始化为1，这里不需要显式设置
#                 else:
#                     # block_size>1：块内双向（标准 Block Diffusion）
#                     mask[:, :, block_start:block_end, block_start:block_end] = 0
# 
#         # 转换为additive mask：0 → 0 (can attend), 1 → -inf (cannot attend)
#         min_val = torch.finfo(dtype).min
#         attention_mask = torch.where(
#             mask == 0,
#             torch.zeros_like(mask),
#             torch.full_like(mask, min_val)
#         )
# 
#         return attention_mask
#     
#     def forward(self, xs, ys, train_mode=True, respond_position_mask=None):
#         """
#         Forward pass for Prompt-Respond setting
#         
#         Args:
#             xs: [B, n_prompt + n_respond, D]
#             ys: [B, n_prompt + n_respond]
#             train_mode: True for training, False for inference
#             respond_position_mask: [B, total_points] boolean tensor marking respond positions.
#                                    If None, assumes respond positions are at the end (sequential mode).
#                                    If provided, uses it to determine which positions are respond pairs
#                                    (non-sequential mode).
#         
#         Returns:
#             If train_mode: (loss, pred_y, t_scalar)
#             Else: pred_y
#         """
#         b, total_points, d = xs.shape
#         device = xs.device
#         
#         # 🔧 使用辅助函数：计算实际 n_respond
#         actual_n_respond = compute_actual_n_respond(total_points, self.n_prompt, respond_position_mask)
#         if respond_position_mask is not None:
#             assert respond_position_mask.shape == (b, total_points), \
#                 f"respond_position_mask shape mismatch: expected {(b, total_points)}, got {respond_position_mask.shape}"
#         assert total_points >= self.n_prompt, \
#             f"Expected at least {self.n_prompt} points (prompt), got {total_points}"
#         assert actual_n_respond <= self.n_respond, \
#             f"Actual respond count {actual_n_respond} exceeds model max {self.n_respond}"
#         
#         # 🔧 使用辅助函数：采样 timestep
#         t_scalar = sample_timestep_for_mdm(
#             b, device, train_mode, 
#             self.mask_epsilon, self.train_mask_ratio, 
#             self.eval_mask_ratio, self.eval_mask_mode
#         )
#         
#         # 🔧 使用辅助函数：生成 masked_indices
#         masked_indices = generate_masked_indices_for_mdm(
#             b, total_points, self.n_prompt, actual_n_respond,
#             t_scalar, device, self.use_prompt_context, respond_position_mask
#         )
#         
#         # ===== Step 3: 构建序列 =====
#         ys_input = ys.clone().float()
#         if ys_input.dim() == 2:
#             ys_input = ys_input.unsqueeze(-1)
#         
#         ys_wide = torch.cat([ys_input, torch.zeros(b, total_points, d - 1, device=device)], dim=2)
#         zs = torch.stack((xs, ys_wide), dim=2).view(b, 2 * total_points, d)
#         
#         # ===== Step 4: Embedding + Time conditioning =====
#         embeds = self._read_in(zs)
#         time_emb = self._time_mlp(t_scalar.view(b, 1, 1).expand(b, 2 * total_points, 1))
#         embeds = embeds + time_emb
#         
#         # ===== Step 5: 应用 mask =====
#         mask_embed = self._read_in(self.mask_embedding.squeeze(0))
#         for i in range(b):
#             if masked_indices[i].any():
#                 # 找到所有被 mask 的位置（包括 prompt 和 respond）
#                 masked_positions = masked_indices[i].nonzero(as_tuple=False).squeeze(-1)
#                 # 转换为序列中的 y 位置索引（y 位置在序列中是奇数索引：1, 3, 5, ...）
#                 full_idx = masked_positions * 2 + 1
#                 embeds[i, full_idx, :] = mask_embed
#         
#         # ===== Step 6: Backbone forward =====
#         # SDAR 模型使用 inputs_embeds 而不是 input_ids
#         # 注意：SDAR 返回 BaseModelOutputWithPast，使用 last_hidden_state 属性
#         
#         # 🆕 创建Block-Causal Attention Mask（如果启用）
#         if self.use_block_diffusion:
#             # 创建block-causal attention mask
#             attention_mask = self._create_block_causal_attention_mask(
#                 total_points=total_points,
#                 n_prompt=self.n_prompt,
#                 block_size=self.block_size,
#                 device=device,
#                 dtype=embeds.dtype
#             )
#             # 扩展到batch size
#             attention_mask = attention_mask.expand(b, -1, -1, -1)
#         else:
#             # 标准Masked Diffusion：不使用attention mask（全局双向）
#             attention_mask = None
#         
#         out = self._backbone(
#             inputs_embeds=embeds,
#             attention_mask=attention_mask,  # 传入block-causal mask（或None）
#             output_hidden_states=False,  # 我们只需要最后一层
#         )
#         h = out.last_hidden_state
#         pred_y_all = self._read_out(h)[:, 0::2, 0]  # [B, total_points]
#         
#         # 🔧 使用辅助函数：提取 respond 部分的预测
#         pred_y = extract_respond_predictions(pred_y_all, self.n_prompt, actual_n_respond, respond_position_mask)
#         
#         if not train_mode:
#             # 推理模式：支持单步、多步或 block-by-block 推理
#             if self.use_block_by_block_inference and self.use_block_diffusion:
#                 # 使用逐个 block 生成（参考 BD3LMSampler）
#                 return self._block_by_block_inference(xs, ys, device, respond_position_mask=respond_position_mask)
#             elif self.use_multistep_inference:
#                 return self._multistep_inference(xs, ys, device, respond_position_mask=respond_position_mask)
#             else:
#                 # 🔧 使用辅助函数：提取 respond 部分的 masked_indices
#                 respond_masked_indices = extract_respond_masked_indices(
#                     masked_indices, self.n_prompt, actual_n_respond, respond_position_mask
#                 )
#                 return pred_y, respond_masked_indices  # 返回 (pred_y, mask)
#         
#         # ===== Step 8: Training Loss（只在 respond 的 masked 位置）=====
#         target = ys.squeeze(-1) if ys.dim() == 3 else ys
#         
#         # 🔧 使用辅助函数：提取 respond target 和 masked_indices
#         respond_target = extract_respond_targets(target, self.n_prompt, actual_n_respond, respond_position_mask)
#         respond_masked_indices = extract_respond_masked_indices(masked_indices, self.n_prompt, actual_n_respond, respond_position_mask)
#         
#         diff = pred_y - respond_target
#         
#         # 只在 respond 部分的 masked 位置计算 loss
#         mask = respond_masked_indices.float()
#         
#         if self.loss_weight_type == "1/t":
#             weight = (1.0 / (t_scalar + 1e-8)).unsqueeze(1)
#             weight = torch.clamp(weight, min=0.1, max=10.0)
#         elif self.loss_weight_type == "ones":
#             weight = torch.ones_like(t_scalar).unsqueeze(1)
#         else:
#             raise ValueError(f"Unknown loss_weight_type: {self.loss_weight_type}")
#         
#         per_sample_loss = (diff.square() * mask * weight).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
#         weighted_loss = per_sample_loss.mean()
#         
#         # 🔧 返回 respond_masked_indices 用于计算 train MSE
#         return weighted_loss, pred_y, t_scalar, respond_masked_indices
#     
#     @torch.no_grad()
#     def _multistep_inference(self, xs, ys, device, respond_position_mask=None):
#         """
#         多步迭代推理（参考 LLaDA/Dream generate.py 逻辑）
#         
#         流程：
#         1. 初始：mask 所有 respond 位置
#         2. 使用 scheduler 预先计算每步 unmask 的数量
#         3. 迭代：每步 forward → 计算 confidence → 选择 top-k 位置 unmask
#         4. 逐步 refine 直到所有位置都被 unmask
#         
#         Args:
#             xs: [B, n_prompt + n_respond, D]
#             ys: [B, n_prompt + n_respond] (用于构建初始序列，但 respond 部分会被 mask)
#             device: device
#             respond_position_mask: [B, total_points] boolean tensor marking respond positions.
#                                    Currently not fully supported for multistep inference - will use
#                                    sequential mode (respond at the end) if None.
#         
#         Returns:
#             (pred_y, initial_mask): 
#                 - pred_y: [B, actual_n_respond] - 最终预测的 respond 部分
#                 - initial_mask: [B, actual_n_respond] - 初始 mask（所有位置都为 True，表示初始时所有位置都被 mask）
#         """
#         # TODO: Full support for non-sequential mode in multistep inference
#         # For now, multistep inference only works in sequential mode
#         if respond_position_mask is not None:
#             raise NotImplementedError("Non-sequential mode is not yet supported for multistep inference")
#         from dllm.utils.generation_utils import get_num_transfer_tokens
#         
#         b, total_points, d = xs.shape
#         actual_n_respond = total_points - self.n_prompt
#         respond_start = self.n_prompt
#         
#         # 初始化：所有 respond 位置都 mask
#         ys_pred = torch.zeros(b, actual_n_respond, device=device)
#         masked_indices = torch.ones(b, actual_n_respond, device=device, dtype=torch.bool)
#         
#         # 保存初始 mask（用于评估时只对初始被 mask 的位置计算 MSE）
#         initial_mask = masked_indices.clone()
#         
#         # 预先计算所有步的 unmask 数量（使用 scheduler，与源码逻辑一致）
#         num_transfer_tokens = get_num_transfer_tokens(
#             mask_index=masked_indices,
#             steps=self.inference_steps,
#             scheduler=self.inference_scheduler,
#             stochastic=False,
#         )
#         effective_steps = num_transfer_tokens.size(1)
#         
#         # 预定义时间步序列（从 1.0 到 epsilon，对应 scheduler 的时间步语义）
#         time_steps = torch.linspace(1.0, self.mask_epsilon, effective_steps + 1, device=device)
#         
#         # 迭代去噪
#         for step in range(effective_steps):
#             # 使用预定义的时间步
#             t_curr = time_steps[step]
#             
#             # 构建当前状态：prompt + 部分预测的 respond
#             ys_current = ys.clone()
#             ys_current[:, respond_start:] = ys_pred
#             
#             # 构建序列
#             ys_input = ys_current.clone().float()
#             if ys_input.dim() == 2:
#                 ys_input = ys_input.unsqueeze(-1)
#             ys_wide = torch.cat([ys_input, torch.zeros(b, total_points, d - 1, device=device)], dim=2)
#             zs = torch.stack((xs, ys_wide), dim=2).view(b, 2 * total_points, d)
#             
#             # Embedding + Time conditioning（使用预定义的时间步）
#             embeds = self._read_in(zs)
#             time_emb = self._time_mlp(t_curr.view(1, 1, 1).expand(b, 2 * total_points, 1))
#             embeds = embeds + time_emb
#             
#             # 应用 mask（只 mask 还未预测的位置）
#             mask_embed = self._read_in(self.mask_embedding.squeeze(0))
#             for i in range(b):
#                 if masked_indices[i].any():
#                     y_positions = masked_indices[i].nonzero(as_tuple=False).squeeze(-1)
#                     full_idx = (respond_start + y_positions) * 2 + 1
#                     embeds[i, full_idx, :] = mask_embed
#             
#             # Forward pass
#             # 🆕 创建Block-Causal Attention Mask（如果启用）
#             if self.use_block_diffusion:
#                 attention_mask = self._create_block_causal_attention_mask(
#                     total_points=total_points,
#                     n_prompt=self.n_prompt,
#                     block_size=self.block_size,
#                     device=device,
#                     dtype=embeds.dtype
#                 ).expand(b, -1, -1, -1)
#             else:
#                 attention_mask = None
#             
#             out = self._backbone(
#                 inputs_embeds=embeds,
#                 attention_mask=attention_mask,  # ✅ 添加 block-causal mask（或None）
#                 output_hidden_states=False,  # 我们只需要最后一层
#             )
#             h = out.last_hidden_state
#             pred_y_all = self._read_out(h)[:, 0::2, 0]  # [B, total_points]
#             pred_y_respond = pred_y_all[:, respond_start:respond_start+actual_n_respond]  # [B, actual_n_respond]
#             
#             # 🔧 使用辅助函数：计算 confidence
#             confidence = compute_multistep_confidence(
#                 pred_y_respond, ys_pred, step, 
#                 self.inference_confidence_alg, device
#             )
#             
#             # 只考虑当前 masked 位置的 confidence
#             confidence = torch.where(
#                 masked_indices, 
#                 confidence, 
#                 torch.tensor(-float('inf'), device=device)
#             )
#             
#             # 根据 scheduler 决定 unmask 的数量和位置
#             for j in range(b):
#                 num_unmask = int(num_transfer_tokens[j, step].item())
#                 if num_unmask > 0 and masked_indices[j].any():
#                     # 选择 confidence 最高的 num_unmask 个位置
#                     available_mask_count = masked_indices[j].sum().item()
#                     k = min(num_unmask, available_mask_count)
#                     if k > 0:
#                         _, top_indices = torch.topk(confidence[j], k=k)
#                         
#                         # Unmask 这些位置：使用预测值
#                         ys_pred[j, top_indices] = pred_y_respond[j, top_indices]
#                         masked_indices[j, top_indices] = False
#         
#         # 返回预测值和初始 mask（用于评估时只对初始被 mask 的位置计算 MSE）
#         return ys_pred, initial_mask
#     
#     @torch.no_grad()
#     def _block_by_block_inference(self, xs, ys, device, respond_position_mask=None):
#         """
#         逐个 block 生成推理（参考 BD3LMSampler 逻辑）
#         
#         流程：
#         1. 外层循环：逐个 block 生成
#            - Forward prefix（prompt + 之前的 blocks），使用 use_cache=True 缓存 KV
#            - Append 新的 block（mask tokens）
#         2. 内层循环：对当前 block 进行多步 diffusion
#            - 只 forward 当前 block，使用 past_key_values 缓存 prefix
#            - 使用 get_num_transfer_tokens 计算每步 unmask 的数量
#            - 逐步 unmask 当前 block 的位置
#         
#         Args:
#             xs: [B, n_prompt + n_respond, D]
#             ys: [B, n_prompt + n_respond] (用于构建初始序列，但 respond 部分会被 mask)
#             device: device
#             respond_position_mask: [B, total_points] boolean tensor marking respond positions.
#                                    Currently only supports sequential mode (respond at the end).
#         
#         Returns:
#             (pred_y, initial_mask): 
#                 - pred_y: [B, actual_n_respond] - 最终预测的 respond 部分
#                 - initial_mask: [B, actual_n_respond] - 初始 mask（所有位置都为 True）
#         """
#         import copy
#         import math
#         
#         # TODO: Full support for non-sequential mode
#         if respond_position_mask is not None:
#             raise NotImplementedError("Non-sequential mode is not yet supported for block-by-block inference")
#         
#         b, total_points, d = xs.shape
#         actual_n_respond = total_points - self.n_prompt
#         respond_start = self.n_prompt
#         
#         # 初始化：所有 respond 位置都 mask
#         ys_pred = torch.zeros(b, actual_n_respond, device=device)
#         masked_indices = torch.ones(b, actual_n_respond, device=device, dtype=torch.bool)
#         initial_mask = masked_indices.clone()
#         
#         # 计算 block 数量（block_size 是以点为单位）
#         block_size_points = self.block_size  # 每个 block 包含的点数
#         num_blocks = math.ceil(actual_n_respond / block_size_points)
#         
#         # 计算每个 block 的 diffusion 步数
#         if self.inference_steps_per_block is not None:
#             steps_per_block = self.inference_steps_per_block
#         else:
#             steps_per_block = math.ceil(self.inference_steps / num_blocks) if num_blocks > 0 else self.inference_steps
#         
#         # 预定义时间步序列（从 1.0 到 epsilon）
#         time_steps = torch.linspace(1.0, self.mask_epsilon, steps_per_block + 1, device=device)
# 
#         # 🔧 修复问题3：使用累积的 KV cache，避免重复计算
#         # 🔧 修复变量作用域问题：在循环外初始化前缀长度变量
#         current_prefix_points = self.n_prompt  # 当前前缀长度（点数），会在每个block生成后递增
#         accumulated_past_kv = None  # 累积的KV cache
# 
#         # ==========================================================
#         # 外层循环：逐个 block 生成
#         # ==========================================================
#         for block_idx in range(num_blocks):
#             # 计算当前 block 的起始和结束位置（以点为单位）
#             block_start_point = block_idx * block_size_points
#             block_end_point = min((block_idx + 1) * block_size_points, actual_n_respond)
#             cur_block_len = block_end_point - block_start_point
# 
#             if cur_block_len <= 0:
#                 break
# 
#             # 🔧 每次循环动态计算前缀序列长度
#             prefix_seq_len = 2 * current_prefix_points
# 
#             # ------------------------------------------------------
#             # 2.1) Prefix handling with KV cache optimization
#             # ------------------------------------------------------
#             if block_idx == 0:
#                 # 第一个block：计算完整的 prompt 的 KV cache
#                 # 构建 prompt 的序列
#                 ys_prefix_input = ys[:, :current_prefix_points].clone().float()
#                 if ys_prefix_input.dim() == 2:
#                     ys_prefix_input = ys_prefix_input.unsqueeze(-1)
#                 ys_prefix_wide = torch.cat([ys_prefix_input, torch.zeros(b, current_prefix_points, d - 1, device=device)], dim=2)
#                 zs_prefix = torch.stack((xs[:, :current_prefix_points], ys_prefix_wide), dim=2).view(b, prefix_seq_len, d)
# 
#                 # Embedding + Time conditioning
#                 embeds_prefix = self._read_in(zs_prefix)
#                 t_prefix = time_steps[0]
#                 time_emb_prefix = self._time_mlp(t_prefix.view(1, 1, 1).expand(b, prefix_seq_len, 1))
#                 embeds_prefix = embeds_prefix + time_emb_prefix
# 
#                 # 创建 prompt 的 attention mask
#                 if self.use_block_diffusion:
#                     attention_mask_prefix = self._create_block_causal_attention_mask(
#                         total_points=current_prefix_points,
#                         n_prompt=self.n_prompt,
#                         block_size=self.block_size,
#                         device=device,
#                         dtype=embeds_prefix.dtype
#                     ).expand(b, -1, -1, -1)
#                 else:
#                     attention_mask_prefix = None
# 
#                 # Forward prompt，获取 KV cache
#                 out_prefix = self._backbone(
#                     inputs_embeds=embeds_prefix,
#                     attention_mask=attention_mask_prefix,
#                     use_cache=True,
#                     output_hidden_states=False,
#                 )
#                 accumulated_past_kv = out_prefix.past_key_values
# 
#             # 使用累积的 past_key_values（包含 prompt + 之前所有生成的 blocks）
#             prefix_past_key_values = accumulated_past_kv
#             
#             # ------------------------------------------------------
#             # 2.2) Append new block of mask tokens
#             # ------------------------------------------------------
#             # 当前 block 的 y 值初始化为 0（mask）
#             block_ys = torch.zeros(b, cur_block_len, device=device)
#             ys_current = ys.clone()
#             ys_current[:, respond_start + block_start_point:respond_start + block_end_point] = block_ys
# 
#             # 构建完整序列（prefix + current block）
#             total_points_current = current_prefix_points + cur_block_len
#             total_seq_len_current = 2 * total_points_current
# 
#             ys_current_input = ys_current[:, :total_points_current].clone().float()
#             if ys_current_input.dim() == 2:
#                 ys_current_input = ys_current_input.unsqueeze(-1)
#             ys_current_wide = torch.cat([ys_current_input, torch.zeros(b, total_points_current, d - 1, device=device)], dim=2)
#             zs_current = torch.stack((xs[:, :total_points_current], ys_current_wide), dim=2).view(b, total_seq_len_current, d)
# 
#             # 当前 block 的 mask 状态
#             block_masked_indices = torch.ones(b, cur_block_len, device=device, dtype=torch.bool)
# 
#             # 预先计算当前 block 的 unmask 数量
#             num_transfer_tokens = get_num_transfer_tokens(
#                 mask_index=block_masked_indices,
#                 steps=steps_per_block,
#                 scheduler=self.inference_scheduler,
#                 stochastic=False,
#             )
#             effective_steps = num_transfer_tokens.size(1)
# 
#             # 创建完整序列的 attention mask，用于当前block的查询
#             if self.use_block_diffusion:
#                 attention_mask_full = self._create_block_causal_attention_mask(
#                     total_points=total_points_current,
#                     n_prompt=self.n_prompt,
#                     block_size=self.block_size,
#                     device=device,
#                     dtype=torch.float32
#                 ).expand(b, -1, -1, -1)
# 
#                 # 提取当前 block 的 attention mask view
#                 # 形状: [B, 1, block_seq_len, total_seq_len_current]
#                 block_seq_start = prefix_seq_len
#                 block_seq_end = total_seq_len_current
#                 attn_block = attention_mask_full[:, :, block_seq_start:block_seq_end, :]
#             else:
#                 attention_mask_full = None
#                 attn_block = None
#             
#             # ======================================================
#             # 3) Inner diffusion loop within the current block
#             # ======================================================
#             for i_step in range(effective_steps):
#                 # 获取当前 block 的 y 值
#                 block_ys_current = ys_current[:, respond_start + block_start_point:respond_start + block_end_point]
#                 
#                 # 检查是否还有 masked 位置
#                 if not block_masked_indices.any():
#                     break
#                 
#                 # 构建当前 block 的序列
#                 block_ys_input = block_ys_current.clone().float()
#                 if block_ys_input.dim() == 2:
#                     block_ys_input = block_ys_input.unsqueeze(-1)
#                 block_ys_wide = torch.cat([block_ys_input, torch.zeros(b, cur_block_len, d - 1, device=device)], dim=2)
#                 block_xs = xs[:, respond_start + block_start_point:respond_start + block_end_point]
#                 zs_block = torch.stack((block_xs, block_ys_wide), dim=2).view(b, 2 * cur_block_len, d)
#                 
#                 # Embedding + Time conditioning
#                 embeds_block = self._read_in(zs_block)
#                 t_curr = time_steps[i_step]
#                 time_emb_block = self._time_mlp(t_curr.view(1, 1, 1).expand(b, 2 * cur_block_len, 1))
#                 embeds_block = embeds_block + time_emb_block
#                 
#                 # 应用 mask（只 mask 还未预测的位置）
#                 mask_embed = self._read_in(self.mask_embedding.squeeze(0))
#                 for i in range(b):
#                     if block_masked_indices[i].any():
#                         y_positions = block_masked_indices[i].nonzero(as_tuple=False).squeeze(-1)
#                         full_idx = y_positions * 2 + 1  # y 位置在 block 序列中是奇数索引
#                         embeds_block[i, full_idx, :] = mask_embed
#                 
#                 # Forward pass：只 forward 当前 block，使用 past_key_values 缓存 prefix
#                 # 🔧 移除copy.deepcopy以提升性能（模型内部不会修改KV cache）
#                 out_block = self._backbone(
#                     inputs_embeds=embeds_block,
#                     attention_mask=attn_block,
#                     past_key_values=prefix_past_key_values,  # 直接传入，不需要deepcopy
#                     use_cache=False,  # 不需要再次缓存
#                     output_hidden_states=False,
#                 )
#                 h_block = out_block.last_hidden_state
#                 pred_y_block_all = self._read_out(h_block)[:, 0::2, 0]  # [B, cur_block_len]
#                 
#                 # 计算 confidence（使用简单的 variance 作为 confidence）
#                 if self.inference_confidence_alg == "entropy":
#                     # 对于连续值，使用预测值的方差作为 confidence 的逆
#                     pred_variance = (pred_y_block_all - block_ys_current).abs()
#                     confidence = -pred_variance  # 负方差，越小越好
#                 else:
#                     # 默认使用绝对值差异
#                     confidence = -(pred_y_block_all - block_ys_current).abs()
#                 
#                 # 只考虑当前 masked 位置的 confidence
#                 confidence = torch.where(
#                     block_masked_indices,
#                     confidence,
#                     torch.tensor(-float('inf'), device=device)
#                 )
#                 
#                 # 根据 scheduler 决定 unmask 的数量和位置
#                 for j in range(b):
#                     num_unmask = int(num_transfer_tokens[j, i_step].item())
#                     if num_unmask > 0 and block_masked_indices[j].any():
#                         available_mask_count = block_masked_indices[j].sum().item()
#                         k = min(num_unmask, available_mask_count)
#                         if k > 0:
#                             _, top_indices = torch.topk(confidence[j], k=k)
#                             
#                             # Unmask 这些位置：使用预测值
#                             block_ys_current[j, top_indices] = pred_y_block_all[j, top_indices]
#                             block_masked_indices[j, top_indices] = False
#                 
#                 # 更新完整序列中的当前 block
#                 ys_current[:, respond_start + block_start_point:respond_start + block_end_point] = block_ys_current
# 
#             # 更新全局预测值
#             ys_pred[:, block_start_point:block_end_point] = block_ys_current
# 
#             # 🔧 修复问题3（续）：在生成当前block后，更新accumulated_past_kv
#             # 用最终的block_ys_current重新forward一次，获取KV cache，然后append
#             if block_idx < num_blocks - 1:  # 不是最后一个block时才需要更新cache（最后一个block后不需要再生成）
#                 # 构建当前block的最终序列
#                 block_ys_final = ys_pred[:, block_start_point:block_end_point].clone().float()
#                 if block_ys_final.dim() == 1:
#                     block_ys_final = block_ys_final.unsqueeze(-1)
#                 if block_ys_final.dim() == 2:
#                     block_ys_final = block_ys_final.unsqueeze(-1)
#                 block_ys_wide_final = torch.cat([block_ys_final, torch.zeros(b, cur_block_len, d - 1, device=device)], dim=2)
#                 block_xs_final = xs[:, respond_start + block_start_point:respond_start + block_end_point]
#                 zs_block_final = torch.stack((block_xs_final, block_ys_wide_final), dim=2).view(b, 2 * cur_block_len, d)
# 
#                 # Embedding + Time conditioning (使用第一个时间步)
#                 embeds_block_final = self._read_in(zs_block_final)
#                 t_final = time_steps[0]
#                 time_emb_block_final = self._time_mlp(t_final.view(1, 1, 1).expand(b, 2 * cur_block_len, 1))
#                 embeds_block_final = embeds_block_final + time_emb_block_final
# 
#                 # Forward当前block，获取KV cache
#                 out_block_final = self._backbone(
#                     inputs_embeds=embeds_block_final,
#                     attention_mask=attn_block,
#                     past_key_values=prefix_past_key_values,  # 直接传入，不需要deepcopy
#                     use_cache=True,  # 启用cache以获取当前block的KV
#                     output_hidden_states=False,
#                 )
# 
#                 # 将当前block的KV cache append到accumulated_past_kv
#                 # out_block_final.past_key_values包含了prefix + 当前block的KV cache
#                 accumulated_past_kv = out_block_final.past_key_values
# 
#                 # 🔧 关键修复：更新前缀长度，供下一个block循环使用
#                 current_prefix_points += cur_block_len
# 
#         # 返回预测值和初始 mask
#         return ys_pred, initial_mask
# 

# ============================================================
# LLaDA Block Diffusion - Prompt-Respond Version
# ============================================================
class LLaDABlockDiffusion(LLaDAPromptRespond):
    """
    LLaDA + Block Diffusion for Prompt-Respond Setting

    基于已验证的 LLaDA backbone，添加 Block-Causal Attention 支持。
    用于验证 Block Diffusion 逻辑是否正确（隔离 SDAR backbone 问题）。

    Training/Eval:
    - Prompt 部分：始终可见，不 mask（全局双向attention）
    - Respond 部分：随机 mask，用于训练和预测（可选block-causal attention）
    - Loss: 只在 respond 的 masked 位置计算

    Block Diffusion特性：
    - use_block_diffusion=False: 标准Masked Diffusion（全局双向attention）
    - use_block_diffusion=True: Block Diffusion（块间因果，块内双向）
      * Prompt区域：全部可见（双向attention）
      * Respond区域：Block-causal attention
        - 块间（Inter-block）：因果（第i块只能看到前i-1块）
        - 块内（Intra-block）：双向（同一块内所有位置互相可见）
    """

    def __init__(
        self,
        n_dims,
        n_positions,
        n_embd=256,
        n_layer=12,
        n_head=8,
        n_prompt=20,
        n_respond=5,
        *,
        mlp_ratio=4.0,
        block_group_size=1,
        mask_epsilon=1e-3,
        loss_weight_type="1/t",
        train_mask_ratio=0.5,
        eval_mask_ratio=1.0,
        eval_mask_mode="fixed",
        use_prompt_context=True,
        # 多步推理优化选项（可选）
        use_multistep_inference=False,
        inference_steps=10,
        inference_scheduler=None,
        inference_confidence_alg="entropy",
        # 可选的训练策略配置
        training_strategy=None,
        # 🆕 Block Diffusion 参数
        use_block_diffusion=False,
        block_size=4,
        **extra,
    ):
        # 调用父类 LLaDAPromptRespond 的初始化
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
            mask_epsilon=mask_epsilon,
            loss_weight_type=loss_weight_type,
            train_mask_ratio=train_mask_ratio,
            eval_mask_ratio=eval_mask_ratio,
            eval_mask_mode=eval_mask_mode,
            use_prompt_context=use_prompt_context,
            use_multistep_inference=use_multistep_inference,
            inference_steps=inference_steps,
            inference_scheduler=inference_scheduler,
            inference_confidence_alg=inference_confidence_alg,
            training_strategy=training_strategy,
            **extra,
        )

        # 🆕 添加 Block Diffusion 参数
        self.use_block_diffusion = use_block_diffusion
        self.block_size = block_size

        # 更新模型名称
        if self.use_block_diffusion:
            self.name = "llada_block_diffusion"
            print(f"[LLaDA Block Diffusion] Enabled: block_size={self.block_size}")
            print(f"  - Prompt: Full bidirectional attention")
            print(f"  - Respond: Block-causal attention (inter-block causal, intra-block bidirectional)")

    def _create_block_causal_attention_bias(
        self,
        total_points: int,
        n_prompt: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        创建Block-Causal Attention Bias for LLaDA backbone

        注意：LLaDA 使用 attention_bias（additive），不是 attention_mask
        - 0 = can attend
        - -inf = cannot attend

        序列结构：[x1, y1, x2, y2, ..., x_prompt, y_prompt | x_respond, y_respond, ...]

        Attention规则：
        1. Prompt区域（前n_prompt个点，即前2*n_prompt个position）：
           - 完全双向attention（所有位置互相可见）
        2. Respond区域（后面的点）：
           - Block-Causal Attention：
             * 块间（Inter-block）：第i块只能看到第0~i-1块（因果）
             * 块内（Intra-block）：同一块内所有位置互相可见（双向）
           - 所有respond块都可以看到完整的prompt区域

        Args:
            total_points: 总点数（n_prompt + n_respond）
            n_prompt: prompt点数
            block_size: 块大小（以点为单位，每个点包含x和y两个position）
            device: torch device
            dtype: dtype for the bias

        Returns:
            attention_bias: [1, 1, seq_len, seq_len] 的attention bias
                           0 = can attend, -inf = cannot attend (additive bias)
        """
        seq_len = 2 * total_points  # 每个点包含x和y
        prompt_len = 2 * n_prompt    # prompt部分的序列长度
        respond_len = seq_len - prompt_len  # respond部分的序列长度

        # 初始化bias为-inf（默认不能attend），然后显式设置0（可以attend）
        bias = torch.full((1, 1, seq_len, seq_len), float('-inf'), device=device, dtype=dtype)

        # === Step 1: Prompt区域（完全双向attention）===
        bias[:, :, :prompt_len, :prompt_len] = 0  # Prompt内所有位置互相可见

        # === Step 2: Respond区域（Block-Causal Attention）===
        if respond_len > 0:
            block_size_pos = 2 * block_size  # 每个块包含block_size个点
            num_respond_blocks = (respond_len + block_size_pos - 1) // block_size_pos

            for block_i in range(num_respond_blocks):
                block_start = prompt_len + block_i * block_size_pos
                block_end = min(prompt_len + (block_i + 1) * block_size_pos, seq_len)

                # 规则1：当前块可以看到整个Prompt区域
                bias[:, :, block_start:block_end, :prompt_len] = 0

                # 规则2：当前块可以看到之前的所有Respond块（因果）
                if block_i > 0:
                    prev_respond_end = prompt_len + block_i * block_size_pos
                    bias[:, :, block_start:block_end, prompt_len:prev_respond_end] = 0

                # 规则3：当前块内部
                if block_size == 1:
                    # block_size=1：块内因果（等价于 AR）
                    if block_start + 1 < block_end:
                        bias[:, :, block_start:block_start + 1, block_start:block_start + 1] = 0  # x_i 看自己
                        bias[:, :, block_start + 1:block_end, block_start:block_end] = 0  # y_i 看 x_i 和自己
                else:
                    # block_size>1：块内双向
                    bias[:, :, block_start:block_end, block_start:block_end] = 0

        return bias

    def forward(self, xs, ys, train_mode=True, respond_position_mask=None):
        """
        Forward pass with optional Block-Causal Attention

        与 LLaDAPromptRespond 的唯一区别：在 backbone forward 时传入 attention_bias
        """
        b, total_points, d = xs.shape
        device = xs.device

        # 🔧 使用辅助函数：计算实际 n_respond
        actual_n_respond = compute_actual_n_respond(total_points, self.n_prompt, respond_position_mask)
        if respond_position_mask is not None:
            assert respond_position_mask.shape == (b, total_points), \
                f"respond_position_mask shape mismatch: expected {(b, total_points)}, got {respond_position_mask.shape}"
        assert total_points >= self.n_prompt, \
            f"Expected at least {self.n_prompt} points (prompt), got {total_points}"
        assert actual_n_respond <= self.n_respond, \
            f"Actual respond count {actual_n_respond} exceeds model max {self.n_respond}"

        # 🔧 采样 timestep 或 mask ratio
        t_scalar, t = sample_timestep_with_strategy(
            b, device, train_mode, self.training_strategy,
            self.num_timesteps, self.mask_epsilon, self.train_mask_ratio,
            self.eval_mask_ratio, self.eval_mask_mode
        )

        # 🔧 使用辅助函数：生成 masked_indices
        masked_indices = generate_masked_indices_for_mdm(
            b, total_points, self.n_prompt, actual_n_respond,
            t_scalar, device, self.use_prompt_context, respond_position_mask
        )

        # ===== Step 3: 构建序列 =====
        ys_input = ys.clone().float()
        if ys_input.dim() == 2:
            ys_input = ys_input.unsqueeze(-1)

        ys_wide = torch.cat([ys_input, torch.zeros(b, total_points, d - 1, device=device)], dim=2)
        zs = torch.stack((xs, ys_wide), dim=2).view(b, 2 * total_points, d)

        # ===== Step 4: Embedding + Time conditioning =====
        embeds = self._read_in(zs)
        time_emb = self._time_mlp(t_scalar.view(b, 1, 1).expand(b, 2 * total_points, 1))
        embeds = embeds + time_emb

        # ===== Step 5: 应用 mask =====
        mask_embed = self._read_in(self.mask_embedding.squeeze(0))
        for i in range(b):
            if masked_indices[i].any():
                masked_positions = masked_indices[i].nonzero(as_tuple=False).squeeze(-1)
                full_idx = masked_positions * 2 + 1
                embeds[i, full_idx, :] = mask_embed

        # ===== Step 6: Backbone forward =====
        # 🆕 创建Block-Causal Attention Bias（如果启用）
        if self.use_block_diffusion:
            attention_bias = self._create_block_causal_attention_bias(
                total_points=total_points,
                n_prompt=self.n_prompt,
                block_size=self.block_size,
                device=device,
                dtype=embeds.dtype
            )
            # 扩展到batch size
            attention_bias = attention_bias.expand(b, -1, -1, -1)
        else:
            # 标准Masked Diffusion：使用全局双向attention
            attention_bias = self._backbone.get_bidirectional_attention_bias(
                seq_len=2 * total_points,
                device=device
            )

        dummy_input_ids = torch.zeros(b, 2 * total_points, dtype=torch.long, device=device)
        out = self._backbone(
            input_ids=dummy_input_ids,
            input_embeddings=embeds,
            attention_bias=attention_bias,  # 🆕 传入 block-causal bias 或 bidirectional bias
            output_hidden_states=True,
        )
        h = out.hidden_states[-1]
        pred_y_all = self._read_out(h)[:, 0::2, 0]  # [B, total_points]

        # 🔧 使用辅助函数：提取 respond 部分的预测
        pred_y = extract_respond_predictions(pred_y_all, self.n_prompt, actual_n_respond, respond_position_mask)

        if not train_mode:
            # 推理模式：支持单步或多步推理
            if self.use_multistep_inference:
                return self._multistep_inference(xs, ys, device, respond_position_mask=respond_position_mask)
            else:
                respond_masked_indices = extract_respond_masked_indices(
                    masked_indices, self.n_prompt, actual_n_respond, respond_position_mask
                )
                return pred_y, respond_masked_indices

        # ===== Step 8: Training Loss =====
        target = ys.squeeze(-1) if ys.dim() == 3 else ys
        respond_target = extract_respond_targets(target, self.n_prompt, actual_n_respond, respond_position_mask)
        respond_masked_indices = extract_respond_masked_indices(masked_indices, self.n_prompt, actual_n_respond, respond_position_mask)

        diff = pred_y - respond_target
        mask = respond_masked_indices.float()
        per_sample_loss = (diff.square() * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        # 使用训练策略计算loss权重
        weighted_loss = compute_loss_weight_with_strategy(
            per_sample_loss, t, t_scalar, self.training_strategy, self.num_timesteps
        )

        return weighted_loss, pred_y, t_scalar, respond_masked_indices


# ============================================================
# BOP-AR: Block-Offset Parallel Autoregressive Model
# ============================================================
class BOPARPromptRespond(LLaDAPromptRespond):
    """
    BOP-AR (ScatDiff): Block-Offset Parallel Autoregressive for Prompt-Respond Setting

    Core Concept:
    - Block Diffusion (SDAR): Autoregressive on **block index** (horizontal generation)
      * Finish block 1, then block 2, then block 3, ...
      * Inter-block causal, intra-block bidirectional

    - BOP-AR (ScatDiff): Autoregressive on **offset within block** (vertical generation)
      * Write all blocks' layer 1, then all blocks' layer 2, ...
      * Inter-block parallel, intra-block offset-causal

    Block Size Parameter:
    - `block_size`: Defines the "vertical generation unit" or "cycle depth"
    - Each block conceptually has `2 * block_size` positions
    - Offset = position % (2 * block_size), range: [0, 2*block_size-1]
    - Examples:
      * block_size=1: offset ∈ {0, 1} → 2 layers (x, y)
      * block_size=5: offset ∈ {0, 1, ..., 9} → 10 layers
      * block_size=10: offset ∈ {0, 1, ..., 19} → 20 layers

    Attention Rule (ScatDiff):
    - Query at position i can see Key at position j if:
      1. j is in Prompt region (always visible), OR
      2. (j % (2*block_size)) <= (i % (2*block_size))

    - This creates "vertical parallelism":
      * All positions with offset=0 are generated simultaneously
      * Then all positions with offset=1 (can see offset=0)
      * Then all positions with offset=2 (can see offset=0,1)
      * And so on...

    Key Advantage:
    - When generating layer k, model can see ALL positions from layers 0..k-1 across ALL blocks
    - Enables global information sharing within each layer
    - More flexible than fixed (x, y) structure

    Inherited from LLaDAPromptRespond:
    - All diffusion logic (timestep sampling, masking strategy, loss computation)
    - Only overrides forward() to use ScatDiff attention bias
    """

    def __init__(self, n_dims, n_positions, n_embd=256, n_layer=12, n_head=8,
                 n_prompt=20, n_respond=5, *, mlp_ratio=4.0, block_group_size=1,
                 block_size=1, **kwargs):
        """
        Initialize BOP-AR (ScatDiff) model.

        Args:
            n_dims: Feature dimension
            n_positions: Maximum sequence length
            n_embd: Embedding dimension
            n_layer: Number of transformer layers
            n_head: Number of attention heads
            n_prompt: Number of prompt pairs
            n_respond: Number of respond pairs
            mlp_ratio: MLP hidden dimension ratio
            block_group_size: Block group size (inherited from LLaDA)
            block_size: Vertical generation unit size (controls offset depth)
                       - Each block has 2 * block_size positions
                       - Offset = position % (2 * block_size)
                       - Larger block_size → more layers in vertical generation
            **kwargs: Other parameters (mask_epsilon, loss_weight_type, etc.)
        """
        # Inherit all parent initialization
        super().__init__(
            n_dims=n_dims, n_positions=n_positions, n_embd=n_embd,
            n_layer=n_layer, n_head=n_head, n_prompt=n_prompt,
            n_respond=n_respond, mlp_ratio=mlp_ratio,
            block_group_size=block_group_size, **kwargs
        )

        # Store block_size for ScatDiff attention
        self.block_size = block_size

        # Validate block_size
        assert block_size >= 1, f"block_size must be >= 1, got {block_size}"

    def _create_scatdiff_attention_bias(self, total_points, n_prompt, block_size, device, dtype):
        """
        Create ScatDiff (BOP-AR) attention bias with dynamic block_size.

        Core Logic (ScatDiff):
        - block_size: User-defined generation cycle depth
        - logical_block_len = 2 * block_size (number of positions in one vertical cycle)
        - offset[i] = i % logical_block_len (range: 0 to 2*block_size-1)
        - Attention rule: Query i can see Key j if:
          1. j is in Prompt region (always visible), OR
          2. (j % logical_block_len) <= (i % logical_block_len)

        This creates "layer-wise" vertical generation:
        - Layer 0: All positions with offset=0 generated simultaneously (parallel)
        - Layer 1: All positions with offset=1 (can see layer 0)
        - Layer k: All positions with offset=k (can see layers 0..k-1)

        Example (block_size=1, logical_block_len=2):
        - offset ∈ {0, 1}
        - offset=0: x positions (see all x's)
        - offset=1: y positions (see all x's and y's)

        Example (block_size=5, logical_block_len=10):
        - offset ∈ {0, 1, 2, ..., 9}
        - 10 layers of vertical generation

        Args:
            total_points: n_prompt + n_respond
            n_prompt: number of prompt pairs
            block_size: vertical generation unit size
            device: torch device
            dtype: torch dtype

        Returns:
            attention_bias: [1, 1, seq_len, seq_len] with 0=attend, -inf=mask
        """
        seq_len = 2 * total_points
        prompt_len = 2 * n_prompt
        logical_block_len = 2 * block_size  # Each vertical cycle has 2*block_size positions

        # Initialize to -inf (all masked)
        bias = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device, dtype=dtype)

        # Calculate offset: determines the layer in vertical generation
        offsets = torch.arange(seq_len, device=device) % logical_block_len

        # Mark prompt region
        is_prompt = torch.arange(seq_len, device=device) < prompt_len

        # Vectorized attention mask construction
        off_i = offsets.view(-1, 1)  # Query offsets [seq_len, 1]
        off_j = offsets.view(1, -1)  # Key offsets [1, seq_len]

        # Core ScatDiff logic: Key is visible if in Prompt OR offset_j <= offset_i
        mask = is_prompt.view(1, -1).expand(seq_len, -1) | (off_j <= off_i)

        # Apply mask (0 = attend, -inf = mask)
        bias[0, 0, mask] = 0

        return bias

    def forward(self, xs, ys, train_mode=True, respond_position_mask=None):
        """
        Forward pass with BOP-AR attention.

        All diffusion logic (timestep sampling, masking, embedding) is inherited from parent.
        Only difference: use BOP-AR attention bias instead of bidirectional.

        Args:
            xs: [b, total_points, d] input features
            ys: [b, total_points] target values
            train_mode: whether in training mode
            respond_position_mask: optional mask for respond positions

        Returns:
            loss: weighted MSE loss
            pred_y: predictions for all y positions
            t_scalar: timestep values
            masked_indices: indices of masked positions
        """
        b, total_points, d = xs.shape
        device = xs.device

        # 🔧 计算实际 n_respond (使用全局辅助函数)
        actual_n_respond = compute_actual_n_respond(total_points, self.n_prompt, respond_position_mask)
        if respond_position_mask is not None:
            assert respond_position_mask.shape == (b, total_points), \
                f"respond_position_mask shape mismatch: expected {(b, total_points)}, got {respond_position_mask.shape}"
        assert total_points >= self.n_prompt, \
            f"Expected at least {self.n_prompt} points (prompt), got {total_points}"
        assert actual_n_respond <= self.n_respond, \
            f"Actual respond count {actual_n_respond} exceeds model max {self.n_respond}"

        # 🔧 采样 timestep 或 mask ratio（使用全局函数）
        t_scalar, t = sample_timestep_with_strategy(
            b, device, train_mode, self.training_strategy,
            self.num_timesteps, self.mask_epsilon, self.train_mask_ratio,
            self.eval_mask_ratio, self.eval_mask_mode
        )

        # 🔧 生成 masked_indices（使用全局函数）
        masked_indices = generate_masked_indices_for_mdm(
            b, total_points, self.n_prompt, actual_n_respond,
            t_scalar, device, self.use_prompt_context, respond_position_mask
        )

        # ===== 构建序列 =====
        ys_input = ys.clone().float()
        if ys_input.dim() == 2:
            ys_input = ys_input.unsqueeze(-1)

        ys_wide = torch.cat([ys_input, torch.zeros(b, total_points, d - 1, device=device)], dim=2)
        zs = torch.stack((xs, ys_wide), dim=2).view(b, 2 * total_points, d)

        # ===== Embedding + Time conditioning =====
        embeds = self._read_in(zs)
        time_emb = self._time_mlp(t_scalar.view(b, 1, 1).expand(b, 2 * total_points, 1))
        embeds = embeds + time_emb

        # ===== 应用 mask =====
        mask_embed = self._read_in(self.mask_embedding.squeeze(0))
        for i in range(b):
            if masked_indices[i].any():
                # 找到所有被 mask 的位置（包括 prompt 和 respond）
                masked_positions = masked_indices[i].nonzero(as_tuple=False).squeeze(-1)
                # 转换为序列中的 y 位置索引（y 位置在序列中是奇数索引：1, 3, 5, ...）
                full_idx = masked_positions * 2 + 1
                embeds[i, full_idx, :] = mask_embed

        # 🆕 创建 ScatDiff (BOP-AR) attention bias（唯一区别）
        attention_bias = self._create_scatdiff_attention_bias(
            total_points=total_points,
            n_prompt=self.n_prompt,
            block_size=self.block_size,
            device=device,
            dtype=embeds.dtype
        )

        # ===== Backbone forward =====
        dummy_input_ids = torch.zeros(b, 2 * total_points, dtype=torch.long, device=device)
        out = self._backbone(
            input_ids=dummy_input_ids,
            input_embeddings=embeds,
            attention_bias=attention_bias,  # 🆕 显式传入 BOP-AR attention bias
            output_hidden_states=True,
        )
        h = out.hidden_states[-1]
        pred_y_all = self._read_out(h)[:, 0::2, 0]  # [B, total_points]

        # 🔧 提取 respond 部分的预测（使用全局函数）
        pred_y = extract_respond_predictions(pred_y_all, self.n_prompt, actual_n_respond, respond_position_mask)

        if not train_mode:
            # 推理模式
            respond_masked_indices = extract_respond_masked_indices(
                masked_indices, self.n_prompt, actual_n_respond, respond_position_mask
            )
            return pred_y, respond_masked_indices

        # ===== Training Loss =====
        target = ys.squeeze(-1) if ys.dim() == 3 else ys

        # 🔧 提取 respond target 和 masked_indices（使用全局函数）
        respond_target = extract_respond_targets(target, self.n_prompt, actual_n_respond, respond_position_mask)
        respond_masked_indices = extract_respond_masked_indices(masked_indices, self.n_prompt, actual_n_respond, respond_position_mask)

        diff = pred_y - respond_target

        # 只在 respond 部分的 masked 位置计算 loss
        mask = respond_masked_indices.float()
        per_sample_loss = (diff.square() * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        # 🔧 使用训练策略计算loss权重（使用全局函数）
        weighted_loss = compute_loss_weight_with_strategy(
            per_sample_loss, t, t_scalar, self.training_strategy, self.num_timesteps
        )

        return weighted_loss, pred_y, t_scalar, respond_masked_indices


# ============================================================
# RBO-AR: Random Block-Order Autoregressive Model
# ============================================================
class RBOARPromptRespond(LLaDAPromptRespond):
    """
    RBO-AR: Random Block-Order Autoregressive for Prompt-Respond Setting

    Core Concept:
    - Block Diffusion (LLaDABlock): Autoregressive on **physical block index** (horizontal generation)
      * Finish block 1, then block 2, then block 3, ...
      * Physical order = Logical order (1→2→3)

    - BOP-AR (ScatDiff): Autoregressive on **offset within block** (vertical generation)
      * Write all blocks' layer 1, then all blocks' layer 2, ...
      * Layer-wise generation (Layer 0→1→2)

    - RBO-AR: Autoregressive on **random priority** (random-order generation)
      * Physical block order ≠ Logical generation order
      * Each block assigned random priority, generate in priority order
      * Example: Physical [B0, B1, B2] → Priority [2, 0, 1] → Generate B1→B2→B0

    Key Innovation:
    - Decouples physical sequence position from logical generation order
    - Each block's generation (from MASK to clear values) is a diffusion denoising step
    - Forms a "random-order diffusion chain":
      * When generating block with priority k, all blocks with priority <k are visible
      * These visible blocks serve as conditioning context for current block denoising

    Paradigm Shift:
    - From "predicting future" to "global recognition"
    - Learns permutation-invariant relationships between data points
    - Enables global interpolation capability

    ICL Mathematical Meaning:
    - Learns permutation invariance: f(x_π(1), ..., x_π(n)) = f(x_1, ..., x_n)
    - Global interpolation: Can generate any block given any subset of other blocks

    Attention Rule (RBO-AR):
    - Query at position i can see Key at position j if:
      1. j is in Prompt region (always visible), OR
      2. priority[j] < priority[i] (j's block generated before i's block, inter-block causal), OR
      3. priority[j] == priority[i] AND j <= i (same block, intra-block causal/AR)
    
    Key Innovation: "Random Block Order + Intra-block AR"
    - Inter-block: Random priority order (permutation-invariant learning)
    - Intra-block: Physical position order (autoregressive causality)

    Block Size Parameter:
    - `block_size`: Number of points per block (controls granularity)
    - Each block has 2 * block_size positions (x and y for each point)
    - Smaller block_size → more blocks → more fine-grained random ordering
    - Larger block_size → fewer blocks → coarser random ordering

    Random Order Control:
    - `random_order`: If True, use random priority assignment (default: True)
    - `priority_seed`: Seed for priority assignment (for reproducibility)
      * If None, use different random order each forward pass
      * If set, use fixed random order throughout training

    Inherited from LLaDAPromptRespond:
    - All diffusion logic (timestep sampling, masking strategy, loss computation)
    - Only overrides forward() to use RBO attention bias with random priorities
    """

    def __init__(self, n_dims, n_positions, n_embd=256, n_layer=12, n_head=8,
                 n_prompt=20, n_respond=5, *, mlp_ratio=4.0, block_group_size=1,
                 block_size=4, random_order=True, priority_seed=None, **kwargs):
        """
        Initialize RBO-AR model.

        Args:
            n_dims: Feature dimension
            n_positions: Maximum sequence length
            n_embd: Embedding dimension
            n_layer: Number of transformer layers
            n_head: Number of attention heads
            n_prompt: Number of prompt pairs
            n_respond: Number of respond pairs
            mlp_ratio: MLP hidden dimension ratio
            block_group_size: Block group size (inherited from LLaDA)
            block_size: Number of points per block (controls granularity)
                       - Each block has 2 * block_size positions
                       - Smaller → more blocks → finer random ordering
            random_order: If True, use random priority assignment
            priority_seed: Seed for priority assignment (None = different each time)
            **kwargs: Other parameters (mask_epsilon, loss_weight_type, etc.)
        """
        # Inherit all parent initialization
        super().__init__(
            n_dims=n_dims, n_positions=n_positions, n_embd=n_embd,
            n_layer=n_layer, n_head=n_head, n_prompt=n_prompt,
            n_respond=n_respond, mlp_ratio=mlp_ratio,
            block_group_size=block_group_size, **kwargs
        )

        # Store RBO-AR specific parameters
        self.block_size = block_size
        self.random_order = random_order
        self.priority_seed = priority_seed

        # Validate parameters
        assert block_size >= 1, f"block_size must be >= 1, got {block_size}"

        # Update model name
        self.name = "rbo_ar"

        print(f"[RBO-AR] Random Block-Order Autoregressive initialized:")
        print(f"  - block_size: {self.block_size} points per block")
        print(f"  - random_order: {self.random_order}")
        print(f"  - priority_seed: {self.priority_seed}")
        print(f"  - Prompt: Always visible (priority -1)")
        print(f"  - Respond: Random block ordering based on priority")

    def _create_rbo_attention_bias(self, total_points, n_prompt, block_size, device, dtype, external_priorities=None):
        """
        Create RBO-AR (Random Block-Order) attention bias with priority assignment.

        Core Logic:
        1. Divide Respond region into blocks (each block has block_size points = 2*block_size positions)
        2. Assign priority to each block:
           - If external_priorities is provided, use it directly (for EBO-AR inference)
           - Otherwise, use random priority assignment (for RBO-AR training)
        3. Attention rule: Query i can see Key j if:
           - j is in Prompt region (priority -1, always visible), OR
           - priority[j] < priority[i] (j's block generated before i's block, inter-block causal), OR
           - priority[j] == priority[i] AND j <= i (same block, intra-block causal/AR)

        Key Innovation: "Random Block Order + Intra-block AR"
        - Inter-block: Priority order (random for training, entropy-guided for inference)
          * Physical position ≠ Logical generation order
          * Each forward pass uses different random priority (unless priority_seed is set)
        - Intra-block: Physical position order (autoregressive causality)
          * Within the same block, positions follow causal order (j <= i)
          * Ensures true autoregressive behavior within blocks

        Args:
            total_points: n_prompt + n_respond
            n_prompt: number of prompt pairs
            block_size: number of points per block
            device: torch device
            dtype: torch dtype
            external_priorities: Optional tensor of shape [num_blocks] with block priorities.
                               If None, generates random priorities. If provided, uses them directly.

        Returns:
            attention_bias: [1, 1, seq_len, seq_len] with 0=attend, -inf=mask
        """
        seq_len = 2 * total_points
        prompt_len = 2 * n_prompt
        respond_len = seq_len - prompt_len

        # Initialize to -inf (all masked)
        bias = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device, dtype=dtype)

        # === Step 1: Assign priorities to each position ===
        # pos_priority[i] = priority of position i
        # - Prompt region: priority = -1 (always visible)
        # - Respond region: priority = block_id's random priority
        pos_priority = torch.full((seq_len,), -1, device=device, dtype=torch.long)

        if respond_len > 0:
            # Calculate number of blocks in Respond region
            block_size_pos = 2 * block_size  # Each block has 2*block_size positions
            num_blocks = (respond_len + block_size_pos - 1) // block_size_pos

            # Priority assignment: use external_priorities if provided, otherwise generate
            if external_priorities is not None:
                # Use external priorities (for EBO-AR inference)
                priorities = external_priorities.to(device)
                assert len(priorities) == num_blocks, \
                    f"external_priorities length {len(priorities)} != num_blocks {num_blocks}"
            else:
                # Generate random priorities (for RBO-AR training)
                if self.random_order:
                    if self.priority_seed is not None:
                        # Use fixed seed for reproducibility
                        generator = torch.Generator(device=device).manual_seed(self.priority_seed)
                        priorities = torch.randperm(num_blocks, device=device, generator=generator)
                    else:
                        # Different random order each forward pass
                        priorities = torch.randperm(num_blocks, device=device)
                else:
                    # Sequential order (for debugging/ablation)
                    priorities = torch.arange(num_blocks, device=device)

            # Assign priorities to each position in Respond region
            for block_idx in range(num_blocks):
                block_start = prompt_len + block_idx * block_size_pos
                block_end = min(prompt_len + (block_idx + 1) * block_size_pos, seq_len)
                pos_priority[block_start:block_end] = priorities[block_idx]

        # === Step 2: Construct attention mask ===
        # Query i can see Key j if:
        # - priority[j] == -1 (Prompt), OR
        # - priority[j] < priority[i] (earlier block, inter-block causal), OR
        # - priority[j] == priority[i] AND idx_j <= idx_i (same block, intra-block causal)

        # 1. 基础索引（物理位置）
        idx_i = torch.arange(seq_len, device=device).view(-1, 1)  # Query 物理索引 [seq_len, 1]
        idx_j = torch.arange(seq_len, device=device).view(1, -1)   # Key 物理索引 [1, seq_len]

        # 2. 优先级比较
        priority_i = pos_priority.view(-1, 1)  # Query priorities [seq_len, 1]
        priority_j = pos_priority.view(1, -1)   # Key priorities [1, seq_len]

        # --- 核心因果规则 ---
        # 规则 A: Prompt 区域始终可见 (priority == -1)
        mask_prompt = (priority_j == -1)

        # 规则 B: 块间因果 (逻辑过去)
        # Key 所属块的生成优先级比 Query 所在的块早
        # 即：priority[j] < priority[i] (j 在逻辑上先于 i 生成)
        mask_inter_block = (priority_j < priority_i) & (priority_j != -1)

        # 规则 C: 块内因果 (物理过去)
        # Key 和 Query 在同一个块（priority[j] == priority[i]），
        # 且 Key 的物理位置在 Query 之前或等于（idx_j <= idx_i）
        # 这确保了块内遵循自回归（AR）逻辑
        mask_intra_block = (priority_j == priority_i) & (idx_j <= idx_i)

        # 合并所有可见条件
        mask = mask_prompt | mask_inter_block | mask_intra_block

        # Apply mask (0 = attend, -inf = mask)
        bias[0, 0, mask] = 0

        return bias

    def _compute_block_entropy(self, h, total_points, n_prompt, block_size, masked_indices, device,
                               pred_variance=None, pred_y_all_mean=None, respond_position_mask=None):
        """
        Compute block-level entropy for EBO-AR inference.

        Supports both sequential and non-sequential modes:
        - Sequential (respond_position_mask=None): respond from n_prompt to total_points
        - Non-sequential (respond_position_mask provided): respond at specified positions

        Supports both regression and classification tasks:
        - Regression (single-dim output): Uses prediction variance as uncertainty proxy
        - Classification (multi-dim logits): Uses Shannon Entropy: H = -sum(p * log(p))

        Args:
            h: Hidden states from backbone [B, seq_len, d_model]
            total_points: n_prompt + n_respond
            n_prompt: number of prompt pairs
            block_size: number of points per block
            masked_indices: [B, total_points] boolean tensor marking masked positions
            device: torch device
            respond_position_mask: [B, total_points] optional boolean tensor marking respond positions (non-sequential mode)

        Returns:
            block_entropies: [B, num_blocks] tensor with entropy sum for each block
            generated_block_mask: [B, num_blocks] boolean tensor marking generated blocks
        """
        b, seq_len, _ = h.shape
        prompt_len = 2 * n_prompt
        respond_len = seq_len - prompt_len
        block_size_pos = 2 * block_size
        num_blocks = (respond_len + block_size_pos - 1) // block_size_pos

        # Extract predictions/logits from hidden states
        readout_output = self._read_out(h)  # [B, seq_len, output_dim]
        output_dim = readout_output.shape[-1]

        # Extract Y positions (odd indices: 1, 3, 5, ...)
        y_readout = readout_output[:, 1::2, :]  # [B, total_points, output_dim]

        # ✅ CRITICAL FIX: Handle respond position mapping correctly
        # For sequential mode: respond is contiguous from n_prompt
        # For non-sequential mode: respond is scattered, use respond_position_mask to identify
        if respond_position_mask is None:
            # Sequential mode (standard): respond from n_prompt to total_points
            y_respond = y_readout[:, n_prompt:, :]  # [B, n_respond, output_dim]
            respond_masked = masked_indices[:, n_prompt:]  # [B, n_respond]
            # Build respond index mapping (sequential: 0->n_prompt, 1->n_prompt+1, ...)
            respond_idx_to_global = torch.arange(n_prompt, total_points, device=device)  # [n_respond]
        else:
            # Non-sequential mode: respond is scattered according to respond_position_mask
            # Extract indices where respond_position_mask is True (using first sample as reference)
            # In non-sequential, all samples have same respond positions (same structure)
            respond_indices = respond_position_mask[0].nonzero(as_tuple=False).squeeze(-1)  # [n_respond]
            y_respond = y_readout[:, respond_indices, :]  # [B, n_respond, output_dim]
            respond_masked = masked_indices[:, respond_indices]  # [B, n_respond]
            respond_idx_to_global = respond_indices  # [n_respond]

        # === Compute uncertainty/entropy based on output type ===
        if output_dim == 1:
            # === Regression Task: Use variance as uncertainty proxy ===
            if pred_variance is not None and pred_y_all_mean is not None:
                # Use variance from multiple forward passes (more accurate)
                if respond_position_mask is None:
                    pred_y_respond = pred_y_all_mean[:, n_prompt:]  # [B, n_respond]
                    variance_respond = pred_variance[:, n_prompt:]  # [B, n_respond]
                else:
                    pred_y_respond = pred_y_all_mean[:, respond_indices]  # [B, n_respond]
                    variance_respond = pred_variance[:, respond_indices]  # [B, n_respond]
                uncertainty = variance_respond  # Use variance directly
            else:
                # Fallback: use absolute value as simple proxy
                pred_y_respond = y_respond.squeeze(-1)  # [B, n_respond]
                uncertainty = torch.abs(pred_y_respond)  # [B, n_respond]
        else:
            # === Classification Task: Use Shannon Entropy (Numerically Stable) ===
            logits = y_respond  # [B, n_respond, vocab_size]

            # Use log_softmax for numerical stability
            log_probs = torch.log_softmax(logits, dim=-1)  # [B, n_respond, vocab_size]
            probs = torch.exp(log_probs)  # [B, n_respond, vocab_size]

            # Compute Shannon Entropy: H = -sum(p * log(p))
            token_entropy = -torch.sum(probs * log_probs, dim=-1)  # [B, n_respond]
            uncertainty = token_entropy  # [B, n_respond]

        # Aggregate uncertainty to block level
        block_entropies = torch.zeros(b, num_blocks, device=device)
        generated_block_mask = torch.zeros(b, num_blocks, dtype=torch.bool, device=device)

        for block_idx in range(num_blocks):
            # Get positions in this block (in respond coordinate system)
            block_start = block_idx * block_size
            block_end = min((block_idx + 1) * block_size, y_respond.shape[1])

            if block_start >= y_respond.shape[1]:
                # Block is beyond respond region
                block_entropies[:, block_idx] = float('inf')
                generated_block_mask[:, block_idx] = True
                continue

            # Get masked positions in this block
            block_masked = respond_masked[:, block_start:block_end]  # [B, block_size]

            # Check if block is fully generated (no masked positions)
            is_generated = ~block_masked.any(dim=1)  # [B]
            generated_block_mask[:, block_idx] = is_generated

            # Sum uncertainty for masked positions in this block
            block_uncertainty = (uncertainty[:, block_start:block_end] * block_masked.float()).sum(dim=1)  # [B]

            # If block is generated, set entropy to a large value (so it won't be selected)
            block_entropies[:, block_idx] = torch.where(
                is_generated,
                torch.full_like(block_uncertainty, float('inf')),
                block_uncertainty
            )

        return block_entropies, generated_block_mask

    @torch.no_grad()
    def generate_ebo(self, xs, ys, device, respond_position_mask=None, num_probe_samples=3):
        """
        EBO-AR (Entropy-based Block Order) generation: adaptive inference loop.

        Supports both sequential and non-sequential modes:
        - Sequential (respond_position_mask=None): respond from n_prompt to total_points
        - Non-sequential (respond_position_mask provided): respond at specified positions

        Core Logic:
        1. Initialize: All respond positions are masked
        2. Iterate (num_blocks steps):
           a. Probing: Forward pass with current priority assignment (optionally multiple times with noise)
           b. Selection: Compute block entropies, select block with minimum entropy
           c. Fixing: Assign priority to selected block
           d. Denoising: Update predictions for selected block

        Args:
            xs: [B, total_points, D] input features
            ys: [B, total_points] target values (used for initialization, respond part will be masked)
            device: torch device
            respond_position_mask: [B, total_points] optional boolean tensor marking respond positions (non-sequential mode)
            num_probe_samples: Number of forward passes with noise for regression variance estimation (default: 3)

        Returns:
            pred_y: [B, actual_n_respond] final predictions for respond region
            confirmed_priorities: List of confirmed priority assignments (for debugging)
        """
        b, total_points, d = xs.shape
        
        # Compute actual n_respond
        actual_n_respond = compute_actual_n_respond(total_points, self.n_prompt, respond_position_mask)
        assert total_points >= self.n_prompt, \
            f"Expected at least {self.n_prompt} points (prompt), got {total_points}"
        assert actual_n_respond <= self.n_respond, \
            f"Actual respond count {actual_n_respond} exceeds model max {self.n_respond}"
        
        # Calculate number of blocks
        block_size_pos = 2 * self.block_size
        respond_len = 2 * actual_n_respond
        num_blocks = (respond_len + block_size_pos - 1) // block_size_pos
        
        # Initialize: all respond positions are masked
        # ✅ Supports both sequential and non-sequential modes
        ys_masked = ys.clone()
        if respond_position_mask is None:
            # Sequential mode: mask all respond positions from n_prompt onward
            ys_masked[:, self.n_prompt:] = 0.0  # Use 0 as placeholder
        else:
            # Non-sequential mode: mask only respond positions marked in respond_position_mask
            ys_masked = torch.where(respond_position_mask.unsqueeze(-1).expand_as(ys_masked),
                                   torch.zeros_like(ys_masked), ys_masked)
        
        # Track confirmed priorities for each block (Batch-independent)
        # Shape: [B, num_blocks], value = priority step (0, 1, 2, ...) or 999 if not generated
        confirmed_priorities = torch.full((b, num_blocks), 999, device=device, dtype=torch.long)
        current_priority_steps = torch.zeros(b, device=device, dtype=torch.long)  # [B]
        
        # === Pre-compute block position indices for efficiency ===
        # Pre-generate block start/end indices to avoid repeated computation
        block_start_indices = torch.arange(num_blocks, device=device) * self.block_size + self.n_prompt
        block_end_indices = torch.minimum(
            block_start_indices + self.block_size,
            torch.full((num_blocks,), total_points, device=device)
        )
        # Create a mask matrix for block positions: [num_blocks, total_points]
        block_position_mask = torch.zeros(num_blocks, total_points, dtype=torch.bool, device=device)
        for block_idx in range(num_blocks):
            start = block_start_indices[block_idx].item()
            end = block_end_indices[block_idx].item()
            if start < total_points:
                block_position_mask[block_idx, start:end] = True
        
        # Pre-compute respond position indices for masking
        # ✅ Supports both sequential and non-sequential modes
        if respond_position_mask is None:
            # Sequential mode: respond positions are at the end
            respond_positions_base = torch.arange(self.n_prompt, total_points, device=device)
            respond_full_indices_base = respond_positions_base * 2 + 1  # Y positions in sequence
        else:
            # Non-sequential mode: will be computed per sample
            respond_positions_base = None
            respond_full_indices_base = None
        
        # Iterate: select and generate one block at a time
        for step in range(num_blocks):
            # === Step 1: Probing - Construct temporary priorities ===
            # For generated blocks, use confirmed priority
            # For ungenerated blocks, use 999 (large value, invisible to current blocks)
            temp_priorities = confirmed_priorities.clone()  # [B, num_blocks]
            
            # === Step 2: Forward pass with temporary priorities ===
            # Build sequence
            ys_input = ys_masked.clone().float()
            if ys_input.dim() == 2:
                ys_input = ys_input.unsqueeze(-1)
            
            ys_wide = torch.cat([ys_input, torch.zeros(b, total_points, d - 1, device=device)], dim=2)
            zs = torch.stack((xs, ys_wide), dim=2).view(b, 2 * total_points, d)
            
            # Embedding + Time conditioning (use t=1.0 for full mask)
            embeds = self._read_in(zs)
            t_scalar = torch.ones(b, device=device)  # Full mask for probing
            time_emb = self._time_mlp(t_scalar.view(b, 1, 1).expand(b, 2 * total_points, 1))
            embeds = embeds + time_emb
            
            # Apply mask to respond region
            # ✅ Supports both sequential and non-sequential modes
            mask_embed = self._read_in(self.mask_embedding.squeeze(0))
            if respond_position_mask is None:
                # Sequential mode: use pre-computed indices
                embeds[:, respond_full_indices_base, :] = mask_embed
            else:
                # Non-sequential mode: compute per sample
                for i in range(b):
                    respond_positions = respond_position_mask[i].nonzero(as_tuple=False).squeeze(-1)
                    respond_positions = respond_positions[respond_positions >= self.n_prompt]
                    if len(respond_positions) > 0:
                        full_idx = respond_positions * 2 + 1
                        embeds[i, full_idx, :] = mask_embed
            
            # Create attention bias with temporary priorities
            # Since each sample may have different priorities, we need to create bias for each sample
            # Pre-compute unique priority patterns to avoid redundant computation
            unique_priorities = {}
            attention_biases = []
            for i in range(b):
                sample_priorities = temp_priorities[i]  # [num_blocks]
                # Create a hash key for the priority pattern
                priorities_key = tuple(sample_priorities.cpu().tolist())
                if priorities_key not in unique_priorities:
                    # Compute bias for this unique priority pattern
                    sample_bias = self._create_rbo_attention_bias(
                        total_points=total_points,
                        n_prompt=self.n_prompt,
                        block_size=self.block_size,
                        device=device,
                        dtype=embeds.dtype,
                        external_priorities=sample_priorities
                    )
                    unique_priorities[priorities_key] = sample_bias
                attention_biases.append(unique_priorities[priorities_key])
            attention_bias = torch.cat(attention_biases, dim=0)  # [B, 1, seq_len, seq_len]
            
            # === Step 2: Forward pass with multiple samples for regression variance ===
            # First, do a single forward pass to check output dimension
            dummy_input_ids = torch.zeros(b, 2 * total_points, dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                attention_bias=attention_bias,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            readout_output = self._read_out(h)  # [B, seq_len, output_dim]
            output_dim = readout_output.shape[-1]
            
            if output_dim == 1 and num_probe_samples > 1:
                # Regression task: run multiple forward passes with noise to compute variance
                pred_samples = []
                # First sample: no noise (deterministic)
                pred_first = readout_output[:, 0::2, 0]  # [B, total_points]
                pred_samples.append(pred_first)
                
                # Additional samples: with small noise
                for sample_idx in range(1, num_probe_samples):
                    # Add small noise to embeddings for variance estimation
                    noise_scale = 0.01  # Small noise scale
                    embeds_noisy = embeds + torch.randn_like(embeds) * noise_scale
                    
                    out_noisy = self._backbone(
                        input_ids=dummy_input_ids,
                        input_embeddings=embeds_noisy,
                        attention_bias=attention_bias,
                        output_hidden_states=True,
                    )
                    h_noisy = out_noisy.hidden_states[-1]
                    pred_noisy = self._read_out(h_noisy)[:, 0::2, 0]  # [B, total_points]
                    pred_samples.append(pred_noisy)
                
                # Stack predictions: [num_probe_samples, B, total_points]
                pred_samples = torch.stack(pred_samples, dim=0)
                # Compute variance across samples: [B, total_points]
                pred_variance = pred_samples.var(dim=0)
                pred_y_all_mean = pred_samples.mean(dim=0)  # [B, total_points]
            else:
                # Single forward pass (classification or single-sample regression)
                pred_y_all_mean = None
                pred_variance = None
            
            # === Step 3: Selection - Compute block entropies ===
            # Create masked_indices for respond region (track which positions are still masked)
            # Initially all respond positions are masked, but we update this as blocks are generated
            if step == 0:
                # First step: all respond positions are masked
                masked_indices = torch.zeros(b, total_points, dtype=torch.bool, device=device)
                if respond_position_mask is None:
                    masked_indices[:, self.n_prompt:] = True
                else:
                    masked_indices = respond_position_mask.clone()
            else:
                # Subsequent steps: update masked_indices based on generated blocks
                # Positions in generated blocks are no longer masked
                masked_indices = torch.zeros(b, total_points, dtype=torch.bool, device=device)
                if respond_position_mask is None:
                    masked_indices[:, self.n_prompt:] = True
                else:
                    masked_indices = respond_position_mask.clone()

                # Unmask positions in already generated blocks (batch-independent)
                # Use pre-computed block position mask for efficiency
                for i in range(b):
                    for block_idx in range(num_blocks):
                        if confirmed_priorities[i, block_idx] < 999:  # Block is generated
                            masked_indices[i] = masked_indices[i] & ~block_position_mask[block_idx]

            # Compute block entropies (with variance for regression if available)
            # ✅ Pass respond_position_mask for correct position mapping
            block_entropies, generated_block_mask = self._compute_block_entropy(
                h, total_points, self.n_prompt, self.block_size, masked_indices, device,
                pred_variance=pred_variance, pred_y_all_mean=pred_y_all_mean,
                respond_position_mask=respond_position_mask
            )
            
            # Select block with minimum entropy (excluding already generated blocks)
            # For each sample in batch, select independently
            best_block_indices = torch.argmin(block_entropies, dim=1)  # [B]
            
            # === Step 4: Fixing - Assign priority to selected block (Batch-independent) ===
            for i in range(b):
                best_idx = best_block_indices[i].item()
                if confirmed_priorities[i, best_idx] == 999:  # Block not yet generated
                    confirmed_priorities[i, best_idx] = current_priority_steps[i].item()
                    current_priority_steps[i] += 1
            
            # === Step 5: Denoising - Update predictions for selected block ===
            # Extract predictions: use mean prediction if available (from multiple samples), otherwise single pass
            if pred_y_all_mean is not None:
                # Use mean from multiple samples (regression with variance estimation)
                pred_y_all = pred_y_all_mean  # [B, total_points]
            else:
                # Single forward pass: extract from readout_output
                if output_dim == 1:
                    # Regression: extract scalar predictions
                    pred_y_all = readout_output[:, 0::2, 0]  # [B, total_points]
                else:
                    # Classification: use argmax or sample from logits
                    logits = readout_output[:, 0::2, :]  # [B, total_points, vocab_size]
                    pred_y_all = torch.argmax(logits, dim=-1).float()  # [B, total_points]
            
            # Update ys_masked for selected blocks
            for i in range(b):
                best_idx = best_block_indices[i].item()
                block_start = self.n_prompt + best_idx * self.block_size
                block_end = min(self.n_prompt + (best_idx + 1) * self.block_size, total_points)
                
                # Extract predictions for this block
                block_pred = pred_y_all[i, block_start:block_end]  # [block_size]
                
                # Update ys_masked (only for positions in this block)
                if block_start < total_points:
                    actual_end = min(block_end, total_points)
                    ys_masked[i, block_start:actual_end] = block_pred[:actual_end-block_start].unsqueeze(-1)
        
        # Final forward pass to get predictions for all blocks
        # Build final sequence with all generated blocks
        ys_input = ys_masked.clone().float()
        if ys_input.dim() == 2:
            ys_input = ys_input.unsqueeze(-1)
        
        ys_wide = torch.cat([ys_input, torch.zeros(b, total_points, d - 1, device=device)], dim=2)
        zs = torch.stack((xs, ys_wide), dim=2).view(b, 2 * total_points, d)
        
        # Embedding + Time conditioning (use t=0.0 for no mask, all revealed)
        embeds = self._read_in(zs)
        t_scalar = torch.zeros(b, device=device)  # No mask for final pass
        time_emb = self._time_mlp(t_scalar.view(b, 1, 1).expand(b, 2 * total_points, 1))
        embeds = embeds + time_emb
        
        # Create attention bias with final confirmed priorities (Batch-independent)
        attention_biases = []
        for i in range(b):
            sample_priorities = confirmed_priorities[i]  # [num_blocks]
            sample_bias = self._create_rbo_attention_bias(
                total_points=total_points,
                n_prompt=self.n_prompt,
                block_size=self.block_size,
                device=device,
                dtype=embeds.dtype,
                external_priorities=sample_priorities
            )
            attention_biases.append(sample_bias)
        attention_bias = torch.cat(attention_biases, dim=0)  # [B, 1, seq_len, seq_len]
        
        # Final forward pass
        dummy_input_ids = torch.zeros(b, 2 * total_points, dtype=torch.long, device=device)
        out = self._backbone(
            input_ids=dummy_input_ids,
            input_embeddings=embeds,
            attention_bias=attention_bias,
            output_hidden_states=True,
        )
        h = out.hidden_states[-1]
        # Extract final predictions: handle both regression and classification
        readout_output = self._read_out(h)  # [B, seq_len, output_dim]
        if readout_output.shape[-1] == 1:
            # Regression: extract scalar predictions
            pred_y_all = readout_output[:, 0::2, 0]  # [B, total_points]
        else:
            # Classification: use argmax or sample from logits
            logits = readout_output[:, 0::2, :]  # [B, total_points, vocab_size]
            pred_y_all = torch.argmax(logits, dim=-1).float()  # [B, total_points]
        
        # Extract final predictions for respond region
        pred_y = extract_respond_predictions(
            pred_y_all, self.n_prompt, actual_n_respond, respond_position_mask
        )
        
        # Convert confirmed_priorities to list format for backward compatibility
        confirmed_priorities_list = [
            [confirmed_priorities[i, j].item() if confirmed_priorities[i, j] < 999 else None
             for j in range(num_blocks)]
            for i in range(b)
        ]
        
        return pred_y, confirmed_priorities_list

    @torch.no_grad()
    def generate_bpd(self, xs, ys, device, respond_position_mask=None, K=2,
                     lambda_penalty=0.0, entropy_tau=1e5, num_probe_samples=3):
        """
        Block-wise Parallel Diffusion (BPD) 推理：分阶段并行去噪 + WeDLM优化技巧。

        核心思想：
        - 逻辑分阶段，阶段内并行
        - 每一步选取K个熵最小的块，同时生成它们
        - 这K个块被赋予相同的priority值，使其组内互不可见
        - 但能看到所有前面阶段的块

        WeDLM优化技巧（默认关闭，不影响原有实验）：
        - Distance Penalty (lambda_penalty): 给左侧块更高优先级，利用"确定性锚点"
        - Confidence Filtering (entropy_tau): 只生成置信度高的块，自适应调整并行度

        与EBO-AR的关系：
        - EBO-AR: K=1，严格串行
        - BPD: K>1，同一阶段并行
        - 全并行: K=num_blocks，一步完成

        Args:
            xs: [B, total_points, D] input features
            ys: [B, total_points] target values (respond part will be masked)
            device: torch device
            respond_position_mask: optional boolean tensor marking respond positions (sequential mode only)
            K: 并行度，每步同时生成K个块
            lambda_penalty: WeDLM Trick 1 - 距离惩罚系数（建议0.01~0.1，默认0.0关闭）
            entropy_tau: WeDLM Trick 2 - 熵阈值（建议0.1~0.3，默认1e5关闭）
            num_probe_samples: 回归方差估计采样数

        Returns:
            pred_y: [B, actual_n_respond] 最终预测
            confirmed_priorities: List of priority assignments for each block
        """
        b, total_points, d = xs.shape

        # Compute actual n_respond
        actual_n_respond = compute_actual_n_respond(total_points, self.n_prompt, respond_position_mask)
        assert total_points >= self.n_prompt, \
            f"Expected at least {self.n_prompt} points (prompt), got {total_points}"
        assert actual_n_respond <= self.n_respond, \
            f"Actual respond count {actual_n_respond} exceeds model max {self.n_respond}"

        # 计算块参数
        block_size_pos = 2 * self.block_size
        respond_len = 2 * actual_n_respond
        num_blocks = (respond_len + block_size_pos - 1) // block_size_pos

        # === 初始化 ===
        # 所有 respond 部分 MASK 掉
        ys_masked = ys.clone()
        ys_masked[:, self.n_prompt:] = 0.0

        # 优先级矩阵：999 表示未生成
        confirmed_priorities = torch.full((b, num_blocks), 999, device=device, dtype=torch.long)
        current_step = 0  # 逻辑步数（对应 Priority Step）

        # 预计算块位置
        block_start_indices = torch.arange(num_blocks, device=device) * self.block_size + self.n_prompt
        block_end_indices = torch.minimum(
            block_start_indices + self.block_size,
            torch.full((num_blocks,), total_points, device=device)
        )

        # 🆕 WeDLM Trick 1: 预计算块索引用于距离惩罚
        # block_indices: [num_blocks] 从0到num_blocks-1
        # 用于给左侧块更高优先级（adjusted_entropy = entropy + lambda_penalty * block_idx）
        block_indices = torch.arange(num_blocks, device=device, dtype=torch.float32)

        # === 并行生成主循环 ===
        while (confirmed_priorities == 999).any():
            # === Step 1: Probing - 前向传播获取隐藏状态 ===
            # 构建输入序列
            ys_input = ys_masked.clone().float()
            if ys_input.dim() == 2:
                ys_input = ys_input.unsqueeze(-1)

            ys_wide = torch.cat([ys_input, torch.zeros(b, total_points, d - 1, device=device)], dim=2)
            zs = torch.stack((xs, ys_wide), dim=2).view(b, 2 * total_points, d)

            # Embedding + Time conditioning (use t=1.0 for full mask)
            embeds = self._read_in(zs)
            t_scalar = torch.ones(b, device=device)  # Full mask for probing
            time_emb = self._time_mlp(t_scalar.view(b, 1, 1).expand(b, 2 * total_points, 1))
            embeds = embeds + time_emb

            # Apply mask to respond region
            mask_embed = self._read_in(self.mask_embedding.squeeze(0))
            respond_positions_base = torch.arange(self.n_prompt, total_points, device=device)
            respond_full_indices_base = respond_positions_base * 2 + 1  # Y positions in sequence
            embeds[:, respond_full_indices_base, :] = mask_embed

            # 创建 attention bias（支持 Batch 内不同优先级）
            attention_biases = []
            for i in range(b):
                sample_bias = self._create_rbo_attention_bias(
                    total_points=total_points,
                    n_prompt=self.n_prompt,
                    block_size=self.block_size,
                    device=device,
                    dtype=embeds.dtype,
                    external_priorities=confirmed_priorities[i]
                )
                attention_biases.append(sample_bias)
            attention_bias = torch.cat(attention_biases, dim=0)  # [B, 1, seq_len, seq_len]

            # === Step 2: Forward pass with multiple samples for regression variance ===
            dummy_input_ids = torch.zeros(b, 2 * total_points, dtype=torch.long, device=device)
            out = self._backbone(
                input_ids=dummy_input_ids,
                input_embeddings=embeds,
                attention_bias=attention_bias,
                output_hidden_states=True,
            )
            h = out.hidden_states[-1]
            readout_output = self._read_out(h)  # [B, seq_len, output_dim]
            output_dim = readout_output.shape[-1]

            if output_dim == 1 and num_probe_samples > 1:
                # Regression task: run multiple forward passes with noise to compute variance
                pred_samples = []
                # First sample: no noise (deterministic)
                pred_first = readout_output[:, 0::2, 0]  # [B, total_points]
                pred_samples.append(pred_first)

                # Additional samples: with small noise
                for sample_idx in range(1, num_probe_samples):
                    # Add small noise to embeddings for variance estimation
                    noise_scale = 0.01  # Small noise scale
                    embeds_noisy = embeds + torch.randn_like(embeds) * noise_scale

                    out_noisy = self._backbone(
                        input_ids=dummy_input_ids,
                        input_embeddings=embeds_noisy,
                        attention_bias=attention_bias,
                        output_hidden_states=True,
                    )
                    h_noisy = out_noisy.hidden_states[-1]
                    pred_noisy = self._read_out(h_noisy)[:, 0::2, 0]  # [B, total_points]
                    pred_samples.append(pred_noisy)

                # Stack predictions: [num_probe_samples, B, total_points]
                pred_samples = torch.stack(pred_samples, dim=0)
                # Compute variance across samples: [B, total_points]
                pred_variance = pred_samples.var(dim=0)
                pred_y_all_mean = pred_samples.mean(dim=0)  # [B, total_points]
            else:
                # Single forward pass (classification or single-sample regression)
                pred_y_all_mean = None
                pred_variance = None

            # === Step 3: Selection - 计算熵并选取Top-K ===
            # 创建 masked_indices 用于标记哪些位置还是空的
            masked_indices = torch.zeros(b, total_points, dtype=torch.bool, device=device)
            masked_indices[:, self.n_prompt:] = True
            # 对已生成的块进行 unmask
            for i in range(b):
                for block_idx in range(num_blocks):
                    if confirmed_priorities[i, block_idx] < 999:  # Block is generated
                        # Mark this block as unmasked
                        for pos_idx in range(self.block_size):
                            point_idx = self.n_prompt + block_idx * self.block_size + pos_idx
                            if point_idx < total_points:
                                masked_indices[i, point_idx] = False

            # 计算块熵
            block_entropies, generated_block_mask = self._compute_block_entropy(
                h, total_points, self.n_prompt, self.block_size, masked_indices, device,
                pred_variance=pred_variance, pred_y_all_mean=pred_y_all_mean,
                respond_position_mask=respond_position_mask
            )

            # === Step 4: Top-K Selection with WeDLM Tricks ===
            # 🆕 WeDLM Trick 1: Distance Penalty - 计算调整后的熵
            # adjusted_entropy = original_entropy + lambda_penalty * block_index
            # 这使得左侧块（小索引）获得更低的调整熵，从而更优先被选择
            adjusted_entropies = block_entropies.clone()  # [B, num_blocks]
            if lambda_penalty > 0:
                # 广播 block_indices 到 [B, num_blocks]
                adjusted_entropies = block_entropies + lambda_penalty * block_indices.unsqueeze(0)

            # 确定本轮实际并行数（处理剩余不足K的情况）
            remaining_counts = (confirmed_priorities == 999).sum(dim=1)  # [B]
            actual_k_per_sample = torch.minimum(
                torch.full_like(remaining_counts, K),
                remaining_counts
            )  # [B] 每个sample的实际K值

            # Mask掉已生成的块（设为极大值）
            adjusted_entropies_masked = adjusted_entropies.clone()
            block_entropies_masked = block_entropies.clone()
            for i in range(b):
                for block_idx in range(num_blocks):
                    if confirmed_priorities[i, block_idx] < 999:  # Already generated
                        adjusted_entropies_masked[i, block_idx] = float('inf')
                        block_entropies_masked[i, block_idx] = float('inf')

            # 对每个sample独立选取topk（基于adjusted_entropies）
            best_k_indices_list = []
            for i in range(b):
                actual_k = actual_k_per_sample[i].item()
                if actual_k > 0:
                    # 使用 adjusted_entropies 选择 Top-K 候选块
                    _, topk_indices = torch.topk(
                        adjusted_entropies_masked[i], k=actual_k, largest=False
                    )

                    # 🆕 WeDLM Trick 2: Confidence Filtering - 只保留置信度高的块
                    if entropy_tau < 1e5:  # 如果启用了置信度过滤
                        # 检查哪些候选块的原始熵低于阈值
                        is_confident = block_entropies[i, topk_indices] < entropy_tau

                        # 找到调整熵最小的块（保底：至少生成1个块）
                        best_idx = torch.argmin(adjusted_entropies_masked[i])
                        is_best = (topk_indices == best_idx)

                        # 最终选择：置信的块 OR 最佳块（保底）
                        to_generate = is_confident | is_best
                        topk_indices = topk_indices[to_generate]

                    best_k_indices_list.append(topk_indices)
                else:
                    best_k_indices_list.append(torch.tensor([], device=device, dtype=torch.long))

            # === Step 5: Parallel Collapse & Priority Sync ===
            # 获取当前预测值
            if pred_y_all_mean is not None:
                # Use mean from multiple samples (regression with variance estimation)
                pred_y_all = pred_y_all_mean  # [B, total_points]
            else:
                # Single forward pass: extract from readout_output
                if output_dim == 1:
                    # Regression: extract scalar predictions
                    pred_y_all = readout_output[:, 0::2, 0]  # [B, total_points]
                else:
                    # Classification: use argmax
                    logits = readout_output[:, 0::2, :]  # [B, total_points, vocab_size]
                    pred_y_all = torch.argmax(logits, dim=-1).float()  # [B, total_points]

            # 这K个块使用相同的current_step（关键：使其组内互不可见）
            for i in range(b):
                for b_idx in best_k_indices_list[i]:
                    idx = b_idx.item()
                    # 填充预测值
                    block_start = self.n_prompt + idx * self.block_size
                    block_end = min(self.n_prompt + (idx + 1) * self.block_size, total_points)

                    if block_start < total_points:
                        actual_end = min(block_end, total_points)
                        block_pred = pred_y_all[i, block_start:actual_end]
                        ys_masked[i, block_start:actual_end] = block_pred.unsqueeze(-1)

                    # 💡 关键：这K个块赋予相同的current_step，使其组内优先级相同
                    confirmed_priorities[i, idx] = current_step

            current_step += 1  # 只有完成这一批K个块，步数才增加

        # === Final Forward Pass ===
        # 构建最终序列（所有块已生成）
        ys_input = ys_masked.clone().float()
        if ys_input.dim() == 2:
            ys_input = ys_input.unsqueeze(-1)

        ys_wide = torch.cat([ys_input, torch.zeros(b, total_points, d - 1, device=device)], dim=2)
        zs = torch.stack((xs, ys_wide), dim=2).view(b, 2 * total_points, d)

        # Embedding + Time conditioning (use t=0.0 for no mask, all revealed)
        embeds = self._read_in(zs)
        t_scalar = torch.zeros(b, device=device)  # No mask for final pass
        time_emb = self._time_mlp(t_scalar.view(b, 1, 1).expand(b, 2 * total_points, 1))
        embeds = embeds + time_emb

        # 创建最终 attention bias（所有块已有优先级分配）
        attention_biases = []
        for i in range(b):
            sample_bias = self._create_rbo_attention_bias(
                total_points=total_points,
                n_prompt=self.n_prompt,
                block_size=self.block_size,
                device=device,
                dtype=embeds.dtype,
                external_priorities=confirmed_priorities[i]
            )
            attention_biases.append(sample_bias)
        attention_bias = torch.cat(attention_biases, dim=0)  # [B, 1, seq_len, seq_len]

        # 最终前向传播
        dummy_input_ids = torch.zeros(b, 2 * total_points, dtype=torch.long, device=device)
        out = self._backbone(
            input_ids=dummy_input_ids,
            input_embeddings=embeds,
            attention_bias=attention_bias,
            output_hidden_states=True,
        )
        h = out.hidden_states[-1]
        # 提取最终预测
        readout_output = self._read_out(h)  # [B, seq_len, output_dim]
        if readout_output.shape[-1] == 1:
            # Regression: extract scalar predictions
            pred_y_all = readout_output[:, 0::2, 0]  # [B, total_points]
        else:
            # Classification: use argmax
            logits = readout_output[:, 0::2, :]  # [B, total_points, vocab_size]
            pred_y_all = torch.argmax(logits, dim=-1).float()  # [B, total_points]

        # 提取 respond 部分的预测
        pred_y = extract_respond_predictions(
            pred_y_all, self.n_prompt, actual_n_respond, respond_position_mask
        )

        # 转换优先级为列表格式（向后兼容）
        confirmed_priorities_list = [
            [confirmed_priorities[i, j].item() if confirmed_priorities[i, j] < 999 else None
             for j in range(num_blocks)]
            for i in range(b)
        ]

        return pred_y, confirmed_priorities_list

    def forward(self, xs, ys, train_mode=True, respond_position_mask=None,
                use_ebo_inference=False, use_bpd=False, bpd_k=2,
                lambda_penalty=0.0, entropy_tau=1e5):
        """
        Forward pass with RBO-AR attention (training) or advanced inference modes.

        Training: Uses random block priorities (RBO-AR)
        Inference: Supports multiple strategies:
          - RBO-AR (default): Random block order
          - EBO-AR (use_ebo_inference=True): Entropy-based serial generation (K=1)
          - BPD (use_bpd=True): Block-wise Parallel Diffusion (K>1)

        Args:
            xs: [b, total_points, d] input features
            ys: [b, total_points] target values
            train_mode: whether in training mode
            respond_position_mask: optional mask for respond positions
            use_ebo_inference: if True, use EBO-AR entropy-guided serial inference
            use_bpd: if True, use BPD parallel inference (requires K specification)
            bpd_k: parallelism degree for BPD (default: 2)
            lambda_penalty: WeDLM distance penalty for BPD (default: 0.0, disabled)
            entropy_tau: WeDLM confidence threshold for BPD (default: 1e5, disabled)

        Returns:
            If train_mode:
                loss: weighted MSE loss
                pred_y: predictions for respond positions
                t_scalar: timestep values
                respond_masked_indices: indices of masked positions in respond region
            Else:
                pred_y: predictions for respond positions
                respond_masked_indices: indices of masked positions in respond region
        """
        b, total_points, d = xs.shape
        device = xs.device

        # 🔧 计算实际 n_respond (使用全局辅助函数)
        actual_n_respond = compute_actual_n_respond(total_points, self.n_prompt, respond_position_mask)
        if respond_position_mask is not None:
            assert respond_position_mask.shape == (b, total_points), \
                f"respond_position_mask shape mismatch: expected {(b, total_points)}, got {respond_position_mask.shape}"
        assert total_points >= self.n_prompt, \
            f"Expected at least {self.n_prompt} points (prompt), got {total_points}"
        assert actual_n_respond <= self.n_respond, \
            f"Actual respond count {actual_n_respond} exceeds model max {self.n_respond}"

        # 🔧 采样 timestep 或 mask ratio（使用全局函数）
        t_scalar, t = sample_timestep_with_strategy(
            b, device, train_mode, self.training_strategy,
            self.num_timesteps, self.mask_epsilon, self.train_mask_ratio,
            self.eval_mask_ratio, self.eval_mask_mode
        )

        # 🔧 生成 masked_indices（使用全局函数）
        masked_indices = generate_masked_indices_for_mdm(
            b, total_points, self.n_prompt, actual_n_respond,
            t_scalar, device, self.use_prompt_context, respond_position_mask
        )

        # ===== 构建序列 =====
        ys_input = ys.clone().float()
        if ys_input.dim() == 2:
            ys_input = ys_input.unsqueeze(-1)

        ys_wide = torch.cat([ys_input, torch.zeros(b, total_points, d - 1, device=device)], dim=2)
        zs = torch.stack((xs, ys_wide), dim=2).view(b, 2 * total_points, d)

        # ===== Embedding + Time conditioning =====
        embeds = self._read_in(zs)
        time_emb = self._time_mlp(t_scalar.view(b, 1, 1).expand(b, 2 * total_points, 1))
        embeds = embeds + time_emb

        # ===== 应用 mask =====
        mask_embed = self._read_in(self.mask_embedding.squeeze(0))
        for i in range(b):
            if masked_indices[i].any():
                # 找到所有被 mask 的位置（包括 prompt 和 respond）
                masked_positions = masked_indices[i].nonzero(as_tuple=False).squeeze(-1)
                # 转换为序列中的 y 位置索引（y 位置在序列中是奇数索引：1, 3, 5, ...）
                full_idx = masked_positions * 2 + 1
                embeds[i, full_idx, :] = mask_embed

        # 🆕 创建 RBO-AR attention bias（唯一区别）
        attention_bias = self._create_rbo_attention_bias(
            total_points=total_points,
            n_prompt=self.n_prompt,
            block_size=self.block_size,
            device=device,
            dtype=embeds.dtype
        )
        # 扩展到batch size
        attention_bias = attention_bias.expand(b, -1, -1, -1)

        # ===== Backbone forward =====
        dummy_input_ids = torch.zeros(b, 2 * total_points, dtype=torch.long, device=device)
        out = self._backbone(
            input_ids=dummy_input_ids,
            input_embeddings=embeds,
            attention_bias=attention_bias,  # 🆕 显式传入 RBO-AR attention bias
            output_hidden_states=True,
        )
        h = out.hidden_states[-1]
        pred_y_all = self._read_out(h)[:, 0::2, 0]  # [B, total_points]

        # 🔧 提取 respond 部分的预测（使用全局函数）
        pred_y = extract_respond_predictions(pred_y_all, self.n_prompt, actual_n_respond, respond_position_mask)

        if not train_mode:
            # 推理模式 - 支持多种策略
            if use_bpd:
                # 🆕 BPD: Block-wise Parallel Diffusion (K>1)
                # 分阶段并行生成，每步选取K个熵最小的块
                # 支持 WeDLM 优化技巧（通过 lambda_penalty 和 entropy_tau 控制）
                pred_y, confirmed_priorities = self.generate_bpd(
                    xs, ys, device, respond_position_mask, K=bpd_k,
                    lambda_penalty=lambda_penalty, entropy_tau=entropy_tau
                )
                respond_masked_indices = extract_respond_masked_indices(
                    masked_indices, self.n_prompt, actual_n_respond, respond_position_mask
                )
                return pred_y, respond_masked_indices
            elif use_ebo_inference:
                # EBO-AR: Entropy-based Block Order inference (serial, K=1)
                # ✅ Supports both sequential and non-sequential modes
                pred_y, confirmed_priorities = self.generate_ebo(
                    xs, ys, device, respond_position_mask
                )
                respond_masked_indices = extract_respond_masked_indices(
                    masked_indices, self.n_prompt, actual_n_respond, respond_position_mask
                )
                return pred_y, respond_masked_indices
            else:
                # RBO-AR: Random Block Order inference (standard)
                respond_masked_indices = extract_respond_masked_indices(
                    masked_indices, self.n_prompt, actual_n_respond, respond_position_mask
                )
                return pred_y, respond_masked_indices

        # ===== Training Loss =====
        target = ys.squeeze(-1) if ys.dim() == 3 else ys

        # 🔧 提取 respond target 和 masked_indices（使用全局函数）
        respond_target = extract_respond_targets(target, self.n_prompt, actual_n_respond, respond_position_mask)
        respond_masked_indices = extract_respond_masked_indices(masked_indices, self.n_prompt, actual_n_respond, respond_position_mask)

        diff = pred_y - respond_target

        # 只在 respond 部分的 masked 位置计算 loss
        mask = respond_masked_indices.float()
        per_sample_loss = (diff.square() * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        # 🔧 使用训练策略计算loss权重（使用全局函数）
        weighted_loss = compute_loss_weight_with_strategy(
            per_sample_loss, t, t_scalar, self.training_strategy, self.num_timesteps
        )

        return weighted_loss, pred_y, t_scalar, respond_masked_indices


# ============================================================
# BAD-AR: Block-level Autoregressive Diffusion
# ============================================================
class BADARPromptRespond(LLaDAPromptRespond):
    """
    BAD-AR: Block-level Autoregressive Diffusion for Prompt-Respond Setting
    
    核心设计理念：
    - 块间 (Inter-block): Diffusion 逻辑。对 Respond 部分进行分块，随机 Mask 若干块。
      不同 Mask 块之间互不可见（避免去噪时的噪声交叉污染）。
    - 块内 (Intra-block): AR 逻辑。每个块内部遵循严格的因果关系。
      这保证了模型对单个数据点 (x,y) 内部特征的极高重建精度。
    - 条件可见性：所有块均可看到完整的 Prompt 和其他未被 Mask 的可见块。
    
    数学原理：
    训练时学习：P(token_j | visible_blocks, intra_block_prefix)
    其中 visible_blocks 包括：Prompt + 所有未mask的块
    
    优势：
    1. 局部绑定更强：在 ICL 任务中，(x,y) 通常是一个整体。
       块内 AR 强制模型在生成 y 时必须紧紧盯着刚刚生成的 x。
    2. 并行度优化：相比全 AR，它大大减少了 KV Cache 的压力。
       只需要在生成选中的块时进行小规模的 AR 推理。
    3. 解决"置换不变性"：因为块间是 Diffusion，模型在训练时见过各种块组合。
    
    Attention 规则：
    对于位置 i (Query) 和位置 j (Key)，j 对 i 可见当且仅当：
    1. j 属于 Prompt 区域（永远可见）
    2. j 属于未被 Mask 的块（Visible Blocks，永远可见）
    3. i 和 j 属于同一个 Masked 块，且 j <= i（Intra-block AR）
    4. 否则不可见（特别是不同 Masked 块之间互不可见）
    """

    def __init__(self, n_dims, n_positions, n_embd=256, n_layer=12, n_head=8,
                 n_prompt=20, n_respond=5, *, mlp_ratio=4.0, block_group_size=1,
                 block_size=4, **kwargs):
        """
        Initialize BAD-AR model.
        
        Args:
            block_size: Number of points per block (controls granularity)
                       - Each block has 2 * block_size positions (x and y for each point)
                       - Smaller → more blocks → finer block-level diffusion
        """
        # Inherit all parent initialization
        super().__init__(
            n_dims=n_dims, n_positions=n_positions, n_embd=n_embd,
            n_layer=n_layer, n_head=n_head, n_prompt=n_prompt,
            n_respond=n_respond, mlp_ratio=mlp_ratio,
            block_group_size=block_group_size, **kwargs
        )
        
        self.block_size = block_size
        self.n_head = n_head  # 🎯 必须保存，用于 Bias 函数获取 Head 数
        self.name = "bad_ar"
        
        print(f"[{self.name}] Initialized: block_size={block_size}, n_head={n_head}")
        print(f"  - Inter-block: Diffusion logic (blocks can be masked/unmasked)")
        print(f"  - Intra-block: AR logic (strict causality within each block)")
        print(f"  - Visible blocks: Always visible (bidirectional)")
        print(f"  - Masked blocks: Isolated from each other, AR within block")

    def _create_bad_ar_attention_bias(self, b, total_points, n_prompt, block_size, 
                                     masked_block_indices, device, dtype, respond_indices_batch=None):
        """
        优化版：减少循环，提高 GPU 利用率
        
        创建 BAD-AR 专用的向量化 Attention Bias (0 = attend, -inf = mask)
        
        性能优化：
        1. 减少 Python 循环，尽量使用向量化操作
        2. 使用 torch.where 替代 expand + 索引赋值
        3. 优化内存分配
        
        Args:
            b: batch size
            total_points: n_prompt + n_respond
            n_prompt: number of prompt pairs
            block_size: number of points per block
            masked_block_indices: [b, num_blocks] bool tensor, True 表示该块被 mask
            device: torch device
            dtype: torch dtype
            respond_indices_batch: Optional list of [b] tensors, each contains actual respond point indices
                                  If None, assumes sequential mode (respond points are at the end)
            
        Returns:
            attention_bias: [b, 1, seq_len, seq_len] additive attention bias
            (LLaDA backbone 会自动广播到 [b, n_head, seq_len, seq_len])
        """
        seq_len = 2 * total_points
        prompt_len = 2 * n_prompt
        idx_range = torch.arange(seq_len, device=device)
        
        # === 1. 建立位置到 Block ID 的映射 (Batch 维度感知) ===
        # pos_block_id: [b, seq_len] 记录每个位置属于哪个块
        pos_block_id = torch.full((b, seq_len), -1, device=device, dtype=torch.long)
        pos_is_masked = torch.zeros((b, seq_len), dtype=torch.bool, device=device)
        
        # 优化：只在需要时循环（non-sequential 模式必须循环，sequential 可以向量化）
        is_sequential = (respond_indices_batch is None)
        
        if is_sequential:
            # Sequential 模式：向量化实现（更快）
            respond_len = seq_len - prompt_len
            num_respond_points = total_points - n_prompt
            num_blocks = (num_respond_points + block_size - 1) // block_size
            
            # 为所有 batch 同时计算（假设 sequential 模式下所有 batch 的 respond 位置相同）
            for block_idx in range(num_blocks):
                # 计算该块对应的点索引范围
                start_point = n_prompt + block_idx * block_size
                end_point = min(start_point + block_size, total_points)
                
                # 每个点对应 x (2*p) 和 y (2*p+1) 两个位置
                block_points = torch.arange(start_point, end_point, device=device)
                block_pos_x = block_points * 2
                block_pos_y = block_points * 2 + 1
                block_pos = torch.cat([block_pos_x, block_pos_y])
                
                # 为所有 batch 同时分配 block_id（向量化）
                pos_block_id[:, block_pos] = block_idx
                
                # 如果该块被 mask，标记状态（向量化）
                if block_idx < masked_block_indices.shape[1]:
                    # 确保广播正确：is_masked [b, 1] -> pos_is_masked [b, len(block_pos)]
                    is_masked = masked_block_indices[:, block_idx]  # [b]
                    pos_is_masked[:, block_pos] = is_masked.unsqueeze(-1)  # [b, 1] 广播到 [b, len(block_pos)]
        else:
            # Non-sequential 模式：必须循环（但尽量减少循环内操作）
            for i in range(b):
                r_indices = respond_indices_batch[i]
                num_r = len(r_indices)
                num_blocks = (num_r + block_size - 1) // block_size
                
                for block_idx in range(num_blocks):
                    b_start = block_idx * block_size
                    b_end = min(b_start + block_size, num_r)
                    block_points = r_indices[b_start:b_end]
                    
                    # 每个点对应 x (2*p) 和 y (2*p+1) 两个位置
                    block_pos_x = block_points * 2
                    block_pos_y = block_points * 2 + 1
                    block_pos = torch.cat([block_pos_x, block_pos_y])
                    
                    pos_block_id[i, block_pos] = block_idx
                    
                    if block_idx < masked_block_indices.shape[1] and masked_block_indices[i, block_idx]:
                        pos_is_masked[i, block_pos] = True
        
        # === 2. 构建可见性规则 (完全向量化) ===
        idx_i = idx_range.view(-1, 1)  # [seq_len, 1]
        idx_j = idx_range.view(1, -1)  # [1, seq_len]
        
        # q_block, k_block: [b, seq_len, 1] / [b, 1, seq_len]
        q_block = pos_block_id.unsqueeze(-1)  # [b, seq_len, 1]
        k_block = pos_block_id.unsqueeze(1)   # [b, 1, seq_len]
        
        # 规则 1: Key 是 Prompt
        is_prompt_k = (k_block == -1)  # [b, 1, seq_len]
        
        # 规则 2: Key 是 Visible Respond Block
        is_visible_block_k = (~pos_is_masked.unsqueeze(1)) & (k_block != -1)  # [b, 1, seq_len]
        
        # 规则 3: 同一个 Masked 块内的 Intra-AR
        same_block = (q_block == k_block)  # [b, seq_len, seq_len]
        is_causal = (idx_j <= idx_i).unsqueeze(0)  # [1, seq_len, seq_len]
        q_masked = pos_is_masked.unsqueeze(-1)  # [b, seq_len, 1]
        k_masked = pos_is_masked.unsqueeze(1)    # [b, 1, seq_len]
        
        is_intra_ar = same_block & q_masked & k_masked & is_causal  # [b, seq_len, seq_len]
        
        # 综合可见性 [b, seq_len, seq_len]
        # 优化：使用广播而不是 expand（更高效）
        is_visible = (
            is_prompt_k.expand(-1, seq_len, -1) |  # [b, seq_len, seq_len]
            is_visible_block_k.expand(-1, seq_len, -1) |  # [b, seq_len, seq_len]
            is_intra_ar  # [b, seq_len, seq_len]
        )
        
        # === 3. 🎯 优化：使用 torch.where 创建 [b, 1, seq_len, seq_len]，让 PyTorch 自动广播到 head 维度 ===
        # 注意：LLaDA backbone 会自动将 [b, 1, seq_len, seq_len] 广播到 [b, n_head, seq_len, seq_len]
        # 这样可以节省 n_head 倍的内存（对于 n_head=8，节省 8 倍内存）
        bias = torch.where(
            is_visible.unsqueeze(1),  # [b, 1, seq_len, seq_len]
            torch.zeros((b, 1, seq_len, seq_len), device=device, dtype=dtype),
            torch.full((b, 1, seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        )
        
        return bias

    def forward(self, xs, ys, train_mode=True, respond_position_mask=None):
        """
        Forward pass with BAD-AR attention (Block-level Diffusion + Intra-block AR).
        
        Args:
            xs: [b, total_points, d] input features
            ys: [b, total_points] target values
            train_mode: whether in training mode
            respond_position_mask: optional mask for respond positions
            
        Returns:
            If train_mode:
                loss: weighted MSE loss
                pred_y: predictions for respond positions
                t_scalar: timestep values
                respond_masked_indices: indices of masked positions in respond region
            Else:
                pred_y: predictions for respond positions
                respond_masked_indices: indices of masked positions in respond region
        """
        b, total_points, d = xs.shape
        device = xs.device
        
        # 🔧 计算实际 n_respond
        actual_n_respond = compute_actual_n_respond(total_points, self.n_prompt, respond_position_mask)
        if respond_position_mask is not None:
            assert respond_position_mask.shape == (b, total_points), \
                f"respond_position_mask shape mismatch: expected {(b, total_points)}, got {respond_position_mask.shape}"
        assert total_points >= self.n_prompt, \
            f"Expected at least {self.n_prompt} points (prompt), got {total_points}"
        assert actual_n_respond <= self.n_respond, \
            f"Actual respond count {actual_n_respond} exceeds model max {self.n_respond}"
        
        # 🔧 采样 timestep（使用全局函数）
        t_scalar, t = sample_timestep_with_strategy(
            b, device, train_mode, self.training_strategy,
            self.num_timesteps, self.mask_epsilon, self.train_mask_ratio,
            self.eval_mask_ratio, self.eval_mask_mode
        )
        
        # === 1. 确定每个 batch 的 Respond 点索引（支持 non-sequential 模式）===
        # 优化：只在 non-sequential 模式下才需要循环
        is_sequential = (respond_position_mask is None)
        
        if is_sequential:
            # Sequential 模式：向量化实现（更快）
            num_respond_points = actual_n_respond
            num_blocks = (num_respond_points + self.block_size - 1) // self.block_size
            respond_indices_batch = None  # 不需要存储，sequential 模式下可以推断
        else:
            # Non-sequential 模式：需要循环获取每个 batch 的索引
            respond_indices_batch = []
            max_num_blocks = 0
            for i in range(b):
                r_indices = respond_position_mask[i].nonzero(as_tuple=False).squeeze(-1)
                respond_indices_batch.append(r_indices)
                num_r = len(r_indices)
                num_blocks = (num_r + self.block_size - 1) // self.block_size
                max_num_blocks = max(max_num_blocks, num_blocks)
            num_blocks = max_num_blocks
        
        # === 2. Block-level Masking: 决定哪些块被 MASK ===
        masked_block_indices = torch.zeros((b, num_blocks), dtype=torch.bool, device=device)
        ratio = t_scalar if train_mode else torch.full((b,), self.eval_mask_ratio, device=device)
        
        if not train_mode and self.eval_mask_mode == "fixed":
            # 评估模式（fixed）：全部 mask（推理时）
            masked_block_indices.fill_(True)
        else:
            # 训练模式或评估模式（sample）：根据 ratio 随机决定
            # 优化：向量化计算 num_to_mask（如果可能）
            if is_sequential:
                # Sequential 模式：所有 batch 的 num_blocks 相同，可以部分向量化
                num_to_mask_per_batch = torch.ceil(num_blocks * ratio).long().clamp(min=1, max=num_blocks)
                for i in range(b):
                    perm = torch.randperm(num_blocks, device=device)
                    masked_block_indices[i, perm[:num_to_mask_per_batch[i]]] = True
            else:
                # Non-sequential 模式：必须循环
                for i in range(b):
                    r_indices = respond_indices_batch[i]
                    num_r = len(r_indices)
                    num_blocks_i = (num_r + self.block_size - 1) // self.block_size
                    num_to_mask = max(1, int(math.ceil(num_blocks_i * ratio[i].item())))
                    num_to_mask = min(num_to_mask, num_blocks_i)
                    perm = torch.randperm(num_blocks_i, device=device)
                    masked_block_indices[i, perm[:num_to_mask]] = True
        
        # === 3. 映射到 Token 级别的 masked_indices（用于 Loss 和 Embedding 替换）===
        masked_indices = torch.zeros((b, total_points), dtype=torch.bool, device=device)
        
        if is_sequential:
            # Sequential 模式：向量化实现
            for block_idx in range(num_blocks):
                if masked_block_indices[:, block_idx].any():
                    start_point = self.n_prompt + block_idx * self.block_size
                    end_point = min(start_point + self.block_size, self.n_prompt + actual_n_respond)
                    # 向量化：为所有被 mask 的 batch 同时设置
                    masked_indices[:, start_point:end_point] = masked_block_indices[:, block_idx:block_idx+1]
        else:
            # Non-sequential 模式：必须循环
            for i in range(b):
                r_indices = respond_indices_batch[i]
                num_r = len(r_indices)
                num_blocks_i = (num_r + self.block_size - 1) // self.block_size
                
                for block_idx in range(num_blocks_i):
                    if block_idx < masked_block_indices.shape[1] and masked_block_indices[i, block_idx]:
                        start_in_r = block_idx * self.block_size
                        end_in_r = min(start_in_r + self.block_size, num_r)
                        target_points = r_indices[start_in_r:end_in_r]
                        masked_indices[i, target_points] = True
        
        # === 构建序列 ===
        ys_input = ys.clone().float()
        if ys_input.dim() == 2:
            ys_input = ys_input.unsqueeze(-1)
        
        ys_wide = torch.cat([ys_input, torch.zeros(b, total_points, d - 1, device=device)], dim=2)
        zs = torch.stack((xs, ys_wide), dim=2).view(b, 2 * total_points, d)
        
        # === Embedding + Time conditioning ===
        embeds = self._read_in(zs)
        time_emb = self._time_mlp(t_scalar.view(b, 1, 1).expand(b, 2 * total_points, 1))
        embeds = embeds + time_emb
        
        # === 4. 应用 Mask Embedding ===
        mask_embed = self._read_in(self.mask_embedding.squeeze(0))
        for i in range(b):
            m_pos = masked_indices[i].nonzero(as_tuple=False).squeeze(-1)
            if len(m_pos) > 0:
                # 转换为序列中的 y 位置索引（y 位置在序列中是奇数索引：1, 3, 5, ...）
                full_idx = m_pos * 2 + 1
                embeds[i, full_idx, :] = mask_embed
        
        # === 5. 创建 BAD-AR attention bias（核心）===
        attention_bias = self._create_bad_ar_attention_bias(
            b=b,
            total_points=total_points,
            n_prompt=self.n_prompt,
            block_size=self.block_size,
            masked_block_indices=masked_block_indices,
            device=device,
            dtype=embeds.dtype,
            respond_indices_batch=respond_indices_batch  # 🆕 支持 non-sequential 模式
        )
        
        # === 6. Backbone Forward ===
        dummy_input_ids = torch.zeros(b, 2 * total_points, dtype=torch.long, device=device)
        out = self._backbone(
            input_ids=dummy_input_ids,
            input_embeddings=embeds,
            attention_bias=attention_bias,  # 🆕 BAD-AR attention bias
            output_hidden_states=True,
        )
        h = out.hidden_states[-1]
        pred_y_all = self._read_out(h)[:, 0::2, 0]  # [B, total_points]
        
        # 🔧 提取 respond 部分的预测
        pred_y = extract_respond_predictions(pred_y_all, self.n_prompt, actual_n_respond, respond_position_mask)
        
        if not train_mode:
            # 推理模式
            respond_masked_indices = extract_respond_masked_indices(
                masked_indices, self.n_prompt, actual_n_respond, respond_position_mask
            )
            return pred_y, respond_masked_indices
        
        # === 7. Training Loss ===
        target = ys.squeeze(-1) if ys.dim() == 3 else ys
        
        # 🔧 提取 respond target 和 masked_indices
        respond_target = extract_respond_targets(target, self.n_prompt, actual_n_respond, respond_position_mask)
        respond_masked_indices = extract_respond_masked_indices(
            masked_indices, self.n_prompt, actual_n_respond, respond_position_mask
        )
        
        diff = pred_y - respond_target
        
        # 只在 respond 部分的 masked 位置计算 loss
        mask = respond_masked_indices.float()
        per_sample_loss = (diff.square() * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        # 🔧 使用训练策略计算 loss 权重
        weighted_loss = compute_loss_weight_with_strategy(
            per_sample_loss, t, t_scalar, self.training_strategy, self.num_timesteps
        )
        
        return weighted_loss, pred_y, t_scalar, respond_masked_indices


# ============================================================
# Model Builder
# ============================================================
def build_model_prompt_respond(conf):
    """
    Build Prompt-Respond models from config
    
    Config should contain:
    - family: "gpt2", "llama", "llada", "dream", "sdar"
    - n_prompt: number of prompt pairs
    - n_respond: number of respond pairs (k)
    - other model hyperparameters
    """
    family = conf["family"]
    n_prompt = conf.get("n_prompt", 20)
    n_respond = conf.get("n_respond", 5)
    
    # 兼容不同命名
    n_layer = conf.get("n_layer", conf.get("n_layers", 6))
    n_head = conf.get("n_head", conf.get("n_heads", 8))
    n_embd = conf.get("n_embd", conf.get("d_model", 256))
    
    if family in ["gpt2", "gptJ", "gptj", "qwen", "qwen2", "qwen2.5", "llama", "llama2", "llama3"]:
        model = TransformerModelPromptRespond(
            n_dims=conf["n_dims"],
            n_positions=conf["n_positions"],
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            type=family,
            n_prompt=n_prompt,
            n_respond=n_respond,
            attention_mode=conf.get("attention_mode", "causal"),  # 🆕 新增
            pretrained=conf.get("pretrained", False),
            model_name_or_path=conf.get("model_name_or_path", None),
        )
        return model
    
    elif family in ["llada", "llada_masked", "llada_prompt_respond"]:
        # 多步推理配置（可选）
        inference_config = conf.get("inference", {})
        use_multistep = inference_config.get("use_multistep_inference", False)
        inference_scheduler = create_inference_scheduler(inference_config)
        
        return LLaDAPromptRespond(
            n_dims=conf["n_dims"],
            n_positions=conf["n_positions"],
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_prompt=n_prompt,
            n_respond=n_respond,
            mlp_ratio=conf.get("mlp_ratio", 4.0),
            block_group_size=conf.get("block_group_size", 1),
            mask_epsilon=conf.get("mask_epsilon", 1e-3),
            loss_weight_type=conf.get("loss_weight_type", "1/t"),
            train_mask_ratio=conf.get("train_mask_ratio", 0.5),
            eval_mask_ratio=conf.get("eval_mask_ratio", 1.0),
            eval_mask_mode=conf.get("eval_mask_mode", "fixed"),  # 评估mask模式: "fixed" (固定mask ratio) 或 "sample" (随机采样)
            use_prompt_context=conf.get("use_prompt_context", True),  # 新增：是否使用 prompt 作为上下文
            # 多步推理参数（可选）
            use_multistep_inference=use_multistep,
            inference_steps=inference_config.get("steps", 10),
            inference_scheduler=inference_scheduler,
            inference_confidence_alg=inference_config.get("confidence_alg", "entropy"),
        )

    elif family in ["llada_block", "llada_block_diffusion"]:
        # 🆕 LLaDA + Block Diffusion
        # 多步推理配置（可选）
        inference_config = conf.get("inference", {})
        use_multistep = inference_config.get("use_multistep_inference", False)
        inference_scheduler = create_inference_scheduler(inference_config)

        return LLaDABlockDiffusion(
            n_dims=conf["n_dims"],
            n_positions=conf["n_positions"],
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_prompt=n_prompt,
            n_respond=n_respond,
            mlp_ratio=conf.get("mlp_ratio", 4.0),
            block_group_size=conf.get("block_group_size", 1),
            mask_epsilon=conf.get("mask_epsilon", 1e-3),
            loss_weight_type=conf.get("loss_weight_type", "1/t"),
            train_mask_ratio=conf.get("train_mask_ratio", 0.5),
            eval_mask_ratio=conf.get("eval_mask_ratio", 1.0),
            eval_mask_mode=conf.get("eval_mask_mode", "fixed"),
            use_prompt_context=conf.get("use_prompt_context", True),
            # 多步推理参数（可选）
            use_multistep_inference=use_multistep,
            inference_steps=inference_config.get("steps", 10),
            inference_scheduler=inference_scheduler,
            inference_confidence_alg=inference_config.get("confidence_alg", "entropy"),
            # 🆕 Block Diffusion 参数
            use_block_diffusion=conf.get("use_block_diffusion", False),
            block_size=conf.get("block_size", 4),
        )

    elif family in ["bop_ar", "bopar", "bop-ar"]:
        # 🆕 BOP-AR (ScatDiff): Block-Offset Parallel Autoregressive
        # 多步推理配置（可选）
        inference_config = conf.get("inference", {})
        use_multistep = inference_config.get("use_multistep_inference", False)
        inference_scheduler = create_inference_scheduler(inference_config)

        return BOPARPromptRespond(
            n_dims=conf["n_dims"],
            n_positions=conf["n_positions"],
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_prompt=n_prompt,
            n_respond=n_respond,
            mlp_ratio=conf.get("mlp_ratio", 4.0),
            block_group_size=conf.get("block_group_size", 1),
            # 🆕 ScatDiff 核心参数：block_size 控制垂直生成深度
            block_size=conf.get("block_size", 1),  # 默认 block_size=1 (2 layers: x, y)
            mask_epsilon=conf.get("mask_epsilon", 1e-3),
            loss_weight_type=conf.get("loss_weight_type", "1/t"),
            train_mask_ratio=conf.get("train_mask_ratio", 0.5),
            eval_mask_ratio=conf.get("eval_mask_ratio", 1.0),
            eval_mask_mode=conf.get("eval_mask_mode", "fixed"),
            use_prompt_context=conf.get("use_prompt_context", True),
            # 多步推理参数（可选）
            use_multistep_inference=use_multistep,
            inference_steps=inference_config.get("steps", 10),
            inference_scheduler=inference_scheduler,
            inference_confidence_alg=inference_config.get("confidence_alg", "entropy"),
        )

    elif family in ["rbo_ar", "rbo-ar", "rboar"]:
        # 🆕 RBO-AR (Random Block-Order Autoregressive): Random priority block generation
        # 多步推理配置（可选）
        inference_config = conf.get("inference", {})
        use_multistep = inference_config.get("use_multistep_inference", False)
        inference_scheduler = create_inference_scheduler(inference_config)

        return RBOARPromptRespond(
            n_dims=conf["n_dims"],
            n_positions=conf["n_positions"],
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_prompt=n_prompt,
            n_respond=n_respond,
            mlp_ratio=conf.get("mlp_ratio", 4.0),
            block_group_size=conf.get("block_group_size", 1),
            # 🆕 RBO-AR 核心参数：block_size 控制块大小（每个块包含 block_size 个点）
            block_size=conf.get("block_size", 4),  # 默认 block_size=4
            random_order=conf.get("random_order", True),  # 是否使用随机优先级
            priority_seed=conf.get("priority_seed", None),  # 优先级种子（None=每次随机）
            mask_epsilon=conf.get("mask_epsilon", 1e-3),
            loss_weight_type=conf.get("loss_weight_type", "1/t"),
            train_mask_ratio=conf.get("train_mask_ratio", 0.5),
            eval_mask_ratio=conf.get("eval_mask_ratio", 1.0),
            eval_mask_mode=conf.get("eval_mask_mode", "fixed"),
            use_prompt_context=conf.get("use_prompt_context", True),
            # 多步推理参数（可选）
            use_multistep_inference=use_multistep,
            inference_steps=inference_config.get("steps", 10),
            inference_scheduler=inference_scheduler,
            inference_confidence_alg=inference_config.get("confidence_alg", "entropy"),
        )

    elif family in ["bad_ar", "bad-ar", "badar"]:
        # 🆕 BAD-AR (Block-level Autoregressive Diffusion): 
        # 块间 Diffusion + 块内 AR 的混合范式
        # 多步推理配置（可选）
        inference_config = conf.get("inference", {})
        use_multistep = inference_config.get("use_multistep_inference", False)
        inference_scheduler = create_inference_scheduler(inference_config)

        return BADARPromptRespond(
            n_dims=conf["n_dims"],
            n_positions=conf["n_positions"],
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_prompt=n_prompt,
            n_respond=n_respond,
            mlp_ratio=conf.get("mlp_ratio", 4.0),
            block_group_size=conf.get("block_group_size", 1),
            # 🆕 BAD-AR 核心参数：block_size 控制块大小（每个块包含 block_size 个点）
            block_size=conf.get("block_size", 4),  # 默认 block_size=4
            mask_epsilon=conf.get("mask_epsilon", 1e-3),
            loss_weight_type=conf.get("loss_weight_type", "1/t"),
            train_mask_ratio=conf.get("train_mask_ratio", 0.5),
            eval_mask_ratio=conf.get("eval_mask_ratio", 1.0),
            eval_mask_mode=conf.get("eval_mask_mode", "fixed"),
            use_prompt_context=conf.get("use_prompt_context", True),
            # 多步推理参数（可选）
            use_multistep_inference=use_multistep,
            inference_steps=inference_config.get("steps", 10),
            inference_scheduler=inference_scheduler,
            inference_confidence_alg=inference_config.get("confidence_alg", "entropy"),
        )

    elif family in ["dream", "dream_prompt_respond"]:
        # 多步推理配置（可选）
        inference_config = conf.get("inference", {})
        use_multistep = inference_config.get("use_multistep_inference", False)
        inference_scheduler = create_inference_scheduler(inference_config)
        
        return DreamPromptRespond(
            n_dims=conf["n_dims"],
            n_positions=conf["n_positions"],
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_prompt=n_prompt,
            n_respond=n_respond,
            loss_weight_type=conf.get("loss_weight_type", "1/t"),
            mask_epsilon=conf.get("mask_epsilon", 1e-3),
            train_mask_ratio=conf.get("train_mask_ratio", 0.5),
            eval_mask_ratio=conf.get("eval_mask_ratio", 1.0),
            eval_mask_mode=conf.get("eval_mask_mode", "fixed"),  # 评估mask模式: "fixed" (固定mask ratio) 或 "sample" (随机采样)
            use_prompt_context=conf.get("use_prompt_context", True),  # 新增：是否使用 prompt 作为上下文
            # 多步推理参数（可选）
            use_multistep_inference=use_multistep,
            use_dllm_generate=inference_config.get("use_dllm_generate", False),  # 🎯 新增
            inference_steps=inference_config.get("steps", 10),
            inference_scheduler=inference_scheduler,
            inference_confidence_alg=inference_config.get("confidence_alg", "entropy"),
            inference_temperature=inference_config.get("temperature", 1.0),  # 🎯 新增
            inference_top_p=inference_config.get("top_p", 1.0),  # 🎯 新增
            inference_top_k=inference_config.get("top_k", 50),  # 🎯 新增
        )
    
#     elif family in ["sdar", "sdar_prompt_respond"]:
#         # 🆕 SDAR 模型支持
#         if not SDAR_AVAILABLE:
#             raise ImportError(
#                 "SDAR model is not available. Please ensure dLLM-RL/models/sdar is accessible. "
#                 "Make sure dLLM-RL directory is in the project root and models/sdar module can be imported."
#             )
#         
#         # 多步推理配置（可选）
#         inference_config = conf.get("inference", {})
#         use_multistep = inference_config.get("use_multistep_inference", False)
#         inference_scheduler = create_inference_scheduler(inference_config)
#         
#         return SDARPromptRespond(
#             n_dims=conf["n_dims"],
#             n_positions=conf["n_positions"],
#             n_embd=n_embd,
#             n_layer=n_layer,
#             n_head=n_head,
#             n_prompt=n_prompt,
#             n_respond=n_respond,
#             mlp_ratio=conf.get("mlp_ratio", 4.0),
#             mask_epsilon=conf.get("mask_epsilon", 1e-3),
#             loss_weight_type=conf.get("loss_weight_type", "1/t"),
#             train_mask_ratio=conf.get("train_mask_ratio", 0.5),
#             eval_mask_ratio=conf.get("eval_mask_ratio", 1.0),
#             eval_mask_mode=conf.get("eval_mask_mode", "fixed"),  # 评估mask模式: "fixed" (固定mask ratio) 或 "sample" (随机采样)
#             use_prompt_context=conf.get("use_prompt_context", True),  # 新增：是否使用 prompt 作为上下文
#             # 🆕 Block Diffusion 参数
#             use_block_diffusion=conf.get("use_block_diffusion", False),
#             block_size=conf.get("block_size", 4),
#             # 多步推理参数（可选）
#             use_multistep_inference=use_multistep,
#             inference_steps=inference_config.get("steps", 10),
#             inference_scheduler=inference_scheduler,
#             inference_confidence_alg=inference_config.get("confidence_alg", "entropy"),
#             # 🆕 Block-by-block inference 参数（可选，默认根据 use_block_diffusion 自动决定）
#             use_block_by_block_inference=conf.get("use_block_by_block_inference", None),
#             inference_steps_per_block=conf.get("inference_steps_per_block", None),
#         )
    
    # 🆕 Duo增强模型支持 (已禁用)
    # elif family in [
    #     "llada_duo_curriculum",
    #     "llada_duo_distillation",
    #     "llada_duo_full",
    #     "dream_duo_curriculum",
    #     "dream_duo_distillation",
    #     "dream_duo_full",
    # ]:
    #     # 导入 Duo 增强模型的构建函数
    #     try:
    #         from models_prompt_respond_duo_enhanced import build_model_prompt_respond_duo
    #         return build_model_prompt_respond_duo(conf)
    #     except ImportError as e:
    #         raise ImportError(
    #             f"Failed to import Duo enhanced models. "
    #             f"Make sure models_prompt_respond_duo_enhanced.py is available. "
    #             f"Original error: {e}"
    #         )
    
    else:
        raise NotImplementedError(f"Unsupported model family: {family}")

        