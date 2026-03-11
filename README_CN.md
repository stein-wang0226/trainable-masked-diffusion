# Blockwise Diffusion Trainability

本仓库提供论文 **“On Trainability of Masked Diffusion Language Models with Blockwise Locality”** 的主要实验代码，用于系统比较 **自回归语言模型（AR-LLMs）** 与 **掩码扩散语言模型（Masked Diffusion LMs, MDMs）** 在多种结构化任务上的 **可训练性（trainability）**，包括：

- **ICL 线性回归任务**（Prompt-Respond In-Context Learning）
- **Sudoku 解题任务**
- （论文里）**路径规划 / Path-finding 任务** —— 训练管线复用本项目代码，图任务的核心实现和数据位于单独的 `dllm-pathfinding` 仓库

---

## 1. 项目概览

本项目重点研究：

- **MDM vs AR 在 ICL 线性映射学习上的可训练性差异**
- **不同 Blockwise 设计（Block Diffusion / Scatter / Jigsaw）对训练稳定性和样本效率的影响**
- **在 Sudoku 等强约束组合任务上，扩散范式相对 AR 的优势**

实验统一基于 **Prompt-Respond** 序列结构：

$$
\text{[Prompt]}: (x_1, y_1), \dots, (x_p, y_p)
\mid
\text{[Respond]}: (x_{p+1}, ?), \dots, (x_{p+r}, ?)
$$

模型需在同一序列中，从 Prompt 的少量示例中学习隐式函数 \(y=f(x)\)，并在 Respond 区间做出预测。

---

## 2. 代码结构

```text
.
├── src/                                   # 核心实验代码
│   ├── train_prompt_respond.py            # ICL / Sudoku / Path-finding 统一训练入口脚本
│   ├── eval_prompt_respond.py             # ICL 评估脚本（线性回归 MSE 等）
│   ├── eval_sudoku.py                     # Sudoku 评估脚本（cell / puzzle accuracy）
│   ├── models_prompt_respond.py           # ICL 模型（LLaDA, Dream, Block, BOP-AR, BAD-AR, AR 等）
│   ├── models_prompt_respond_sudoku.py    # Sudoku 模型
│   ├── models_prompt_respond_pathfinding.py # Path-finding 模型（封装 core-nebula 图逻辑）
│   ├── tasks.py                           # ICL 任务定义（线性回归等）
│   ├── tasks_sudoku.py                    # Sudoku 任务定义
│   ├── tasks_pathfinding.py               # Path-finding 任务定义（Star-Graph 等）
│   ├── samplers.py                        # ICL 数据采样器（Gaussian / Unit-Norm 等）
│   ├── samplers_sudoku.py                 # Sudoku 数据采样器
│   ├── samplers_pathfinding.py            # Path-finding 数据采样器（读取图数据文本）
│   ├── train_utils.py                     # 通用训练工具（日志、保存、调度等）
│   ├── curriculum.py                      # 课程学习配置（dims / points 课程）
│   ├── dllm-pathfinding/                  # Path-finding 核心实现（原 dllm-pathfinding 仓库的 core-nebula）
│   │   └── core-nebula/                   # 图生成、tokenizer、参考配置等
│   └── conf/                              # 全部实验配置（YAML）
│       ├── prompt_respond_*.yaml          # 单实验 ICL 配置
│       ├── experiments_batch/             # ICL 批量实验配置
│       ├── llada_block/                   # Block Diffusion 配置（论文中的 SDAR/Block 系列）
│       ├── BOP-AR/                        # Scatter Diffusion 配置（论文中的 Scatter）
│       ├── BAD-AR/                        # Jigsaw Diffusion 配置（论文中的 Jigsaw）
│       ├── sudoku_experiments*/           # Sudoku 统一协议配置
│       └── pathfinding_experiments*/      # Path-finding 对应的 ICL 设定（调用 core-nebula 逻辑）
├── dllm/                                  # 通用扩散语言模型库（LLaDA / Dream 等）
├── dllm_rl/                               # Block Diffusion / SDAR 等 RL 实现
├── run_batch_experiments.py               # ICL 批量实验脚本（本地 / 集群通用）
├── run_batch_experiments.sh               # Nebula 集群批量提交脚本（可选）
├── boot.py, boot_ddp.py                   # 集群环境依赖修复 + 启动脚本（可选）
├── cluster*.json                          # 集群资源配置（GPU 拓扑等）
├── optimal_params.json                    # 部分实验的最优超参记录
└── requirements.txt                       # 依赖列表（精简版）
```

> 说明：  
>
> - **ICL + Sudoku + Path-finding 的完整训练 / 评估管线全部在本仓库内**。  
> - Path-finding 任务依赖的 Star-Graph 图生成、tokenization 等逻辑已经内嵌在 `src/dllm-pathfinding/core-nebula/` 下，并通过 `models_prompt_respond_pathfinding.py` / `tasks_pathfinding.py` / `samplers_pathfinding.py` 与统一的 Prompt-Respond 训练入口对接。

---

## 3. 环境安装

### 3.1 依赖安装

推荐使用 Python 3.10+ 与虚拟环境：

```bash
cd Diffu-ICL_副本

python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` 中只保留了最小化的依赖栈（Transformers / Accelerate / WandB / Numpy / Matplotlib 等），适合论文复现。

### 3.2 硬件需求

- 单机 / 多机 GPU 均可  
- ICL 主实验在 24GB 以上显存的单卡即可完成  
- Sudoku / 大型 batch 实验推荐多卡或集群（可使用 `run_batch_experiments.py` 或 `run_batch_experiments.sh`）


### 3.3 克隆外部核心依赖库 (重要)

本项目依赖以下两个外部仓库。由于本项目直接引用其代码逻辑，请在根目录下手动执行克隆：

dllm: 掩码扩散模型（LLaDA/Dream）的基础实现。

```bash
git clone https://github.com/ZHZisZZ/dllm.git dllm
git clone https://github.com/gregorbachmann/Next-Token-Failures.git dllm_rl

```

---

## 4. ICL 线性回归实验（核心）

### 4.1 任务说明

- 输入维度：\(d \in \{10, 15, 20\}\)  
- Prompt 数：\(p \in \{10, 20\}\)  
- Respond 数：\(r = 20\)  
- 函数族：线性回归 \(y = W x + b\)（可带噪 / 稀疏 / Unit-Norm 等变体）  
- 目标：在 In-Context 条件下恢复隐式线性映射，评估 MSE。

Prompt-Respond 序列形式为：

$$
\text{[Prompt]}: (x_1, y_1), \dots, (x_p, y_p)
\mid
\text{[Respond]}: (x_{p+1}, ?), \dots, (x_{p+r}, ?)
$$

模型在不更新参数的前提下，从 Prompt 中学习函数，再对 Respond 中的输入进行泛化。

### 4.2 单实验示例

以 **LLaDA（MDM）+ Sequential ICL** 为例：

```bash
python src/train_prompt_respond.py \
  --config src/conf/prompt_respond_llada_formal.yaml
```

常见变体（不同训练设置对应不同 config 文件，可根据论文表/图映射）：

```bash
# Non-Sequential ICL（打乱 Prompt 次序）
python src/train_prompt_respond.py \
  --config src/conf/prompt_respond_llada_non_sequential.yaml

# 带 Unit-Norm 的线性回归（消除 AR 的模长捷径）
python src/train_prompt_respond.py \
  --config src/conf/prompt_respond_llada_formal_unitnorm.yaml
```

### 4.3 批量实验（论文主结果）

论文中 ICL 线性回归的大规模对比实验由：

```bash
# 运行默认批量 ICL 实验矩阵（AR / LLaDA / Block / BOP-AR / BAD-AR）
python run_batch_experiments.py

# 只跑 Block Diffusion（LLaDA Block）
python run_batch_experiments.py --model llada_block

# Debug：只跑前 5 个实验
python run_batch_experiments.py --limit 5
```

各 YAML 配置文件位于 `src/conf/experiments_batch/` 等目录，其命名中编码了维度 D / Prompt P / Block size 等信息，可与论文表格一一对应。

此外，`run_batch_experiments.py` 也可以只用于 **批量生成 config 文件** 而不实际启动训练，例如：

```bash
python run_batch_experiments.py --generate-only
```

会将自动生成的配置写入 `src/conf/experiments_batch/`、`src/conf/llada_block/`、`src/conf/BOP-AR/`、`src/conf/BAD-AR/` 等目录，方便你按需手工挑选或修改。

---

## 5. Sudoku 解题实验

### 5.1 任务说明

- 输入：部分填充的 9×9 Sudoku 棋盘（quiz）  
- 输出：完整解（solution），需同时满足行 / 列 / 3×3 子格互斥约束  
- 序列化：使用 **163-token 协议**（Nebula tokenizer），并注入行/列/子格的坐标信息作为拓扑归纳偏置

Sudoku 数据来自 Kaggle 公共数据集 [Kaggle Sudoku dataset](https://www.kaggle.com/datasets/bryanpark/sudoku)。

模型需要在强组合约束下，生成满足所有约束的完整棋盘，我们主要关注：

- **Cell Accuracy**：81 个格子逐格正确率  
- **Sudoku Accuracy**：整盘完全正确比例（严格 CSP 指标）

### 5.2 训练示例

Sudoku 相关配置位于：

- `src/conf/sudoku_experiments_standard/`  
- `src/conf/sudoku_experiments_batch_comp/`  
- 以及若干 `sudoku_experiments*` 目录

示例（统一协议 + Dream MDM）：

```bash
python src/train_prompt_respond.py \
  --config src/conf/sudoku_experiments_standard/sudoku_dream_unified_P0_R1_seed42.yaml
```

支持的 Sudoku 模型 family（见 `models_prompt_respond_sudoku.py`）包括：

- `sudoku_dream` / `sudoku_llada`（MDM）
- `sudoku_ar`（自回归基线）
- `sudoku_llada_block`（Block Diffusion）
- `sudoku_bopar` / `sudoku_badar ` / `sudoku_rboar`（Scatter / Jigsaw / Random Block-Order 等）

### 5.3 评估

```bash
python src/eval_sudoku.py \
  --checkpoint <path_to_checkpoint.pt> \
  --num_eval_examples 1000
```

评估脚本会输出：

- **Cell Accuracy**：逐格准确率  
- **Sudoku Accuracy**：整盘完全正确比例

---

## 6. Path-finding 实验（说明）

论文中的 Star-Graph Path-finding 实验：

- 使用本仓库的训练框架（`train_prompt_respond.py` + ICL 设定）  
- Path-finding 的核心模型与图逻辑代码已内置在 `src/dllm-pathfinding/core-nebula/` 目录下（对应原始 `dllm-pathfinding` 仓库的 `core-nebula/`）  
- 本仓库中的 `src/conf/pathfinding_experiments*/` 目录给出了 ICL 设定和配置模板

Path-finding 任务的数据与图构造基于 GitHub 仓库 [Next-Token-Failures](https://github.com/gregorbachmann/Next-Token-Failures) 及其对应论文（参见该仓库首页链接的 arXiv 条目）。

实践建议：

- **推荐做法**：直接进入 `src/dllm-pathfinding/`，按该子目录下的 `README.md` 与 `core-nebula/` 中的配置运行 / 修改 Path-finding 相关代码，更贴近原始实现与论文设置。  
- **本仓库一侧**：`train_prompt_respond.py + src/conf/pathfinding_experiments*/` 复用了同一图数据与 tokenizer，将 Path-finding 任务嵌入统一的 Prompt-Respond ICL 框架中，用于与 ICL / Sudoku 实验做统一对比。

---

## 7. 模型与超参数（简要）

### 7.1 典型模型尺寸

- 以 `big` 配置为主：
  - `n_embd = 384`  
  - `n_layers = 16`  
  - `n_heads = 12`

### 7.2 训练步数（线性回归 ICL）

- d = 10：约 `1e6` steps  
- d = 15：约 `1.5e6` steps  
- d = 20：约 `2e6` steps  

### 7.3 优化设置

- `batch_size = 64`  
- `learning_rate = 1e-4`  
- `weight_decay = 0.0`  
- `save_every_steps = 5000`  
- `log_interval = 200`

### 7.4 Diffusion / Blockwise 关键超参

不同 family 的关键 hyper-parameters 可在各自 YAML 中找到，通常包括：

- 掩码扩散：
  - `mask_epsilon`  
  - `loss_weight_type = "1/t"`  
  - `train_mask_ratio` / `eval_mask_ratio`  
- Block / Scatter / Jigsaw：
  - `block_size`  
  - `use_block_diffusion` 等

---

## 8. 复现建议

- **快速 sanity check**：  
  - 先用 d=10, 较少训练步数（例如 1e5）跑通 ICL 配置  
  - Sudoku 上先用小模型与小数据子集  
- **对齐论文主结果**：  
  - 使用本 README 中给出的维度 / 步数 / 模型尺寸  
  - 根据 `src/conf/` 中 YAML 文件名选择与论文表格/图对应的配置  
  - 使用 `run_batch_experiments.py` 统一跑完 ICL 线性回归实验矩阵

如果你需要按“论文具体某张表/图 → 对应 config + 运行命令”的精确对照表，可以在 issue 或 README 中补充，我们建议在 `docs/` 或 README 末尾单独开一节记录映射关系。

---

## 9. 引用

如本仓库对你的研究有帮助，请在论文中引用：

> *On Trainability of Masked Diffusion Language Models with Blockwise Locality*  
> 