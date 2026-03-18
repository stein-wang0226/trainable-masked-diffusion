

# Blockwise Diffusion Trainability

This repository provides the official implementation for the paper **"On Trainability of Masked Diffusion Language Models with Blockwise Locality"**. 

The project aims to systematically compare the **trainability** of **Autoregressive Language Models (AR-LLMs)** and **Masked Diffusion Language Models (MDMs)** across several structured tasks.

---

## 1. Overview

Key research areas:
- **MDM vs AR**: Comparing convergence speed and sample efficiency in In-Context Learning (ICL).
- **Blockwise Locality**: Evaluating the impact of design choices (Block Diffusion / Scatter / Jigsaw) on training stability.
- **Combinatorial Tasks**: Demonstrating the advantages of the diffusion paradigm over AR in tasks like Sudoku.

All experiments follow a unified **Prompt-Respond** sequence format:
$$ \text{[Prompt]}: (x_1, y_1), \dots, (x_p, y_p) \mid \text{[Respond]}: (x_{p+1}, ?), \dots, (x_{p+r}, ?) $$

---

## 2. Code Structure

```text
.
├── src/                                   # Core experiment code
│   ├── train_prompt_respond.py            # Unified trainer for ICL/Sudoku/Path-finding
│   ├── eval_prompt_respond.py             # ICL evaluation (MSE)
│   ├── eval_sudoku.py                     # Sudoku evaluation (Cell/Puzzle Acc)
│   ├── models_prompt_respond.py           # Model definitions (LLaDA, Block, AR, etc.)
│   ├── conf/                              # Experiment configurations (YAML)
│   └── dllm-pathfinding/                  # Path-finding logic (core-nebula)
├── dllm/                                  # [External] Placeholder for dllm repo
├── dllm_rl/                               # [External] Placeholder for dllm_rl repo
├── run_batch_experiments.py               # Script for batch processing
├── requirements.txt                       # Dependency list
└── ...
```

------

## 3. Installation

### 3.1 Prerequisites

Python 3.10+ is recommended:

```
bash


pip install -r requirements.txt
```

### 3.2 External Dependencies (Mandatory)

The following repositories must be cloned manually into the root directory:

1. **Clone dllm (Base MDM library)**:

   ```
   bash
   
   
   git clone https://github.com/ZHZisZZ/dllm.git dllm
   ```

2. **Clone dllm_rl (Path-finding/Next-Token-Failures)**:

   ```
   bash
   
   
   git clone https://github.com/gregorbachmann/Next-Token-Failures.git dllm_rl
   ```

------

## 4. Running Experiments

### 4.1 ICL Linear Regression

- **Single Experiment:**

  ```
  bash
  
  
  python src/train_prompt_respond.py --config src/conf/prompt_respond_llada_formal.yaml
  ```

- **Batch Reproduction:**

  ```
  bash
  
  
  python run_batch_experiments.py --model all
  ```

### 4.2 Sudoku

- **Training:**

  ```
  bash
  
  
  python src/train_prompt_respond.py --config src/conf/sudoku_experiments_standard/sudoku_dream_unified_P0_R1_seed42.yaml
  ```

- **Evaluation:**

  ```
  bash
  
  
  python src/eval_sudoku.py --checkpoint <path_to_ckpt>
  ```

### 4.3 Path-finding

The core graph logic is integrated in `src/dllm-pathfinding/core-nebula/`.

- Training can be launched using configs in `src/conf/pathfinding_experiments/`.

------

## 5. Citation

If you use this code in your research, please cite our paper:

```
bibtex


@article{blockwise_diffusion_2024,
  title={On Trainability of Masked Diffusion Language Models with Blockwise Locality},
  author={...},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}

