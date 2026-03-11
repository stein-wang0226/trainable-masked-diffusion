import argparse
from typing import Dict, Any
import yaml
import json

def parse_arguments() -> Dict[str, Any]:
    """
    解析命令行参数 + （可选）YAML 配置文件。
    若传入 --config，将自动加载 YAML 并与命令行参数合并。
    """
    parser = argparse.ArgumentParser(description="Unified training configuration")

    # === 可选 YAML 配置路径 ===
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to YAML config file. Command line args override YAML."
    )

    # Model parameters
    parser.add_argument("--model-family", choices=["gpt2", "lstm", "gptJ", "llada", "dream"], help="Model family")
    parser.add_argument("--n_positions", type=int, help="Number of positions (context length)")
    parser.add_argument("--n_dims", type=int, help="Input dimension")
    parser.add_argument("--n_embd", type=int, help="Embedding dimension")
    parser.add_argument("--n_layer", type=int, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, help="Number of attention heads")
    # === Model load options ===
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="If set, load pretrained weights from HuggingFace (e.g., Qwen2.5, LLaMA3)"
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default=None,
        help="Optional model name or local path, e.g., 'Qwen/Qwen2.5-7B-Instruct' or './checkpoints/llama3'"
    )

    # Curriculum parameters
    parser.add_argument("--curriculum-dims-start", type=int)
    parser.add_argument("--curriculum-dims-end", type=int)
    parser.add_argument("--curriculum-dims-inc", type=int)
    parser.add_argument("--curriculum-dims-interval", type=int)

    parser.add_argument("--curriculum-points-start", type=int)
    parser.add_argument("--curriculum-points-end", type=int)
    parser.add_argument("--curriculum-points-inc", type=int)
    parser.add_argument("--curriculum-points-interval", type=int)

    # Training parameters
    parser.add_argument("--task", choices=["linear_regression", "sparse_linear_regression",
                                           "linear_classification", "relu_2nn_regression", "decision_tree"])
    parser.add_argument("--task-kwargs", type=str, default="{}", help="Additional task-specific arguments (as JSON string)")
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--num_training_examples", type=int, default=10000)

    parser.add_argument("--data", choices=["gaussian", "uniform"])
    parser.add_argument("--w-type", choices=["gaussian", "uniform"], default="gaussian")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.00)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--save-every-steps", type=int, default=1000)
    parser.add_argument("--keep-every-steps", type=int, default=-1)
    parser.add_argument("--resume-id", type=str, default=None)
    
# # 在 parse_arguments() 里其它 parser.add_argument 之后，新增：
#     parser.add_argument(
#         "--sampler_mode",
#         choices=["fixed_w_dynamic_x", "dynamic_w_fixed_x", "dynamic_both", "fixed_both"],
#         default="fixed_w_dynamic_x",
#     )

    # Optional curriculum settings
    parser.add_argument("--if-two-distribution", action="store_true")
    parser.add_argument("--if-random-shuffle-2distribution", action="store_true")
    parser.add_argument("--w-distribution1", choices=["gaussian", "uniform"], default="gaussian")
    parser.add_argument("--w-distribution2", choices=["gaussian", "uniform"], default="uniform")

    # Wandb
    parser.add_argument("--wandb-project", type=str, default="ICL-dllm")
    # TODO: replace with your own WandB entity if you use WandB logging
    parser.add_argument("--wandb-entity", type=str, default="your-wandb-entity")
    parser.add_argument("--wandb-name", type=str, default="linear_regression_standard")
    parser.add_argument("--wandb-notes", type=str, default="Test run")
    parser.add_argument("--wandb-log-every-steps", type=int, default=100)

    # Output & test
    parser.add_argument("--out-dir", type=str, help="Output directory")
    parser.add_argument("--test-run", action="store_true")

    # === 解析 ===
    args = parser.parse_args()
    args_dict = vars(args)

    # === 如果指定了 config，加载 YAML 并合并 ===
    if args.config:
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)
        # 命令行优先覆盖 YAML
        for k, v in args_dict.items():
            if v is not None:
                yaml_config[k] = v

        # 对 task-kwargs 进行解析，确保其是字典
        if 'task-kwargs' in yaml_config and isinstance(yaml_config['task-kwargs'], str):
            yaml_config['task-kwargs'] = json.loads(yaml_config['task-kwargs'])

        return yaml_config
    else:
        # 对 task-kwargs 进行解析
        if 'task-kwargs' in args_dict and isinstance(args_dict['task-kwargs'], str):
            args_dict['task-kwargs'] = json.loads(args_dict['task-kwargs'])

        return args_dict
