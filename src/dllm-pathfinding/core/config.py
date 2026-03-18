from dataclasses import dataclass, field
from typing import Optional, List
from simple_parsing.helpers import Serializable

@dataclass
class ModelConfig(Serializable):
    # Backbone architecture (Shared)
    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 12
    vocab_size: int = 50257 # Default GPT-2 size, can be overridden
    n_positions: int = 1024
    dropout: float = 0.1
    
    # Model Type
    family: str = "ar" # "ar" (Autoregressive) or "dream" (Diffusion)
    
    # Diffusion Specific (Only used if family="dream")
    diffusion_steps: int = 1000
    noise_schedule: str = "cosine" 

@dataclass
class DataConfig(Serializable):
    # Task type
    task: str = "graph" # "graph" or "sudoku"
    
    # Pathfinding task specific
    num_nodes: int = 50
    deg: int = 2
    path_len: int = 5
    
    # File paths (Can be local or OSS paths handled by nebulactl/training script)
    train_data_path: str = "data/train.txt"
    test_data_path: str = "data/test.txt"
    
    # Tokenizer
    tokenizer_path: Optional[str] = None # If None, use default or build one

    # Block-wise generation training (Data side)
    block_size: int = 0 # 0 means no blocking (full sequence)


@dataclass
class TrainingConfig(Serializable):
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    
    max_steps: int = 100000
    warmup_steps: int = 1000
    save_every_steps: int = 5000
    eval_every_steps: int = 1000
    
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16" # "no", "fp16", "bf16"
    dtype: str = "bfloat16" # Used by training script
    
    seed: int = 42
    
    # Task type for loss calculation
    task_type: str = "autoregressive" # "autoregressive" or "diffusion"
    
    # Evaluation
    test_batches: Optional[int] = None # Number of batches to test, None for all
    
    # Resuming
    resume_id: Optional[str] = None

@dataclass
class WandbConfig(Serializable):
    project: str = "pathfinding-nebula"
    entity: str = "in-context" # Change as needed
    name: Optional[str] = None
    group: Optional[str] = None
    notes: Optional[str] = None
    log_every_steps: int = 10

@dataclass
class NebulaConfig(Serializable):
    # Nebula platform specific configurations
    oss_bucket: str = "nebula-pathfinding"
    remote_sync: bool = False

@dataclass
class Config(Serializable):
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    nebula: NebulaConfig = field(default_factory=NebulaConfig)
    
    out_dir: str = "./outputs"
    test_run: bool = False

@dataclass
class MultiDiffusionConfig(Config):
    # Extension of base config to include paradigm selection
    paradigm: str = "block" # "block", "scatter", "jigsaw"
    block_size: int = 4 # Block size for the paradigms