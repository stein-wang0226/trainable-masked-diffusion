import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config, Qwen2ForCausalLM, Qwen2Config, LlamaForCausalLM, LlamaConfig
import os
import sys

# Assuming dllm is installed or available in python path
try:
    from dllm.pipelines.dream.models.configuration_dream import DreamConfig
    from dllm.pipelines.dream.models.modeling_dream import DreamModel
    
    from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig, BlockType
    from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM
    from dllm.core.samplers import MDLMSampler
    
    # Try importing JigsawModel (assumes multi-diffusion is in path)
    try:
        from jigsaw_diffusion import JigsawModel
    except ImportError:
        JigsawModel = None

    # Try importing ScatterModel (assumes multi-diffusion is in path)
    try:
        from scatter_diffusion import ScatterModel
    except ImportError:
        ScatterModel = None

    # Try importing BlockDiffusionModel (assumes multi-diffusion is in path)
    try:
        from block_diffusion import BlockDiffusionModel
    except ImportError:
        BlockDiffusionModel = None

except ImportError as e:
    print(f"ERROR: Could not import Dream/LLaDA Config/Model from 'dllm': {e}")
    print(f"Current PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not Set')}")
    print(f"Current sys.path: {sys.path}")
    print(f"Current working directory: {os.getcwd()}")
    # List contents of dllm directory if it exists
    if os.path.exists("dllm"):
        print(f"Contents of './dllm': {os.listdir('dllm')}")
    # Re-raise to prevent confusing NameError later
    raise ImportError("dllm is required for this training task. Please ensure it is in the PYTHONPATH.") from e

def build_model(conf):
    model_args = {
        "n_positions": conf.n_positions,
        "n_embd": conf.n_embd,
        "n_layer": conf.n_layer,
        "n_head": conf.n_head,
        "vocab_size": conf.vocab_size,
    }
    
    # Handle Qwen specific args scaling if needed, or just pass generic ones
    # Qwen uses 'hidden_size' instead of 'n_embd', 'num_hidden_layers' instead of 'n_layer'
    # We will map them inside the model wrapper classes.

    if conf.family == "gpt2" or conf.family == "ar": # Default to GPT2 for AR if not specified
        model = TransformerModel(**model_args)
    elif conf.family == "qwen":
        model = QwenModel(**model_args)
    elif conf.family == "llama":
        model = LlamaModel(**model_args)
    elif conf.family == "dream":
        # For Dream, we add a dedicated [MASK] token.
        model_args["vocab_size"] += 1
        model = DreamDlmModel(**model_args)
    elif conf.family == "llada":
        # For LLaDA, we also add a dedicated [MASK] token.
        model_args["vocab_size"] += 1
        model = LladaDlmModel(**model_args, dropout=conf.dropout)
    elif conf.family == "jigsaw":
        # Jigsaw Model (wraps LLaDA backbone but different logic)
        model_args["vocab_size"] += 1
        # Retrieve block_size from conf if available, default to 4
        block_size = getattr(conf, 'block_size', 4)
        model = JigsawDlmModel(**model_args, block_size=block_size, dropout=conf.dropout)
    elif conf.family == "scatter":
        # Scatter Model
        model_args["vocab_size"] += 1
        block_size = getattr(conf, 'block_size', 4) # Default block size
        model = ScatterDlmModel(**model_args, block_size=block_size, dropout=conf.dropout, family=conf.family)
    elif conf.family == "block":
        # Block Diffusion Model
        model_args["vocab_size"] += 1
        block_size = getattr(conf, 'block_size', 4)
        model = BlockDlmModel(**model_args, block_size=block_size, dropout=conf.dropout)
    else:
        raise NotImplementedError(f"Model family '{conf.family}' not implemented.")

    return model


class TransformerModel(nn.Module):
    """Autoregressive model using GPT-2 backbone."""
    def __init__(self, vocab_size, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=True,  # use_cache must be True for generation
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_positions = n_positions

        # Use the full GPT2 model with Language Modeling Head
        self._backbone = GPT2LMHeadModel(configuration)

    def forward(self, xs, ys=None, task_type="autoregressive"):
        # The GPT2LMHeadModel returns an object containing the logits
        return self._backbone(input_ids=xs).logits

    def generate(self, prefix, **kwargs):
        """Wraps the backbone's generate method."""
        # The generate method is available on the LMHeadModel
        return self._backbone.generate(input_ids=prefix, **kwargs)

class QwenModel(nn.Module):
    """Autoregressive model using Qwen2 backbone."""
    def __init__(self, vocab_size, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(QwenModel, self).__init__()
        configuration = Qwen2Config(
            vocab_size=vocab_size,
            max_position_embeddings=n_positions,
            hidden_size=n_embd,
            intermediate_size=4 * n_embd,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            num_key_value_heads=n_head,
            use_cache=True, # Must be True for generation
        )
        self.name = f"qwen_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_positions = n_positions

        self._backbone = Qwen2ForCausalLM(configuration)

    def forward(self, xs, ys=None, task_type="autoregressive"):
        return self._backbone(input_ids=xs).logits

    def generate(self, prefix, **kwargs):
        """Wraps the backbone's generate method."""
        return self._backbone.generate(input_ids=prefix, **kwargs)


class LlamaModel(nn.Module):
    """Autoregressive model using Llama backbone."""
    def __init__(self, vocab_size, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(LlamaModel, self).__init__()
        configuration = LlamaConfig(
            vocab_size=vocab_size,
            max_position_embeddings=n_positions,
            hidden_size=n_embd,
            intermediate_size=4 * n_embd,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            num_key_value_heads=n_head,
            use_cache=True, 
        )
        self.name = f"llama_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_positions = n_positions

        self._backbone = LlamaForCausalLM(configuration)

    def forward(self, xs, ys=None, task_type="autoregressive"):
        return self._backbone(input_ids=xs).logits

    def generate(self, prefix, **kwargs):
        """Wraps the backbone's generate method."""
        return self._backbone.generate(input_ids=prefix, **kwargs)


class DreamDlmModel(nn.Module):
    """Diffusion-style model using Dream backbone."""
    def __init__(self, vocab_size, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(DreamDlmModel, self).__init__()
        self.family = "dream"
        
        # The vocab_size passed here is already incremented by 1 for the [MASK] token
        mask_token_id = vocab_size - 1
        # The '$' token ID from the tokenizer is at num_nodes + 3.
        # We need to be careful assuming vocab structure, but based on legacy:
        # vocab_size passed to this function = (num_nodes + 4) + 1.
        # So mask_token_id = num_nodes + 4.
        # pad_token_id is usually '$'.
        pad_token_id = vocab_size - 2

        configuration = DreamConfig(
            vocab_size=vocab_size,
            max_position_embeddings=n_positions,
            hidden_size=n_embd,
            intermediate_size=4 * n_embd,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            num_key_value_heads=n_head,
            use_cache=False,
            # Set pad_token_id to the ID of the '$' token to avoid collision with node 0.
            # Set mask_token_id to our new dedicated ID.
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            # Set a default eos_token_id as well for the generation canvas
            eos_token_id=0,
        )
        self.name = f"dream_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_positions = n_positions
        self.mask_token_id = mask_token_id # Store for use in training

        # The DreamModel from dllm is the main component
        self._backbone = DreamModel(configuration)

    def forward(self, xs, ys=None, task_type="diffusion"):
        return self._backbone(input_ids=xs).logits

    def generate(self, prefix, **kwargs):
        """
        Wraps the DreamSampler.sample method for diffusion-based generation.
        """
        from dllm.pipelines.dream.sampler import DreamSampler, DreamSamplerConfig
        
        # The DreamSampler needs a tokenizer-like object with a mask_token_id.
        class DreamTokenizerWrapper:
            def __init__(self, mask_token_id, eos_token_id):
                self.mask_token_id = mask_token_id
                self.eos_token_id = eos_token_id
        
        tokenizer_wrapper = DreamTokenizerWrapper(
            mask_token_id=self.mask_token_id,
            eos_token_id=self._backbone.config.eos_token_id
        )

        # Build prompts list
        prompts = [p for p in prefix]
        
        # Create sampler
        sampler = DreamSampler(model=self._backbone, tokenizer=tokenizer_wrapper)
        
        # Prepare sampling arguments
        sampling_kwargs = {
            'steps': kwargs.get('steps', 50),
            'alg': kwargs.get('alg', 'entropy'),
            'temperature': kwargs.get('temperature', 0.0),
            'top_k': kwargs.get('top_k', 50), # Default to 50 if not specified
            'max_new_tokens': kwargs.get('max_new_tokens', 20),
            'return_dict': kwargs.get('return_dict', False),
        }

        # The DreamSampler.sample method takes a list of prompts
        generated_output = sampler.sample(
            inputs=prompts,
            **sampling_kwargs
        )
        
        return generated_output


class LladaDlmModel(nn.Module):
    """Diffusion-style model using LLaDA backbone."""
    def __init__(self, vocab_size, n_positions, n_embd=128, n_layer=12, n_head=4, dropout=0.1):
        super(LladaDlmModel, self).__init__()
        self.family = "llada"
        
        # The vocab_size passed here is already incremented by 1 for the [MASK] token
        mask_token_id = vocab_size - 1
        pad_token_id = vocab_size - 2

        # LLaDAConfig uses 'd_model' instead of 'hidden_size' (handled by property)
        # and 'max_sequence_length' instead of 'max_position_embeddings'
        configuration = LLaDAConfig(
            vocab_size=vocab_size,
            max_sequence_length=n_positions,
            d_model=n_embd,
            mlp_ratio=4,
            n_layers=n_layer,
            n_heads=n_head,
            use_cache=False,
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            eos_token_id=0, # Default EOS
            block_type=BlockType.llama, # Use Llama blocks as per request/convention
            activation_type="silu", # Use SiLU because LLaDALlamaBlock manually implements the Gating, preventing double splitting by SwiGLU class
            rope=True,
            alibi=False,
            attention_dropout=dropout,
            residual_dropout=dropout,
            embedding_dropout=dropout,
        )
        self.name = f"llada_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_positions = n_positions
        self.mask_token_id = mask_token_id

        self._backbone = LLaDAModelLM(configuration)

    def forward(self, xs, ys=None, task_type="diffusion"):
        # LLaDAModelLM returns CausalLMOutputWithPast, we want logits
        return self._backbone(input_ids=xs).logits

    def generate(self, prefix, **kwargs):
        """
        Wraps the MDLMSampler for LLaDA generation.
        """
        # Sampler needs a tokenizer-like object
        class LLaDATokenizerWrapper:
            def __init__(self, mask_token_id, eos_token_id, bos_token_id=0):
                self.mask_token_id = mask_token_id
                self.eos_token_id = eos_token_id
                self.bos_token_id = bos_token_id
            
            def decode(self, token_ids, skip_special_tokens=False):
                # Placeholder decode if needed by sampler logging, unlikely for raw generation
                return ""
                
        tokenizer_wrapper = LLaDATokenizerWrapper(
            mask_token_id=self.mask_token_id,
            eos_token_id=self._backbone.config.eos_token_id
        )

        sampler = MDLMSampler(model=self._backbone.model, tokenizer=tokenizer_wrapper)
        
        # Prepare inputs as list of tensors
        prompts = [p for p in prefix]
        
        # Sampling arguments
        steps = kwargs.get('steps', 64)
        max_new_tokens = kwargs.get('max_new_tokens', 20)
        
        # MDLMSampler.sample expects specific args
        generated_ids = sampler.sample(
            inputs=prompts,
            steps=steps,
            max_new_tokens=max_new_tokens,
            # Block size for semi-autoregressive generation, 0 or matches max_new_tokens for full diffusion
            block_size=kwargs.get('block_size', 0) if kwargs.get('block_size', 0) > 0 else max_new_tokens,
            temperature=kwargs.get('temperature', 0.0),
            cfg_scale=kwargs.get('cfg_scale', 0.0),
            right_shift_logits=True,
        )
        
        return generated_ids

class JigsawDlmModel(nn.Module):
    """Wrapper for Jigsaw Model."""
    def __init__(self, vocab_size, n_positions, n_embd=128, n_layer=12, n_head=4, block_size=4, dropout=0.1):
        super(JigsawDlmModel, self).__init__()
        self.family = "jigsaw"
        if JigsawModel is None:
             raise ImportError("JigsawModel could not be imported. Ensure multi-diffusion is in path.")
             
        self.mask_token_id = vocab_size - 1
        self._backbone = JigsawModel(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout
        )
        self.name = self._backbone.name

    def forward(self, xs, ys=None, task_type="jigsaw"):
        # Jigsaw model handles masking internally and returns (loss, logits)
        return self._backbone(xs, ys, task_type=task_type)

    def generate(self, prefix, **kwargs):
        # Forward generation parameters to Jigsaw generate
        return self._backbone.generate(prefix, **kwargs)

class ScatterDlmModel(nn.Module):
    """Wrapper for Scatter Model."""
    def __init__(self, vocab_size, n_positions, n_embd=128, n_layer=12, n_head=4, block_size=4, dropout=0.1, family="scatter"):
        super(ScatterDlmModel, self).__init__()
        self.family = family
        if ScatterModel is None:
             raise ImportError("ScatterModel could not be imported. Ensure multi-diffusion is in path.")
        
        self.mask_token_id = vocab_size - 1
        
        # Unified ScatterModel handles both generic and Sudoku logic internally
        self._backbone = ScatterModel(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout
        )
        self.name = self._backbone.name

    def forward(self, xs, ys=None, task_type="scatter"):
        # Scatter model handles masking internally and returns (loss, logits)
        return self._backbone(xs, ys, task_type=task_type)

    def generate(self, prefix, **kwargs):
        # Forward generation parameters to Scatter generate
        return self._backbone.generate(prefix, **kwargs)

class BlockDlmModel(nn.Module):
    """Wrapper for Block Diffusion Model."""
    def __init__(self, vocab_size, n_positions, n_embd=128, n_layer=12, n_head=4, block_size=4, dropout=0.1):
        super(BlockDlmModel, self).__init__()
        self.family = "block"
        if BlockDiffusionModel is None:
             raise ImportError("BlockDiffusionModel could not be imported. Ensure multi-diffusion is in path.")
             
        self.mask_token_id = vocab_size - 1
        self._backbone = BlockDiffusionModel(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout
        )
        self.name = self._backbone.name

    def forward(self, xs, ys=None, task_type="block"):
        return self._backbone(xs, ys, task_type=task_type)

    def generate(self, prefix, **kwargs):
        return self._backbone.generate(prefix, **kwargs)