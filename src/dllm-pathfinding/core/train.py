import os
import sys
import uuid
import math
import json
from contextlib import nullcontext

import torch
import yaml
import dataclasses
from tqdm import tqdm
from simple_parsing import ArgumentParser
from torch.utils.data import DataLoader
import wandb

# Ensure we can import from local modules and dllm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add the path to dllm which is at ../dllm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../dllm')))

from config import Config
from models import build_model
from dataset import get_dataset, sequence_accuracy, token_accuracy

try:
    from dllm.pipelines.dream.models.configuration_dream import DreamConfig
    from dllm.pipelines.dream.models.modeling_dream import DreamModel
    from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig
    from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM
    from dllm.core.schedulers import LinearAlphaScheduler
except ImportError as e:
    # Fallback or placeholder if dllm is not strictly installed as a package
    # This expects the user to have set up PYTHONPATH correctly
    import sys
    print(f"Warning: dllm import failed. Diffusion training will fail. Error: {e}")
    print(f"Current sys.path: {sys.path}")
    # Try to find dllm in parent directories if not found (Hack for development)
    # This is handled usually by the training script, but let's be safe.
    pass

def check_dllm_availability(family):
    if family == 'dream' and 'DreamModel' not in globals():
        raise ImportError(
            "The 'dllm' library is required for the 'dream' model family but could not be imported. "
            "Please ensure it is installed or in your PYTHONPATH."
        )
    if family == 'llada' and 'LLaDAModelLM' not in globals():
        raise ImportError(
            "The 'dllm' library is required for the 'llada' model family but could not be imported. "
            "Please ensure it is installed or in your PYTHONPATH."
        )

# LR scheduler from the original repository
def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

@torch.no_grad()
def evaluate_generative(model, loader, args, max_batches=None):
    """Computes sequence-level accuracy using generative evaluation (no teacher-forcing)."""
    model.eval()
    total_correct = 0
    total_count = 0
    for i, (xs, ys) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        prefix_len_masked = (ys[0] == -100).sum().item()  # This is len(prefix) - 1
        true_prefix_len = prefix_len_masked + 1

        prefix = xs[:, :true_prefix_len].cuda()
        
        true_path_full = torch.cat([xs, ys[:, -1].unsqueeze(1)], dim=1).cuda()

        # Prepare generation arguments
        gen_kwargs = {
            "max_new_tokens": args.data.path_len, # Use data config for path len
        }
        # For AR models, we might need a pad_token_id if sequences had variable length
        # Assuming '$' is pad/special token
        if args.model.family not in ['dream', 'llada']:
             gen_kwargs["pad_token_id"] = loader.dataset.tokenizer.encoder.get("$", -1)

        generated_tokens = model.generate(prefix, **gen_kwargs)

        # For dream/llada model, the output may be longer due to canvas padding, truncate to match.
        if args.model.family in ['dream', 'llada']:
            # The generate function returns a canvas/sequence.
            # We need to slice the relevant part from the end to compare.
            generated_tokens = generated_tokens[:, -true_path_full.shape[1]:]

        is_correct = torch.all(generated_tokens == true_path_full, dim=1)
        total_correct += is_correct.sum().item()
        total_count += xs.size(0)

    model.train()
    return total_correct / total_count if total_count > 0 else 0.0

@torch.no_grad()
def evaluate_forced(model, loader, max_batches=None):
    """Computes sequence-level accuracy using teacher-forcing.""" 
    model.eval()
    total_acc = 0
    num_batches = 0
    for i, (xs, ys) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        xs, ys = xs.cuda(), ys.cuda()
        output = model(xs, ys)
        total_acc += sequence_accuracy(output, ys)
        num_batches += 1
    model.train()
    return total_acc / num_batches if num_batches > 0 else 0.0

@torch.no_grad()
def compute_test_loss(model, loader, args, max_batches=None):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    num_batches = 0
    
    # Diffusion setup
    scheduler = None
    mask_token_id = None
    time_epsilon = 1e-3
    if args.training.task_type == 'diffusion':
        scheduler = LinearAlphaScheduler()
        mask_token_id = model.mask_token_id

    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16 if args.training.dtype == 'float16' else torch.bfloat16)

    for i, (xs, ys) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        xs, ys = xs.cuda(), ys.cuda()
        
        if args.training.task_type == 'diffusion':
            b_size = xs.shape[0]
            full_sequence = torch.cat([xs, ys[:, -1].unsqueeze(1)], dim=1)
            seq_len = full_sequence.shape[1]

            t = time_epsilon + (1 - time_epsilon) * torch.rand(b_size, device=xs.device)
            p_mask = 1 - scheduler(t).unsqueeze(1).expand(b_size, seq_len)

            prefix_len = (ys[0] == -100).sum().item() + 1 
            
            target_part_mask = torch.zeros_like(full_sequence, dtype=torch.bool)
            target_part_mask[:, prefix_len:] = True

            masked_indices = (torch.rand_like(full_sequence, dtype=torch.float32) < p_mask) & target_part_mask

            noised_input_ids = full_sequence.clone()
            noised_input_ids[masked_indices] = mask_token_id

            diffusion_ys = full_sequence.clone()
            diffusion_ys[~masked_indices] = -100 # Ignore non-masked positions

            with ctx:
                output = model(noised_input_ids, task_type='diffusion')
                output = torch.cat([output[:, :1], output[:, :-1]], dim=1)
                loss = loss_fn(output.view(-1, output.size(-1)), diffusion_ys.view(-1))
        else:
            with ctx:
                output = model(xs, ys, task_type=args.training.task_type)
                loss = loss_fn(output.view(-1, output.size(-1)), ys.view(-1))
        
        total_loss += loss.item()
        num_batches += 1

    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0

def train(args):
    """Main training and evaluation loop."""
    best_acc = -1.0
    
    # 1. Create Datasets and Dataloaders
    train_dataset = get_dataset(args.data.train_data_path, args.data.num_nodes)
    test_dataset = get_dataset(args.data.test_data_path, args.data.num_nodes)

    train_loader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.training.batch_size, shuffle=True)

    # 2. Update Config and Build Model
    # Update vocab_size from tokenizer
    args.model.vocab_size = train_dataset.tokenizer.vocab_size
    print(f"Vocab size set to: {args.model.vocab_size}")
    
    # Determine task_type from model family if not explicitly set (optional safety)
    if args.model.family in ['dream', 'llada']:
        args.training.task_type = 'diffusion'
    else:
        args.training.task_type = 'autoregressive'
    
    check_dllm_availability(args.model.family)

    model = build_model(args.model)
    model.cuda()
    model.train()

    # 3. Setup Optimizer, Loss, and Mixed Precision
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.training.learning_rate, weight_decay=args.training.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.training.dtype == 'float16'))
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16 if args.training.dtype == 'float16' else torch.bfloat16)

    # Diffusion-specific setup
    if args.training.task_type == 'diffusion':
        scheduler = LinearAlphaScheduler()
        mask_token_id = model.mask_token_id
        time_epsilon = 1e-3

    # Setup Log File
    import shutil
    # Use a local temporary directory for logging to avoid slow OSS writes
    local_log_dir = "./tmp_logs"
    if not os.path.exists(local_log_dir):
        os.makedirs(local_log_dir)
    
    # Ensure output directory exists (on OSS)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    log_filename = "train_log.jsonl"
    local_log_path = os.path.join(local_log_dir, log_filename)
    oss_log_path = os.path.join(args.out_dir, log_filename)
    
    log_file = open(local_log_path, "a")

    try:
        # 4. Training Loop
        pbar = tqdm(range(args.training.max_steps))
        data_iter = iter(train_loader)
        
        warmup_iters = args.training.warmup_steps
        lr_decay_iters = args.training.max_steps
        min_lr = 1e-5

        for i in pbar:
            lr = get_lr(i, args.training.learning_rate, warmup_iters, lr_decay_iters, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            try:
                xs, ys = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                xs, ys = next(data_iter)

            xs, ys = xs.cuda(), ys.cuda()

            t = torch.tensor([0.0], device=xs.device) # Default t for logging
            
            if args.training.task_type == 'diffusion':
                # --- Diffusion Loss Calculation ---
                b_size, seq_len_minus_1 = xs.shape
                
                # Reconstruct full sequence
                full_sequence = torch.cat([xs, ys[:, -1].unsqueeze(1)], dim=1)
                seq_len = full_sequence.shape[1]

                t = time_epsilon + (1 - time_epsilon) * torch.rand(b_size, device=xs.device)
                p_mask = 1 - scheduler(t).unsqueeze(1).expand(b_size, seq_len)

                prefix_len = (ys[0] == -100).sum().item() + 1 
                
                target_part_mask = torch.zeros_like(full_sequence, dtype=torch.bool)
                target_part_mask[:, prefix_len:] = True

                masked_indices = (torch.rand_like(full_sequence, dtype=torch.float32) < p_mask) & target_part_mask

                noised_input_ids = full_sequence.clone()
                noised_input_ids[masked_indices] = mask_token_id

                diffusion_ys = full_sequence.clone()
                diffusion_ys[~masked_indices] = -100 

                with ctx:
                    output = model(noised_input_ids, task_type='diffusion')
                    # Shift output for prediction
                    output = torch.cat([output[:, :1], output[:, :-1]], dim=1)
                    loss = loss_fn(output.view(-1, output.size(-1)), diffusion_ys.view(-1))
            else:
                # --- Autoregressive Loss Calculation ---
                with ctx:
                    output = model(xs, ys, task_type=args.training.task_type)
                    loss = loss_fn(output.view(-1, output.size(-1)), ys.view(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Log to file
            log_entry = {
                "step": i,
                "loss": loss.item(),
                "t_mean": t.mean().item(),
                "lr": lr
            }
            log_file.write(json.dumps(log_entry) + "\n")

            pbar.set_description(f"Loss: {loss.item():.4f}, LR: {lr:.6f}")

            # 5. Logging and Evaluation
            if i > 0 and i % args.wandb.log_every_steps == 0 and not args.test_run:
                # Basic logging
                metrics = {
                    "train_loss": loss.item(),
                    "learning_rate": lr,
                }
                
                # Periodic Evaluation
                if i % args.training.eval_every_steps == 0:
                    # Sync logs to OSS
                    try:
                        shutil.copy2(local_log_path, oss_log_path)
                    except Exception as e:
                        print(f"Warning: Failed to sync logs to OSS: {e}")

                    test_batches = args.training.test_batches
                    gen_batches = max(1, int(test_batches // 4)) if test_batches else 4
                    
                    test_generative_acc = evaluate_generative(model, test_loader, args, max_batches=gen_batches)
                    train_generative_acc = evaluate_generative(model, train_loader, args, max_batches=gen_batches)
                    
                    if args.model.family != 'dream':
                        test_forced_acc = evaluate_forced(model, test_loader, max_batches=test_batches)
                    else:
                        test_forced_acc = 0.0 # Not really applicable or comparable

                    test_loss = compute_test_loss(model, test_loader, args, max_batches=test_batches)
                    
                    metrics.update({
                        "test_loss": test_loss,
                        "test_generate_acc": test_generative_acc,
                        "train_generate_acc": train_generative_acc,
                        "test_forced_sequence_accuracy": test_forced_acc,
                    })
                    
                    print(f"Step {i}: Test Acc: {test_generative_acc:.4f}, Train Acc: {train_generative_acc:.4f}")

                    # Checkpointing logic
                    last_model_dir = os.path.join(args.out_dir, "last")
                    if not os.path.exists(last_model_dir):
                        os.makedirs(last_model_dir)
                    model._backbone.save_pretrained(last_model_dir)

                    if test_generative_acc > best_acc:
                        best_acc = test_generative_acc
                        best_model_dir = os.path.join(args.out_dir, "best_gen_acc")
                        if not os.path.exists(best_model_dir):
                            os.makedirs(best_model_dir)
                        print(f"\nNew best model found! Generative Accuracy: {best_acc:.4f}. Saving model to {best_model_dir}\n")
                        model._backbone.save_pretrained(best_model_dir)

                wandb.log(metrics, step=i)
        
        # Save the last model at the end of training
        last_model_dir = os.path.join(args.out_dir, "last")
        if not os.path.exists(last_model_dir):
            os.makedirs(last_model_dir)
        print(f"\nSaving last model to {last_model_dir}\n")
        model._backbone.save_pretrained(last_model_dir)

    finally:
        log_file.close()
        # Final sync
        try:
            shutil.copy2(local_log_path, oss_log_path)
            print(f"\nTraining log synced to {oss_log_path}")
        except Exception as e:
            print(f"Warning: Failed to sync logs to OSS: {e}")
        print(f"\nTraining log saved locally to {local_log_path}")

def main(config):
    # Setup WandB
    if not config.test_run:
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
            
        run_id = config.training.resume_id or wandb.util.generate_id()
        wandb.init(
            dir=config.out_dir,
            project=config.wandb.project,
            entity=config.wandb.entity,
            config=dataclasses.asdict(config),
            notes=config.wandb.notes,
            name=config.wandb.name,
            resume="allow",
            id=run_id
        )
        config.training.resume_id = run_id
    else:
        print("Test run mode: WandB logging disabled.")

    train(config)

    print("Training complete.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")
    
    # Support --config as an alias for --config_path
    # In some versions of simple-parsing, add_config_path_arg is a method.
    # In others, it might be a boolean or missing.
    config_path_manual = None
    if hasattr(parser, "add_config_path_arg") and callable(parser.add_config_path_arg):
        parser.add_config_path_arg("--config")
    else:
        # If not available as a method, add it manually
        parser.add_argument("--config", type=str, dest="config_path_manual", help="Path to a YAML config file")
    
    args = parser.parse_args()
    
    # Handle config loading
    if hasattr(args, "config_path_manual") and args.config_path_manual:
        config = Config.load(args.config_path_manual)
    else:
        config = args.config
    
    print(f"Running with: {config}")

    if not config.test_run:
        if config.training.resume_id is None:
            config.training.resume_id = str(uuid.uuid4())

        # Debugging OSS mount issues
        print(f"DEBUG: Target config.out_dir: {config.out_dir}")
        base_mount = "/path/to/your/data"  # TODO: Replace with your actual data mount path
        if os.path.exists(base_mount):
            print(f"DEBUG: {base_mount} exists.")
            try:
                print(f"DEBUG: Contents of {base_mount}: {os.listdir(base_mount)}")
            except Exception as e:
                print(f"DEBUG: Error listing {base_mount}: {e}")
        else:
            print(f"DEBUG: {base_mount} does NOT exist. Check OSS mounting configuration.")

        # Create sub-directory for this run
        out_dir = os.path.join(config.out_dir, config.training.resume_id)
        print(f"DEBUG: Attempting to create out_dir: {out_dir}")
        if not os.path.exists(out_dir):
            try:
                os.makedirs(out_dir)
            except OSError as e:
                print(f"DEBUG: os.makedirs failed with error: {e}")
                # Optional: try to see which part of the path failed
                curr = ""
                for part in out_dir.split(os.sep):
                    curr += os.sep + part
                    if part and not os.path.exists(curr):
                        print(f"DEBUG: Failed at path component: {curr}")
                raise e
        config.out_dir = out_dir

        # Dump config
        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(dataclasses.asdict(config), yaml_file, default_flow_style=False)
    else:
        # Use a temp dir for test runs to avoid clutter
        config.out_dir = os.path.join(config.out_dir, "test_run")
        if not os.path.exists(config.out_dir):
            os.makedirs(config.out_dir)

    main(config)