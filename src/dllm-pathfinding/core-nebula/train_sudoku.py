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
from dataset import get_sudoku_dataset, sequence_accuracy

try:
    from dllm.pipelines.dream.models.configuration_dream import DreamConfig
    from dllm.pipelines.dream.models.modeling_dream import DreamModel
    from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig
    from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM
    from dllm.core.schedulers import LinearAlphaScheduler
except ImportError as e:
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
    
    # Sudoku output length is always 81
    TARGET_LEN = 81

    for i, (xs, ys) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        # In SudokuDataset, ys has -100 for prefix.
        prefix_len_masked = (ys[0] == -100).sum().item() 
        true_prefix_len = prefix_len_masked + 1

        prefix = xs[:, :true_prefix_len].cuda()
        
        # True full sequence: xs is input (without last token of target), ys has last token.
        # However, cleaner to just reconstruct from xs and ys target part.
        # xs: [p1...=...t_last-1]
        # ys: [p2...=...t_last]
        # We need full sequence [p1...=...t1...t_last]
        # xs has everything except the very last token of the target.
        # ys has the very last token at the end.
        true_path_full = torch.cat([xs, ys[:, -1].unsqueeze(1)], dim=1).cuda()

        # Prepare generation arguments
        gen_kwargs = {
            "max_new_tokens": TARGET_LEN,
        }
        
        if args.model.family not in ['dream', 'llada']:
             gen_kwargs["pad_token_id"] = loader.dataset.tokenizer.encoder.get("$", -1)

        generated_tokens = model.generate(prefix, **gen_kwargs)

        if args.model.family in ['dream', 'llada']:
            # The dream/llada generate function returns a canvas (potentially padded or aligned).
            # We slice the relevant part from the end to ensure we compare the solution.
            generated_tokens = generated_tokens[:, -true_path_full.shape[1]:]

        # Compare only the target part (the 81 digits)
        # generated_tokens matches full sequence length usually?
        # If model.generate returns full sequence including prefix.
        if generated_tokens.shape[1] == true_path_full.shape[1]:
             is_correct = torch.all(generated_tokens == true_path_full, dim=1)
        else:
             # If lengths mismatch, something is wrong, count as failure
             is_correct = torch.zeros(xs.size(0), device=xs.device, dtype=torch.bool)

        total_correct += is_correct.sum().item()
        total_count += xs.size(0)

    model.train()
    return total_correct / total_count if total_count > 0 else 0.0

@torch.no_grad()
def compute_test_loss(model, loader, args, max_batches=None):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    num_batches = 0
    
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
            diffusion_ys[~masked_indices] = -100 

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
    print(f"Loading Sudoku Datasets from {args.data.train_data_path} and {args.data.test_data_path}")
    train_dataset = get_sudoku_dataset(args.data.train_data_path)
    test_dataset = get_sudoku_dataset(args.data.test_data_path)

    train_loader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.training.batch_size, shuffle=True)

    # 2. Update Config and Build Model
    args.model.vocab_size = train_dataset.tokenizer.vocab_size
    print(f"Vocab size set to: {args.model.vocab_size}")
    
    if args.model.family in ['dream', 'llada']:
        args.training.task_type = 'diffusion'
    else:
        args.training.task_type = 'autoregressive'
    
    check_dllm_availability(args.model.family)

    model = build_model(args.model)
    model.cuda()
    model.train()

    # 3. Setup Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.training.learning_rate, weight_decay=args.training.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.training.dtype == 'float16'))
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16 if args.training.dtype == 'float16' else torch.bfloat16)

    if args.training.task_type == 'diffusion':
        scheduler = LinearAlphaScheduler()
        mask_token_id = model.mask_token_id
        time_epsilon = 1e-3

    # Log setup
    import shutil
    local_log_dir = "./tmp_logs"
    if not os.path.exists(local_log_dir):
        os.makedirs(local_log_dir)
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    log_filename = "train_log.jsonl"
    local_log_path = os.path.join(local_log_dir, log_filename)
    oss_log_path = os.path.join(args.out_dir, log_filename)
    
    log_file = open(local_log_path, "a")

    try:
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
            
            # Loss Calculation (Same as train.py)
            t = torch.tensor([0.0], device=xs.device) 
            
            if args.training.task_type == 'diffusion':
                b_size, _ = xs.shape
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
                    output = torch.cat([output[:, :1], output[:, :-1]], dim=1)
                    loss = loss_fn(output.view(-1, output.size(-1)), diffusion_ys.view(-1))
            else:
                with ctx:
                    output = model(xs, ys, task_type=args.training.task_type)
                    loss = loss_fn(output.view(-1, output.size(-1)), ys.view(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            log_entry = {
                "step": i,
                "loss": loss.item(),
                "t_mean": t.mean().item(),
                "lr": lr
            }
            log_file.write(json.dumps(log_entry) + "\n")
            pbar.set_description(f"Loss: {loss.item():.4f}, LR: {lr:.6f}")

            # Evaluation
            if i > 0 and i % args.wandb.log_every_steps == 0 and not args.test_run:
                # Basic metrics
                metrics = {
                    "train_loss": loss.item(),
                    "learning_rate": lr,
                }
                
                # Periodic Full Evaluation
                if i % args.training.eval_every_steps == 0:
                    try:
                        shutil.copy2(local_log_path, oss_log_path)
                    except Exception as e:
                        print(f"Warning: Failed to sync logs: {e}")

                    test_batches = args.training.test_batches
                    gen_batches = max(1, int(test_batches // 4)) if test_batches else 4
                    
                    # Test Set Metrics
                    test_loss = compute_test_loss(model, test_loader, args, max_batches=test_batches)
                    test_gen_acc = evaluate_generative(model, test_loader, args, max_batches=gen_batches)
                    
                    # Train Set Metrics (Evaluate on subset of train)
                    train_gen_acc = evaluate_generative(model, train_loader, args, max_batches=gen_batches)
                    
                    metrics.update({
                        "test_loss": test_loss,
                        "test_generate_acc": test_gen_acc,
                        "train_generate_acc": train_gen_acc
                    })
                    
                    print(f"Step {i}: Test Acc: {test_gen_acc:.4f}, Train Acc: {train_gen_acc:.4f}")

                    # Checkpointing
                    last_model_dir = os.path.join(args.out_dir, "last")
                    if not os.path.exists(last_model_dir): os.makedirs(last_model_dir)
                    model._backbone.save_pretrained(last_model_dir)

                    if test_gen_acc > best_acc:
                        best_acc = test_gen_acc
                        best_model_dir = os.path.join(args.out_dir, "best_gen_acc")
                        if not os.path.exists(best_model_dir): os.makedirs(best_model_dir)
                        model._backbone.save_pretrained(best_model_dir)

                wandb.log(metrics, step=i)
        
        last_model_dir = os.path.join(args.out_dir, "last")
        if not os.path.exists(last_model_dir): os.makedirs(last_model_dir)
        model._backbone.save_pretrained(last_model_dir)

    finally:
        log_file.close()
        try:
            shutil.copy2(local_log_path, oss_log_path)
        except Exception:
            pass

def main(config):
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
        config.out_dir = os.path.join(config.out_dir, "test_run")
        if not os.path.exists(config.out_dir):
            os.makedirs(config.out_dir)

    train(config)
    print("Training complete.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")
    
    if hasattr(parser, "add_config_path_arg") and callable(parser.add_config_path_arg):
        parser.add_config_path_arg("--config")
    else:
        parser.add_argument("--config", type=str, dest="config_path_manual", help="Path to a YAML config file")
    
    args = parser.parse_args()
    
    if hasattr(args, "config_path_manual") and args.config_path_manual:
        config = Config.load(args.config_path_manual)
    else:
        config = args.config
    
    # Create out_dir if not test_run
    if not config.test_run:
        if config.training.resume_id is None:
            config.training.resume_id = str(uuid.uuid4())
        out_dir = os.path.join(config.out_dir, config.training.resume_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        config.out_dir = out_dir
        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(dataclasses.asdict(config), yaml_file, default_flow_style=False)

    main(config)
