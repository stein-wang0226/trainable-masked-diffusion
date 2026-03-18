import os
import sys
import uuid
import math
import json
import dataclasses
from contextlib import nullcontext

import torch
import yaml
from tqdm import tqdm
from simple_parsing import ArgumentParser
from torch.utils.data import DataLoader
import wandb
import shutil

# Ensure we can import from local modules, dllm, and multi-diffusion
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.abspath(os.path.join(current_dir, '../dllm')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../multi-diffusion')))

from config import Config
from models import build_model
from dataset import get_dataset, get_sudoku_dataset, sequence_accuracy

try:
    from dllm.core.schedulers import LinearAlphaScheduler
except ImportError:
    pass

def check_dllm_availability(family):
    if family in ['dream', 'llada', 'jigsaw']:
        # These families depend on dllm components (LLaDA backbone for Jigsaw)
        try:
            import dllm
        except ImportError:
             raise ImportError(
                f"The 'dllm' library is required for the '{family}' model family but could not be imported. "
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
    """Computes sequence-level and token-level accuracy using generative evaluation."""
    model.eval()
    total_correct = 0
    total_count = 0
    total_token_correct = 0
    total_token_count = 0
    
    # Determine generation length based on task
    # Dynamic calculation based on ground truth length ensures alignment
    # prefix_len_masked is calculated inside loop, but we need it here?
    # No, we can calc per batch.
    
    for i, (xs, ys) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        # Identify prefix length
        # For both datasets, ys has -100 for prefix positions.
        prefix_len_masked = (ys[0] == -100).sum().item()
        true_prefix_len = prefix_len_masked + 1

        prefix = xs[:, :true_prefix_len].cuda()
        
        # Reconstruct true full sequence for comparison
        # xs is [p...t_last-1], ys is [p+1...t_last]
        true_path_full = torch.cat([xs, ys[:, -1].unsqueeze(1)], dim=1).cuda()

        # Dynamic gen_len
        gen_len = true_path_full.shape[1] - true_prefix_len

        gen_kwargs = {
            "max_new_tokens": gen_len,
        }
        
        # Add pad_token_id for AR models if needed
        if args.model.family not in ['dream', 'llada', 'jigsaw']:
             gen_kwargs["pad_token_id"] = loader.dataset.tokenizer.encoder.get("$", -1)

        generated_tokens = model.generate(prefix, **gen_kwargs)

        # Truncate/Align output for comparison
        # Most models (dream, llada, jigsaw, block, scatter) return the full sequence (prefix + generated)
        # Standard AR models (qwen, llama) return only the generated tokens or the full sequence depending on implementation.
        # We ensure Start-alignment and truncate to the expected Ground Truth length.
        if generated_tokens.shape[1] > true_path_full.shape[1]:
            generated_tokens = generated_tokens[:, :true_path_full.shape[1]]
        elif generated_tokens.shape[1] < true_path_full.shape[1]:
            # This handles cases where AR models might return only generated tokens
            # We prepend the prefix to align them for the sequence-level check
            if generated_tokens.shape[1] == gen_len:
                generated_tokens = torch.cat([prefix, generated_tokens], dim=1)
        
        # Compare Sequence Level
        if generated_tokens.shape[1] == true_path_full.shape[1]:
             is_correct = torch.all(generated_tokens == true_path_full, dim=1)
        else:
             # Length mismatch
             is_correct = torch.zeros(xs.size(0), device=xs.device, dtype=torch.bool)
        
        # Compare Token Level (Only Response Part)
        min_len = min(generated_tokens.shape[1], true_path_full.shape[1])
        if min_len > true_prefix_len:
            # Slicing from true_prefix_len to compare only the generated response
            gen_resp = generated_tokens[:, true_prefix_len:min_len]
            true_resp = true_path_full[:, true_prefix_len:min_len]
            token_matches = (gen_resp == true_resp).sum().item()
            total_token_correct += token_matches
            
            # Count only target tokens
            total_token_count += true_resp.numel()

        total_correct += is_correct.sum().item()
        total_count += xs.size(0)

    model.train()
    seq_acc = total_correct / total_count if total_count > 0 else 0.0
    token_acc = total_token_correct / total_token_count if total_token_count > 0 else 0.0
    return seq_acc, token_acc

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
            diffusion_ys[~masked_indices] = -100 

            with ctx:
                output = model(noised_input_ids, task_type='diffusion')
                output = torch.cat([output[:, :1], output[:, :-1]], dim=1)
                loss = loss_fn(output.view(-1, output.size(-1)), diffusion_ys.view(-1))
        
        elif args.training.task_type == 'jigsaw':
            # Jigsaw model returns (loss, logits)
            with ctx:
                loss, _ = model(xs, ys, task_type='jigsaw')
        
        else:
            # Autoregressive
            with ctx:
                output = model(xs, ys, task_type='autoregressive')
                loss = loss_fn(output.view(-1, output.size(-1)), ys.view(-1))
        
        total_loss += loss.item()
        num_batches += 1

    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0

def train(args):
    """Main training and evaluation loop."""
    best_acc = -1.0
    
    # 1. Dataset Selection
    if args.data.task == "sudoku":
        print(f"Loading Sudoku Datasets...")
        train_dataset = get_sudoku_dataset(args.data.train_data_path)
        test_dataset = get_sudoku_dataset(args.data.test_data_path)
    else:
        print(f"Loading Graph Pathfinding Datasets (Nodes: {args.data.num_nodes})...")
        train_dataset = get_dataset(args.data.train_data_path, args.data.num_nodes)
        test_dataset = get_dataset(args.data.test_data_path, args.data.num_nodes)

    train_loader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.training.batch_size, shuffle=True)

    # 2. Update Config and Build Model
    args.model.vocab_size = train_dataset.tokenizer.vocab_size
    print(f"Vocab size set to: {args.model.vocab_size}")
    
    # Pass block_size from DataConfig to ModelConfig if needed (for Jigsaw)
    if hasattr(args.data, 'block_size') and args.data.block_size > 0:
        args.model.block_size = args.data.block_size
        print(f"Passing block_size={args.model.block_size} to model builder.")

    # Determine Task Type
    if args.model.family in ['dream', 'llada']:
        args.training.task_type = 'diffusion'
    elif args.model.family in ['jigsaw', 'bad_ar', 'rbo_ar', 'bop_ar', 'scatter', 'block']: # Future proofing
        args.training.task_type = 'jigsaw' # Or generic 'multi'
    else:
        args.training.task_type = 'autoregressive'
    
    print(f"Model Family: {args.model.family}, Task Type: {args.training.task_type}")
    
    check_dllm_availability(args.model.family)

    model = build_model(args.model)
    model.cuda()
    model.train()

    # 3. Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.training.learning_rate, weight_decay=args.training.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.training.dtype == 'float16'))
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16 if args.training.dtype == 'float16' else torch.bfloat16)

    # Diffusion setup
    scheduler = None
    mask_token_id = None
    time_epsilon = 1e-3
    if args.training.task_type == 'diffusion':
        scheduler = LinearAlphaScheduler()
        mask_token_id = model.mask_token_id

    # Logging
    local_log_dir = "./tmp_logs"
    if not os.path.exists(local_log_dir): os.makedirs(local_log_dir)
    if not os.path.exists(args.out_dir): os.makedirs(args.out_dir)
        
    log_filename = "train_log.jsonl"
    local_log_path = os.path.join(local_log_dir, log_filename)
    oss_log_path = os.path.join(args.out_dir, log_filename)
    
    log_file = open(local_log_path, "a")

    # 4. Training Loop
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
            t_log = 0.0

            if args.training.task_type == 'diffusion':
                # --- Diffusion Training Logic ---
                b_size = xs.shape[0]
                full_sequence = torch.cat([xs, ys[:, -1].unsqueeze(1)], dim=1)
                seq_len = full_sequence.shape[1]

                t = time_epsilon + (1 - time_epsilon) * torch.rand(b_size, device=xs.device)
                p_mask = 1 - scheduler(t).unsqueeze(1).expand(b_size, seq_len)
                t_log = t.mean().item()

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
            
            elif args.training.task_type == 'jigsaw':
                # --- Jigsaw / Multi-Diffusion Logic ---
                with ctx:
                    # Model handles masking and loss calculation
                    loss, _ = model(xs, ys, task_type='jigsaw')
            
            else:
                # --- Autoregressive Logic ---
                with ctx:
                    output = model(xs, ys, task_type='autoregressive')
                    loss = loss_fn(output.view(-1, output.size(-1)), ys.view(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            log_entry = {"step": i, "loss": loss.item(), "lr": lr, "t_mean": t_log}
            log_file.write(json.dumps(log_entry) + "\n")
            pbar.set_description(f"Loss: {loss.item():.4f}, LR: {lr:.6f}")

            # 5. Evaluation
            if i > 0 and i % args.wandb.log_every_steps == 0 and not args.test_run:
                metrics = {"train_loss": loss.item(), "learning_rate": lr}
                
                if i % args.training.eval_every_steps == 0:
                    try:
                        shutil.copy2(local_log_path, oss_log_path)
                    except Exception: pass

                    test_batches = args.training.test_batches
                    gen_batches = max(1, int(test_batches // 4)) if test_batches else 4
                    
                    test_loss = compute_test_loss(model, test_loader, args, max_batches=test_batches)
                    test_gen_seq_acc, test_gen_token_acc = evaluate_generative(model, test_loader, args, max_batches=gen_batches)
                    train_gen_seq_acc, train_gen_token_acc = evaluate_generative(model, train_loader, args, max_batches=gen_batches)
                    
                    metrics.update({
                        "test_loss": test_loss,
                        "test_generate_acc": test_gen_seq_acc,
                        "test_token_acc": test_gen_token_acc,
                        "train_generate_acc": train_gen_seq_acc,
                        "train_token_acc": train_gen_token_acc
                    })
                    print(f"Step {i}: Test Seq Acc: {test_gen_seq_acc:.4f}, Test Token Acc: {test_gen_token_acc:.4f}")

                    # Checkpointing
                    last_model_dir = os.path.join(args.out_dir, "last")
                    if not os.path.exists(last_model_dir): os.makedirs(last_model_dir)
                    model._backbone.save_pretrained(last_model_dir)

                    if test_gen_seq_acc > best_acc:
                        best_acc = test_gen_seq_acc
                        best_model_dir = os.path.join(args.out_dir, "best_gen_acc")
                        if not os.path.exists(best_model_dir): os.makedirs(best_model_dir)
                        model._backbone.save_pretrained(best_model_dir)

                wandb.log(metrics, step=i)
        
        # Save final model
        last_model_dir = os.path.join(args.out_dir, "last")
        if not os.path.exists(last_model_dir): os.makedirs(last_model_dir)
        model._backbone.save_pretrained(last_model_dir)

    finally:
        log_file.close()
        try:
            shutil.copy2(local_log_path, oss_log_path)
        except Exception: pass

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
        if not os.path.exists(config.out_dir): os.makedirs(config.out_dir)

    train(config)

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
    
    if not config.test_run:
        if config.training.resume_id is None:
            config.training.resume_id = str(uuid.uuid4())
        out_dir = os.path.join(config.out_dir, config.training.resume_id)
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        config.out_dir = out_dir
        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(dataclasses.asdict(config), yaml_file, default_flow_style=False)

    main(config)
