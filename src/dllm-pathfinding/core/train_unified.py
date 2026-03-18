import os
import sys
import uuid
import math
import json
import dataclasses
import shutil
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
from simple_parsing import ArgumentParser
from torch.utils.data import DataLoader
import wandb

# Ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.abspath(os.path.join(current_dir, '../dllm')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../multi-diffusion')))

from config import MultiDiffusionConfig, Config
from models import build_model
from dataset import get_dataset, get_sudoku_dataset

try:
    from dllm.core.schedulers import LinearAlphaScheduler
except ImportError:
    pass

def calculate_flops(model, batch_size, seq_len):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Approximation: 6 * N * B * L (Forward=2N, Backward=4N)
    flops = 6 * total_params * batch_size * seq_len
    return total_params, flops

def check_dllm_availability(family):
    if family in ['dream', 'llada', 'jigsaw', 'block', 'scatter']:
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
    """
    Computes sequence-level and token-level accuracy using generative evaluation.
    Handles different model return types (full sequence vs generated only).
    """
    model.eval()
    total_seq_correct = 0
    total_seq_count = 0
    total_token_correct = 0
    total_token_count = 0
    
    for i, (xs, ys) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        # xs: [B, L-1] (full sequence excluding last)
        # ys: [B, L-1] (full sequence shifted by 1, with -100 for prefix)
        
        # Identify prefix length from ys mask
        # ys[b, i] == -100 means i is part of prefix (or ignored).
        # We assume standard prefix masking: prefix is masked, target is not.
        # The first non -100 index is the start of the target prediction.
        # So prefix length is the count of -100s.
        # Note: ys is shifted. ys[i] corresponds to prediction at xs[i].
        # If ys[i] is target, then xs[i] is the last token of prefix (or prev target).
        
        # Let's rely on the assumption that prefix tokens are marked with -100 in ys.
        # prefix_len_masked = number of -100s. 
        # This is the length of the prefix in the *target* sequence.
        # The input prefix to the model should be xs[:, :prefix_len_masked+1] 
        # (since xs is shifted left by 1 relative to prediction).
        # Wait, let's verify dataset.py logic.
        # Graph: x = full[:-1], y = full[1:]. y[:len(prefix)-1] = -100.
        # So y has len(prefix)-1 masked tokens.
        # We want to feed `prefix` string to generate.
        # The prefix string length is len(prefix).
        # So we need xs[:, :len(prefix)].
        
        prefix_mask_count = (ys[0] == -100).sum().item()
        true_prefix_len = prefix_mask_count + 1 # +1 because y is shifted
        
        prefix = xs[:, :true_prefix_len].cuda()
        
        # Reconstruct full ground truth sequence [B, L]
        true_full_seq = torch.cat([xs, ys[:, -1].unsqueeze(1)], dim=1).cuda()
        
        # Calculate how many tokens to generate
        gen_len = true_full_seq.shape[1] - true_prefix_len

        gen_kwargs = {
            "max_new_tokens": gen_len,
        }
        
        # Add pad_token_id for AR models if needed
        # Dream/Llada/Multi usually handle this or don't need it for fixed canvas
        if args.model.family not in ['dream', 'llada', 'jigsaw', 'block', 'scatter']:
             gen_kwargs["pad_token_id"] = loader.dataset.tokenizer.encoder.get("$", -1)

        # GENERATE
        # Returns: [B, L_out]
        generated_output = model.generate(prefix, **gen_kwargs)

        # Standardize output to [B, L_full]
        # Most models (multi, dream, llada) return full canvas.
        # AR models might return only new tokens or full sequence. 
        # HuggingFace default is full sequence.
        
        if generated_output.shape[1] < true_full_seq.shape[1]:
            # If shorter (e.g. AR returning only new tokens), prepend prefix
            # Assuming it aligns with end of prefix
            if generated_output.shape[1] == gen_len:
                generated_output = torch.cat([prefix, generated_output], dim=1)
        
        # Truncate if longer (e.g. padded)
        if generated_output.shape[1] > true_full_seq.shape[1]:
            if args.model.family in ['dream', 'llada']:
                generated_output = generated_output[:, -true_full_seq.shape[1]:]
            else:
                generated_output = generated_output[:, :true_full_seq.shape[1]]
            
        # 1. Sequence Accuracy
        # Exact match of the full sequence (or just target part? usually full implies correct prefix preservation)
        if generated_output.shape == true_full_seq.shape:
            is_correct = torch.all(generated_output == true_full_seq, dim=1)
            total_seq_correct += is_correct.sum().item()
        else:
            # Length mismatch -> Wrong
            pass
        total_seq_count += xs.size(0)
        
        # 2. Token Accuracy (Target Part Only)
        # We compare from true_prefix_len to end
        if generated_output.shape[1] >= true_full_seq.shape[1]: 
            # Ensure valid length for slicing
            gen_target = generated_output[:, true_prefix_len:]
            true_target = true_full_seq[:, true_prefix_len:]
            
            token_matches = (gen_target == true_target).sum().item()
            total_token_correct += token_matches
            total_token_count += true_target.numel()

    model.train()
    
    seq_acc = total_seq_correct / total_seq_count if total_seq_count > 0 else 0.0
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
    
    # Check if we need external diffusion logic
    # Internal logic models: block, scatter, jigsaw
    internal_loss_families = ['block', 'scatter', 'jigsaw']
    is_internal_loss = args.model.family in internal_loss_families
    
    if args.training.task_type == 'diffusion' and not is_internal_loss:
        scheduler = LinearAlphaScheduler()
        mask_token_id = model.mask_token_id

    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16 if args.training.dtype == 'float16' else torch.bfloat16)

    for i, (xs, ys) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        xs, ys = xs.cuda(), ys.cuda()
        
        with ctx:
            if is_internal_loss:
                # Models that compute loss internally
                # They handle their own masking and logic
                loss, _ = model(xs, ys, task_type=args.model.family)
                
            elif args.training.task_type == 'diffusion':
                # External diffusion logic (Dream, LLaDA basic)
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

                output = model(noised_input_ids, task_type='diffusion')
                # Standard diffusion shift for loss calc
                output = torch.cat([output[:, :1], output[:, :-1]], dim=1)
                loss = loss_fn(output.view(-1, output.size(-1)), diffusion_ys.view(-1))
                
            else:
                # Autoregressive
                output = model(xs, ys, task_type='autoregressive')
                loss = loss_fn(output.view(-1, output.size(-1)), ys.view(-1))
        
        total_loss += loss.item()
        num_batches += 1

    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0

def train(args):
    # 1. Dataset Selection
    print(f"Loading {args.data.task.upper()} Datasets...")
    if args.data.task == "sudoku":
        train_dataset = get_sudoku_dataset(args.data.train_data_path)
        test_dataset = get_sudoku_dataset(args.data.test_data_path)
    else: # graph
        train_dataset = get_dataset(args.data.train_data_path, args.data.num_nodes)
        test_dataset = get_dataset(args.data.test_data_path, args.data.num_nodes)

    train_loader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.training.batch_size, shuffle=True)

    # 2. Setup Model Config
    args.model.vocab_size = train_dataset.tokenizer.vocab_size
    print(f"Vocab size set to: {args.model.vocab_size}")
    
    # Inject block_size if applicable
    if hasattr(args.data, 'block_size') and args.data.block_size > 0:
        args.model.block_size = args.data.block_size
        print(f"Passing block_size={args.model.block_size} to model builder.")

    # Determine Task Type and Internal Loss Flag
    internal_loss_families = ['block', 'scatter', 'jigsaw']
    is_internal_loss = args.model.family in internal_loss_families
    
    if args.model.family in ['dream', 'llada']:
        args.training.task_type = 'diffusion'
    elif is_internal_loss:
        args.training.task_type = args.model.family # e.g. 'block'
    else:
        args.training.task_type = 'autoregressive'
    
    print(f"Model Family: {args.model.family}")
    print(f"Training Task Type: {args.training.task_type}")
    print(f"Internal Loss Calculation: {is_internal_loss}")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    check_dllm_availability(args.model.family)

    model = build_model(args.model)
    model.to(device)
    model.train()

    # 3. Optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.training.learning_rate, weight_decay=args.training.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.training.dtype == 'float16' and device.type == 'cuda'))
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16 if args.training.dtype == 'float16' else torch.bfloat16) if device.type == 'cuda' else nullcontext()

    # Setup for External Diffusion
    scheduler = None
    mask_token_id = None
    time_epsilon = 1e-3
    if args.training.task_type == 'diffusion' and not is_internal_loss:
        scheduler = LinearAlphaScheduler()
        mask_token_id = model.mask_token_id

    # Logging setup
    local_log_dir = "./tmp_logs"
    if not os.path.exists(local_log_dir): os.makedirs(local_log_dir)
    if not os.path.exists(args.out_dir): os.makedirs(args.out_dir)
    
    log_filename = "train_log.jsonl"
    local_log_path = os.path.join(local_log_dir, log_filename)
    oss_log_path = os.path.join(args.out_dir, log_filename)
    log_file = open(local_log_path, "a")

    best_acc = -1.0
    
    # FLOPs Tracking
    measured_step_flops = None
    cumulative_flops = 0.0

    try:
        pbar = tqdm(range(args.training.max_steps))
        data_iter = iter(train_loader)
        
        warmup_iters = args.training.warmup_steps
        lr_decay_iters = args.training.max_steps
        min_lr = 1e-5

        for i in pbar:
            # LR Schedule
            lr = get_lr(i, args.training.learning_rate, warmup_iters, lr_decay_iters, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Data
            try:
                xs, ys = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                xs, ys = next(data_iter)
            xs, ys = xs.cuda(), ys.cuda()

            # Profiling logic at step 10
            if i == 10:
                prof_ctx = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True)
            else:
                prof_ctx = nullcontext()

            with prof_ctx as prof:
                # Forward & Loss
                with ctx:
                    if is_internal_loss:
                        # Model handles masking and loss
                        loss, _ = model(xs, ys, task_type=args.model.family)
                    
                    elif args.training.task_type == 'diffusion':
                        # External Diffusion masking logic
                        b_size = xs.shape[0]
                        full_sequence = torch.cat([xs, ys[:, -1].unsqueeze(1)], dim=1)
                        seq_len = full_sequence.shape[1]

                        t_val = time_epsilon + (1 - time_epsilon) * torch.rand(b_size, device=xs.device)
                        p_mask = 1 - scheduler(t_val).unsqueeze(1).expand(b_size, seq_len)

                        prefix_len = (ys[0] == -100).sum().item() + 1 
                        target_part_mask = torch.zeros_like(full_sequence, dtype=torch.bool)
                        target_part_mask[:, prefix_len:] = True

                        masked_indices = (torch.rand_like(full_sequence, dtype=torch.float32) < p_mask) & target_part_mask

                        noised_input_ids = full_sequence.clone()
                        noised_input_ids[masked_indices] = mask_token_id

                        diffusion_ys = full_sequence.clone()
                        diffusion_ys[~masked_indices] = -100 

                        output = model(noised_input_ids, task_type='diffusion')
                        output = torch.cat([output[:, :1], output[:, :-1]], dim=1)
                        loss = loss_fn(output.view(-1, output.size(-1)), diffusion_ys.view(-1))
                    
                    else:
                        # Autoregressive
                        output = model(xs, ys, task_type='autoregressive')
                        loss = loss_fn(output.view(-1, output.size(-1)), ys.view(-1))

                # Backward
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Post-Profiling processing
            if i == 10:
                events = prof.key_averages()
                train_step_flops = sum([e.flops for e in events])
                measured_step_flops = train_step_flops
                
                forward_flops = train_step_flops / 3.0
                
                # Estimate Inference FLOPs
                is_ar = (args.training.task_type == 'autoregressive')
                if is_ar:
                    # AR: path_len steps
                    if args.data.task == "sudoku":
                        n_gen = 81
                    else:
                        n_gen = args.data.path_len
                        
                    inference_flops = forward_flops * n_gen
                    model_type_str = f"Autoregressive ({n_gen} tokens)"
                else:
                    # Diffusion/Iterative: diffusion_steps
                    steps = args.model.diffusion_steps
                    inference_flops = forward_flops * steps
                    model_type_str = f"Iterative ({args.model.family}, {steps} steps)"
                
                # Print to stdout
                print(f"\n[FLOPs Profiling @ Step {i}]")
                print(f"  Train Step FLOPs (Fwd+Bwd): {train_step_flops:.4e}")
                print(f"  Forward FLOPs (Approx):     {forward_flops:.4e}")
                print(f"  Single Inference FLOPs:     {inference_flops:.4e} [{model_type_str}]")
                
                # Write to file
                try:
                    flops_file_path = os.path.join(args.out_dir, "flops_results.txt")
                    with open(flops_file_path, "w") as f_flops:
                        f_flops.write(f"Model Family: {args.model.family}\n")
                        f_flops.write(f"Train Step FLOPs (Fwd+Bwd): {train_step_flops:.4e}\n")
                        f_flops.write(f"Forward FLOPs (Approx): {forward_flops:.4e}\n")
                        f_flops.write(f"Single Inference FLOPs: {inference_flops:.4e}\n")
                        f_flops.write(f"Inference Type: {model_type_str}\n")
                except Exception as e:
                    print(f"Warning: Failed to write FLOPs results to file: {e}")
                
                # Backfill cumulative FLOPs for steps 0-10
                # FLOPs are per-batch for the training step.
                cumulative_flops += train_step_flops * (i + 1)
            
            elif measured_step_flops is not None:
                cumulative_flops += measured_step_flops

            # File Logging
            log_entry = {"step": i, "loss": loss.item(), "lr": lr}
            log_file.write(json.dumps(log_entry) + "\n")
            pbar.set_description(f"Loss: {loss.item():.4f}")

            # WandB & Eval
            if i > 0 and i % args.wandb.log_every_steps == 0 and not args.test_run:
                metrics = {"train_loss": loss.item(), "learning_rate": lr}
                
                if i % args.training.eval_every_steps == 0:
                    # Sync log
                    try: shutil.copy2(local_log_path, oss_log_path)
                    except Exception: pass

                    # Evaluation
                    test_batches = args.training.test_batches
                    gen_batches = max(1, int(test_batches // 4)) if test_batches else 4
                    
                    test_loss = compute_test_loss(model, test_loader, args, max_batches=test_batches)
                    test_seq_acc, test_token_acc = evaluate_generative(model, test_loader, args, max_batches=gen_batches)
                    train_seq_acc, train_token_acc = evaluate_generative(model, train_loader, args, max_batches=gen_batches)
                    
                    metrics.update({
                        "test_loss": test_loss,
                        "test_seq_acc": test_seq_acc,
                        "test_token_acc": test_token_acc,
                        "train_seq_acc": train_seq_acc,
                        "train_token_acc": train_token_acc
                    })
                    
                    print(f"\nStep {i}: Test Seq Acc: {test_seq_acc:.4f}, Token Acc: {test_token_acc:.4f}")

                    # Save Latest
                    last_model_dir = os.path.join(args.out_dir, "last")
                    if not os.path.exists(last_model_dir): os.makedirs(last_model_dir)
                    model._backbone.save_pretrained(last_model_dir)

                    # Save Best
                    if test_seq_acc > best_acc:
                        best_acc = test_seq_acc
                        best_model_dir = os.path.join(args.out_dir, "best_gen_acc")
                        if not os.path.exists(best_model_dir): os.makedirs(best_model_dir)
                        model._backbone.save_pretrained(best_model_dir)
                        print(f"New Best Model Saved (Acc: {best_acc:.4f})")

                wandb.log(metrics, step=i)
        
        # Save Final
        last_model_dir = os.path.join(args.out_dir, "last")
        if not os.path.exists(last_model_dir): os.makedirs(last_model_dir)
        model._backbone.save_pretrained(last_model_dir)

    finally:
        log_file.close()
        try: shutil.copy2(local_log_path, oss_log_path)
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
        if not os.path.exists(config.out_dir):
            os.makedirs(config.out_dir)

    train(config)

if __name__ == "__main__":
    parser = ArgumentParser()
    # Use MultiDiffusionConfig to cover all possible fields
    parser.add_arguments(MultiDiffusionConfig, dest="config")
    
    if hasattr(parser, "add_config_path_arg") and callable(parser.add_config_path_arg):
        parser.add_config_path_arg("--config")
    else:
        parser.add_argument("--config", type=str, dest="config_path_manual", help="Path to a YAML config file")
    
    args = parser.parse_args()
    
    if hasattr(args, "config_path_manual") and args.config_path_manual:
        config = MultiDiffusionConfig.load(args.config_path_manual)
    else:
        config = args.config
    
    if not config.test_run:
        if config.training.resume_id is None:
            config.training.resume_id = str(uuid.uuid4())
        
        # Ensure out_dir subfolder
        out_dir = os.path.join(config.out_dir, config.training.resume_id)
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        config.out_dir = out_dir
        
        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            import yaml
            yaml.dump(dataclasses.asdict(config), yaml_file, default_flow_style=False)

    main(config)
