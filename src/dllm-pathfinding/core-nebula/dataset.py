import torch
from torch.utils.data import Dataset
import os
from typing import List, Tuple, Dict
from tokenizer import NumeralTokenizer, SudokuTokenizer

def token_accuracy(ys_pred, ys):
    """Calculates batch-wise TOKEN-LEVEL accuracy."""
    preds = torch.argmax(ys_pred, dim=-1)
    mask = (ys != -100)
    correct_preds = (preds[mask] == ys[mask]).sum().item()
    total_preds = mask.sum().item()
    return correct_preds / total_preds if total_preds > 0 else 0.0


def sequence_accuracy(ys_pred, ys):
    """Calculates batch-wise SEQUENCE-LEVEL accuracy."""
    preds = torch.argmax(ys_pred, dim=-1)
    mask = (ys != -100)
    
    # Check correctness for each token
    correct_tokens = (preds == ys) | ~mask

    # A sequence is correct only if all its relevant tokens are correct
    correct_sequences = torch.all(correct_tokens, dim=1)
    
    return correct_sequences.sum().item() / ys.size(0)

def prefix_target_list(filename: str) -> List[Tuple[str, str]]:
    """Load graphs from a file and split them into prefix and target."""
    data_list = []
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Data file not found: {filename}")
        
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if '=' not in line:
            continue
        try:
            parts = line.strip().split('=')
            # Handle cases where target might be empty or multiple '='? 
            # Assuming strictly one '=' based on legacy logic: "prefix, target = line.strip().split('=')"
            if len(parts) == 2:
                prefix, target = parts
                data_list.append((prefix + '=', target))
        except ValueError:
            continue
            
    return data_list

class PathFindingDataset(Dataset):
    def __init__(self, data_path: str, num_nodes: int):
        self.tokenizer = NumeralTokenizer(num_nodes)
        self.data = self._load_data(data_path)

    def _load_data(self, data_path: str) -> List[Dict[str, List[int]]]:
        """Loads and tokenizes the dataset from a file."""
        raw_data = prefix_target_list(data_path)
        tokenized_data = []
        for prefix_str, target_str in raw_data:
            prefix_tokens = self.tokenizer.encode(prefix_str)
            target_tokens = self.tokenizer.encode(target_str)
            tokenized_data.append({
                'prefix': prefix_tokens,
                'target': target_tokens
            })
        return tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        prefix = sample['prefix']
        target = sample['target']
        
        # Combine prefix and target
        full_sequence = torch.tensor(prefix + target, dtype=torch.long)
        
        # Create inputs (x) and targets (y)
        # x is the sequence up to the last token
        # y is the sequence shifted by one
        x = full_sequence[:-1]
        y = full_sequence[1:].clone()
        
        y[:len(prefix)-1] = -100
        
        return x, y

class SudokuDataset(Dataset):
    def __init__(self, data_path: str):
        self.tokenizer = SudokuTokenizer()
        self.data = self._load_data(data_path)

    def _load_data(self, data_path: str) -> List[Dict[str, List[int]]]:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        tokenized_data = []
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' not in line: continue
                parts = line.split('=')
                if len(parts) == 2:
                    q, a = parts
                    # q is 81 chars, a is 81 chars
                    # prefix: q + '='
                    prefix_str = q + '='
                    target_str = a
                    
                    tokenized_data.append({
                        'prefix': self.tokenizer.encode(prefix_str),
                        'target': self.tokenizer.encode(target_str)
                    })
        return tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        prefix = sample['prefix']
        target = sample['target']
        
        full_sequence = torch.tensor(prefix + target, dtype=torch.long)
        
        x = full_sequence[:-1]
        y = full_sequence[1:].clone()
        
        # Mask prefix in y
        # We want to predict starting from the first token of target.
        # x ending at '=' corresponds to y at position len(prefix)-1.
        # We want y[len(prefix)-1] to be active.
        # So we mask indices < len(prefix)-1.
        y[:len(prefix)-1] = -100
        
        return x, y

class SudokuCSVDataset(Dataset):
    def __init__(self, data_path: str):
        self.tokenizer = SudokuTokenizer()
        self.data = self._load_data(data_path)

    def _load_data(self, data_path: str) -> List[Dict[str, List[int]]]:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        tokenized_data = []
        import csv
        with open(data_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None) # skip header
            for row in reader:
                if len(row) < 2: continue
                q, a = row[0], row[1]
                # q is 81 chars, a is 81 chars
                # prefix: q + '='
                prefix_str = q + '='
                target_str = a
                
                tokenized_data.append({
                    'prefix': self.tokenizer.encode(prefix_str),
                    'target': self.tokenizer.encode(target_str)
                })
        return tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        prefix = sample['prefix']
        target = sample['target']
        
        full_sequence = torch.tensor(prefix + target, dtype=torch.long)
        
        x = full_sequence[:-1]
        y = full_sequence[1:].clone()
        
        y[:len(prefix)-1] = -100
        
        return x, y

def get_dataset(data_path: str, num_nodes: int) -> PathFindingDataset:
    return PathFindingDataset(data_path, num_nodes)

def get_sudoku_dataset(data_path: str) -> Dataset:
    if data_path.endswith('.csv'):
        return SudokuCSVDataset(data_path)
    return SudokuDataset(data_path)