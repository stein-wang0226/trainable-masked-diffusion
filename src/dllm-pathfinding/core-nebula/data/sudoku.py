import random
import numpy as np
import os
from tqdm import tqdm
import copy
import multiprocessing
from functools import partial

class SudokuGenerator:
    def __init__(self):
        pass

    def get_candidates(self, board, row, col):
        candidates = set(range(1, 10))
        # Remove row
        candidates -= set(board[row])
        # Remove col
        candidates -= set(board[i][col] for i in range(9))
        # Remove box
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                val = board[start_row + i][start_col + j]
                if val in candidates:
                    candidates.remove(val)
        return list(candidates)

    def solve_mrv(self, board, count_only=False, limit=1):
        """
        Backtracking solver with Minimum Remaining Values (MRV) heuristic.
        """
        min_len = 10
        best_cell = None
        
        # Find cell with fewest candidates
        empty_found = False
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    empty_found = True
                    cands = self.get_candidates(board, r, c)
                    if len(cands) == 0:
                        return 0 if count_only else False
                    if len(cands) < min_len:
                        min_len = len(cands)
                        best_cell = (r, c)
                        if min_len == 1: 
                            break
            if min_len == 1:
                break
        
        if not empty_found:
            return 1 if count_only else True

        row, col = best_cell
        candidates = self.get_candidates(board, row, col)
        
        count = 0
        for num in candidates:
            board[row][col] = num
            if count_only:
                c = self.solve_mrv(board, count_only=True, limit=limit-count)
                count += c
                if count >= limit:
                    board[row][col] = 0 # Backtrack
                    return count
            else:
                if self.solve_mrv(board, count_only=False):
                    return True
            board[row][col] = 0
            
        return count if count_only else False

    def fill_random(self, board):
        """Fills an empty board with a valid random Sudoku solution."""
        for i in range(81):
            row, col = i // 9, i % 9
            if board[row][col] == 0:
                nums = list(range(1, 10))
                random.shuffle(nums)
                for num in nums:
                    if self.is_valid_move(board, row, col, num):
                        board[row][col] = num
                        if self.fill_random(board):
                            return True
                        board[row][col] = 0
                return False
        return True

    def is_valid_move(self, board, row, col, num):
        for i in range(9):
            if board[row][i] == num: return False
            if board[i][col] == num: return False
        sr, sc = (row // 3) * 3, (col // 3) * 3
        for i in range(3):
            for j in range(3):
                if board[sr + i][sc + j] == num: return False
        return True

    def generate_full_board(self):
        board = [[0]*9 for _ in range(9)]
        self.fill_random(board)
        return board

    def remove_numbers(self, board, difficulty=0.5):
        """
        Removes numbers to create a puzzle with a unique solution.
        """
        # Map difficulty to number of holes.
        # Easy: ~25 holes. Hard: ~55 holes.
        min_holes = 25
        max_holes = 58 
        target_holes = int(min_holes + (max_holes - min_holes) * difficulty)
        
        puzzle = [row[:] for row in board]
        positions = list(range(81))
        random.shuffle(positions)
        
        holes = 0
        for pos in positions:
            if holes >= target_holes:
                break
            
            r, c = pos // 9, pos % 9
            val = puzzle[r][c]
            puzzle[r][c] = 0
            
            # Check uniqueness using MRV solver
            check_bd = [row[:] for row in puzzle]
            if self.solve_mrv(check_bd, count_only=True, limit=2) != 1:
                puzzle[r][c] = val # Not unique, put back
            else:
                holes += 1
        
        return puzzle

# Worker function needs to be at module level for multiprocessing pickling
def generate_single_pair(difficulty):
    """Worker function to generate a single sudoku pair."""
    # Re-seed random per process to avoid identical results if forked
    # (Though Python 3.7+ usually handles this, it's safer)
    np.random.seed()
    random.seed()
    
    gen = SudokuGenerator()
    full_board = gen.generate_full_board()
    puzzle = gen.remove_numbers(full_board, difficulty)
    
    flat_full = [item for sublist in full_board for item in sublist]
    flat_puzzle = [item if item != 0 else '$' for sublist in puzzle for item in sublist]
    
    return "".join(map(str, flat_puzzle)), "".join(map(str, flat_full))

def generate_and_save_sudoku(n_train, n_test, output_dir='./core-nebula/data/datasets/sudoku', 
                             difficulty=0.5, batch_size=1000, num_workers=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    diff_str = f"{difficulty:.1f}".replace('.', '')
    
    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1) # Leave one core for system
    
    print(f"Starting generation with {num_workers} workers.")

    def save_split_parallel(n, filename):
        file_path = os.path.join(output_dir, filename)
        print(f"Generating {filename} with difficulty {difficulty}...")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            with multiprocessing.Pool(processes=num_workers) as pool:
                # Use imap_unordered for better performance as order doesn't matter
                # Create a generator of arguments (difficulty repeated n times)
                # chunksize adjustment can help with performance
                chunksize = 10 
                iterator = pool.imap_unordered(generate_single_pair, [difficulty] * n, chunksize=chunksize)
                
                for i, (q, a) in tqdm(enumerate(iterator), total=n):
                    f.write(f"{q}={a}\n")
                    if (i + 1) % batch_size == 0:
                        f.flush()

    save_split_parallel(n_train, f'sudoku_train_{n_train}_diff{diff_str}.txt')
    save_split_parallel(n_test, f'sudoku_test_{n_test}_diff{diff_str}.txt')

if __name__ == "__main__":
    # You can adjust difficulty here
    # 0.8 is very slow, so parallelism is highly recommended
    generate_and_save_sudoku(300000, 30000, difficulty=0.8)