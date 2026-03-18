from typing import List, Dict

class NumeralTokenizer:
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        # Define encoder and decoder as a dictionary
        self.encoder: Dict[str, int] = {str(i): i for i in range(num_nodes)}
        self.encoder['|'] = num_nodes
        self.encoder['='] = num_nodes + 1
        self.encoder['/'] = num_nodes + 2
        self.encoder['$'] = num_nodes + 3 # Special token for teacherless training

        self.vocab_size = num_nodes + 4

        self.decoder: Dict[int, str] = {i: str(i) for i in range(num_nodes)}
        self.decoder[num_nodes] = '|'
        self.decoder[num_nodes + 1] = '='
        self.decoder[num_nodes + 2] = '/'
        self.decoder[num_nodes + 3] = '$'
        self.decoder[-1] = '' # For ignoring prefix tokens in loss

    def encode(self, s: str) -> List[int]:
        """Converts a string representation of a graph problem into a list of integer tokens."""
        # Manually parse the string instead of splitting to handle numbers and symbols
        out = []
        i = 0
        while i < len(s):
            if s[i] == ',':
                i += 1
                continue
            
            # Check for multi-digit numbers
            num_str = ''
            j = i
            while j < len(s) and s[j].isdigit():
                num_str += s[j]
                j += 1
            
            if num_str:
                out.append(self.encoder[num_str])
                i = j
            else:
                # Handle single character symbols
                out.append(self.encoder[s[i]])
                i += 1
        return out

    def decode(self, tokens: List[int]) -> List[str]:
        """Converts a list of integer tokens back to a string."""
        return [self.decoder.get(token, '') for token in tokens]

    def decode_to_string(self, tokens: List[int]) -> str:
        """Helper to decode directly to a single string."""
        return "".join(self.decode(tokens))

class SudokuTokenizer:
    def __init__(self):
        # Tokens: 1-9, $, =
        self.chars = [str(i) for i in range(1, 10)] + ['$', '=']
        self.encoder = {c: i for i, c in enumerate(self.chars)}
        self.encoder['0'] = self.encoder['$'] # Map 0 to $ as placeholder
        self.decoder = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, s: str) -> List[int]:
        """Converts sudoku string to tokens."""
        return [self.encoder[c] for c in s if c in self.encoder]

    def decode(self, tokens: List[int]) -> List[str]:
        """Converts tokens back to list of chars."""
        return [self.decoder.get(token, '') for token in tokens]

    def decode_to_string(self, tokens: List[int]) -> str:
        """Decodes to string."""
        return "".join(self.decode(tokens))