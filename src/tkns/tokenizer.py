import os
import json
import numpy as np


class RNASequenceTokenizer:
    """
    Tokenizer class for RNA sequences using a predefined vocabulary.
    Converts RNA sequences to token IDs and vice versa, supporting special tokens.
    Optimized with precompiled lookups.
    """
    def __init__(self, vocabulary: dict = None):
        # Try to load vocabulary from nucleotide2id.json if no vocabulary is provided
        if vocabulary is None:
            dir_path = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(dir_path, "nucleotide2id.json")
            if os.path.exists(file_path):
                with open(file_path, "r") as file:
                    vocabulary = json.load(file)
            else:
                raise ValueError("Vocabulary must be provided or nucleotide2id.json must exist in the directory.")

        if not vocabulary:
            raise ValueError("Vocabulary must be a non-empty dictionary.")

        self.vocabulary = vocabulary
        self.token_to_id = vocabulary
        self.id_to_token = {idx: token for token, idx in vocabulary.items()}

        # Create a precompiled lookup array for fast encoding
        max_char_code = max(ord(char) for char in vocabulary if len(char) == 1)
        self.lookup_array = np.full((max_char_code + 1,), -1, dtype=np.int32)
        for char, idx in vocabulary.items():
            if len(char) == 1:  # Only process single-character tokens
                self.lookup_array[ord(char)] = idx

    def encode(self, sequence: str) -> list:
        """
        Encodes an RNA sequence into a list of token IDs.
        Ensures the input sequence is capitalized.
        Optimized with precompiled lookup.
        """
        sequence = sequence.upper()  # Capitalize the sequence
        try:
            return [self.lookup_array[ord(char)] for char in sequence]
        except IndexError:
            raise ValueError("Invalid character in sequence.")

    def decode(self, ids: list) -> str:
        """
        Decodes a list of token IDs back into an RNA sequence.
        """
        try:
            return ''.join(self.id_to_token[i] for i in ids)
        except KeyError as e:
            raise ValueError(f"Invalid token ID {e.args[0]} in the input list.")

    def get_vocabulary(self) -> dict:
        """
        Returns the full vocabulary with token-to-ID mappings.
        """
        return self.token_to_id

    def is_valid_sequence(self, sequence: str) -> bool:
        """
        Checks if a sequence contains only valid tokens.
        Optimized with precompiled lookup.
        """
        sequence = sequence.upper()
        return all(0 <= ord(char) < len(self.lookup_array) and self.lookup_array[ord(char)] != -1 for char in sequence)

