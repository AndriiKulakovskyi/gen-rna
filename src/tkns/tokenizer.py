import os
import pandas as pd
from typing import List, Optional
from collections import OrderedDict
from transformers.tokenization_utils import PreTrainedTokenizer


class Tokenizer(PreTrainedTokenizer):
    """
    A custom tokenizer for processing RNA nucleotide sequences with support for both autoregressive generation
    with defined Start/End markers and unconditional generation.

    Args:
        unique_nucleotides: List of valid tokens (nucleotides or special characters).
        bos_token: Token representing the beginning of a sequence.
        eos_token: Token representing the end of a sequence.
        pad_token: Token used for padding sequences.
        unk_token: Token for unknown characters.
        additional_special_tokens: Additional special tokens to include.
        do_upper_case: Whether to convert input sequences to uppercase.

    Example Usage:
        >>> tokenizer = Tokenizer("src/tnks/unique_nucleotides.csv", unk_token="<unk>")
        >>> tokenizer("ACGU")
        {'input_ids': [0, 1, 2, 3], 'attention_mask': [1, 1, 1, 1]}
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        bos_token: Optional[str] = "<bos>",
        eos_token: Optional[str] = "<eos>",
        pad_token: Optional[str] = "<pad>",
        unk_token: Optional[str] = "<unk>",
        additional_special_tokens: Optional[List[str]] = None,
        do_upper_case: bool = True,
        **kwargs,
    ):
        # Load unique nucleotides from the CSV file
        df = pd.read_csv(vocab_file, header=None)
        unique_nucleotides = df[0].tolist()

        # Map tokens to IDs and vice versa
        self._id_to_token = OrderedDict(enumerate(unique_nucleotides))
        self._token_to_id = OrderedDict({tok: idx for idx, tok in enumerate(unique_nucleotides)})

        additional_special_tokens = additional_special_tokens or []

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.do_upper_case = do_upper_case

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize input text into individual characters.

        This method splits the input text into a list of single-character tokens.
        It is particularly useful for nucleotide sequences, where each character
        (e.g., A, C, G, U) is treated as a separate token.

        Args:
            text: Input string to tokenize.

        Returns:
            List[str]: List of single-character tokens.
        """
        if not isinstance(text, str):
            raise ValueError("Input text must be a string.")

        if self.do_upper_case:
            text = text.upper()

        # Split text into individual characters, handling special cases like spaces and non-standard characters
        return [char for char in text]

    def encode(
        self,
        text: str,
        include_special_tokens: bool = True,
        **kwargs
    ) -> List[int]:
        """
        Encode the input text into token IDs.

        Args:
            text: The input text to encode.
            include_special_tokens: Whether to include special tokens (e.g., bos/eos).
            **kwargs: Additional arguments for customization.

        Returns:
            List[int]: The encoded token IDs.
        """
        token_ids = [self._convert_token_to_id(token) for token in self._tokenize(text)]
        if include_special_tokens:
            token_ids = self.build_inputs_with_special_tokens(token_ids, include_special_tokens=True)
        return token_ids

    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token to its corresponding ID."""
        return self._token_to_id.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        """Convert an ID back to its corresponding token."""
        return self._id_to_token.get(index, self.unk_token)

    def build_inputs_with_special_tokens(
        self,
        token_ids: List[int],
        include_special_tokens: bool = True
    ) -> List[int]:
        """
        Build input sequence with optional special tokens.

        Args:
            token_ids: List of token IDs for the sequence.
            include_special_tokens: Whether to include bos/eos tokens.

        Returns:
            List[int]: Encoded sequence with or without special tokens.
        """
        if not include_special_tokens:
            return token_ids

        bos = [self.bos_token_id]
        eos = [self.eos_token_id]

        return bos + token_ids + eos

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        include_bos_eos: bool = False
    ) -> str:
        """
        Decode a sequence of token IDs into a string, optionally skipping special tokens.

        Args:
            token_ids: List of token IDs to decode.
            skip_special_tokens: Whether to remove special tokens from the decoded output.
            include_bos_eos: Whether to include bos/eos tokens in the output.

        Returns:
            str: Decoded string.
        """
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in {
                self.bos_token_id, self.eos_token_id, self.pad_token_id
            }:
                continue
            tokens.append(self._convert_id_to_token(token_id))

        if include_bos_eos:
            tokens = ([self.bos_token] if self.bos_token_id in token_ids else []) + tokens
            if self.eos_token_id in token_ids:
                tokens.append(self.eos_token)

        return "".join(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> str:
        """Save the vocabulary to a file."""
        filename = (filename_prefix + "-" if filename_prefix else "") + "vocab.txt"
        vocab_path = os.path.join(save_directory, filename)
        with open(vocab_path, "w", encoding="utf-8") as file:
            file.write("\n".join(self._id_to_token.values()))
        return vocab_path

    def get_vocab(self) -> dict:
        """Return the tokenizer vocabulary."""
        return self._token_to_id.copy()

    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self._id_to_token)
