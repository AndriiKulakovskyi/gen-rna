import h5py, torch
from torch.utils.data import Dataset


class RNACentral(Dataset):
    """
    Dataset class for RNACentral sequences. Supports
    loading pre-tokenized sequences from an H5 file.
    """
    def __init__(self, tokenizer, max_length, h5_file_path=None, lines=None):
        """
        Args:
            tokenizer: Tokenizer instance for encoding sequences.
            max_length (int): Fixed sequence length for padding/truncation.
            h5_file_path (str, optional): Path to the H5 file containing pre-tokenized sequences.
            lines (list, optional): List of RNA sequences (strings) for on-the-fly tokenization.

        Notes:
            Either `h5_file_path` or `lines` must be provided, but not both.
        """
        if not h5_file_path and not lines:
            raise ValueError("Either `h5_file_path` or `lines` must be provided.")
        if h5_file_path and lines:
            raise ValueError("Provide only one of `h5_file_path` or `lines`.")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = self.tokenizer.vocabulary["[PAD]"]
        self.from_h5 = h5_file_path is not None

        if self.from_h5:
            # Load pre-tokenized sequences from the H5 file
            self.h5_file = h5py.File(h5_file_path, 'r')
            self.tokenized_sequences = self.h5_file['tokenized_sequences']
            self.num_samples = self.tokenized_sequences.shape[0]
        else:
            # Store lines for on-the-fly tokenization
            if not isinstance(lines, list) or not lines:
                raise ValueError("`lines` should be a non-empty list of RNA sequences.")
            self.lines = lines
            self.num_samples = len(lines)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        """
        Retrieve a tokenized sequence.
        Args:
            idx (int): Index of the sequence.
        Returns:
            torch.Tensor: Tokenized sequence.
        """
        if self.from_h5:
            # Load pre-tokenized sequence from H5 file
            token_ids = self.tokenized_sequences[idx][:self.max_length]
            token_ids = list(token_ids)  # Convert from array to list for consistency
        else:
            # Tokenize the RNA sequence on-the-fly
            line = self.lines[idx]
            token_ids = self.tokenizer.encode(line)

        # Truncate or pad the sequence to the maximum length
        token_ids = (token_ids[:self.max_length] + 
                     [self.pad_token_id] * max(0, self.max_length - len(token_ids)))

        return torch.tensor(token_ids, dtype=torch.long)

    def close_h5_file(self):
        """Close the H5 file if it was opened."""
        if self.from_h5 and hasattr(self, 'h5_file'):
            self.h5_file.close()