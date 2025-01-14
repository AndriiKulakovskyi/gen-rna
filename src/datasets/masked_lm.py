import torch
from torch.utils.data import Dataset


class MLMDataset(Dataset):
    """
    Dataset class for Masked Language Modeling (MLM).
    Prepares RNA sequences for MLM training by tokenizing, truncating, and padding.
    """
    def __init__(self, lines, tokenizer, max_length):
        """
        Args:
            lines (list): List of RNA sequences (strings).
            tokenizer: Tokenizer instance for encoding sequences.
            max_length (int): Fixed sequence length for padding/truncation.
        """
        if not lines or not isinstance(lines, list):
            raise ValueError("`lines` should be a non-empty list of RNA sequences.")
        
        self.lines = lines
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = self.tokenizer.vocabulary["[PAD]"]

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.lines)

    def __getitem__(self, idx):
        """
        Tokenize, truncate/pad, and return a sequence as input and target.
        Args:
            idx (int): Index of the sequence.
        Returns:
            tuple: (tokenized_input_ids, tokenized_labels)
        """
        line = self.lines[idx]

        # Tokenize the RNA sequence
        token_ids = self.tokenizer.encode(line)

        # Truncate or pad the sequence to the maximum length
        
        token_ids = (
            token_ids[:self.max_length] +
            [self.pad_token_id] * max(0, self.max_length - len(token_ids))
        )

        # Labels are identical to the input at this stage
        labels = token_ids.copy()

        # Return as tensors
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_fn(batch, mask_token_id, mask_prob, pad_token_id):
    """
    Collation function for batching and applying MLM masking.
    Args:
        batch: List of (input_ids, labels) tuples from dataset.
        mask_token_id (int): Token ID for [MASK].
        mask_prob (float): Probability of masking a token (15% typically).
        pad_token_id (int): Token ID for padding.
    Returns:
        input_ids (torch.Tensor): Masked input IDs.
        labels (torch.Tensor): Labels with unmasked tokens set to -100.
    """
    # Stack already padded sequences directly
    input_ids = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])

    # Generate random masking for MLM
    rand = torch.rand(input_ids.shape)
    mlm_mask = (rand < mask_prob) & (input_ids != pad_token_id)

    # Create masked input IDs and labels
    masked_input_ids = input_ids.clone()
    masked_input_ids[mlm_mask] = mask_token_id  # Replace with [MASK] token ID

    masked_labels = labels.clone()
    masked_labels[~mlm_mask] = -100  # Set non-masked tokens to -100

    return masked_input_ids, masked_labels
