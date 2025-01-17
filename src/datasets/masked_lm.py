import torch


def collate_fn_mlm(batch, mask_token_id, mask_prob, pad_token_id, no_mask_tokens, n_tokens, randomize_prob, no_change_prob):
    """
    Collation function for batching and applying MLM masking with detailed masking logic.
    
    Args:
        batch: List of tokenized sequences from the dataset.
        mask_token_id (int): Token ID for [MASK].
        mask_prob (float): Probability of masking a token (15% typically).
        pad_token_id (int): Token ID for padding.
        no_mask_tokens (list): List of token IDs that should not be masked.
        n_tokens (int): Total number of tokens (used for random token generation).
        randomize_prob (float): Probability of replacing with a random token.
        no_change_prob (float): Probability of leaving the original token unchanged.

    Returns:
        input_ids (torch.Tensor): Masked input IDs.
        labels (torch.Tensor): Labels with unmasked tokens set to padding ID.
    """
    # Stack sequences from the batch
    input_ids = torch.stack(batch)
    labels = input_ids.clone()  # Labels initialized as a copy of input IDs

    # Create mask based on probabilities
    mask = torch.rand(input_ids.shape) < mask_prob
    for token in no_mask_tokens + [pad_token_id, mask_token_id]:
        mask &= (input_ids != token)

    # Apply masking logic
    unchanged_mask = mask & (torch.rand(input_ids.shape) < no_change_prob)
    random_token_mask = mask & ~unchanged_mask & (torch.rand(input_ids.shape) < randomize_prob)
    mask_token_mask = mask & ~unchanged_mask & ~random_token_mask

    # Replace tokens with [MASK], random tokens, or leave unchanged
    random_tokens = torch.randint(0, n_tokens, input_ids.shape, device=input_ids.device)
    input_ids[random_token_mask] = random_tokens[random_token_mask]
    input_ids[mask_token_mask] = mask_token_id

    # Set labels for masked tokens, others to pad_token_id
    labels[~mask] = pad_token_id

    return input_ids, labels
