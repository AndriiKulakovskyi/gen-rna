from typing import List
from dataclasses import dataclass


@dataclass
class BERTConfig:
    dim: int
    n_heads: int
    attn_dropout: float
    mlp_dropout: float
    depth: int
    vocab_size: int
    max_len: int
    pad_token_id: int
    mask_token_id: int


@dataclass
class TrainingConfig:
    batch_size: int
    lr: float
    n_epochs: int
    max_seq_length: int
    device: str
    gradient_clip: float = 1.0
    log_steps: int = 500
    save_steps: int = 100000
    pad_token_id: int = 0
    mask_token_id: int = 1
    mask_prob: float = 0.15
    no_mask_tokens: List[int] = None
    n_tokens: int = 0
    randomize_prob: float = 0.1
    no_change_prob: float = 0.1