import random
import pandas as pd
from tqdm import tqdm
from typing import List
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score

from src.tkns.tokenizer import RNASequenceTokenizer
from src.models.bert.bert import BERT
from src.models.bert.cgf import BERTConfig, TrainingConfig
from src.datasets.masked_lm import collate_fn_mlm
from src.datasets.rna_central import RNACentral
from src.trainer.mlm import Trainer


def set_seed(seed: int = 11):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_mml(config: BERTConfig, train_config: TrainingConfig, tokenizer: RNASequenceTokenizer):
    set_seed()

    # Load data
    tokenizer = RNASequenceTokenizer()
    writer = SummaryWriter(log_dir="./logs/mlm_training")

    

    # Create datasets
    dataset_train = RNACentral(h5_file_path="/home/andrii/Documents/genrna/data/tokenized_sequences_train.h5", tokenizer=tokenizer, max_length=train_config.max_seq_length)
    dataset_test = RNACentral(h5_file_path="/home/andrii/Documents/genrna/data/tokenized_sequences_test.h5", tokenizer=tokenizer, max_length=train_config.max_seq_length)

    # Create custom collate function
    custom_collate_fn = partial(collate_fn_mlm,
                                pad_token_id=tokenizer.vocabulary["[PAD]"],
                                mask_token_id=tokenizer.vocabulary["[MASK]"],
                                mask_prob=train_config.mask_prob,
                                no_mask_tokens=train_config.no_mask_tokens,
                                n_tokens=train_config.n_tokens,
                                randomize_prob=train_config.randomize_prob,
                                no_change_prob=train_config.no_change_prob)

    # Create data loaders
    dataloader_train = DataLoader(dataset_train, batch_size=train_config.batch_size, collate_fn=custom_collate_fn)
    dataloader_test = DataLoader(dataset_test, batch_size=train_config.batch_size, collate_fn=custom_collate_fn)

    # Load model
    model = BERT(config)
    model.to(train_config.device)

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=train_config.lr)
    scheduler = OneCycleLR(optimizer, max_lr=train_config.lr, steps_per_epoch=len(dataloader_train), epochs=train_config.n_epochs)

    # Criterion
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocabulary["[PAD]"])

    # Initialize the Trainer
    trainer = Trainer(model=model, train_dataloader=dataloader_train,
                      val_dataloader=dataloader_test, tokenizer=tokenizer,
                      criterion=criterion, optimizer=optimizer,
                      scheduler=scheduler, config=train_config, writer=writer)
    # Train the model
    trainer.train()


if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = RNASequenceTokenizer()

    bert_config = BERTConfig(dim=256, n_heads=8, attn_dropout=0.1,
                            mlp_dropout=0.1, depth=6, vocab_size=21,
                            max_len=512, pad_token_id=0, mask_token_id=1)

    train_config = TrainingConfig(batch_size=32, lr=6e-5, n_epochs=2, max_seq_length=512,
                                device="cuda", log_steps=500, save_steps=100000,
                                pad_token_id=tokenizer.vocabulary["[PAD]"], mask_token_id=tokenizer.vocabulary["[MASK]"],
                                mask_prob=0.15, no_mask_tokens=[], n_tokens=len(tokenizer.vocabulary),
                                randomize_prob=0.1, no_change_prob=0.1)

    train_mml(bert_config, train_config, tokenizer)