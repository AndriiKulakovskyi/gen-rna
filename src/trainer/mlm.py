import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.bert.cgf import TrainingConfig


class Trainer:
    def __init__(self, 
                 model: nn.Module, 
                 train_dataloader: DataLoader, 
                 val_dataloader: DataLoader,
                 tokenizer,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 config: TrainingConfig,
                 writer):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.writer = writer
        self.global_step = 0

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0

        for i, (input_ids, labels) in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1} Training")):
            input_ids, labels = input_ids.to(self.config.device), labels.to(self.config.device)

            # Forward pass
            logits = self.model(input_ids)["logits"]

            # Compute loss (transpose to match CrossEntropyLoss dimensions)
            loss = self.criterion(logits.transpose(1, 2), labels)
            total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Scheduler step
            self.scheduler.step()

            # Log loss periodically
            if self.global_step % self.config.log_steps == 0:
                avg_loss = total_loss / (i + 1)
                self.writer.add_scalar("Loss/Train", avg_loss, self.global_step)
                print(f"Step {self.global_step} | Training Loss: {avg_loss:.4f}")

            # Save the model periodically
            if self.global_step % self.config.save_steps == 0:
                model_path = f"mml_model_step_{self.global_step}.pt"
                torch.save(self.model.state_dict(), model_path)
                print(f"Model saved at step {self.global_step}: {model_path}")

            self.global_step += 1

        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss

    def validate_epoch(self, epoch: int):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for i, (input_ids, labels) in tqdm(self.val_dataloader, desc=f"Epoch {epoch + 1} Validation"):
                input_ids, labels = input_ids.to(self.config.device), labels.to(self.config.device)

                # Forward pass
                logits = self.model(input_ids)["logits"]

                # Compute loss
                loss = self.criterion(logits.transpose(1, 2), labels)
                total_loss += loss.item()

                # Log loss periodically during validation
                if i % self.config.log_steps == 0:
                    avg_loss = total_loss / (i + 1)
                    self.writer.add_scalar("Loss/Validation", avg_loss, self.global_step)
                    print(f"Step {self.global_step} | Validation Loss: {avg_loss:.4f}")

        avg_loss = total_loss / len(self.val_dataloader)
        return avg_loss

    def train(self):
        best_val_loss = float("inf")

        for epoch in range(self.config.n_epochs):
            # Train and validate
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)

            print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "mml_best_model.pt")
                print("Best model saved.")
