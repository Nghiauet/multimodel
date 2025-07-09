import torch
import torch.nn.functional as F
import math
from .base_trainer import BaseTrainer


class iGPTTrainer(BaseTrainer):
    """iGPT specific trainer with learning rate scheduling"""
    
    def __init__(self, model, optimizer, device, vocab_size, sequence_length, 
                 gradient_clip_norm=1.0, warmup_steps=1000):
        super().__init__(model, optimizer, device, gradient_clip_norm)
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.warmup_steps = warmup_steps
        self.scheduler = None
        self.step_count = 0
    
    def setup_scheduler(self, train_loader, num_epochs):
        """Setup learning rate scheduler"""
        total_steps = len(train_loader) * num_epochs
        
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            else:
                decay_ratio = (step - self.warmup_steps) / (total_steps - self.warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, train_loader, epoch, total_epochs):
        """Train for one epoch with learning rate scheduling"""
        if self.scheduler is None:
            self.setup_scheduler(train_loader, total_epochs)
        
        self.model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            data = batch.to(self.device)
            
            # Prepare input and target sequences
            input_seq = data[:, :-1]  # All tokens except the last one
            targets = data[:, 1:]     # All tokens except the first one
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(input_seq)
            
            # Compute loss
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size), 
                targets.reshape(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.gradient_clip_norm
                )
            
            # Update parameters
            self.optimizer.step()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Record loss
            epoch_losses.append(loss.item())
            self.train_losses.append(loss.item())
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{total_epochs}, "
                      f"Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        return sum(epoch_losses) / len(epoch_losses)
    
    def _compute_loss(self, batch):
        """Compute iGPT loss for evaluation"""
        data = batch
        input_seq = data[:, :-1]
        targets = data[:, 1:]
        
        logits = self.model(input_seq)
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size), 
            targets.reshape(-1)
        )
        return loss