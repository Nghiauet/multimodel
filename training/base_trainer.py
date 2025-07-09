import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """Base trainer class with common training functionality"""
    
    def __init__(self, model, optimizer, device, gradient_clip_norm=1.0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.gradient_clip_norm = gradient_clip_norm
        
        self.train_losses = []
        self.test_losses = []
    
    def train_epoch(self, train_loader, epoch, total_epochs):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            batch = self._move_batch_to_device(batch)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass and compute loss
            loss = self._compute_loss(batch)
            
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
            
            # Record loss
            epoch_losses.append(loss.item())
            self.train_losses.append(loss.item())
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{total_epochs}, "
                      f"Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        return np.mean(epoch_losses)
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = self._move_batch_to_device(batch)
                loss = self._compute_loss(batch)
                total_loss += loss.item()
                n_batches += 1
        
        avg_loss = total_loss / n_batches
        self.test_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader, test_loader, num_epochs):
        """Full training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        
        # Initial evaluation
        initial_loss = self.evaluate(test_loader)
        print(f"Initial test loss: {initial_loss:.4f}")
        
        # Training loop
        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss = self.train_epoch(train_loader, epoch, num_epochs)
            
            # Evaluate on test set
            test_loss = self.evaluate(test_loader)
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Test Loss: {test_loss:.4f}")
        
        print("Training completed!")
        return np.array(self.train_losses), np.array(self.test_losses)
    
    @abstractmethod
    def _compute_loss(self, batch):
        """Compute loss for a batch (to be implemented by subclasses)"""
        pass
    
    def _move_batch_to_device(self, batch):
        """Move batch to device"""
        if isinstance(batch, (list, tuple)):
            return [item.to(self.device) if hasattr(item, 'to') else item for item in batch]
        else:
            return batch.to(self.device) if hasattr(batch, 'to') else batch