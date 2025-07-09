import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import math


def evaluate_model(model, data_loader, sequence_length, vocab_size, device):
    """
    Evaluates model performance on a dataset.
    
    Args:
        model: The iGPT model
        data_loader: DataLoader containing tokenized sequences (already includes BOS token)
        sequence_length: Length of token sequences including <bos>
        vocab_size: Size of vocabulary
        device: Device to run evaluation on
        
    Returns:
        Average loss (negative log-likelihood) per dimension
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)  # Shape: (batch_size, sequence_length)
            batch_size = data.size(0)
            
            # Data already includes BOS token at the beginning
            # Create input sequence (all tokens except the last one)
            input_seq = data[:, :-1]  # Shape: (batch_size, sequence_length-1)
            
            # Create targets (all tokens except the first BOS token)
            targets = data[:, 1:]  # Shape: (batch_size, sequence_length-1)
            
            # Forward pass
            logits = model(input_seq)  # Shape: (batch_size, sequence_length-1, vocab_size)
            
            # Compute loss
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), reduction='sum')
            
            total_loss += loss.item()
            total_samples += batch_size * (sequence_length - 1)
    
    return total_loss / total_samples



def train_igpt(model, train_loader, test_loader, sequence_length, vocab_size, 
               device, num_epochs, learning_rate):
    """
    Trains the iGPT model.
    
    Args:
        model: The iGPT model to train
        train_loader: DataLoader for training data (already includes BOS token)
        test_loader: DataLoader for test data (already includes BOS token)
        sequence_length: Length of token sequences including <bos>
        vocab_size: Size of vocabulary
        device: Device to train on
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        
    Returns:
        train_losses: Array of training losses per minibatch
        test_losses: Array of test losses per epoch
    """
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler with warmup and cosine decay
    warmup_steps = 1000
    total_steps = len(train_loader) * num_epochs
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Initialize arrays to store losses
    train_losses = []
    test_losses = [evaluate_model(model, test_loader, sequence_length, vocab_size, device)]
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        for (batch_idx,data) in enumerate(train_loader):
            data = data.to(device)  # Shape: (batch_size, sequence_length)
            batch_size = data.size(0)
            
            # Data already includes BOS token at the beginning
            # Create input sequence (all tokens except the last one)
            input_seq = data[:, :-1]  # Shape: (batch_size, sequence_length-1)
            
            # Create targets (all tokens except the first BOS token)
            targets = data[:, 1:]  # Shape: (batch_size, sequence_length-1)
            
            # Forward pass
            logits = model(input_seq)  # Shape: (batch_size, sequence_length-1, vocab_size)
            
            # Compute loss
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Record loss
            train_losses.append(loss.item())
            epoch_losses.append(loss.item())
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Evaluate on test set after each epoch
        test_loss = evaluate_model(model, test_loader, sequence_length, vocab_size, device)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1}/{num_epochs} completed. Test Loss: {test_loss:.4f}")
    
    return np.array(train_losses), np.array(test_losses)