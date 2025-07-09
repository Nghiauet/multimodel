import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class DataProcessor:
    """Unified data processing utilities for VQ-VAE and iGPT training"""
    
    @staticmethod
    def normalize_images(images, method='vqvae'):
        """
        Normalize images for training
        
        Args:
            images: numpy array of images
            method: 'vqvae' ([-1, 1]) or 'standard' ([0, 1])
        """
        if method == 'vqvae':
            # Normalize to [-1, 1] range for VQ-VAE
            return images.astype(np.float32) / 127.5 - 1.0
        elif method == 'standard':
            # Normalize to [0, 1] range
            return images.astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def prepare_vqvae_data(train_data, test_data, batch_size=128):
        """
        Prepare data for VQ-VAE training
        
        Args:
            train_data: Training images (H, W, C)
            test_data: Test images (H, W, C)
            batch_size: Batch size for data loaders
            
        Returns:
            train_loader, test_loader
        """
        # Normalize and convert to tensors
        train_data = DataProcessor.normalize_images(train_data, method='vqvae')
        test_data = DataProcessor.normalize_images(test_data, method='vqvae')
        
        # Convert to PyTorch tensors and permute to (C, H, W)
        train_tensor = torch.FloatTensor(train_data).permute(0, 3, 1, 2)
        test_tensor = torch.FloatTensor(test_data).permute(0, 3, 1, 2)
        
        # Create datasets and data loaders
        train_dataset = TensorDataset(train_tensor)
        test_dataset = TensorDataset(test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    @staticmethod
    def add_quantize_method(vqvae_model, device):
        """
        Add quantize method to VQ-VAE model for compatibility
        
        Args:
            vqvae_model: VQ-VAE model instance
            device: PyTorch device
        """
        def quantize(self, x):
            """Quantize images to discrete tokens"""
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x).to(device)
                if x.max() > 1:
                    x = x / 127.5 - 1.0
                x = x.permute(0, 3, 1, 2)
            return self.get_tokens(x)
        
        # Bind the method to the model instance
        vqvae_model.quantize = quantize.__get__(vqvae_model, type(vqvae_model))
        return vqvae_model