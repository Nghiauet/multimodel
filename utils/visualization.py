import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
import os


class Visualizer:
    """Unified visualization utilities for the multimodal project"""
    
    @staticmethod
    def show_image_grid(images, nrow=10, title="Images", figsize=(12, 8), 
                       normalize=True, save_path=None):
        """
        Display images in a grid using torchvision's make_grid
        
        Args:
            images: numpy array or torch tensor of images
            nrow: number of images per row
            title: title for the plot
            figsize: figure size
            normalize: whether to normalize images to [0, 1]
            save_path: path to save the figure
        """
        # Convert to torch tensor if needed
        if isinstance(images, np.ndarray):
            # Handle different input formats
            if images.ndim == 3:
                images = images[np.newaxis, :, :, :]
            
            # Convert to tensor and permute to (B, C, H, W)
            if normalize and images.max() > 1:
                images = images.astype(np.float32) / 255.0
            images = torch.FloatTensor(images).permute(0, 3, 1, 2)
        
        # Create grid
        grid_img = make_grid(images, nrow=nrow, normalize=normalize, pad_value=1)
        
        # Plot
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.axis('off')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        plt.show()
    
    @staticmethod
    def show_image_text_grid(image_text_pairs, title="Image-Text Pairs", 
                            figsize=(12, 12), save_path=None):
        """
        Display image-text pairs in a 3x3 grid
        
        Args:
            image_text_pairs: list of (image, text) tuples
            title: title for the plot
            figsize: figure size
            save_path: path to save the figure
        """
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        for i, (img, text) in enumerate(image_text_pairs[:9]):
            row, col = i // 3, i % 3
            
            # Handle different image formats
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            elif img.max() > 1:
                img = img.astype(np.float32) / 255.0
            
            axes[row, col].imshow(img)
            axes[row, col].set_title(text, fontsize=8, wrap=True)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        plt.show()
    
    @staticmethod
    def show_data_samples(data, texts=None, num_samples=9, title="Data Samples",
                         figsize=(10, 10), save_path=None):
        """
        Display random data samples with optional text labels
        
        Args:
            data: array of images
            texts: optional array of text labels
            num_samples: number of samples to display
            title: title for the plot
            figsize: figure size
            save_path: path to save the figure
        """
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        indices = np.random.choice(len(data), size=num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            row, col = i // 3, i % 3
            img = data[idx]
            
            # Handle different data formats
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            elif img.max() > 1:
                img = img.astype(np.float32) / 255.0
            
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            
            if texts is not None:
                axes[row, col].set_title(texts[idx], fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        plt.show()
    
    @staticmethod
    def plot_training_curves(train_losses, test_losses, title="Training Curves",
                            save_path=None):
        """
        Plot training and test losses
        
        Args:
            train_losses: array of training losses
            test_losses: array of test losses
            title: title for the plot
            save_path: path to save the figure
        """
        plt.figure(figsize=(10, 6))
        
        n_epochs = len(test_losses) - 1
        x_train = np.linspace(0, n_epochs, len(train_losses))
        x_test = np.arange(n_epochs + 1)
        
        plt.plot(x_train, train_losses, label="Train Loss", alpha=0.7)
        plt.plot(x_test, test_losses, label="Test Loss", marker='o', markersize=4)
        plt.legend()
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        plt.show()
    
    @staticmethod
    def plot_combined_training_curves(vqvae_train_losses, vqvae_test_losses,
                                    igpt_train_losses, igpt_test_losses,
                                    save_path=None):
        """
        Plot VQ-VAE and iGPT training curves in subplots
        
        Args:
            vqvae_train_losses: VQ-VAE training losses
            vqvae_test_losses: VQ-VAE test losses
            igpt_train_losses: iGPT training losses
            igpt_test_losses: iGPT test losses
            save_path: path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # VQ-VAE plot
        n_epochs_vqvae = len(vqvae_test_losses) - 1
        x_train_vqvae = np.linspace(0, n_epochs_vqvae, len(vqvae_train_losses))
        x_test_vqvae = np.arange(n_epochs_vqvae + 1)
        
        ax1.plot(x_train_vqvae, vqvae_train_losses, label="Train Loss", alpha=0.7)
        ax1.plot(x_test_vqvae, vqvae_test_losses, label="Test Loss", marker='o', markersize=4)
        ax1.set_title("VQ-VAE Training")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # iGPT plot
        n_epochs_igpt = len(igpt_test_losses) - 1
        x_train_igpt = np.linspace(0, n_epochs_igpt, len(igpt_train_losses))
        x_test_igpt = np.arange(n_epochs_igpt + 1)
        
        ax2.plot(x_train_igpt, igpt_train_losses, label="Train Loss", alpha=0.7)
        ax2.plot(x_test_igpt, igpt_test_losses, label="Test Loss", marker='o', markersize=4)
        ax2.set_title("iGPT Training")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        plt.show()