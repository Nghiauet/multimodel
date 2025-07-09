import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os

# Import all necessary modules
from VQ_VAE.model import VQVAE, compute_loss, evaluate_VQVAE
from iGPT.model import iGPT
from iGPT.tokenizer import Tokenizer
from iGPT.train import train_igpt, evaluate_model
from iGPT.create_dataset import create_dataset
from iGPT.sample import generate_conditional_samples_from_text, generate_conditional_samples_from_image, generate_unconditional_samples

def load_data(data_path):
    """Load data from pickle file"""
    with open(data_path, 'rb') as f:
        return pickle.load(f)

def visualize_data(data, texts=None, num_samples=9, title="Data Samples"):
    """Visualize data samples"""
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
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
    plt.show()

def train_vqvae(train_data, test_data, device, config):
    """Train VQ-VAE model"""
    print("=== Training VQ-VAE ===")
    
    # Prepare data (normalize to [-1, 1])
    train_data = train_data.astype(np.float32) / 127.5 - 1.0
    test_data = test_data.astype(np.float32) / 127.5 - 1.0
    
    # Convert to tensors and create data loaders
    train_tensor = torch.FloatTensor(train_data).permute(0, 3, 1, 2)
    test_tensor = torch.FloatTensor(test_data).permute(0, 3, 1, 2)
    
    train_dataset = TensorDataset(train_tensor)
    test_dataset = TensorDataset(test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize model
    model = VQVAE(
        dim=config['vqvae_dim'], 
        K=config['vqvae_K'], 
        D=config['vqvae_D']
    ).to(device)
    
    # Add quantize method to model for compatibility
    def quantize(self, x):
        """Quantize images to discrete tokens"""
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(device)
            if x.max() > 1:
                x = x / 127.5 - 1.0
            x = x.permute(0, 3, 1, 2)
        return self.get_tokens(x)
    
    # Bind the method to the model instance
    model.quantize = quantize.__get__(model, VQVAE)
    
    optimizer = optim.Adam(model.parameters(), lr=config['vqvae_lr'])
    
    train_losses = []
    test_losses = []
    
    # Initial evaluation
    test_loss = evaluate_VQVAE(model, test_loader)
    test_losses.append(test_loss)
    print(f"Initial VQ-VAE Test Loss: {test_loss:.4f}")
    
    # Training loop
    model.train()
    for epoch in range(config['vqvae_epochs']):
        epoch_losses = []
        
        for batch_idx, (x,) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            
            z_e, z_q, recon_x = model(x)
            loss = compute_loss(x, recon_x, z_e, z_q)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            train_losses.append(loss.item())
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{config['vqvae_epochs']}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Evaluate after each epoch
        test_loss = evaluate_VQVAE(model, test_loader)
        test_losses.append(test_loss)
        
        avg_train_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{config['vqvae_epochs']} - Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/vqvae_model.pth')
    print("VQ-VAE model saved to checkpoints/vqvae_model.pth")
    
    return model, train_losses, test_losses

def train_multimodal_igpt(train_data, test_data, train_texts, test_texts, vqvae_model, device, config):
    """Train multimodal iGPT model"""
    print("\n=== Training Multimodal iGPT ===")
    
    # Create tokenizer
    text_tokenizer = Tokenizer(train_texts, vqvae_model.n_embeddings)
    
    # Calculate sequence length and vocab size
    sequence_length = config['sequence_length']
    total_vocab_size = vqvae_model.n_embeddings + len(text_tokenizer.all_words)
    
    print(f"VQ-VAE embeddings: {vqvae_model.n_embeddings}")
    print(f"Text vocabulary size: {len(text_tokenizer.all_words)}")
    print(f"Total vocabulary size: {total_vocab_size}")
    print(f"Sequence length: {sequence_length}")
    
    # Create datasets
    train_loader = create_dataset(train_data, train_texts, vqvae_model, text_tokenizer, config['batch_size'])
    test_loader = create_dataset(test_data, test_texts, vqvae_model, text_tokenizer, config['batch_size'])
    
    # Initialize model
    model = iGPT(
        vocab_size=total_vocab_size,
        context_length=sequence_length,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        dropout=config['dropout']
    ).to(device)
    
    # Train model
    train_losses, test_losses = train_igpt(
        model, train_loader, test_loader, 
        sequence_length, total_vocab_size, device,
        config['igpt_epochs'], config['igpt_lr']
    )
    
    # Save model
    torch.save(model.state_dict(), 'checkpoints/igpt_model.pth')
    print("iGPT model saved to checkpoints/igpt_model.pth")
    
    return model, text_tokenizer, train_losses, test_losses

def generate_samples(model, text_tokenizer, vqvae_model, test_data, test_texts, device, num_samples=9):
    """Generate samples from the trained model"""
    print("\n=== Generating Samples ===")
    
    # Prepare test prompts
    image_test_prompt = test_data[:num_samples]
    text_test_prompt = test_texts[:num_samples]
    
    # Generate different types of samples
    print("Generating text-conditioned samples...")
    samples_text_conditioned = generate_conditional_samples_from_text(
        model, text_tokenizer, vqvae_model, text_test_prompt, device
    )
    
    print("Generating image-conditioned samples...")
    samples_image_conditioned = generate_conditional_samples_from_image(
        model, text_tokenizer, vqvae_model, image_test_prompt, device
    )
    
    print("Generating unconditional samples...")
    samples_unconditioned = generate_unconditional_samples(
        model, text_tokenizer, vqvae_model, device, num_samples=num_samples
    )
    
    return samples_image_conditioned, samples_text_conditioned, samples_unconditioned

def visualize_samples(samples_image_conditioned, samples_text_conditioned, samples_unconditioned):
    """Visualize generated samples"""
    print("\n=== Visualizing Results ===")
    
    def plot_samples(samples, title, filename):
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle(title, fontsize=16)
        
        for i in range(9):
            row, col = i // 3, i % 3
            img, text = samples[i]
            
            axes[row, col].imshow(img)
            axes[row, col].set_title(text, fontsize=8)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Plot different types of samples
    plot_samples(samples_image_conditioned, "Image-Conditioned Samples", "results/image_conditioned_samples.png")
    plot_samples(samples_text_conditioned, "Text-Conditioned Samples", "results/text_conditioned_samples.png")
    plot_samples(samples_unconditioned, "Unconditional Samples", "results/unconditional_samples.png")

def plot_training_curves(vqvae_train_losses, vqvae_test_losses, igpt_train_losses, igpt_test_losses):
    """Plot training curves"""
    print("\n=== Plotting Training Curves ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # VQ-VAE training loss
    axes[0, 0].plot(vqvae_train_losses)
    axes[0, 0].set_title('VQ-VAE Training Loss')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # VQ-VAE test loss
    axes[0, 1].plot(vqvae_test_losses)
    axes[0, 1].set_title('VQ-VAE Test Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    # iGPT training loss
    axes[1, 0].plot(igpt_train_losses)
    axes[1, 0].set_title('iGPT Training Loss')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True)
    
    # iGPT test loss
    axes[1, 1].plot(igpt_test_losses)
    axes[1, 1].set_title('iGPT Test Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('results/training_curves.png')
    plt.show()

def main():
    """Main function to run the complete pipeline"""
    print("=== Multimodal VQ-VAE + iGPT Training Pipeline ===")
    
    # Configuration
    config = {
        # VQ-VAE config
        'vqvae_dim': 256,
        'vqvae_K': 128,
        'vqvae_D': 256,
        'vqvae_epochs': 30,
        'vqvae_lr': 1e-3,
        
        # iGPT config
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 4,
        'sequence_length': 58,  # 49 (image tokens) + 6 (text tokens) + 2 (special tokens) + 1 (BOS)
        'igpt_epochs': 30,
        'igpt_lr': 1e-3,
        'dropout': 0.1,
        
        # General config
        'batch_size': 128,
    }
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("\n=== Loading Data ===")
    
    # Load VQ-VAE training data
    print("Loading VQ-VAE training data...")
    vqvae_data = load_data('data/mnist_colored.pkl')
    train_data_vqvae, test_data_vqvae = vqvae_data['train'], vqvae_data['test']
    
    print(f"VQ-VAE train data shape: {train_data_vqvae.shape}")
    print(f"VQ-VAE test data shape: {test_data_vqvae.shape}")
    
    # Load multimodal data
    print("Loading multimodal data...")
    multimodal_data = load_data('data/colored_mnist_with_text.pkl')
    train_data_mm, test_data_mm = multimodal_data['train_images'], multimodal_data['test_images']
    train_texts, test_texts = multimodal_data['train_texts'], multimodal_data['test_texts']
    
    print(f"Multimodal train data shape: {train_data_mm.shape}")
    print(f"Multimodal test data shape: {test_data_mm.shape}")
    print(f"Number of training texts: {len(train_texts)}")
    print(f"Number of test texts: {len(test_texts)}")
    
    # Visualize data
    print("\n=== Visualizing Data ===")
    visualize_data(train_data_vqvae, title="VQ-VAE Training Data")
    visualize_data(train_data_mm, train_texts, title="Multimodal Training Data")
    
    # Train VQ-VAE
    vqvae_model, vqvae_train_losses, vqvae_test_losses = train_vqvae(
        train_data_vqvae, test_data_vqvae, device, config
    )
    
    # Train multimodal iGPT
    igpt_model, text_tokenizer, igpt_train_losses, igpt_test_losses = train_multimodal_igpt(
        train_data_mm, test_data_mm, train_texts, test_texts, vqvae_model, device, config
    )
    
    # Generate samples
    samples_image_conditioned, samples_text_conditioned, samples_unconditioned = generate_samples(
        igpt_model, text_tokenizer, vqvae_model, test_data_mm, test_texts, device
    )
    
    # Visualize results
    visualize_samples(samples_image_conditioned, samples_text_conditioned, samples_unconditioned)
    
    # Plot training curves
    plot_training_curves(vqvae_train_losses, vqvae_test_losses, igpt_train_losses, igpt_test_losses)
    
    print("\n=== Training Complete! ===")
    print("Models saved in 'checkpoints/' directory")
    print("Results saved in 'results/' directory")

if __name__ == "__main__":
    main()