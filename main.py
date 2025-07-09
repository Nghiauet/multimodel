import torch
import torch.optim as optim
import numpy as np
import os

# Import model modules
from VQ_VAE.model import VQVAE
from iGPT.model import iGPT
from iGPT.tokenizer import Tokenizer
from iGPT.create_dataset import create_dataset
from iGPT.sample import generate_conditional_samples_from_text, generate_conditional_samples_from_image, generate_unconditional_samples

# Import utility functions
from utils.utils import load_pickled_data, load_colored_mnist_text
from utils.data_processor import DataProcessor
from utils.visualization import Visualizer

# Import training modules
from training import VQVAETrainer, iGPTTrainer



def train_vqvae(train_data, test_data, device, config):
    """Train VQ-VAE model using the new training framework"""
    print("=== Training VQ-VAE ===")
    
    # Prepare data using unified data processor
    train_loader, test_loader = DataProcessor.prepare_vqvae_data(
        train_data, test_data, config['batch_size']
    )
    
    # Initialize model
    model = VQVAE(
        dim=config['vqvae_dim'], 
        K=config['vqvae_K'], 
        D=config['vqvae_D']
    ).to(device)
    
    # Add quantize method for compatibility
    model = DataProcessor.add_quantize_method(model, device)
    
    # Initialize optimizer and trainer
    optimizer = optim.Adam(model.parameters(), lr=config['vqvae_lr'])
    trainer = VQVAETrainer(model, optimizer, device)
    
    # Train the model
    train_losses, test_losses = trainer.train(
        train_loader, test_loader, config['vqvae_epochs']
    )
    
    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/vqvae_model.pth')
    print("VQ-VAE model saved to checkpoints/vqvae_model.pth")
    
    return model, train_losses, test_losses

def train_multimodal_igpt(train_data, test_data, train_texts, test_texts, vqvae_model, device, config):
    """Train multimodal iGPT model using the new training framework"""
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
    
    # Initialize optimizer and trainer
    optimizer = optim.Adam(model.parameters(), lr=config['igpt_lr'])
    trainer = iGPTTrainer(model, optimizer, device, total_vocab_size, sequence_length)
    
    # Train the model
    train_losses, test_losses = trainer.train(
        train_loader, test_loader, config['igpt_epochs']
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
    """Visualize generated samples using unified visualization"""
    print("\n=== Visualizing Results ===")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Visualize different types of samples
    Visualizer.show_image_text_grid(
        samples_image_conditioned, 
        title="Image-Conditioned Samples",
        save_path="results/image_conditioned_samples.png"
    )
    
    Visualizer.show_image_text_grid(
        samples_text_conditioned, 
        title="Text-Conditioned Samples",
        save_path="results/text_conditioned_samples.png"
    )
    
    Visualizer.show_image_text_grid(
        samples_unconditioned, 
        title="Unconditional Samples",
        save_path="results/unconditional_samples.png"
    )

def plot_training_curves(vqvae_train_losses, vqvae_test_losses, igpt_train_losses, igpt_test_losses):
    """Plot training curves using unified visualization"""
    print("\n=== Plotting Training Curves ===")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Use unified visualization for combined training curves
    Visualizer.plot_combined_training_curves(
        vqvae_train_losses, vqvae_test_losses,
        igpt_train_losses, igpt_test_losses,
        save_path="results/training_curves.png"
    )

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
    train_data_vqvae, test_data_vqvae = load_pickled_data('data/mnist_colored.pkl')
    
    print(f"VQ-VAE train data shape: {train_data_vqvae.shape}")
    print(f"VQ-VAE test data shape: {test_data_vqvae.shape}")
    
    # Load multimodal data
    print("Loading multimodal data...")
    train_data_mm, test_data_mm, train_texts, test_texts = load_colored_mnist_text('data/colored_mnist_with_text.pkl')
    
    print(f"Multimodal train data shape: {train_data_mm.shape}")
    print(f"Multimodal test data shape: {test_data_mm.shape}")
    print(f"Number of training texts: {len(train_texts)}")
    print(f"Number of test texts: {len(test_texts)}")
    
    # Visualize data
    print("\n=== Visualizing Data ===")
    Visualizer.show_image_grid(train_data_vqvae, title="VQ-VAE Training Data")
    Visualizer.show_data_samples(train_data_mm, train_texts, title="Multimodal Training Data")
    
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