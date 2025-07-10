# Multimodal VQ-VAE + iGPT Training Pipeline

A comprehensive implementation of multimodal deep learning combining Vector Quantized Variational Autoencoders (VQ-VAE) for image tokenization with image-based Generative Pre-trained Transformers (iGPT) for joint image-text generation.

## Model Architecture Overview

### VQ-VAE (Vector Quantized Variational Autoencoder)

The VQ-VAE learns discrete representations of images through vector quantization:

```
Input Image (32x32x3) → Encoder → Continuous Features → Quantization → Discrete Tokens → Decoder → Reconstructed Image
```

#### Architecture Components:

**Encoder**:
- Conv2d layers with stride-2 downsampling: 32x32 → 16x16 → 8x8
- Two residual blocks for feature refinement
- Output: 8x8x256 feature maps

**Vector Quantization**:
- Codebook: 128 learnable embeddings of dimension 256
- Distance-based token assignment: `argmin(||z_e - e_k||²)`
- Straight-through estimator for gradient flow

**Decoder**:
- Two residual blocks for feature processing
- TransposeConv2d layers: 8x8 → 16x16 → 32x32
- Output: Reconstructed 32x32x3 image

#### Key Implementation Details:

```python
def _quantize_features(self, z_e):
    """Centralized quantization logic"""
    batch_size, channels, height, width = z_e.shape
    z_e_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, channels)
    
    # Compute distances to codebook vectors
    distances = torch.sum((z_e_flat.unsqueeze(1) - self.codebook.weight.unsqueeze(0))**2, dim=-1)
    tokens = torch.argmin(distances, dim=-1)
    
    # Quantize and reshape
    z_q_flat = self.codebook(tokens)
    z_q = z_q_flat.view(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
    
    return z_q, tokens.view(batch_size, height, width)
```

### iGPT (image Generative Pre-trained Transformer)

The iGPT processes sequences of discrete tokens (image + text) using a decoder-only transformer architecture:

```
Token Sequence → Embedding → Positional Encoding → Transformer Layers → Output Projection → Next Token Prediction
```

#### Architecture Components:

**Multi-Head Self-Attention**:
- Scaled dot-product attention with causal masking
- 4 attention heads, 128-dimensional model
- KV-cache optimization for efficient inference

**Transformer Decoder**:
- 4 transformer layers with pre-norm architecture
- Feed-forward networks with GELU activation
- Residual connections and layer normalization

**KV-Cache Implementation**:
```python
def forward(self, x, use_cache=False, past_key_values=None):
    # Use cached key-value pairs for efficient generation
    if use_cache and self.cached_k is not None:
        # Concatenate current k,v with cached k,v
        k = torch.cat([self.cached_k, k], dim=2)
        v = torch.cat([self.cached_v, v], dim=2)
        self.cached_k = k
        self.cached_v = v
    
    # Apply attention with causal masking
    output = self.attention(q, k, v, mask)
    return output, (k, v)
```

### Multimodal Token Processing

The system processes three types of tokens:

1. **Image Tokens**: VQ-VAE quantized representations (49 tokens from 7x7 grid)
2. **Text Tokens**: Word-level tokenization with vocabulary mapping
3. **Special Tokens**: BOS, end-of-image, end-of-text markers

#### Token Sequence Format:
```
[BOS] [IMAGE_TOKENS] [EOI] [TEXT_TOKENS] [EOT]
  1   +      49      +  1  +      6     +  1  = 58 tokens
```

## Loss Functions

### VQ-VAE Loss

The VQ-VAE training uses a composite loss function:

```python
def compute_loss(x, recon_x, z_e, z_q):
    # Reconstruction loss - MSE between input and reconstructed image
    recon_loss = F.mse_loss(recon_x, x)
    
    # VQ loss - moves codebook vectors towards encoder outputs
    vq_loss = F.mse_loss(z_q.detach(), z_e)
    
    # Commitment loss - encourages encoder outputs to commit to codebook
    commit_loss = F.mse_loss(z_e, z_q.detach())
    
    # Combined loss with commitment weight
    total_loss = recon_loss + vq_loss + 0.25 * commit_loss
    return total_loss
```

**Loss Components**:
- **Reconstruction Loss**: Ensures faithful image reconstruction
- **VQ Loss**: Updates codebook vectors towards encoder outputs
- **Commitment Loss**: Prevents encoder outputs from growing arbitrarily

### iGPT Loss

The iGPT uses standard autoregressive language modeling loss:

```python
def compute_loss(input_tokens, target_tokens, model, vocab_size):
    # Forward pass through transformer
    logits = model(input_tokens)  # [batch_size, seq_len, vocab_size]
    
    # Cross-entropy loss for next token prediction
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size), 
        target_tokens.reshape(-1)
    )
    return loss
```

## Training Process

### Two-Stage Training Pipeline

#### Stage 1: VQ-VAE Training
```python
# 1. Data preparation
train_data = normalize_images(train_data, method='vqvae')  # [-1, 1] range
train_loader = DataLoader(train_data, batch_size=128)

# 2. Model initialization  
vqvae = VQVAE(dim=256, K=128, D=256)
optimizer = Adam(vqvae.parameters(), lr=1e-3)

# 3. Training loop
for epoch in range(30):
    for batch in train_loader:
        z_e, z_q, recon_x = vqvae(batch)
        loss = compute_loss(batch, recon_x, z_e, z_q)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vqvae.parameters(), max_norm=1.0)
        optimizer.step()
```

#### Stage 2: iGPT Training
```python
# 1. Create multimodal dataset
text_tokenizer = Tokenizer(train_texts, vqvae.n_embeddings)
train_loader = create_dataset(train_images, train_texts, vqvae, text_tokenizer)

# 2. Model initialization
vocab_size = vqvae.n_embeddings + len(text_tokenizer.all_words)
igpt = iGPT(vocab_size=vocab_size, context_length=58, d_model=128, n_heads=4, n_layers=4)
optimizer = Adam(igpt.parameters(), lr=1e-3)

# 3. Training with learning rate scheduling
scheduler = create_cosine_scheduler(optimizer, warmup_steps=1000)
for epoch in range(30):
    for batch in train_loader:
        input_seq = batch[:, :-1]  # All tokens except last
        targets = batch[:, 1:]     # All tokens except first
        
        logits = igpt(input_seq)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(igpt.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
```

## Sample Generation

### Three Generation Modes

#### 1. Text-Conditioned Image Generation
```python
def generate_from_text(model, text_tokenizer, vqvae, text_prompt, device):
    # Tokenize text prompt
    text_tokens = text_tokenizer.text_encode(text_prompt)
    
    # Create input sequence: [BOS] [EOI] [TEXT_TOKENS] [EOT]
    input_seq = torch.cat([
        torch.tensor([text_tokenizer.bos_token]),
        torch.tensor([text_tokenizer.end_of_image_token]),
        text_tokens,
        torch.tensor([text_tokenizer.end_of_text_token])
    ])
    
    # Generate image tokens autoregressively
    with torch.no_grad():
        for pos in range(49):  # 7x7 image tokens
            logits = model(input_seq.unsqueeze(0))
            next_token = torch.multinomial(F.softmax(logits[0, -1, :], dim=-1), 1)
            input_seq = torch.cat([input_seq, next_token])
    
    # Decode image tokens to image
    image_tokens = input_seq[-49:].view(7, 7)
    generated_image = vqvae.decode_tokens(image_tokens.unsqueeze(0))
    
    return generated_image
```

#### 2. Image-Conditioned Text Generation
```python
def generate_from_image(model, text_tokenizer, vqvae, image, device):
    # Tokenize image using VQ-VAE
    image_tokens = vqvae.get_tokens(image.unsqueeze(0)).flatten()
    
    # Create input sequence: [BOS] [EOT] [IMAGE_TOKENS] [EOI]
    input_seq = torch.cat([
        torch.tensor([text_tokenizer.bos_token]),
        torch.tensor([text_tokenizer.end_of_text_token]),
        image_tokens,
        torch.tensor([text_tokenizer.end_of_image_token])
    ])
    
    # Generate text tokens
    generated_tokens = []
    for pos in range(6):  # Max 6 text tokens
        logits = model(input_seq.unsqueeze(0))
        next_token = torch.multinomial(F.softmax(logits[0, -1, :], dim=-1), 1)
        
        if next_token == text_tokenizer.end_of_text_token:
            break
            
        generated_tokens.append(next_token)
        input_seq = torch.cat([input_seq, next_token])
    
    # Decode text tokens to text
    generated_text = text_tokenizer.text_decode(generated_tokens)
    return generated_text
```

#### 3. Unconditional Generation
Generates random image-text pairs by sampling from the model distribution without conditioning.

## Quick Start Guide

### Installation
```bash
pip install torch torchvision numpy matplotlib
```

### Data Preparation
Place your data files in the `data/` directory:
- `mnist_colored.pkl`: VQ-VAE training data (colored MNIST images)
- `colored_mnist_with_text.pkl`: Multimodal training data (images + text descriptions)

### Running the Training Pipeline

#### Option 1: Full Pipeline (Recommended)
```bash
python main.py
```

This runs the complete two-stage training process:
1. **Data Loading**: Loads and visualizes both datasets
2. **VQ-VAE Training**: Trains for 30 epochs, saves to `checkpoints/vqvae_model.pth`
3. **iGPT Training**: Trains for 30 epochs, saves to `checkpoints/igpt_model.pth`
4. **Sample Generation**: Generates all three types of samples
5. **Visualization**: Creates plots and saves to `results/` directory

#### Option 2: Custom Training
```python
from training import VQVAETrainer, iGPTTrainer
from utils.data_processor import DataProcessor
from utils.utils import load_pickled_data, load_colored_mnist_text

# Load data
train_data, test_data = load_pickled_data('data/mnist_colored.pkl')
train_data_mm, test_data_mm, train_texts, test_texts = load_colored_mnist_text('data/colored_mnist_with_text.pkl')

# Prepare data loaders
train_loader, test_loader = DataProcessor.prepare_vqvae_data(train_data, test_data, batch_size=128)

# Initialize and train VQ-VAE
vqvae = VQVAE(dim=256, K=128, D=256).to(device)
optimizer = torch.optim.Adam(vqvae.parameters(), lr=1e-3)
trainer = VQVAETrainer(vqvae, optimizer, device)
train_losses, test_losses = trainer.train(train_loader, test_loader, num_epochs=30)

# Train iGPT (after VQ-VAE training)
text_tokenizer = Tokenizer(train_texts, vqvae.n_embeddings)
train_loader_igpt = create_dataset(train_data_mm, train_texts, vqvae, text_tokenizer, batch_size=128)
test_loader_igpt = create_dataset(test_data_mm, test_texts, vqvae, text_tokenizer, batch_size=128)

vocab_size = vqvae.n_embeddings + len(text_tokenizer.all_words)
igpt = iGPT(vocab_size=vocab_size, context_length=58, d_model=128, n_heads=4, n_layers=4).to(device)
optimizer_igpt = torch.optim.Adam(igpt.parameters(), lr=1e-3)
trainer_igpt = iGPTTrainer(igpt, optimizer_igpt, device, vocab_size, 58)
train_losses_igpt, test_losses_igpt = trainer_igpt.train(train_loader_igpt, test_loader_igpt, num_epochs=30)
```

### Configuration

Modify training parameters in `main.py`:

```python
config = {
    # VQ-VAE Configuration
    'vqvae_dim': 256,        # Encoder/decoder channels
    'vqvae_K': 128,          # Codebook size (number of discrete codes)
    'vqvae_D': 256,          # Codebook vector dimension
    'vqvae_epochs': 30,      # Training epochs
    'vqvae_lr': 1e-3,        # Learning rate
    
    # iGPT Configuration  
    'd_model': 128,          # Transformer hidden dimension
    'n_heads': 4,            # Number of attention heads
    'n_layers': 4,           # Number of transformer layers
    'sequence_length': 58,   # Maximum sequence length
    'igpt_epochs': 30,       # Training epochs
    'igpt_lr': 1e-3,         # Learning rate
    'dropout': 0.1,          # Dropout rate
    
    # General
    'batch_size': 128,       # Batch size for training
}
```

### Output Files

After training, you'll find:

**Checkpoints**:
- `checkpoints/vqvae_model.pth`: Trained VQ-VAE model
- `checkpoints/igpt_model.pth`: Trained iGPT model

**Results**:
- `results/training_curves.png`: Training loss plots
- `results/image_conditioned_samples.png`: Image→text samples
- `results/text_conditioned_samples.png`: Text→image samples  
- `results/unconditional_samples.png`: Unconditional samples

## Technical Implementation Details

### Memory Optimization
- **Gradient Clipping**: Prevents exploding gradients (max_norm=1.0)
- **KV-Cache**: Reduces computation during autoregressive generation
- **Batch Processing**: Efficient data loading and processing

### Training Stability
- **Learning Rate Scheduling**: Cosine decay with warmup for iGPT
- **Loss Balancing**: Weighted combination of VQ-VAE loss components
- **Gradient Normalization**: Stable training across different model components

### Code Architecture
- **Modular Design**: Separate trainer classes for each model type
- **Unified Data Processing**: Consistent data preparation pipeline
- **Extensible Framework**: Easy to add new model types and training strategies

This implementation provides a complete, production-ready system for multimodal generation with clear separation of concerns and robust training procedures.