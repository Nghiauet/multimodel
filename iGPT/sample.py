def generate_conditional_samples_from_text(model, text_tokenizer, vqvae, text_prompts, device, max_length=58):
    """
    Generate images conditioned on text prompts.
    
    Args:
        model: Trained iGPT model
        text_tokenizer: Text tokenizer
        vqvae: VQVAE model for decoding image tokens
        text_prompts: List of text strings to condition on
        device: Device to run on
        max_length: Maximum sequence length
        
    Returns:
        List of (image, text) tuples
    """
    model.eval()
    samples = []
    
    with torch.no_grad():
        for text_prompt in text_prompts:
            # Start with BOS token and end_of_image token, then text tokens, then end_of_text token
            text_tokens = text_tokenizer.text_encode(text_prompt)
            input_seq = torch.cat([
                torch.tensor([text_tokenizer.bos_token]),
                torch.tensor([text_tokenizer.end_of_image_token]),
                text_tokens,
                torch.tensor([text_tokenizer.end_of_text_token])
            ]).unsqueeze(0).to(device)
            
            # Generate 49 image tokens
            for _ in range(49):  # 7x7 = 49 image tokens
                logits = model(input_seq)
                next_token_logits = logits[0, -1, :]
                
                # Restrict to image tokens only (0 to vqvae.n_embeddings-1)
                mask = torch.zeros_like(next_token_logits)
                mask[:vqvae.n_embeddings] = 1
                next_token_logits = next_token_logits * mask + (1 - mask) * (-1e9)
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append to sequence
                input_seq = torch.cat([input_seq, next_token.unsqueeze(0)], dim=1)
            
            # Extract image tokens and decode
            image_tokens = input_seq[0, -49:].cpu().numpy().reshape(7, 7)
            decoded_image = vqvae.decode(image_tokens.reshape(1, 7, 7))[0]
            
            samples.append((decoded_image, text_prompt))
    
    return samples

def generate_conditional_samples_from_image(model, text_tokenizer, vqvae, image_prompts, device, max_length=58):
    """
    Generate text conditioned on image prompts.
    
    Args:
        model: Trained iGPT model
        text_tokenizer: Text tokenizer
        vqvae: VQVAE model for encoding image tokens
        image_prompts: Array of images to condition on
        device: Device to run on
        max_length: Maximum sequence length
        
    Returns:
        List of (image, text) tuples
    """
    model.eval()
    samples = []
    
    with torch.no_grad():
        for image_prompt in image_prompts:
            # Quantize the image
            image_tokens = vqvae.quantize(image_prompt.reshape(1, *image_prompt.shape))[0].flatten()
            
            # Start with BOS token, end_of_text token, image tokens, then end_of_image token
            input_seq = torch.cat([
                torch.tensor([text_tokenizer.bos_token]),
                torch.tensor([text_tokenizer.end_of_text_token]),
                torch.tensor(image_tokens),
                torch.tensor([text_tokenizer.end_of_image_token])
            ]).unsqueeze(0).to(device)
            
            # Generate text tokens (typically 6 words based on the dataset)
            generated_text_tokens = []
            for _ in range(6):  # Assuming 6 words per text description
                logits = model(input_seq)
                next_token_logits = logits[0, -1, :]
                
                # Restrict to text tokens only (excluding special tokens)
                mask = torch.zeros_like(next_token_logits)
                # Text tokens start from vqvae.n_embeddings + 1 (excluding BOS which is 0)
                for word, token_id in text_tokenizer.word_to_id.items():
                    if word not in ['<end_of_text>', '<end_of_image>']:
                        mask[token_id] = 1
                
                next_token_logits = next_token_logits * mask + (1 - mask) * (-1e9)
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                generated_text_tokens.append(next_token.item())
                
                # Append to sequence
                input_seq = torch.cat([input_seq, next_token.unsqueeze(0)], dim=1)
            
            # Decode text
            generated_text = text_tokenizer.text_decode(generated_text_tokens)
            
            samples.append((image_prompt, generated_text))
    
    return samples

def generate_unconditional_samples(model, text_tokenizer, vqvae, device, num_samples=9, max_length=58):
    """
    Generate unconditional samples (both text and images).
    
    Args:
        model: Trained iGPT model
        text_tokenizer: Text tokenizer
        vqvae: VQVAE model for decoding
        device: Device to run on
        num_samples: Number of samples to generate
        max_length: Maximum sequence length
        
    Returns:
        List of (image, text) tuples
    """
    model.eval()
    samples = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Start with BOS token
            input_seq = torch.tensor([text_tokenizer.bos_token]).unsqueeze(0).to(device)
            
            # First, decide which modality to start with
            logits = model(input_seq)
            next_token_logits = logits[0, -1, :]
            
            # Only allow end_of_image or end_of_text tokens
            mask = torch.zeros_like(next_token_logits)
            mask[text_tokenizer.end_of_image_token] = 1
            mask[text_tokenizer.end_of_text_token] = 1
            next_token_logits = next_token_logits * mask + (1 - mask) * (-1e9)
            
            probs = torch.softmax(next_token_logits, dim=-1)
            modality_token = torch.multinomial(probs, 1)
            input_seq = torch.cat([input_seq, modality_token.unsqueeze(0)], dim=1)
            
            if modality_token.item() == text_tokenizer.end_of_image_token:
                # Generate text first, then image
                
                # Generate 6 text tokens
                for _ in range(6):
                    logits = model(input_seq)
                    next_token_logits = logits[0, -1, :]
                    
                    # Restrict to text tokens
                    mask = torch.zeros_like(next_token_logits)
                    for word, token_id in text_tokenizer.word_to_id.items():
                        if word not in ['<end_of_text>', '<end_of_image>']:
                            mask[token_id] = 1
                    
                    next_token_logits = next_token_logits * mask + (1 - mask) * (-1e9)
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    input_seq = torch.cat([input_seq, next_token.unsqueeze(0)], dim=1)
                
                # Add end_of_text token
                input_seq = torch.cat([input_seq, torch.tensor([text_tokenizer.end_of_text_token]).unsqueeze(0).to(device)], dim=1)
                
                # Generate 49 image tokens
                for _ in range(49):
                    logits = model(input_seq)
                    next_token_logits = logits[0, -1, :]
                    
                    # Restrict to image tokens
                    mask = torch.zeros_like(next_token_logits)
                    mask[:vqvae.n_embeddings] = 1
                    next_token_logits = next_token_logits * mask + (1 - mask) * (-1e9)
                    
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    input_seq = torch.cat([input_seq, next_token.unsqueeze(0)], dim=1)
                
                # Extract text and image
                text_tokens = input_seq[0, 2:8].cpu().numpy()  # Skip BOS, end_of_image, get 6 text tokens
                image_tokens = input_seq[0, -49:].cpu().numpy().reshape(7, 7)
                
            else:  # end_of_text_token
                # Generate image first, then text
                
                # Generate 49 image tokens
                for _ in range(49):
                    logits = model(input_seq)
                    next_token_logits = logits[0, -1, :]
                    
                    # Restrict to image tokens
                    mask = torch.zeros_like(next_token_logits)
                    mask[:vqvae.n_embeddings] = 1
                    next_token_logits = next_token_logits * mask + (1 - mask) * (-1e9)
                    
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    input_seq = torch.cat([input_seq, next_token.unsqueeze(0)], dim=1)
                
                # Add end_of_image token
                input_seq = torch.cat([input_seq, torch.tensor([text_tokenizer.end_of_image_token]).unsqueeze(0).to(device)], dim=1)
                
                # Generate 6 text tokens
                for _ in range(6):
                    logits = model(input_seq)
                    next_token_logits = logits[0, -1, :]
                    
                    # Restrict to text tokens
                    mask = torch.zeros_like(next_token_logits)
                    for word, token_id in text_tokenizer.word_to_id.items():
                        if word not in ['<end_of_text>', '<end_of_image>']:
                            mask[token_id] = 1
                    
                    next_token_logits = next_token_logits * mask + (1 - mask) * (-1e9)
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    input_seq = torch.cat([input_seq, next_token.unsqueeze(0)], dim=1)
                
                # Extract image and text
                image_tokens = input_seq[0, 2:51].cpu().numpy().reshape(7, 7)  # Skip BOS, end_of_text, get 49 image tokens
                text_tokens = input_seq[0, -6:].cpu().numpy()  # Get last 6 text tokens
            
            # Decode
            decoded_image = vqvae.decode(image_tokens.reshape(1, 7, 7))[0]
            decoded_text = text_tokenizer.text_decode(text_tokens)
            
            samples.append((decoded_image, decoded_text))
    
    return samples