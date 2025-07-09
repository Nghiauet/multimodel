def create_dataset(images, texts, vqvae, text_tokenizer, batch_size):
    # create a dataset of images and texts
    dataset = []
    bos_token = text_tokenizer.bos_token
    end_of_image_token = text_tokenizer.end_of_image_token
    end_of_text_token = text_tokenizer.end_of_text_token
 
    print(f"Creating dataset from {len(images)} samples...")
    
    # Pre-tokenize all text data at once for efficiency
    print("Pre-tokenizing all text data...")
    all_text_tokens = [text_tokenizer.text_encode(text) for text in texts]
    
    # Batch process images for VQVAE quantization
    print("Batch processing images...")
    batch_size_process = 128
    all_image_tokens = []

    for i in range(0, len(images), batch_size_process):
        batch_end = min(i + batch_size_process, len(images))
        batch_images = images[i:batch_end]
        
        # Process batch of images
        batch_image_tokens = vqvae.quantize(batch_images)
        
        # Flatten each image's tokens and store
        for j in range(batch_image_tokens.shape[0]):
            image_tokens_flat = batch_image_tokens[j].flatten()
            all_image_tokens.append(image_tokens_flat)
        
        if i % (batch_size_process * 1000) == 0:
            print(f"Processed {min(i + batch_size_process, len(images))}/{len(images)} images ({min(i + batch_size_process, len(images))/len(images)*1000:.1f}%)")
    
    # Create special token tensors once
    bos_tensor = torch.tensor([bos_token])
    end_of_image_tensor = torch.tensor([end_of_image_token])
    end_of_text_tensor = torch.tensor([end_of_text_token])
    
    print("Assembling dataset...")
    for idx in range(len(texts)):
        text_tokens = all_text_tokens[idx]
        image_tokens_flat = all_image_tokens[idx]
        
        if idx % 2 == 0:
            # text followed by image
            complete_tokens = torch.cat((bos_tensor, end_of_image_tensor, text_tokens, end_of_text_tensor, image_tokens_flat))
            dataset.append(complete_tokens)
        else:
            # image followed by text
            complete_tokens = torch.cat((bos_tensor, end_of_text_tensor, image_tokens_flat, end_of_image_tensor, text_tokens))
            dataset.append(complete_tokens)
    
    print(f"Dataset creation complete! Total samples: {len(dataset)}")
    print(f"Creating DataLoader with batch_size={batch_size}")
    
    # create dataloader
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)