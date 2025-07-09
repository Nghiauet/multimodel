def run_igpt(train_data, test_data, image_shape, train_text, test_text, image_test_prompt, text_test_prompt, vqvae):
    """
    train_data: A (n_train, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    image_shape: tuple (H, W, C) The shape of the images in the dataset, indicating height, width, and number of color channels.
    train_text: list[str] Text data associated with each training image.
    test_text: list[str] Text data associated with each test image.
    image_test_prompt: (9, H, W, C) Image data used for generating conditional text samples during testing.
    text_test_prompt: list of 9 strings Text prompts used for generating conditional image samples during testing.
    vqvae: a vqvae model, trained on the relevant dataset

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a list of 9 (image, text), corresponding to the image conditioned samples
    - a list of 9 (image, text), corresponding to the text conditions samples
    - a list of 9 (image, text), corresponding to unconditional samples
    """
    # Fix the offset parameter for the tokenizer - it should be the vocab_size, not 0
    text_tokenizer = Tokenizer(train_text, vqvae.n_embeddings)
    
    H, W, C = image_shape
    batch_size = 128
    learning_rate = 1e-3
    num_epochs = 30
    d_model = 128
    n_heads = 4
    n_layers = 4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # determine sequence length and vocab size
    sequence_length = 58 # 49 + 6 +2 + 1
    # Total vocab size should include both image tokens and text tokens
    total_vocab_size = vqvae.n_embeddings + len(text_tokenizer.all_words)
    # get subset of data to test first 
    train_loader = create_dataset(train_data, train_text, vqvae, text_tokenizer, batch_size)
    test_loader = create_dataset(test_data, test_text, vqvae, text_tokenizer, batch_size)
    
    model = iGPT(total_vocab_size, sequence_length, d_model, n_heads, n_layers).to(device)
    train_losses, test_losses = train_igpt(model, train_loader, test_loader, 
                                            sequence_length, total_vocab_size, device,
                                            num_epochs, learning_rate)
    

    # Generate samples
    samples_text_conditioned = generate_conditional_samples_from_text(
        model, text_tokenizer, vqvae, text_test_prompt, device
    )
    
    samples_image_conditioned = generate_conditional_samples_from_image(
        model, text_tokenizer, vqvae, image_test_prompt, device
    )
    
    samples_unconditioned = generate_unconditional_samples(
        model, text_tokenizer, vqvae, device, num_samples=9
    )
    return train_losses, test_losses, samples_image_conditioned, samples_text_conditioned, samples_unconditioned