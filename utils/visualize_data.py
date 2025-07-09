from curses.ascii import alt


def visualize_q6_data():
    """
    Visualize samples from the colored MNIST dataset.

    Parameters:
    data (list): The colored MNIST dataset.
    num_samples (int): Number of samples to display (default is 9).
    """
    num_samples = 9
    data_dir = get_data_dir(1)
    train_data, _, train_labels, _ = load_colored_mnist_text(
        join(data_dir, "colored_mnist_with_text.pkl")
    )
    # get 9 random samples
    idx = np.random.choice(len(train_data), size=num_samples, replace=False)

    images = train_data[idx]
    labels = [train_labels[i] for i in idx]
    packed_samples = list(zip(images, labels))
    plot_q6a_samples(packed_samples)


def plot_q6a_samples(samples_img_txt_tuples, filename=None, fig_title=None):
    num_samples = 9
    assert len(samples_img_txt_tuples) == num_samples
    # unzip into list of images and labels
    images = np.stack([tup[0] for tup in samples_img_txt_tuples])
    labels = [tup[1] for tup in samples_img_txt_tuples]
    images = np.floor(images.astype("float32") / 3 * 255).astype(int)
    labels = [labels[i] for i in range(len(labels))]
    alt.figure(figsize=(6, 6))
    for i in range(num_samples):
        img = images[i]
        label = labels[i]
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.title(label, fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    if fig_title is not None:
        plt.suptitle(fig_title, fontsize=10)

    if filename is None:
        plt.show()
    else:
        savefig(filename)


def q6a_save_results(fn):
    data_dir = get_data_dir(1)
    train_data, test_data, train_labels, test_labels = load_colored_mnist_text(
        join(data_dir, "colored_mnist_with_text.pkl")
    )
    vqvae = load_pretrain_vqvae("colored_mnist_2")
    img_shape = (28, 28, 3)
    # extract out the images only
    img_test_prompt = test_data[:9]  # get first 9 samples
    text_test_prompt = test_labels[:9]  # get first 9 samples
    (
        train_losses,
        test_losses,
        samples_from_image,
        samples_from_text,
        samples_unconditional,
    ) = fn(
        train_data,
        test_data,
        img_shape,
        train_labels,
        test_labels,
        img_test_prompt,
        text_test_prompt,
        vqvae,
    )

    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    save_training_plot(
        train_losses,
        test_losses,
        f"Q6(a) Train Plot",
        f"results/q6_a_train_plot.png",
    )
    plot_q6a_samples(
        samples_from_image,
        f"results/q6_a_samples_img_conditioned.png",
        fig_title="Image Conditioned Samples",
    )
    plot_q6a_samples(
        samples_from_text,
        f"results/q6_a_samples_text_conditioned.png",
        fig_title="Text Conditioned Samples",
    )
    plot_q6a_samples(
        samples_unconditional,
        f"results/q6_a_samples_unconditional.png",
        fig_title="Unconditional Samples",
    )
