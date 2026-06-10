import os
import torch
import matplotlib.pyplot as plt


def plot(
    *images,
    titles: list[str] | None = None,
    size: float = 3,
    save_path: str | None = None,
    dpi: int = 300,
) -> None:
    """
    Display one or more images in a single horizontal row.

    The function accepts PyTorch tensors in common image formats and
    automatically moves them to CPU before plotting. Batched tensors are
    displayed by taking the first image in the batch.

    Supported input shapes are:

    - ``[H, W]``: grayscale image.
    - ``[1, H, W]``: single-channel image.
    - ``[3, H, W]``: RGB image.
    - ``[1, 1, H, W]``: batched single-channel image.
    - ``[1, 3, H, W]``: batched RGB image.
    - ``[B, 1, H, W]``: batched single-channel images.
    - ``[B, 3, H, W]``: batched RGB images.

    Parameters
    ----------
    *images : torch.Tensor
        One or more images to display.

    titles : list[str] | None, default=None
        Optional list of titles. If provided, it must have the same length
        as the number of images.

    size : float, default=3
        Size multiplier used to determine the figure size. The final figure
        size is ``(size * number_of_images, size)``.

    save_path : str | None, default=None
        Optional path where the plotted figure is saved. If None, the figure
        is only displayed.

    dpi : int, default=300
        Resolution used when saving the figure.

    Raises
    ------
    ValueError
        If no images are provided.

    ValueError
        If the number of titles does not match the number of images.

    ValueError
        If an image has an unsupported shape.

    Examples
    --------
    >>> plot(img)

    >>> plot(
    ...     clean,
    ...     blurred,
    ...     noisy,
    ...     titles=["Clean", "Blurred", "Noisy"],
    ... )

    >>> plot(
    ...     reconstruction,
    ...     save_path="results/reconstruction.png",
    ... )
    """

    n_images = len(images)

    if n_images == 0:
        raise ValueError("At least one image must be provided.")

    if titles is not None and len(titles) != n_images:
        raise ValueError(
            f"Expected {n_images} titles, got {len(titles)}."
        )

    fig, axes = plt.subplots(
        1,
        n_images,
        figsize=(size * n_images, size),
    )

    if n_images == 1:
        axes = [axes]

    for index, (ax, image) in enumerate(zip(axes, images)):

        if not isinstance(image, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor, got {type(image).__name__}."
            )

        image = image.detach().cpu()

        if image.ndim == 4:
            image = image[0]

        if image.ndim == 3 and image.shape[0] == 1:
            image = image.squeeze(0)

        if image.ndim == 2:
            ax.imshow(
                image,
                cmap="gray",
            )

        elif image.ndim == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)
            image = image.clamp(0, 1)

            ax.imshow(image)

        else:
            raise ValueError(
                f"Unsupported image shape: {tuple(image.shape)}."
            )

        if titles is not None:
            ax.set_title(titles[index])

        ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        directory = os.path.dirname(save_path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        plt.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
        )

    plt.show()