import numpy as np
import random
import torch
import torchvision.transforms.functional as trfm_func
import torchvision.transforms as trfm
from typing import Tuple


def rotate(input_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random 90-degree rotations to a batch of images.

    Args:
        input_data: Image batch with shape (batch, channels, height, width)

    Returns:
        Tuple containing:
        - Rotated image batch with same shape as input
        - Array of rotation angles (0, 1, 2, or 3 * 90 degrees)
    """
    # Generate random rotation angles (0, 1, 2, or 3 * 90 degrees)
    rotation_angles = np.random.randint(0, 4, input_data.shape[0])

    # Perform rotations with NumPy's rot90
    rotated_data = np.stack([
        np.rot90(img, angle, axes=(1, 2))
        for img, angle in zip(input_data, rotation_angles)
    ])

    return rotated_data, rotation_angles


def horizontal_flip(input_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly flip half of the images horizontally.

    Args:
        input_data: Image batch with shape (batch, channels, height, width)

    Returns:
        Tuple containing:
        - Modified image batch
        - Binary array indicating which images were flipped
    """
    # Create a copy to avoid modifying the original data
    modified_data = input_data.copy()

    # Randomly select indices to flip
    indices = np.random.choice(
        input_data.shape[0],
        size=input_data.shape[0] // 2,
        replace=False
    )

    # Track flipped images
    flipped_mask = np.zeros(input_data.shape[0], dtype=int)

    # Apply horizontal flip
    for index in indices:
        modified_data[index] = trfm_func.hflip(torch.as_tensor(modified_data[index])).numpy()
        flipped_mask[index] = 1

    return modified_data, flipped_mask


def cropping(input_data: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Apply random cropping and resize to original image dimensions.

    Args:
        input_data: Image batch with shape (batch, channels, height, width)

    Returns:
        Tuple containing:
        - Resized image batch
        - Crop parameters (top, left, height, width)
    """
    # Randomize crop parameters
    top = np.random.randint(0, input_data.shape[2] // 2)
    left = np.random.randint(0, input_data.shape[3] // 2)
    height = np.random.randint(input_data.shape[2] // 2, input_data.shape[2] - top)
    width = np.random.randint(input_data.shape[3] // 2, input_data.shape[3] - left)

    # Crop and resize
    cropped = trfm_func.crop(torch.as_tensor(input_data), top, left, height, width)
    cropped_resized = trfm_func.resize(
        cropped,
        [input_data.shape[2], input_data.shape[3]],
        antialias=True
    )

    return cropped_resized.numpy(), (top, left, height, width)


def gauss_blur(input_data: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Apply Gaussian blur to image batch with random kernel size.

    Args:
        input_data: Image batch with shape (batch, channels, height, width)

    Returns:
        Tuple containing:
        - Blurred image batch
        - Gaussian kernel dimensions (height, width)
    """
    # Generate odd-sized kernel dimensions
    gau_h = random.randrange(1, 11, 2)
    gau_w = random.randrange(1, 11, 2)

    blurred = trfm_func.gaussian_blur(torch.as_tensor(input_data), [gau_h, gau_w])

    return blurred.numpy(), (gau_h, gau_w)


def color_distortions(input_data: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Apply random color distortions to image batch.

    Args:
        input_data: Image batch with shape (batch, channels, height, width)

    Returns:
        Tuple containing:
        - Color-distorted image batch
        - Placeholder transformation parameters
    """

    brightness = random.random()
    contrast = random.random()
    saturation = random.random()
    hue = random.random() / 2

    transform = trfm.ColorJitter(brightness, contrast, saturation, hue)
    distorted = transform.forward(torch.as_tensor(input_data))

    return distorted.numpy(), (brightness, contrast, saturation, hue)


def masking(input_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mask 75% of image batch by randomly removing 16x16 pixel segments.

    Args:
        input_data: Image batch with shape (batch, channels, height, width)

    Returns:
        Tuple containing:
        - Masked image batch
        - Binary mask indicating removed regions
    """
    # Randomize patch selection
    patch_indices = np.arange(256)
    np.random.shuffle(patch_indices)

    # Calculate patch dimensions
    patch_height = input_data.shape[2] // 16
    patch_width = input_data.shape[3] // 16

    # Initialize mask and masked data
    mask = np.ones((input_data.shape[0], 1, input_data.shape[2], input_data.shape[3]))
    masked_data = np.zeros_like(input_data)

    # Apply masking to selected patches
    for patch_index in patch_indices[:int(len(patch_indices) * 0.25)]:
        row = patch_index // 16 * patch_height
        col = patch_index % 16 * patch_width

        mask[:, :, row:row + patch_height, col:col + patch_width] = 0
        masked_data[:, :, row:row + patch_height, col:col + patch_width] = \
            input_data[:, :, row:row + patch_height, col:col + patch_width]

    return masked_data, mask