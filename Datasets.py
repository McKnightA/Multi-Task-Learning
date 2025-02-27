from pathlib import Path
from typing import Tuple, Union, List
import numpy as np
import torch
from datasets import load_dataset
from torch import Tensor
from PIL import Image


def resize_image_batch(
        images: List[Image.Image],
        target_size: Union[int, Tuple[int, int]]
) -> List[np.ndarray]:
    """
    Resize a batch of images to a target size and convert to numpy arrays.

    Handles both color and grayscale images, ensuring all output images
    have three dimensions (height, width, channels).

    Args:
        images: List of PIL Image objects to be resized
        target_size: Either a single integer for square output, or
                    a tuple of (height, width) for rectangular output

    Returns:
        List of numpy arrays with shape (height, width, channels),
        where channels is 1 for grayscale and 3 for RGB images
    """
    # Convert single integer size to tuple if needed
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    processed_images = []

    for image in images:
        # Resize the image
        resized_image = image.resize(target_size)

        # Convert to numpy array
        image_array = np.array(resized_image)

        # Handle grayscale images by adding channel dimension
        if len(image_array.shape) < 3:
            image_array = np.expand_dims(image_array, axis=-1)

        processed_images.append(image_array)

    return processed_images


def pad_image_batch(
        images: List[np.ndarray],
        target_size: Union[int, Tuple[int, int]]
) -> np.ndarray:
    """
    Pad a batch of images to a consistent size with centered alignment.

    Pads images with zeros to reach the target size while maintaining
    the original image content in the center of the padded area.

    Args:
        images: List of numpy arrays with shape (height, width, channels)
        target_size: Desired output size as (height, width)

    Returns:
        numpy.ndarray: Padded image batch with shape
                      (batch_size, target_height, target_width, channels)
    """
    # Convert single integer size to tuple if needed
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    batch_size = len(images)
    target_height, target_width = target_size
    channels = 3  # RGB images

    # Initialize output array with zeros
    padded_batch = np.zeros(
        (batch_size, target_height, target_width, channels),
        dtype=np.uint8
    )

    for idx, image in enumerate(images):
        # Calculate padding sizes
        height_padding = target_height - image.shape[0]
        width_padding = target_width - image.shape[1]

        # Calculate padding boundaries for centered alignment
        top_pad = height_padding // 2
        bottom_pad = top_pad + image.shape[0]
        left_pad = width_padding // 2
        right_pad = left_pad + image.shape[1]

        # Insert image into padded array
        padded_batch[idx, top_pad:bottom_pad, left_pad:right_pad, :] = image

    return padded_batch


def stack_and_transpose_images(
        images: list
) -> np.ndarray:
    """
    Stack and transpose images to PyTorch expected format.
    Args:
        images: List of images
    Returns:
        numpy.ndarray: Stacked and transposed images with shape
                      (batch_size, channels, height, width)
    """
    # Stack images along batch dimension
    stacked = np.stack(images, axis=0)
    # Transpose from (B, H, W, C) to (B, C, H, W)
    return stacked.transpose(0, 3, 1, 2)


class Cifar10:
    """
    CIFAR-10 dataset loader and preprocessor for deep learning tasks.

    This class handles loading and preprocessing of the CIFAR-10 dataset,
    supporting both training and test splits. It includes functionality for
    loading image batches and their corresponding labels, with support for
    image resizing and format conversion.

    Reference: https://huggingface.co/datasets/cifar10
    """

    def __init__(
            self,
            device: str = "cpu"
    ) -> None:
        """
        Initialize the CIFAR-10 dataset.

        Args:
            device: Device to load the tensors on ('cpu' or 'cuda')
        """
        self.data_path = Path("D:\\cifar10")
        self.device = device
        self.name = "CIFAR-10"
        self.num_classes = 10

        # Load training and test datasets
        self.train_dataset = load_dataset(
            str(self.data_path),
            split="train",
            streaming=True
        )
        self.test_dataset = load_dataset(
            str(self.data_path),
            split="test",
            streaming=True
        )

    def process_batch(
            self,
            batch: dict,
            target_size: Union[int, Tuple[int, int]]
    ) -> Tuple[np.ndarray, Tensor]:
        """
        Process a batch of images and labels from the dataset.

        Args:
            batch: Dictionary containing 'img' and 'label' keys
                  - 'img': List of images
                  - 'label': List of class labels
            target_size: Desired output size for images as (height, width)

        Returns:
            Tuple containing:
            - Processed image batch as numpy array with shape
              (batch_size, channels, height, width)
            - Labels as torch.Tensor with shape (batch_size,)
        """
        # Resize images to target size
        resized_images = resize_image_batch(
            batch["img"],
            target_size
        )

        # Stack and transpose images to (B, C, H, W) format
        processed_images = stack_and_transpose_images(resized_images)

        # Convert labels to tensor
        labels = torch.as_tensor(
            batch['label'],
            dtype=torch.long,
            device=self.device
        )

        return processed_images, labels


class Cifar100:
    """
    CIFAR-100 dataset loader and preprocessor for deep learning tasks.

    This class handles loading and preprocessing of the CIFAR-100 dataset,
    supporting both training and test splits. It includes functionality for
    loading image batches and their corresponding fine-grained labels, with
    support for image resizing and format conversion.

    Reference: https://huggingface.co/datasets/cifar100
    """

    def __init__(
            self,
            device: str = "cpu"
    ) -> None:
        """
        Initialize the CIFAR-100 dataset.

        Args:
            device: Device to load the tensors on ('cpu' or 'cuda')
        """
        self.data_path = Path("D:\\cifar100")
        self.device = device
        self.name = "CIFAR-100"
        self.num_classes = 100

        # Load training and test datasets
        self.train_dataset = load_dataset(
            str(self.data_path),
            split="train",
            streaming=True
        )
        self.test_dataset = load_dataset(
            str(self.data_path),
            split="test",
            streaming=True
        )

    def process_batch(
            self,
            batch: dict,
            target_size: Union[int, Tuple[int, int]]
    ) -> Tuple[np.ndarray, Tensor]:
        """
        Process a batch of images and labels from the dataset.

        Args:
            batch: Dictionary containing 'img' and 'fine_label' keys
                  - 'img': List of images
                  - 'fine_label': List of fine-grained class labels
            target_size: Desired output size for images as (height, width)

        Returns:
            Tuple containing:
            - Processed image batch as numpy array with shape
              (batch_size, channels, height, width)
            - Labels as torch.Tensor with shape (batch_size,)
        """
        # Resize images to target size
        resized_images = resize_image_batch(
            batch["img"],
            target_size
        )

        # Stack and transpose images to (B, C, H, W) format
        processed_images = stack_and_transpose_images(resized_images)

        # Convert labels to tensor
        labels = torch.as_tensor(
            batch['fine_label'],
            dtype=torch.long,
            device=self.device
        )

        return processed_images, labels


class Beans:
    """
    Beans disease classification dataset loader and preprocessor.

    This class handles loading and preprocessing of the Beans dataset from
    AI Lab Makerere, supporting both training and validation splits. The dataset
    contains images of bean plants with 3 classes: healthy, angular leaf spot,
    and bean rust.

    Reference: https://huggingface.co/datasets/beans
    """

    def __init__(
            self,
            device: str = "cpu"
    ) -> None:
        """
        Initialize the Beans dataset.

        Args:
            device: Device to load the tensors on ('cpu' or 'cuda')
        """
        self.data_path = Path("D:\\beans")
        self.device = device
        self.name = "Beans"
        self.num_classes = 3

        # Load training and validation datasets
        self.train_dataset = load_dataset(
            str(self.data_path),
            split="train",
            streaming=True
        )
        self.test_dataset = load_dataset(
            str(self.data_path),
            split="validation",
            streaming=True
        )

    def process_batch(
            self,
            batch: dict,
            target_size: Union[int, Tuple[int, int]]
    ) -> Tuple[np.ndarray, Tensor]:
        """
        Process a batch of images and labels from the dataset.

        Args:
            batch: Dictionary containing 'image' and 'labels' keys
                  - 'image': List of bean plant images
                  - 'labels': List of disease classification labels
                    (0: healthy, 1: angular leaf spot, 2: bean rust)
            target_size: Desired output size for images as (height, width)

        Returns:
            Tuple containing:
            - Processed image batch as numpy array with shape
              (batch_size, channels, height, width)
            - Labels as torch.Tensor with shape (batch_size,)
        """
        # Resize images to target size
        resized_images = resize_image_batch(
            batch["image"],
            target_size
        )

        # Stack and transpose images to (B, C, H, W) format
        processed_images = stack_and_transpose_images(resized_images)

        # Convert labels to tensor
        labels = torch.as_tensor(
            batch['labels'],
            dtype=torch.long,
            device=self.device
        )

        return processed_images, labels


class Svhn:
    """
    Street View House Numbers (SVHN) dataset loader and preprocessor.

    This class handles loading and preprocessing of the SVHN dataset from
    Stanford University, supporting both training and test splits. The dataset
    contains real-world images of digits from house numbers in Google Street
    View images, with 10 classes (0-9).

    Reference: https://huggingface.co/datasets/svhn
    """

    def __init__(
            self,
            device: str = "cpu"
    ) -> None:
        """
        Initialize the SVHN dataset.

        Args:
            device: Device to load the tensors on ('cpu' or 'cuda')
        """
        self.data_path = Path("D:\\svhn")
        self.device = device
        self.name = "SVHN"
        self.num_classes = 10

        # Load training and test datasets
        self.train_dataset = load_dataset(
            str(self.data_path),
            split="train",
            streaming=True
        )
        self.test_dataset = load_dataset(
            str(self.data_path),
            split="test",
            streaming=True
        )

    def process_batch(
            self,
            batch: dict,
            target_size: Union[int, Tuple[int, int]]
    ) -> Tuple[np.ndarray, Tensor]:
        """
        Process a batch of images and labels from the dataset.

        Args:
            batch: Dictionary containing 'image' and 'label' keys
                  - 'image': List of digit images from street numbers
                  - 'label': List of digit labels (0-9)
            target_size: Desired output size for images as (height, width)

        Returns:
            Tuple containing:
            - Processed image batch as numpy array with shape
              (batch_size, channels, height, width)
            - Labels as torch.Tensor with shape (batch_size,)
        """
        # Resize images to target size
        resized_images = resize_image_batch(
            batch["image"],
            target_size
        )

        # Stack and transpose images to (B, C, H, W) format
        processed_images = stack_and_transpose_images(resized_images)

        # Convert labels to tensor
        labels = torch.as_tensor(
            batch['label'],
            dtype=torch.long,
            device=self.device
        )

        return processed_images, labels


class Fmd:
    """
    The Flickr Material Database (FMD) consists of color photographs
    of surfaces belonging to one of ten common material categories:
    fabric, foliage, glass, leather, metal, paper, plastic, stone,
    water, and wood. There are 100 images in each category, 50
    close-ups and 50 regular views. Each image contains surfaces
    belonging to a single material category in the foreground and
    was selected manually from approximately 50 candidates to ensure
    a variety of illumination conditions, compositions, colors,
    textures, surface shapes, material sub-types, and object associations.
    Reference: https://huggingface.co/datasets/mcimpoi/fmd_materials

    L. Sharan, R. Rosenholtz, and E. H. Adelson, "Accuracy and speed of
    material categorization in real-world images", Journal of Vision,
    vol. 14, no. 9, article 12, 2014
    """
    def __init__(
            self,
            device: str = "cpu"
    ) -> None:
        """
        Initialize the FMD Materials dataset.
        Args:
            device: Device to load the tensors on ('cpu' or 'cuda')
        """
        self.data_path = Path("D:\\mcimpoi___fmd_materials")
        self.device = device
        self.name = "fmd"
        self.num_classes = 10
        # Load training and test datasets
        self.train_dataset = load_dataset(
            str(self.data_path),
            split="train",
            streaming=True
        )
        self.test_dataset = load_dataset(
            str(self.data_path),
            split="test",
            streaming=True
        )

    def process_batch(
            self,
            batch: dict,
            target_size: Union[int, Tuple[int, int]]
    ) -> Tuple[np.ndarray, Tensor]:
        """
        Process a batch of images and labels from the dataset.
        Args:
            batch: Dictionary containing 'image' and 'label' keys
                  - 'image': List of material images
                  - 'label': List of digit labels (0-9)
            target_size: Desired output size for images as (height, width)
        Returns:
            Tuple containing:
            - Processed image batch as numpy array with shape
              (batch_size, channels, height, width)
            - Labels as torch.Tensor with shape (batch_size,)
        """
        # Resize images to target size
        resized_images = resize_image_batch(
            batch["image"],
            target_size
        )

        # some images may have c == 1
        for i, img in enumerate(resized_images):
            if img.shape[-1] == 1:
                resized_images[i] = np.stack([img, img, img], axis=2).squeeze(axis=-1)

        # Stack and transpose images to (B, C, H, W) format
        processed_images = stack_and_transpose_images(resized_images)
        # Convert labels to tensor
        labels = torch.as_tensor(
            batch['label'],
            dtype=torch.long,
            device=self.device
        )
        return processed_images, labels


class Pokemon:
    """
        The Pokemon Classification dataset consists of color images of 110 different pokemon.
        Reference: https://huggingface.co/datasets/keremberke/pokemon-classification

        @misc{ pokedex_dataset,
            title = { Pokedex Dataset },
            type = { Open Source Dataset },
            author = { Lance Zhang },
            howpublished = { \\url{ https://universe.roboflow.com/robert-demo-qvail/pokedex } },
            url = { https://universe.roboflow.com/robert-demo-qvail/pokedex },
            journal = { Roboflow Universe },
            publisher = { Roboflow },
            year = { 2022 },
            month = { dec },
            note = { visited on 2023-01-14 },
        }
        """

    def __init__(
            self,
            device: str = "cpu"
    ) -> None:
        """
        Initialize the Pokemon dataset.
        Args:
            device: Device to load the tensors on ('cpu' or 'cuda')
        """
        self.data_path = Path("D:\\keremberke__pokemon_classification")
        self.device = device
        self.name = "pokemon"
        self.num_classes = 150

        # Load training and test datasets
        # Note: the test set contains classes that aren't represented in the training set
        self.train_dataset = load_dataset(
            str(self.data_path),
            split="train",
            streaming=True
        )
        self.test_dataset = load_dataset(
            str(self.data_path),
            split="validation",
            streaming=True
        )

    def process_batch(
            self,
            batch: dict,
            target_size: Union[int, Tuple[int, int]]
    ) -> Tuple[np.ndarray, Tensor]:
        """
        Process a batch of images and labels from the dataset.
        Args:
            batch: Dictionary containing 'image' and 'label' keys
                  - 'image': List of pokemon images
                  - 'label': List of digit labels (0-109)
            target_size: Desired output size for images as (height, width)
        Returns:
            Tuple containing:
            - Processed image batch as numpy array with shape
              (batch_size, channels, height, width)
            - Labels as torch.Tensor with shape (batch_size,)
        """
        # Resize images to target size
        resized_images = resize_image_batch(
            batch["image"],
            target_size
        )
        # Stack and transpose images to (B, C, H, W) format
        processed_images = stack_and_transpose_images(resized_images)

        # Convert labels to tensor
        labels = torch.as_tensor(
            batch["labels"],
            dtype=torch.long,
            device=self.device
        )

        return processed_images, labels


