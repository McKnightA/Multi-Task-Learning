from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torchmetrics
from torchmetrics.functional import clustering, image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import DataAugments as da


# self supervised ----------------------------------------------------------------------------------------------
class Rotation(nn.Module):
    """
    A self-supervised rotation prediction task module for multi-task learning.

    This module implements the rotation prediction task inspired by
    https://arxiv.org/abs/1803.07728. It randomly rotates input images and
    trains a network to predict the rotation angle from the set {0°, 90°, 180°, 270°}.
    """

    def __init__(
            self,
            embed_features: int,
            task_head: nn.Module,  # Type could be more specific based on task_head implementation
            device: str = "cpu",
            *args,
            **kwargs
    ) -> None:
        """
        Initialize the rotation prediction task.

        Args:
            embed_features: Dimension of the embedded feature space
            task_head: Callable that constructs the task-specific neural network head.
                      Should accept (input_dim, num_classes, device) as arguments.
            device: Device to run the computations on ('cpu' or 'cuda')
            *args, **kwargs: Additional arguments passed to parent class
        """
        super().__init__(*args, **kwargs)
        self.name = 'Rotation'
        self.device = device

        # Initialize task-specific head for 4-way classification (0°, 90°, 180°, 270°)
        self.task_head = task_head(embed_features, output_dim=4, device=device)
        self.loss = nn.CrossEntropyLoss()
        self.labels: torch.Tensor | None = None

    def pretreat(self, input_data: np.ndarray) -> torch.Tensor:
        """
        Prepare input data by applying random rotations and converting to tensors.

        Args:
            input_data: Batch of raw images as numpy array
                       Expected shape: (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Rotated images as normalized tensors (pixel values in [0,1])
        """
        rotated, self.labels = da.rotate(input_data)  # Apply random rotations

        # Convert to tensor and normalize
        rotated = torch.tensor(
            rotated / 255,
            dtype=torch.float,
            device=self.device,
            requires_grad=True
        )
        self.labels = torch.as_tensor(
            self.labels,
            dtype=torch.long,
            device=self.device
        )

        return rotated

    def generate_loss(self, embedded_data: torch.Tensor) -> torch.Tensor:
        """
        Generate the task loss from embedded features.

        Args:
            embedded_data: Embedded features from the backbone network

        Returns:
            torch.Tensor: Cross entropy loss between rotation predictions and true angles
        """
        predictions = self.task_head(embedded_data)
        loss = self.loss(predictions, self.labels)

        # Clear labels after loss computation to free memory
        self.labels = None

        return loss

    def check_performance(
            self,
            input_data: np.ndarray,
            backbone: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate the rotation prediction performance.

        Args:
            input_data: Batch of raw images as numpy array
            backbone: Feature extraction backbone network

        Returns:
            Dict containing performance metrics:
                - 'accuracy': Classification accuracy
                - 'f1': Macro-averaged F1 score
        """
        treated = self.pretreat(input_data)
        embedded = backbone(treated)
        predictions = self.task_head(embedded)

        accuracy = torchmetrics.functional.accuracy(
            predictions,
            self.labels,
            task="multiclass",
            num_classes=4
        )
        f1 = torchmetrics.functional.f1_score(
            predictions,
            self.labels,
            task="multiclass",
            num_classes=4,
            average="macro"
        )

        return {
            "accuracy": accuracy.cpu(),
            "f1": f1.cpu()
        }

    def forward(
            self,
            input_data: np.ndarray,
            backbone: nn.Module
    ) -> torch.Tensor:
        """
        Forward pass for the rotation prediction task.

        Args:
            input_data: Batch of raw images as numpy array
            backbone: Feature extraction backbone network

        Returns:
            torch.Tensor: Task loss
        """
        treated = self.pretreat(input_data)
        embedded = backbone(treated)
        loss = self.generate_loss(embedded)
        return loss


class Colorization(nn.Module):
    """
    A self-supervised colorization prediction task module for multi-task learning.

    This module implements the colorization prediction task inspired by
    https://arxiv.org/abs/1603.08511. It predicts the a* and b* channels of the Lab
    color space from the L channel, treating it as a classification problem with
    quantized color values.

    Note: Soft-encoding of labels as mentioned in the paper is pending implementation.
    """

    def __init__(
            self,
            embed_features: int,
            task_head: nn.Module,
            device: str = "cpu",
            *args,
            **kwargs
    ) -> None:
        """
        Initialize the colorization prediction task.

        Args:
            embed_features: Dimension of the embedded feature space
            task_head: Callable that constructs the task-specific neural network head.
                      Should accept (input_dim, num_classes, device) as arguments.
            device: Device to run the computations on ('cpu' or 'cuda')
            *args, **kwargs: Additional arguments passed to parent class"""
        super().__init__(*args, **kwargs)
        self.name = 'Colorization'
        self.device = device
        self.desired_precision = 32

        self.harmonization = nn.Conv2d(1, 3, (1, 1), device=device)
        self.task_head = task_head(embed_features, self.desired_precision * 2, device)

        # Initialize with ones
        self.a_balance_weights = torch.ones(self.desired_precision, device=device)
        self.b_balance_weights = torch.ones(self.desired_precision, device=device)

        # Storage for labels and normalization parameters
        self.a_labels: torch.Tensor | None = None
        self.b_labels: torch.Tensor | None = None
        self.a_norm_params: Dict[str, torch.Tensor] | None = None
        self.b_norm_params: Dict[str, torch.Tensor] | None = None

    def _quantize_channel(
            self,
            channel_data: torch.Tensor,
            balance_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Vectorized quantization of color channel values with cumulative weight updates.

        Args:
            channel_data: Color channel data to quantize
            balance_weights: Current cumulative balance weights to update

        Returns:
            Tuple containing:
            - Quantized labels
            - Normalization parameters
        """
        # Calculate normalization parameters
        channel_min = torch.min(channel_data)
        channel_max = torch.max(channel_data)

        # Normalize to [0, 1] range
        normalized = (channel_data - channel_min) / (channel_max - channel_min)

        # Quantize to desired precision
        quantized = (normalized * (self.desired_precision - 1)).long()

        # Update balance weights using bincount (vectorized operation)
        # Add the new counts to existing weights
        batch_weights = torch.bincount(
            quantized.flatten(),
            minlength=self.desired_precision
        ).to(self.device)
        balance_weights.add_(batch_weights)

        norm_params = {
            'min': channel_min,
            'max': channel_max
        }

        return quantized, norm_params

    def pretreat(self, input_data: np.ndarray) -> torch.Tensor:
        """
        Prepare input data by converting to Lab color space and quantizing a*b* values.

        Args:
            input_data: Batch of RGB images as numpy array
                       Expected shape: (batch_size, channels, height, width)
                       Expected range: [0, 255]

        Returns:
            torch.Tensor: Harmonized L channel data, shape (batch_size, 3, height, width)
                         Range: [0, 1] after sigmoid activation
        """
        # Convert and normalize input data
        input_data = torch.tensor(
            rgb2lab(
                (input_data / 255).transpose(0, 2, 3, 1)
            ).transpose(0, 3, 1, 2),
            dtype=torch.float,
            device=self.device
        )

        # Split channels
        l_data = (input_data[:, 0:1] - 50) / 100
        a_data = input_data[:, 1]
        b_data = input_data[:, 2]

        # Quantize both channels and update cumulative weights
        self.a_labels, self.a_norm_params = self._quantize_channel(a_data, self.a_balance_weights)
        self.b_labels, self.b_norm_params = self._quantize_channel(b_data, self.b_balance_weights)

        # Process L channel
        harmonized_data = torch.sigmoid(self.harmonization(l_data))

        return harmonized_data

    def generate_loss(self, embedded_data: torch.Tensor) -> torch.Tensor:
        """
        Generate the task loss from embedded features.

        Args:
            embedded_data: Embedded features from the backbone network

        Returns:
            torch.Tensor: Average of cross entropy losses for a* and b* channel predictions
        """
        output = self.task_head(embedded_data)

        # Add small constant to prevent division by zero and log of zero
        eps = 1e-5

        # Weight calculation using cumulative weights
        a_weights = 1 / torch.log(self.a_balance_weights + eps)
        b_weights = 1 / torch.log(self.b_balance_weights + eps)

        # Calculate losses
        a_loss = nn.CrossEntropyLoss(weight=a_weights)(
            output[:, :self.desired_precision],
            self.a_labels
        )
        b_loss = nn.CrossEntropyLoss(weight=b_weights)(
            output[:, self.desired_precision:],
            self.b_labels
        )

        loss = (a_loss + b_loss) / 2

        # Clear tensors to free memory
        self.a_labels = self.b_labels = None

        return loss

    def check_performance(
            self,
            input_data: np.ndarray,
            backbone: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate the colorization performance using image quality metrics.

        The original paper evaluates using a "colorization turing test" (human opinion)
        and AUC. Here we use RASE and Universal Image Quality Index as numerical metrics.

        Args:
            input_data: Batch of RGB images as numpy array
            backbone: Feature extraction backbone network

        Returns:
            Dict containing performance metrics:
                - 'Relative Average Spectral Error': RASE metric
                - 'Universal Image Quality Index': Image quality metric
        """
        batch = self.pretreat(input_data)
        latent = backbone(batch)
        predictions = self.task_head(latent)

        # Prepare original image data
        original = torch.tensor(input_data, device=self.device) / 255
        original_lab = torch.tensor(
            rgb2lab(input_data.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2),
            device=self.device
        )
        base_l = original_lab[:, 0:1]

        # Unnormalize predictions using stored parameters
        predictions_a = torch.argmax(predictions[:, :self.desired_precision], dim=1, keepdim=True)
        predictions_a = (predictions_a / (self.desired_precision - 1)) * \
                        (self.a_norm_params['max'] - self.a_norm_params['min']) + \
                        self.a_norm_params['min']

        predictions_b = torch.argmax(predictions[:, self.desired_precision:], dim=1, keepdim=True)
        predictions_b = (predictions_b / (self.desired_precision - 1)) * \
                        (self.b_norm_params['max'] - self.b_norm_params['min']) + \
                        self.b_norm_params['min']

        # Reconstruct and convert back to RGB
        reconstructed = torch.cat((base_l, predictions_a, predictions_b), dim=1)
        reconstructed = reconstructed.permute(0, 2, 3, 1)
        reconstructed = torch.tensor(
            lab2rgb(reconstructed.cpu()),
            device=self.device,
            dtype=torch.float32
        ).permute(0, 3, 1, 2)

        # Calculate metrics
        rase = image.relative_average_spectral_error(reconstructed, original)
        im_qual = image.universal_image_quality_index(reconstructed, original)

        return {
            "Relative Average Spectral Error": rase,
            "Universal Image Quality Index": im_qual
        }

    def forward(
            self,
            input_data: np.ndarray,
            backbone: nn.Module
    ) -> torch.Tensor:
        """
        Forward pass for the colorization prediction task.

        Args:
            input_data: Batch of RGB images as numpy array
            backbone: Feature extraction backbone network

        Returns:
            torch.Tensor: Task loss
        """
        treated = self.pretreat(input_data)
        embedded = backbone(treated)
        loss = self.generate_loss(embedded)
        return loss


class Contrastive(nn.Module):
    """
    A self-supervised contrastive learning task module for multi-task learning.

    This module implements the contrastive learning approach inspired by
    https://arxiv.org/abs/2002.05709 (SimCLR). It learns representations by
    maximizing agreement between differently augmented views of the same image
    via a contrastive loss in the latent space.

    The augmentation pipeline is crucial for performance and includes:
    - Random horizontal flips
    - Random crops
    - Color distortions
    - Gaussian blur
    """

    def __init__(
            self,
            embed_features: int,
            task_head: nn.Module,  # Type could be more specific based on task_head implementation
            device: str = "cpu",
            temperature: float = 0.1,
            *args,
            **kwargs
    ) -> None:
        """
        Initialize the contrastive learning task.

        Args:
            embed_features: Dimension of the embedded feature space
            task_head: Callable that constructs the task-specific neural network head.
                      Should output a 256-dimensional representation.
            device: Device to run the computations on ('cpu' or 'cuda')
            temperature: Temperature parameter for similarity scaling (default: 0.1)
            *args, **kwargs: Additional arguments passed to parent class
        """
        super().__init__(*args, **kwargs)
        self.name = "Contrastive"
        self.device = device

        # Augmentation pipeline following paper recommendations
        # "The combination of random crop and color distortion is crucial
        # to achieve a good performance"
        self.augments: List = [
            da.horizontal_flip,
            da.cropping,
            da.color_distortions,
            da.gauss_blur
        ]

        # Project embeddings to 256-dim space for contrastive learning
        self.task_head = task_head(embed_features, 256, device)
        self.temperature = temperature

    def pretreat(self, input_data: np.ndarray) -> torch.Tensor:
        """
        Generate two augmented views of the input data using the augmentation pipeline.

        Note: The augmentation order is important:
        horizontal flip -> crop -> color distortion -> blur

        Args:
            input_data: Batch of images as numpy array
                       Expected shape: (batch_size, channels, height, width)
                       Expected range: [0, 255]

        Returns:
            torch.Tensor: Concatenated augmented views
                         Shape: (2 * batch_size, channels, height, width)
                         Range: [0, 1]
        """
        # Generate first augmented view
        aug_data_1 = input_data.copy() / 255
        for augment in self.augments:
            aug_data_1, _ = augment(aug_data_1)

        # Generate second augmented view
        aug_data_2 = input_data.copy() / 255
        for augment in self.augments:
            aug_data_2, _ = augment(aug_data_2)

        # Concatenate both views
        aug_data = torch.cat(
            (torch.Tensor(aug_data_1),
             torch.Tensor(aug_data_2)),
            dim=0
        ).float().requires_grad_(True).to(self.device)

        return aug_data

    def generate_loss(self, embedded_data: torch.Tensor) -> torch.Tensor:
        """
        Generate the NT-Xent (normalized temperature-scaled cross entropy) loss.

        The loss encourages representations of different augmented views of the
        same image to be similar, while pushing representations of different
        images apart.

        Args:
            embedded_data: Embedded features from the backbone network
                         Shape: (2 * batch_size, embed_dim)

        Returns:
            torch.Tensor: Contrastive loss value
        """
        # Get normalized representations
        output = self.task_head(embedded_data)

        # Split augmented views and normalize
        aug1 = nn.functional.normalize(output[:output.shape[0] // 2])
        aug2 = nn.functional.normalize(output[output.shape[0] // 2:])
        out = torch.cat((aug1, aug2), dim=0)
        n_samples = out.shape[0]

        # Compute similarity matrix
        sim = torch.matmul(out, out.T)
        scaled_sim = torch.exp(sim / self.temperature)

        # Create mask to exclude self-similarities
        mask = ~torch.eye(n_samples, dtype=torch.bool, device=self.device)

        # Compute negative similarities (denominator)
        neg = torch.sum(scaled_sim * mask, dim=-1)

        # Compute positive similarities (numerator)
        pos = torch.exp(torch.sum(aug1 * aug2, dim=-1) / self.temperature)
        pos = torch.cat((pos, pos), dim=0)

        # Compute NT-Xent loss
        loss = -torch.log(pos / neg).mean()

        return loss

    def check_performance(
            self,
            input_data: np.ndarray,
            labels: np.ndarray,
            backbone: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate the quality of learned representations using clustering metrics.

        While the original paper evaluates using linear probing on ImageNet,
        we use clustering metrics to assess how well the learned representations
        separate different classes.

        Args:
            input_data: Batch of images as numpy array
            labels: Ground truth labels for the input data
            backbone: Feature extraction backbone network

        Returns:
            Dict containing clustering metrics:
                - 'Davies Bouldin Score': Lower is better, measures cluster compactness
                - 'Dunn Index': Higher is better, measures cluster separation and compactness
        """
        # Get feature representations
        embedded = backbone(torch.tensor(input_data, dtype=torch.float, device=self.device))
        vector_representation = self.task_head(embedded)

        # Calculate clustering metrics
        db_score = clustering.davies_bouldin_score(vector_representation, labels)
        dunn_index = clustering.dunn_index(vector_representation, labels)

        return {
            "Davies Bouldin Score": db_score.cpu(),
            "Dunn Index": dunn_index.cpu()
        }

    def forward(
            self,
            input_data: np.ndarray,
            backbone: nn.Module
    ) -> torch.Tensor:
        """
        Forward pass for the contrastive learning task.

        Args:
            input_data: Batch of images as numpy array
            backbone: Feature extraction backbone network

        Returns:
            torch.Tensor: Contrastive loss value
        """
        treated = self.pretreat(input_data)
        embedded = backbone(treated)
        loss = self.generate_loss(embedded)
        return loss


class MaskedAutoEncoding(nn.Module):
    """
    A self-supervised masked autoencoding task module for multi-task learning.

    This module implements the masked autoencoding approach inspired by
    https://arxiv.org/abs/2111.06377 (MAE). It learns representations by
    reconstructing randomly masked patches of the input image.
    """

    def __init__(
            self,
            embed_features: int,
            task_head: nn.Module,
            device: str = "cpu",
            *args,
            **kwargs
    ) -> None:
        """
        Initialize the masked autoencoding task.

        Args:
            embed_features: Dimension of the embedded feature space
            task_head: Callable that constructs the task-specific decoder head.
                      Should output reconstructed image patches.
            device: Device to run the computations on ('cpu' or 'cuda')
            *args, **kwargs: Additional arguments passed to parent class
        """
        super().__init__(*args, **kwargs)
        self.name = "Masked Auto Encoding"
        self.device = device

        # Harmonize masked image (3 channels) + mask (1 channel) into 3 channels
        self.harmonization = nn.Conv2d(4, 3, (1, 1), device=device)
        # Decoder head for reconstruction
        self.task_head = task_head(embed_features, 3, device)

        self.loss = nn.MSELoss()
        self.labels: Tuple[torch.Tensor, torch.Tensor] | None = None

    def pretreat(self, input_data: np.ndarray) -> torch.Tensor:
        """
        Prepare input data by applying masking and normalization.

        Note: While augmentations can be helpful, they're not necessary for MAE.
        Cropping and horizontal flipping can be beneficial, but color jittering
        should be avoided.

        Args:
            input_data: Batch of images as numpy array
                       Expected shape: (batch_size, channels, height, width)
                       Expected range: [0, 255]

        Returns:
            torch.Tensor: Harmonized data combining masked image and mask
                         Shape: (batch_size, 3, height, width)
                         Range: [0, 1] after sigmoid activation
        """
        # Normalize input to [0, 1]
        input_data = input_data / 255

        # Apply random masking
        masked_image, mask = da.masking(input_data)

        # Store mask and original image as labels for reconstruction
        self.labels = (
            torch.tensor(mask, dtype=torch.float, device=self.device, requires_grad=True),
            torch.tensor(input_data, dtype=torch.float, device=self.device, requires_grad=True)
        )

        # Combine masked image and mask
        combo = torch.cat((
            torch.tensor(masked_image, dtype=torch.float, device=self.device),
            self.labels[0]
        ), dim=1).requires_grad_(True)

        # Harmonize to 3 channels and normalize
        harmonized_data = self.harmonization(combo)
        harmonized_data = torch.nn.functional.sigmoid(harmonized_data)

        return harmonized_data

    def generate_loss(self, embedded_data: torch.Tensor) -> torch.Tensor:
        """
        Generate the reconstruction loss for masked regions.

        The loss is computed only on masked regions, following the MAE approach
        of focusing the reconstruction task on the missing information.

        Args:
            embedded_data: Embedded features from the backbone network

        Returns:
            torch.Tensor: MSE reconstruction loss on masked regions
        """
        assert self.labels is not None, "Labels not set, run pretreat first"

        # Generate reconstruction
        output = self.task_head(embedded_data)
        output = nn.functional.sigmoid(output)

        # Compute loss only on masked regions
        loss = self.loss(
            output * self.labels[0],  # Predicted values in masked regions
            self.labels[1] * self.labels[0]  # True values in masked regions
        )

        # Clear labels to free memory
        self.labels = None

        return loss

    def check_performance(
            self,
            input_data: np.ndarray,
            backbone: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate reconstruction quality using image similarity metrics.

        While the original paper evaluates using linear probing and fine-tuning
        on ImageNet, we use direct image reconstruction metrics to assess the
        model's performance.

        Args:
            input_data: Batch of images as numpy array
            backbone: Feature extraction backbone network

        Returns:
            Dict containing image quality metrics:
                - 'Relative Average Spectral Error': RASE metric
                - 'Universal Image Quality Index': Image quality metric
        """
        original = torch.tensor(input_data, device=self.device) / 255

        batch = self.pretreat(input_data)
        embedded = backbone(batch)
        reconstructed = torch.nn.functional.sigmoid(
            self.task_head(embedded)
        )

        rase = image.relative_average_spectral_error(reconstructed, original)
        im_qual = image.universal_image_quality_index(reconstructed, original)

        return {
            "Relative Average Spectral Error": rase,
            "Universal Image Quality Index": im_qual
        }

    def forward(
            self,
            input_data: np.ndarray,
            backbone: nn.Module
    ) -> torch.Tensor:
        """
        Forward pass for the masked autoencoding task.

        Args:
            input_data: Batch of images as numpy array
            backbone: Feature extraction backbone network

        Returns:
            torch.Tensor: Reconstruction loss
        """
        treated = self.pretreat(input_data)
        embedded = backbone(treated)
        loss = self.generate_loss(embedded)
        return loss


# supervised ----------------------------------------------------------------------------------------
class Classification(nn.Module):
    """
    This module implements a standard classification task. It consists
    of a task-specific head that projects embedded features to class logits,
    followed by cross-entropy loss computation.
    """

    def __init__(
            self,
            embed_dim: int,
            task_head: nn.Module,
            classes: int,
            name: str,
            device: str = "cpu",
            *args,
            **kwargs
    ) -> None:
        """
        Initialize the classification task.

        Args:
            embed_dim: Dimension of the embedded feature space
            task_head: Callable that constructs the task-specific classification head.
                      Should output class logits.
            classes: Number of target classes
            name: Task-specific name prefix (will be appended with " Classification")
            device: Device to run the computations on ('cpu' or 'cuda')
            *args, **kwargs: Additional arguments passed to parent class
        """
        super().__init__(*args, **kwargs)
        self.name = f"{name} Classification"
        self.device = device

        self.num_classes = classes
        self.task_head = task_head(embed_dim, classes, device)
        self.loss = nn.CrossEntropyLoss()

    def pretreat(self, input_data: np.ndarray) -> torch.Tensor:
        """
        Prepare input data by normalizing and converting to tensor.

        Args:
            input_data: Batch of images as numpy array
                       Expected shape: (batch_size, channels, height, width)
                       Expected range: [0, 255]

        Returns:
            torch.Tensor: Normalized input tensor
                         Range: [0, 1]
        """
        return torch.tensor(
            input_data,
            dtype=torch.float,
            device=self.device,
            requires_grad=True
        ) / 255

    def generate_loss(
            self,
            embed_data: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate the classification loss from embedded features.

        Args:
            embed_data: Embedded features from the backbone network
            labels: Ground truth class labels

        Returns:
            torch.Tensor: Cross entropy loss
        """
        prediction = self.task_head(embed_data)
        loss = self.loss(prediction, labels.to(self.device))
        return loss

    def check_performance(
            self,
            input_data: np.ndarray,
            labels: torch.Tensor,
            backbone: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate classification performance using accuracy and F1 score.

        Args:
            input_data: Batch of images as numpy array
            labels: Ground truth class labels
            backbone: Feature extraction backbone network

        Returns:
            Dict containing performance metrics:
                - 'accuracy': Classification accuracy
                - 'f1': Macro-averaged F1 score
        """
        treated = self.pretreat(input_data)
        embedded = backbone(treated)
        predictions = self.task_head(embedded)

        accuracy = torchmetrics.functional.accuracy(
            predictions,
            labels.to(self.device),
            task="multiclass",
            num_classes=self.num_classes
        )

        f1 = torchmetrics.functional.f1_score(
            predictions,
            labels.to(self.device),
            task="multiclass",
            num_classes=self.num_classes,
            average="macro"
        )

        return {
            "accuracy": accuracy.cpu(),
            "f1": f1.cpu()
        }

    def forward(
            self,
            input_data: np.ndarray,
            labels: torch.Tensor,
            backbone: nn.Module
    ) -> torch.Tensor:
        """
        Forward pass for the classification task.

        Args:
            input_data: Batch of images as numpy array
            labels: Ground truth class labels
            backbone: Feature extraction backbone network

        Returns:
            torch.Tensor: Classification loss
        """
        treated = self.pretreat(input_data)
        embedded = backbone(treated)
        loss = self.generate_loss(embedded, labels)
        return loss


# multitask ---------------------------------------------------------------------------------------
class AveragedLossMultiTask(nn.Module):
    """
    A multi-task learning module that combines multiple tasks using unweighted loss averaging.

    This module serves as a baseline approach for multi-task learning by running each task
    independently and computing their unweighted average loss. The tasks can be either
    self-supervised or supervised tasks.

    Note:
        This is a naive approach intended as a baseline for comparison with more
        sophisticated multi-task learning strategies.
    """

    def __init__(
            self,
            tasks: List[nn.Module],
            device: str = "cpu",
            *args,
            **kwargs
    ) -> None:
        """
        Initialize the multi-task learning module.

        Args:
            tasks: List of task modules, each implementing pretreat(), generate_loss(),
                  and forward() methods
            device: Computation device ('cpu' or 'cuda')
            *args, **kwargs: Additional arguments passed to parent class
        """
        super().__init__(*args, **kwargs)

        # Create concatenated name from all task names
        self.name = "+".join(task.name for task in tasks)

        self.tasks = tasks
        self.device = device

        # Store batch sizes for each task's processed data
        self.batch_shapes: List[int] = []

        # Collect parameters from all tasks for optimization
        self.task_parameters = []
        for task in self.tasks:
            self.task_parameters.extend(list(task.parameters()))

    def pretreat(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Prepare input data for all tasks by applying their respective preprocessing.

        Args:
            input_data: Raw input batch to be processed by each task

        Returns:
            Concatenated tensor of processed inputs from all tasks
        """
        # Process input data through each task's pretreatment
        processed_batches = [task.pretreat(input_data) for task in self.tasks]

        # Store batch sizes for later loss computation
        self.batch_shapes = [batch.shape[0] for batch in processed_batches]

        # Concatenate all processed batches
        combined_batch = torch.concatenate(processed_batches, dim=0)

        return combined_batch

    def generate_loss(
            self,
            embeddings: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            clear_task_labels: bool = True
    ) -> torch.Tensor:
        """
        Compute average loss across all tasks using their embedded features.

        Args:
            embeddings: Embedded features from the backbone network
            labels: Optional labels for supervised tasks
            clear_task_labels: Whether to clear task-specific labels after loss computation

        Returns:
            Average loss across all tasks
        """
        task_losses = []
        start_idx = 0

        # Compute loss for each task using its corresponding embedded features
        for task_idx, task in enumerate(self.tasks):
            # Get batch size for current task
            batch_size = self.batch_shapes[task_idx]
            end_idx = start_idx + batch_size

            # Extract embeddings for current task
            task_embeddings = embeddings[start_idx:end_idx]

            try:
                # Try self-supervised task loss computation
                task_loss = task.generate_loss(task_embeddings, clear_task_labels)
            except TypeError:
                # Fall back to supervised task loss computation
                task_loss = task.generate_loss(task_embeddings, labels, clear_task_labels)

            task_losses.append(task_loss)
            start_idx = end_idx

        # Compute mean loss across all tasks
        average_loss = torch.mean(torch.tensor(
            task_losses,
            device=self.device,
            requires_grad=True
        ))

        return average_loss

    def check_performance(
            self,
            input_data: torch.Tensor,
            labels: torch.Tensor,
            backbone: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate the performance of all tasks.

        Args:
            input_data: Raw input batch
            labels: Ground truth class labels
            backbone: Feature extraction network

        Returns:
            Performance metrics of each task
        """
        results = {}
        for task in self.tasks:

            try:
                result = task.check_performance(input_data, backbone)
            except TypeError:
                # Handle tasks that require labels
                result = task.check_performance(input_data, labels, backbone)

            for metric in result:
                results[f"{task.name} {metric}"] = result[metric]

        return results

    def forward(
            self,
            input_data: torch.Tensor,
            backbone: nn.Module
    ) -> torch.Tensor:
        """
        Perform forward pass for all tasks and compute their average loss.

        Args:
            input_data: Raw input batch
            backbone: Feature extraction network

        Returns:
            Average loss across all tasks
        """
        # Compute individual task losses
        task_losses = [task.forward(input_data, backbone) for task in self.tasks]

        if len(task_losses) > 1:
            # Multiple tasks: stack losses and compute mean
            average_loss = torch.mean(torch.stack(task_losses))
        elif len(task_losses) == 1:
            # Single task: compute mean of the single loss
            average_loss = torch.mean(task_losses[0])
        else:
            # No tasks: return zero loss (should not happen in practice)
            print("Warning: No tasks provided to compute loss")
            average_loss = torch.tensor(0.0, device=self.device)

        return average_loss
