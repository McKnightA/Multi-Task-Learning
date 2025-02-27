import torch
import torch.nn as nn
from typing import Tuple, Callable, Dict, Any, Union, List
from torch import Tensor
import torch.nn.functional as F


class BasicNonLinearPredictor(nn.Module):
    """
    A prediction head that transforms embedded representations into target outputs.

    The network consists of two linear layers with batch normalization and ReLU
    activation between them. The first layer expands the dimension to an
    intermediate size of 512, while the second layer projects to the desired
    output dimension.
    """

    def __init__(
            self,
            embed_dim: int,
            output_dim: int,
            device: str = "cpu",
            *args,
            **kwargs
    ) -> None:
        """
        Initialize the predictor.

        Args:
            embed_dim: Dimension of the input embedding space
            output_dim: Dimension of the output prediction space
            device: Device to run the computations on ('cpu' or 'cuda')
            *args, **kwargs: Additional arguments passed to parent class
        """
        super().__init__(*args, **kwargs)
        self.layers = self._build_network(embed_dim, output_dim, device)

    def _build_network(
            self,
            embed_dim: int,
            output_dim: int,
            device: str
    ) -> nn.Sequential:
        """
        Construct the prediction network architecture.

        Args:
            embed_dim: Dimension of the input embedding space
            output_dim: Dimension of the output prediction space
            device: Device to run the computations on

        Returns:
            Sequential container of the complete prediction network
        """
        return nn.Sequential(
            # Expansion layer with batch norm and activation
            nn.Linear(embed_dim, 512, device=device),
            nn.BatchNorm1d(512, device=device),
            nn.ReLU(),

            # Final projection to output dimension
            nn.Linear(512, output_dim, device=device)
        )

    def forward(self, input_data: Tensor) -> Tensor:
        """
        Forward pass of the predictor.

        Args:
            input_data: Input embedding tensor
                       Shape: (batch_size, embed_dim)

        Returns:
            Tensor: Predicted output
                   Shape: (batch_size, output_dim)
        """
        return self.layers(input_data)


# -----------------------------------------------------------------
class Cifar10Encoder(nn.Module):
    """
    A convolutional neural network encoder for CIFAR-10 sized images.

    This encoder architecture is based on the design presented in:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8745428

    The network consists of three convolutional blocks, each containing:
    - Two convolutional layers with batch normalization and ReLU activation
    - Max pooling and dropout for regularization
    Finally, a linear projection layer maps to the desired embedding dimension.
    """

    def __init__(
            self,
            embed_dim: int,
            device: str = 'cpu',
            *args,
            **kwargs
    ) -> None:
        """
        Initialize the CIFAR-10 encoder.

        Args:
            embed_dim: Dimension of the output embedding space
            device: Device to run the computations on ('cpu' or 'cuda')
            *args, **kwargs: Additional arguments passed to parent class

        Notes:
            Expected input image size: 32x32 pixels
            Expected input channels: 3 (RGB)
        """
        super().__init__(*args, **kwargs)
        self.expected_input_size = 32
        self.layers = self._build_network(embed_dim, device)

    def _create_conv_block(
            self,
            in_channels: int,
            out_channels: int,
            device: str
    ) -> nn.Sequential:
        """
        Create a convolutional block with batch normalization and regularization.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            device: Device to run the computations on

        Returns:
            Sequential container of layers forming a convolutional block
        """
        return nn.Sequential(
            # First convolution with batch norm and activation
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
                device=device
            ),
            nn.BatchNorm2d(out_channels, device=device),
            nn.ReLU(),

            # Second convolution with batch norm and activation
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                device=device
            ),
            nn.BatchNorm2d(out_channels, device=device),
            nn.ReLU(),

            # Downsampling and regularization
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25)
        )

    def _build_network(
            self,
            embed_dim: int,
            device: str
    ) -> nn.Sequential:
        """
        Construct the complete encoder network architecture.

        Args:
            embed_dim: Dimension of the output embedding space
            device: Device to run the computations on

        Returns:
            Sequential container of the complete encoder network
        """
        return nn.Sequential(
            # Three convolutional blocks with increasing channels
            self._create_conv_block(3, 32, device),  # Input -> 32 channels
            self._create_conv_block(32, 64, device),  # 32 -> 64 channels
            self._create_conv_block(64, 128, device),  # 64 -> 128 channels

            # Final embedding projection
            nn.Flatten(),
            nn.Linear(512, embed_dim, device=device),
            nn.BatchNorm1d(embed_dim, device=device),
            nn.Dropout(p=0.25)
        )

    def forward(self, input_data: Tensor) -> Tensor:
        """
        Forward pass of the encoder.

        Args:
            input_data: Input image tensor
                       Expected shape: (batch_size, 3, 32, 32)
                       Expected range: [0, 1]

        Returns:
            Tensor: Embedded representation of the input
                   Shape: (batch_size, embed_dim)
        """
        return self.layers(input_data)


class Cifar10Projector(nn.Module):
    """
    A projector network that transforms embedded representations back into image space.

    This projector architecture reverses the structure presented in:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8745428

    The network consists of:
    1. An initial linear projection with batch normalization
    2. Three transposed convolution blocks that progressively upsample the features
    3. Interpolation operations between blocks to achieve desired spatial dimensions

    The network reconstructs images through a series of upsampling and transposed
    convolution operations, eventually producing an output with the target number
    of channels and spatial dimensions matching CIFAR-10 images.
    """

    def __init__(
            self,
            embed_dim: int,
            output_channels: int = 3,
            device: str = 'cpu',
            *args,
            **kwargs
    ) -> None:
        """
        Initialize the CIFAR-10 projector.

        Args:
            embed_dim: Dimension of the input embedding space
            output_channels: Number of channels in the output image (e.g., 3 for RGB)
            device: Device to run the computations on ('cpu' or 'cuda')
            *args, **kwargs: Additional arguments passed to parent class

        Notes:
            The network progressively upsamples from 2x2 to 32x32 spatial dimensions
        """
        super().__init__(*args, **kwargs)

        # Initialize network components
        self.initial_projection = self._build_initial_projection(embed_dim, device)
        self.conv_block1 = self._create_conv_block(128, 64, device)
        self.conv_block2 = self._create_conv_block(64, 32, device)
        self.final_block = self._build_final_block(32, output_channels, device)

    def _build_initial_projection(
            self,
            embed_dim: int,
            device: str
    ) -> nn.Sequential:
        """
        Build the initial linear projection layer.

        Args:
            embed_dim: Dimension of the input embedding
            device: Device to run the computations on

        Returns:
            Sequential container with linear projection and activation
        """
        return nn.Sequential(
            nn.Linear(embed_dim, 512, device=device),
            nn.BatchNorm1d(512, device=device),
            nn.ReLU()
        )

    def _create_conv_block(
            self,
            in_channels: int,
            out_channels: int,
            device: str
    ) -> nn.Sequential:
        """
        Create a transposed convolution block with batch normalization and regularization.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            device: Device to run the computations on

        Returns:
            Sequential container of layers forming a transposed convolution block
        """
        return nn.Sequential(
            # First transposed convolution
            nn.ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size=(3, 3),
                device=device
            ),
            nn.BatchNorm2d(in_channels, device=device),
            nn.ReLU(),

            # Second transposed convolution
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                padding=(1, 1),
                device=device
            ),
            nn.BatchNorm2d(out_channels, device=device),
            nn.ReLU(),

            nn.Dropout(0.25)
        )

    def _build_final_block(
            self,
            in_channels: int,
            out_channels: int,
            device: str
    ) -> nn.Sequential:
        """
        Build the final transposed convolution block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            device: Device to run the computations on

        Returns:
            Sequential container with final transposed convolutions
        """
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size=(3, 3),
                device=device
            ),
            nn.BatchNorm2d(in_channels, device=device),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                padding=(1, 1),
                device=device
            )
        )

    def forward(self, input_data: Tensor) -> Tensor:
        """
        Forward pass of the projector.

        Args:
            input_data: Input embedding tensor
                       Shape: (batch_size, embed_dim)

        Returns:
            Tensor: Reconstructed image tensor
                   Shape: (batch_size, output_channels, height, width)
        """
        # Initial projection and reshaping
        out = self.initial_projection(input_data)
        out = torch.reshape(out, (out.shape[0], 128, 2, 2))

        # Progressive upsampling and feature transformation
        out = F.interpolate(out, (4, 4))
        out = self.conv_block1(out)

        out = F.interpolate(out, (13, 13))
        out = self.conv_block2(out)

        out = F.interpolate(out, (30, 30))
        out = self.final_block(out)

        return out


# -------------------------------------------------------------------
class MobileViTv3Encoder(nn.Module):
    """
    MobileViTv3 encoder implementation based on the architecture described in:
    https://arxiv.org/abs/2209.15159

    This encoder combines the efficiency of MobileNet's inverted residuals with
    transformer-style attention using unfolded image patches. The architecture
    progressively increases channel dimensions while reducing spatial dimensions,
    using a combination of convolutional and attention-based processing.
    """

    def __init__(
            self,
            embed_dim: int,
            device: str = 'cpu',
            *args,
            **kwargs
    ) -> None:
        """
        Initialize the MobileViTv3 encoder.

        Args:
            embed_dim: Dimension of the output embedding space
            width_multiplier: Multiplier for scaling channel dimensions
            patch_size: Size of patches for unfolding (height, width)
            ffn_multiplier: Multiplier for FFN hidden dimensions
            mv2_exp_mult: Expansion multiplier for MobileNetV2 blocks
            device: Device to run the computations on
            *args, **kwargs: Additional arguments passed to parent class

        Notes:
            Expected input image size: 256x256 pixels
            Expected input channels: 3 (RGB)
        """
        super().__init__(*args, **kwargs)
        self.expected_input_size = 256

        width_multiplier = 0.5
        patch_size = (2, 2)
        ffn_multiplier = 2.0
        mv2_exp_mult = 2.0

        # Store configuration for network construction
        self.config = self._create_network_config(
            width_multiplier=width_multiplier,
            patch_size=patch_size,
            ffn_multiplier=ffn_multiplier,
            mv2_exp_mult=mv2_exp_mult
        )

        # Build the network
        self.layers = self._build_network(
            embed_dim=embed_dim,
            device=device
        )

    def _create_network_config(
            self,
            width_multiplier: float,
            patch_size: Tuple[int, int],
            ffn_multiplier: float,
            mv2_exp_mult: float
    ) -> Dict[str, Any]:
        """
        Create configuration dictionary for network construction.

        Args:
            width_multiplier: Multiplier for scaling channel dimensions
            patch_size: Size of patches for unfolding
            ffn_multiplier: Multiplier for FFN hidden dimensions
            mv2_exp_mult: Expansion multiplier for MobileNetV2 blocks

        Returns:
            Dictionary containing network configuration parameters
        """
        # Define scaled channel dimensions
        channels = [
            int(max(16, min(64, int(32 * width_multiplier)))),
            int(64 * width_multiplier),
            int(128 * width_multiplier),
            int(256 * width_multiplier),
            int(384 * width_multiplier),
            int(512 * width_multiplier)
        ]

        # Define attention dimensions
        attn_dims = [
            int(128 * width_multiplier),
            int(192 * width_multiplier),
            int(256 * width_multiplier)
        ]

        return {
            'channels': channels,
            'attn_dims': attn_dims,
            'patch_size': patch_size,
            'ffn_multiplier': ffn_multiplier,
            'mv2_exp_mult': mv2_exp_mult
        }

    def _conv_2d(
            self,
            inp: int,
            oup: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            bias: bool = False,
            norm: bool = True,
            act: bool = True,
            device: str = "cpu"
    ) -> nn.Sequential:
        """
        Create a convolutional layer with optional normalization and activation.

        Args:
            inp: Number of input channels
            oup: Number of output channels
            kernel_size: Size of the convolutional kernel
            stride: Stride of the convolution
            padding: Padding size
            groups: Number of groups for grouped convolution
            bias: Whether to include bias
            norm: Whether to include batch normalization
            act: Whether to include activation
            device: Device to place the layer on

        Returns:
            Sequential container with conv2d, optional batch norm, and optional activation
        """
        layers = []
        layers.append(
            nn.Conv2d(
                inp, oup, kernel_size, stride, padding,
                groups=groups, bias=bias, device=device
            )
        )
        if norm:
            layers.append(nn.BatchNorm2d(oup, device=device))
        if act:
            layers.append(nn.SiLU())
        return nn.Sequential(*layers)

    def _build_network(
            self,
            embed_dim: int,
            device: str
    ) -> nn.Sequential:
        """
        Construct the complete encoder network architecture.

        Args:
            embed_dim: Dimension of the output embedding space
            device: Device to run the computations on

        Returns:
            Sequential container of the complete encoder network
        """
        channels = self.config['channels']
        attn_dims = self.config['attn_dims']
        patch_size = self.config['patch_size']
        ffn_mult = self.config['ffn_multiplier']
        mv2_mult = self.config['mv2_exp_mult']

        return nn.Sequential(
            # Initial convolution
            self._conv_2d(3, channels[0], kernel_size=3, stride=2, padding=1, device=device),

            # Stage 1: Single MV2 block
            InvertedResidual(channels[0], channels[1], stride=1, expand_ratio=mv2_mult,
                             conv_layer=self._conv_2d, device=device),

            # Stage 2: Two MV2 blocks
            InvertedResidual(channels[1], channels[2], stride=2, expand_ratio=mv2_mult,
                             conv_layer=self._conv_2d, device=device),
            InvertedResidual(channels[2], channels[2], stride=1, expand_ratio=mv2_mult,
                             conv_layer=self._conv_2d, device=device),

            # Stage 3: MV2 + MobileViT block
            InvertedResidual(channels[2], channels[3], stride=2, expand_ratio=mv2_mult,
                             conv_layer=self._conv_2d, device=device),
            MobileViTBlockv3(channels[3], attn_dims[0], ffn_mult, 2, patch_size,
                             conv_layer=self._conv_2d, device=device),

            # Stage 4: MV2 + MobileViT block
            InvertedResidual(channels[3], channels[4], stride=2, expand_ratio=mv2_mult,
                             conv_layer=self._conv_2d, device=device),
            MobileViTBlockv3(channels[4], attn_dims[1], ffn_mult, 4, patch_size,
                             conv_layer=self._conv_2d, device=device),

            # Stage 5: MV2 + MobileViT block
            InvertedResidual(channels[4], channels[5], stride=2, expand_ratio=mv2_mult,
                             conv_layer=self._conv_2d, device=device),
            MobileViTBlockv3(channels[5], attn_dims[2], ffn_mult, 3, patch_size,
                             conv_layer=self._conv_2d, device=device),

            # Global average pooling and final projection
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], embed_dim, device=device)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the encoder.

        Args:
            x: Input image tensor
               Expected shape: (batch_size, 3, 128, 128)
               Expected range: [0, 1]

        Returns:
            Tensor: Embedded representation of the input
                   Shape: (batch_size, embed_dim)
        """
        return self.layers(x)


class MobileViTv3Projector(nn.Module):
    """
    MobileViTv3Projector for generating image reconstructions from embedded features.
    Based on inverting https://arxiv.org/abs/2209.15159

    This module projects embedded features to image space using a MobileViTv3-based
    architecture with transposed convolutions for upsampling.
    """

    def __init__(
            self,
            embed_dim: int,
            output_channels: int = 3,
            device: str = 'cpu'
    ) -> None:
        """
        Initialize the MobileViTv3Projector.

        Args:
            embed_dim: Dimension of the embedded features
            output_channels: Number of channels in output image (default: 3 for RGB)
            device: Device to run the computations on ('cpu' or 'cuda')
        """
        super().__init__()

        # Set output size for final reconstructed image
        self.output_size = (256, 256)

        # Network architecture parameters
        width_multiplier = 0.5  # Controls network width/capacity
        patch_size = (2, 2)  # Size of patches for transformer blocks
        ffn_multiplier = 2.0  # Expansion factor for feed-forward networks
        mv2_exp_mult = 2.0  # Expansion factor for MobileNetV2 blocks

        # Create network configuration with scaled dimensions
        self.config = self._create_network_config(
            width_multiplier, patch_size, ffn_multiplier, mv2_exp_mult
        )

        # Initial linear projection from embedding to feature space
        self.initial_projection = nn.Sequential(
            nn.Linear(embed_dim, self.config['channels'][0], device=device),
            nn.BatchNorm1d(self.config['channels'][0], device=device),
            nn.SiLU()
        )

        # 1x1 convolution to match channel dimensions after reshaping
        self.channel_matching = nn.Conv2d(
            self.config['channels'][0] // 16,  # Channels after reshaping to 4x4 spatial dims
            self.config['channels'][0],  # Target channel dimension
            kernel_size=(1, 1),
            device=device
        )

        # Build decoder network with progressive upsampling
        self.layers = self._build_network(output_channels, device)

    def _create_network_config(
            self,
            width_multiplier: float,
            patch_size: Tuple[int, int],
            ffn_multiplier: float,
            mv2_exp_mult: float
    ) -> Dict[str, Union[List[int], Tuple[int, int], float]]:
        """
        Create network configuration with scaled dimensions.

        Args:
            width_multiplier: Factor to scale channel dimensions
            patch_size: Size of patches for transformer blocks
            ffn_multiplier: Expansion factor for feed-forward networks
            mv2_exp_mult: Expansion factor for MobileNetV2 blocks

        Returns:
            Dictionary containing network configuration parameters
        """
        # Scale channel dimensions based on width multiplier
        channels = [
            int(512 * width_multiplier),  # Stage 5
            int(384 * width_multiplier),  # Stage 4
            int(256 * width_multiplier),  # Stage 3
            int(128 * width_multiplier),  # Stage 2
            int(64 * width_multiplier),  # Stage 1
            # Final layer with minimum channel constraint
            int(max(16, min(64, int(32 * width_multiplier))))
        ]

        # Scale attention dimensions for transformer blocks
        attn_dims = [
            int(256 * width_multiplier),  # Stage 5
            int(192 * width_multiplier),  # Stage 4
            int(128 * width_multiplier)  # Stage 3
        ]

        return {
            'channels': channels,
            'attn_dims': attn_dims,
            'patch_size': patch_size,
            'ffn_multiplier': ffn_multiplier,
            'mv2_exp_mult': mv2_exp_mult
        }

    def _conv_transpose_2d(
            self,
            inp: int,
            oup: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 0,
            output_padding: int = 0,
            groups: int = 1,
            bias: bool = False,
            norm: bool = True,
            act: bool = True,
            device: str = "cpu"
    ) -> nn.Sequential:
        """
        Create a transposed convolution block with optional normalization and activation.

        Args:
            inp: Number of input channels
            oup: Number of output channels
            kernel_size: Size of convolution kernel
            stride: Stride of convolution
            padding: Padding added to input
            output_padding: Additional padding for output shape
            groups: Number of blocked connections from input to output channels
            bias: Whether to add learnable bias
            norm: Whether to include batch normalization
            act: Whether to include activation function
            device: Device to run the computations on

        Returns:
            Sequential module containing transposed convolution with optional BN and activation
        """
        layers = []

        # Add transposed convolution layer
        layers.append(
            nn.ConvTranspose2d(
                inp, oup, kernel_size, stride, padding, output_padding,
                groups=groups, bias=bias, device=device
            )
        )

        # Add batch normalization if requested
        if norm:
            layers.append(nn.BatchNorm2d(oup, device=device))

        # Add activation function if requested
        if act:
            layers.append(nn.SiLU())

        return nn.Sequential(*layers)

    def _build_network(self, output_channels: int, device: str) -> nn.Sequential:
        """
        Build the decoder network with MobileViT blocks and transposed convolutions.

        Args:
            output_channels: Number of channels in the output image
            device: Device to run the computations on

        Returns:
            Sequential module containing the complete decoder network
        """
        # Extract configuration parameters
        channels = self.config['channels']
        attn_dims = self.config['attn_dims']
        patch_size = self.config['patch_size']
        ffn_mult = self.config['ffn_multiplier']
        mv2_mult = self.config['mv2_exp_mult']

        # Build decoder network in reverse order (from smallest to largest spatial dimensions)
        return nn.Sequential(
            # Stage 5 (reversed): MobileViT block + MV2
            MobileViTBlockv3(
                channels[0], attn_dims[0], ffn_mult, 3, patch_size,
                conv_layer=self._conv_transpose_2d, device=device
            ),
            InvertedResidualTranspose(
                channels[0], channels[1], stride=2, expand_ratio=mv2_mult,
                conv_layer=self._conv_transpose_2d, device=device
            ),

            # Stage 4 (reversed): MobileViT block + MV2
            MobileViTBlockv3(
                channels[1], attn_dims[1], ffn_mult, 4, patch_size,
                conv_layer=self._conv_transpose_2d, device=device
            ),
            InvertedResidualTranspose(
                channels[1], channels[2], stride=2, expand_ratio=mv2_mult,
                conv_layer=self._conv_transpose_2d, device=device
            ),

            # Stage 3 (reversed): MobileViT block + MV2
            MobileViTBlockv3(
                channels[2], attn_dims[2], ffn_mult, 2, patch_size,
                conv_layer=self._conv_transpose_2d, device=device
            ),
            InvertedResidualTranspose(
                channels[2], channels[3], stride=2, expand_ratio=mv2_mult,
                conv_layer=self._conv_transpose_2d, device=device
            ),

            # Stage 2 (reversed): Two MV2 blocks
            InvertedResidualTranspose(
                channels[3], channels[3], stride=1, expand_ratio=mv2_mult,
                conv_layer=self._conv_transpose_2d, device=device
            ),
            InvertedResidualTranspose(
                channels[3], channels[4], stride=2, expand_ratio=mv2_mult,
                conv_layer=self._conv_transpose_2d, device=device
            ),

            # Stage 1 (reversed): Single MV2 block
            InvertedResidualTranspose(
                channels[4], channels[5], stride=1, expand_ratio=mv2_mult,
                conv_layer=self._conv_transpose_2d, device=device
            ),

            # Final convolution to output channels with upsampling
            self._conv_transpose_2d(
                channels[5], output_channels, kernel_size=3, stride=2,
                padding=1, device=device
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the MobileViTv3Projector.

        Args:
            x: Input tensor of embedded features [batch_size, embed_dim]

        Returns:
            Reconstructed image tensor [batch_size, output_channels, height, width]
        """
        # Project embedded features to initial feature space
        out = self.initial_projection(x)

        # Reshape to add spatial dimensions (4x4)
        out = out.view(out.size(0), -1, 4, 4)

        # Match channel dimensions for decoder network
        out = self.channel_matching(out)

        # Pass through decoder network
        out = self.layers(out)

        # Ensure output has correct spatial dimensions
        if out.shape[-2:] != self.output_size:
            out = F.interpolate(
                out,
                size=self.output_size,
                mode='bilinear',
                align_corners=False
            )

        return out


class InvertedResidual(nn.Module):
    """
    Inverted Residual block from MobileViTv3, implementing the inverted bottleneck structure
    with optional residual connection.
    """

    def __init__(
            self,
            inp: int,
            oup: int,
            stride: int,
            expand_ratio: int,
            conv_layer: Callable,
            device: str = "cpu"
    ) -> None:
        """
        Initialize the Inverted Residual block.

        Args:
            inp: Number of input channels
            oup: Number of output channels
            stride: Stride for the 3x3 depthwise convolution
            expand_ratio: Expansion ratio for the inverted bottleneck
            conv_layer: Function to create convolution layers with normalization and activation
            device: Device to place the module on
        """
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        self.block = nn.Sequential()
        if expand_ratio != 1:
            self.block.add_module('exp_1x1',
                                  conv_layer(inp, hidden_dim, kernel_size=1, stride=1, padding=0, device=device))
        self.block.add_module('conv_3x3', conv_layer(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1,
                                                     groups=hidden_dim, device=device))
        self.block.add_module('red_1x1',
                              conv_layer(hidden_dim, oup, kernel_size=1, stride=1, padding=0, act=False, device=device))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Inverted Residual block.

        Args:
            x: Input tensor

        Returns:
            Output tensor with residual connection if applicable
        """
        if self.use_res_connect:
            return x + self.block(x)
        return self.block(x)


class InvertedResidualTranspose(nn.Module):
    """
    Inverted Residual block with transposed convolutions for upsampling.

    This block implements an inverted bottleneck with depthwise separable
    convolutions, suitable for decoder networks. It expands the channel
    dimensions, applies a depthwise transposed convolution, and then
    projects back to the desired output channels.
    """

    def __init__(
            self,
            inp: int,
            oup: int,
            stride: int,
            expand_ratio: int,
            conv_layer: Callable,
            device: str = "cpu"
    ) -> None:
        """
        Initialize the InvertedResidualTranspose block.

        Args:
            inp: Number of input channels
            oup: Number of output channels
            stride: Stride for the depthwise convolution (1 or 2)
            expand_ratio: Channel expansion factor for hidden dimension
            conv_layer: Callable factory function to create convolution layers
            device: Device to run the computations on ('cpu' or 'cuda')
        """
        super().__init__()

        # Store stride for later use
        self.stride = stride

        # Validate stride value
        if stride not in [1, 2]:
            raise ValueError(f"Stride must be 1 or 2, got {stride}")

        # Calculate expanded dimension for bottleneck
        hidden_dim = int(round(inp * expand_ratio))

        # Determine if residual connection should be used
        # Only use when stride=1 and input/output channels match
        self.use_res_connect = (self.stride == 1 and inp == oup)

        # Build the block as a sequential module
        self.block = nn.Sequential()

        # Step 1: Expand channels using 1x1 convolution
        self.block.add_module(
            'channel_expansion',
            conv_layer(
                inp,
                hidden_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                act=False,
                device=device
            )
        )

        # Step 2: Depthwise transposed convolution with stride 1 or 2
        if stride == 2:
            # Upsampling with stride 2
            self.block.add_module(
                'depthwise_transpose',
                conv_layer(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,  # Required for stride 2 to match output size
                    groups=hidden_dim,  # Makes it depthwise
                    device=device
                )
            )
        else:
            # No upsampling (stride 1)
            self.block.add_module(
                'depthwise_transpose',
                conv_layer(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=hidden_dim,  # Makes it depthwise
                    device=device
                )
            )

        # Step 3: Project back to output channels using 1x1 convolution
        # Skip this step if expand_ratio is 1 (no expansion was done)
        if expand_ratio != 1:
            self.block.add_module(
                'channel_projection',
                conv_layer(
                    hidden_dim,
                    oup,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    device=device
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the InvertedResidualTranspose block.

        Args:
            x: Input tensor [batch_size, inp, height, width]

        Returns:
            Output tensor [batch_size, oup, height*stride, width*stride]
            if stride=2, otherwise same spatial dimensions as input
        """
        if self.use_res_connect:
            # Add residual connection when input/output dimensions match
            return x + self.block(x)

        # Otherwise just return the block output
        return self.block(x)


class LinearSelfAttention(nn.Module):
    """
    Linear Self Attention module for MobileViTv3, implementing an efficient attention mechanism
    with linear complexity.
    """

    def __init__(
            self,
            embed_dim: int,
            conv_layer: Callable,
            attn_dropout: float = 0,
            device: str = "cpu"
    ) -> None:
        """
        Initialize the Linear Self Attention module.

        Args:
            embed_dim: Dimension of the embedding space
            conv_layer: Function to create convolution layers
            attn_dropout: Dropout rate for attention weights
            device: Device to place the module on
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.qkv_proj = conv_layer(embed_dim, 1 + 2 * embed_dim, kernel_size=1, bias=True, norm=False, act=False,
                                   device=device)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = conv_layer(embed_dim, embed_dim, kernel_size=1, bias=True, norm=False, act=False, device=device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Linear Self Attention module.

        Args:
            x: Input tensor

        Returns:
            Tensor after self-attention computation
        """
        qkv = self.qkv_proj(x)
        q, k, v = torch.split(qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1)

        context_score = F.softmax(q, dim=-1)
        context_score = self.attn_dropout(context_score)

        context_vector = k * context_score
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        out = F.relu(v) * context_vector.expand_as(v)
        out = self.out_proj(out)
        return out


class LinearAttnFFN(nn.Module):
    """
    Linear Attention Feed Forward Network module for MobileViTv3, combining
    linear attention with a position-wise feed-forward network.
    """

    def __init__(
            self,
            embed_dim: int,
            ffn_latent_dim: int,
            conv_layer: Callable,
            dropout: float = 0,
            attn_dropout: float = 0,
            device: str = "cpu"
    ) -> None:
        """
        Initialize the Linear Attention FFN module.

        Args:
            embed_dim: Dimension of the embedding space
            ffn_latent_dim: Hidden dimension of the feed-forward network
            conv_layer: Function to create convolution layers
            dropout: Dropout rate for FFN
            attn_dropout: Dropout rate for attention
            device: Device to place the module on
        """
        super().__init__()
        self.pre_norm_attn = nn.Sequential(
            nn.GroupNorm(num_channels=embed_dim, eps=1e-5, affine=True, num_groups=1, device=device),
            LinearSelfAttention(embed_dim, conv_layer, attn_dropout, device=device),
            nn.Dropout(dropout)
        )

        self.pre_norm_ffn = nn.Sequential(
            nn.GroupNorm(num_channels=embed_dim, eps=1e-5, affine=True, num_groups=1, device=device),
            conv_layer(embed_dim, ffn_latent_dim, kernel_size=1, stride=1, bias=True, norm=False, act=True,
                       device=device),
            nn.Dropout(dropout),
            conv_layer(ffn_latent_dim, embed_dim, kernel_size=1, stride=1, bias=True, norm=False, act=False,
                       device=device),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Linear Attention FFN module.

        Args:
            x: Input tensor

        Returns:
            Tensor after attention and feed-forward computation
        """
        x = x + self.pre_norm_attn(x)
        x = x + self.pre_norm_ffn(x)
        return x


class MobileViTBlockv3(nn.Module):
    """
    MobileViTv3 block implementing the core transformation combining local and global representations.
    """

    def __init__(
            self,
            inp: int,
            attn_dim: int,
            ffn_multiplier: float,
            attn_blocks: int,
            patch_size: Tuple[int, int],
            conv_layer: Callable,
            device: str = "cpu"
    ) -> None:
        """
        Initialize the MobileViTv3 block.

        Args:
            inp: Number of input channels
            attn_dim: Dimension of attention mechanism
            ffn_multiplier: Multiplier for FFN hidden dimension
            attn_blocks: Number of attention blocks
            patch_size: Size of patches for unfolding (height, width)
            conv_layer: Function to create convolution layers
            device: Device to place the module on
        """
        super().__init__()
        self.patch_h, self.patch_w = patch_size

        # Local representation
        self.local_rep = nn.Sequential(
            conv_layer(inp, inp, kernel_size=3, stride=1, padding=1, groups=inp, device=device),
            conv_layer(inp, attn_dim, kernel_size=1, stride=1, norm=False, act=False, device=device)
        )

        # Global representation
        self.global_rep = nn.Sequential()
        ffn_dims = [int((ffn_multiplier * attn_dim) // 16 * 16)] * attn_blocks
        for i in range(attn_blocks):
            ffn_dim = ffn_dims[i]
            self.global_rep.add_module(
                f'LinearAttnFFN_{i}',
                LinearAttnFFN(attn_dim, ffn_dim, conv_layer, device=device)
            )
        self.global_rep.add_module(
            'LayerNorm2D',
            nn.GroupNorm(num_channels=attn_dim, eps=1e-5, affine=True, num_groups=1, device=device)
        )

        self.conv_proj = conv_layer(2 * attn_dim, inp, kernel_size=1, stride=1, padding=0, act=False, device=device)

    def _unfold(self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        """
        Unfold the feature map into patches.

        Args:
            feature_map: Input feature map tensor

        Returns:
            Tuple of unfolded patches and original output size
        """
        batch_size, in_channels, img_h, img_w = feature_map.shape
        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w)
        )
        patches = patches.reshape(batch_size, in_channels, self.patch_h * self.patch_w, -1)
        return patches, (img_h, img_w)

    def _fold(self, patches: Tensor, output_size: Tuple[int, int]) -> Tensor:
        """
        Fold patches back into a feature map.

        Args:
            patches: Tensor of patches
            output_size: Size of the output feature map

        Returns:
            Folded feature map tensor
        """
        batch_size, in_dim, patch_size, n_patches = patches.shape
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)
        return F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MobileViTv3 block.

        Args:
            x: Input tensor

        Returns:
            Transformed tensor combining local and global features
        """
        res = x.clone()
        fm_conv = self.local_rep(x)
        x, output_size = self._unfold(fm_conv)
        x = self.global_rep(x)
        x = self._fold(patches=x, output_size=output_size)
        x = self.conv_proj(torch.cat((x, fm_conv), dim=1))
        return x + res

# ------------------------------------------------------------------------
