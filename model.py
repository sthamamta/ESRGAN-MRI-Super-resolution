
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torch.nn.utils import spectral_norm
from torchvision.models.vgg import vgg19
# from torchvision.models.feature_extraction import create_feature_extractor
# import torchvision.models.feature_extraction.create_feature_extractor as create_feature_extractor
from torchvision import transforms

__all__ = [
    "ResidualDenseBlock", "ResidualResidualDenseBlock",
    "Discriminator", "Generator",
    "ContentLoss"
]


class ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(ResidualDenseBlock, self).__init__()
        self.conv1 =spectral_norm( nn.Conv2d(channels + growth_channels * 0, growth_channels, (3, 3), (1, 1), (1, 1)))
        self.conv2 =spectral_norm( nn.Conv2d(channels + growth_channels * 1, growth_channels, (3, 3), (1, 1), (1, 1)))
        self.conv3 = spectral_norm(nn.Conv2d(channels + growth_channels * 2, growth_channels, (3, 3), (1, 1), (1, 1)))
        self.conv4 = spectral_norm(nn.Conv2d(channels + growth_channels * 3, growth_channels, (3, 3), (1, 1), (1, 1)))
        self.conv5 = spectral_norm(nn.Conv2d(channels + growth_channels * 4, channels, (3, 3), (1, 1), (1, 1)))

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

        # Initialize model weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))
        out = torch.mul(out5, 0.2)
        out = torch.add(out, identity)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class ResidualResidualDenseBlock(nn.Module):
    """Multi-layer residual dense convolution block.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(ResidualResidualDenseBlock, self).__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(channels, growth_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = torch.mul(out, 0.2)
        out = torch.add(out, identity)

        return out


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 128 x 128
            nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 64 x 64
            nn.Conv2d(64, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 32 x 32
            nn.Conv2d(128, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 16 x 16
            nn.Conv2d(256, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 8 x 8
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 4 x 4
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


class Generator(nn.Module):
    def __init__(self,num_block=7) -> None:
        super(Generator, self).__init__()
        # The first layer of convolutional layer.
        self.conv1 = nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1))

        # Feature extraction backbone network.
        trunk = []
        self.num_block=num_block
        for _ in range(self.num_block):
            trunk.append(ResidualResidualDenseBlock(64, 32))
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks.
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))

        # Upsampling convolutional layer.
        self.upsampling1 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )
        self.upsampling2 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        # Reconnect a layer of convolution block after upsampling.
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        # Output layer.
        self.conv4 = nn.Conv2d(64, 1, (3, 3), (1, 1), (1, 1))

    # The model should be defined in the Torch.script method.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)
        out = self.trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)

        # out = self.upsampling1(F.interpolate(out, scale_factor=1, mode="nearest"))
        # out = self.upsampling2(F.interpolate(out, scale_factor=2, mode="nearest"))

        #replaced to make the size of input and output same
        out = self.upsampling1(out)
        out = self.upsampling2(out)

        out = self.conv3(out)
        out = self.conv4(out)

        out = torch.clamp_(out, 0.0, 1.0)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


# class ContentLoss(nn.Module):
#     """Constructs a content loss function based on the VGG19 network.
#     Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.

#     Paper reference list:
#         -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
#         -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
#         -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.

#      """

#     def __init__(self, feature_model_extractor_node: str,
#                  feature_model_normalize_mean: list,
#                  feature_model_normalize_std: list) -> None:
#         super(ContentLoss, self).__init__()
#         # Get the name of the specified feature extraction node
#         self.feature_model_extractor_node = feature_model_extractor_node
#         # Load the VGG19 model trained on the ImageNet dataset.
#         model = models.vgg19(True)
#         # Extract the thirty-fifth layer output in the VGG19 model as the content loss.
#         self.feature_extractor = create_feature_extractor(model, [feature_model_extractor_node])
#         # set to validation mode
#         self.feature_extractor.eval()

#         # The preprocessing method of the input data.
#         # This is the VGG model preprocessing method of the ImageNet dataset
#         self.normalize = transforms.Normalize(feature_model_normalize_mean, feature_model_normalize_std)

#         # Freeze model parameters.
#         for model_parameters in self.feature_extractor.parameters():
#             model_parameters.requires_grad = False

#     def forward(self, sr_tensor: torch.Tensor, hr_tensor: torch.Tensor) -> torch.Tensor:
#         # Standardized operations
#         sr_tensor = self.normalize(sr_tensor)
#         hr_tensor = self.normalize(hr_tensor)

#         sr_feature = self.feature_extractor(sr_tensor)[self.feature_model_extractor_node]
#         hr_feature = self.feature_extractor(hr_tensor)[self.feature_model_extractor_node]

#         # Find the feature map difference between the two images
#         content_loss = F.l1_loss(sr_feature, hr_feature)

#         return content_loss



class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()

        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def forward(self, high_resolution_tensor, fake_high_resolution_tensor):
        high_resolution_tensor = torch.cat([high_resolution_tensor, high_resolution_tensor, high_resolution_tensor], dim=1)
        fake_high_resolution_tensor = torch.cat([fake_high_resolution_tensor, fake_high_resolution_tensor, fake_high_resolution_tensor], dim=1)
        perception_loss = self.l1_loss(self.loss_network(high_resolution_tensor), self.loss_network(fake_high_resolution_tensor))
        return perception_loss