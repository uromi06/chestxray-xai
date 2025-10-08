<<<<<<< HEAD
"""
model.py
--------
Defines the CNN architecture for classification.

We use torchvision's ResNet18 pretrained on ImageNet.
The final fully connected (fc) layer is replaced with a single output neuron,
since this is a binary classification problem (NORMAL vs PNEUMONIA).

Author: Urmi Bhattacaryya
Date: September 2025

"""

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Bin(nn.Module):
    def __init__(self):
        super().__init__()
        # Load ResNet18 with pretrained ImageNet weights
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features

        # Replace last layer: instead of 1000 classes -> 1 logit
        backbone.fc = nn.Linear(in_features, 1)

        self.model = backbone

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x (Tensor): input batch [B, 3, H, W]
        Returns:
            logits (Tensor): [B], unnormalized logit per image
        """
        return self.model(x).squeeze(1)
=======
"""
model.py
--------
Defines the CNN architecture for classification.

We use torchvision's ResNet18 pretrained on ImageNet.
The final fully connected (fc) layer is replaced with a single output neuron,
since this is a binary classification problem (NORMAL vs PNEUMONIA).

Author: Urmi Bhattacaryya
Date: September 2025

"""

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Bin(nn.Module):
    def __init__(self):
        super().__init__()
        # Load ResNet18 with pretrained ImageNet weights
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = backbone.fc.in_features

        # Replace last layer: instead of 1000 classes -> 1 logit
        backbone.fc = nn.Linear(in_features, 1)

        self.model = backbone

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x (Tensor): input batch [B, 3, H, W]
        Returns:
            logits (Tensor): [B], unnormalized logit per image
        """
        return self.model(x).squeeze(1)
>>>>>>> 797ec99 (Clean repo and apply final .gitignore)
