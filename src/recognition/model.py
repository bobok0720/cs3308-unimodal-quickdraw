"""Model definition for recognition."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_resnet18(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
