"""Digit-position classifiers for five-character CAPTCHA images."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models as tv_models


class CAPTCHAClassifier(nn.Module):
    def __init__(self, image_size=(80, 260), num_digits=5, num_classes=10):
        super().__init__()
        self.image_size = tuple(image_size)
        self.num_digits = num_digits
        self.num_classes = num_classes

    def prepare(self, images):
        if images.dtype == torch.uint8:
            images = images.float().div(255.0)
        if tuple(images.shape[-2:]) != self.image_size:
            images = F.interpolate(images, size=self.image_size, mode="bilinear", align_corners=False)
        return (images - 0.5) / 0.5

    def compute_loss(self, logits, targets):
        return F.cross_entropy(logits.float(), targets), None


class TorchvisionResNet(CAPTCHAClassifier):
    def __init__(self, depth=18, image_size=(80, 260), num_digits=5, num_classes=10, spatial_head=False):
        super().__init__(image_size, num_digits, num_classes)
        constructor = {18: tv_models.resnet18, 34: tv_models.resnet34}.get(depth)
        if constructor is None:
            raise ValueError("depth must be 18 or 34")
        self.backbone = constructor(weights=None)
        if spatial_head:
            self.backbone.avgpool = nn.Identity()
            height, width = self.image_size
            in_features = self.backbone.fc.in_features * ((height + 31) // 32) * ((width + 31) // 32)
        else:
            in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_digits * num_classes)

    def forward(self, images):
        logits = self.backbone(self.prepare(images))
        return logits.reshape(images.shape[0], self.num_classes, self.num_digits)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = nn.Identity() if stride == 1 and in_channels == channels else nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, stride, bias=False), nn.BatchNorm2d(channels)
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.activation(x + residual)


class CAPTCHAResNet(CAPTCHAClassifier):
    """Compact ResNet-18 style model preserving the final spatial feature map."""

    def __init__(self, image_size=(80, 260), num_digits=5, num_classes=10, layers=(2, 2, 2, 2)):
        super().__init__(image_size, num_digits, num_classes)
        self.in_channels = 16
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(16, layers[0], stride=1)
        self.layer2 = self._make_layer(32, layers[1], stride=2)
        self.layer3 = self._make_layer(64, layers[2], stride=2)
        self.layer4 = self._make_layer(128, layers[3], stride=2)
        height, width = self.image_size
        self.classifier = nn.Linear(128 * ((height + 15) // 16) * ((width + 15) // 16), num_classes * num_digits)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(self, channels, count, stride):
        blocks = [BasicBlock(self.in_channels, channels, stride)]
        self.in_channels = channels
        blocks.extend(BasicBlock(channels, channels) for _ in range(1, count))
        return nn.Sequential(*blocks)

    def forward(self, images):
        x = self.stem(self.prepare(images))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        logits = self.classifier(torch.flatten(x, 1))
        return logits.reshape(images.shape[0], self.num_classes, self.num_digits)


def _downsample(size, stride):
    return (size - 1) // stride + 1


class CAPTCHASequenceNet(CAPTCHAClassifier):
    """Residual encoder + BiLSTM + per-position attention.

    The flatten head of ``CAPTCHAResNet`` ties each output digit to fixed
    image columns, so it breaks when the gaps between digits differ from the
    training data. Here the encoder keeps a width-wise sequence of column
    features (height is folded into channels, no global pooling), a BiLSTM
    gives every column left/right context, and five learned queries attend
    over the sequence to pick out the first, second, ... fifth digit wherever
    it actually is.
    """

    def __init__(self, image_size=(80, 260), num_digits=5, num_classes=10, widths=(32, 64, 128, 256), hidden=256):
        super().__init__(image_size, num_digits, num_classes)
        self.in_channels = widths[0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Height is reduced 16x in total, width only 8x, to keep column resolution.
        self.layer1 = self._make_layer(widths[0], 2, stride=1)
        self.layer2 = self._make_layer(widths[1], 2, stride=2)
        self.layer3 = self._make_layer(widths[2], 2, stride=2)
        self.layer4 = self._make_layer(widths[3], 2, stride=(2, 1))
        height, _ = self.image_size
        feature_height = _downsample(_downsample(_downsample(height // 2, 2), 2), 2)
        self.project = nn.Sequential(
            nn.Linear(widths[3] * feature_height, hidden), nn.LayerNorm(hidden), nn.ReLU(inplace=True), nn.Dropout(0.1)
        )
        self.sequence = nn.LSTM(hidden, hidden // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.queries = nn.Parameter(torch.randn(num_digits, hidden) * hidden**-0.5)
        self.key = nn.Linear(hidden, hidden)
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, num_classes))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(self, channels, count, stride):
        blocks = [BasicBlock(self.in_channels, channels, stride)]
        self.in_channels = channels
        blocks.extend(BasicBlock(channels, channels) for _ in range(1, count))
        return nn.Sequential(*blocks)

    def forward(self, images):
        x = self.stem(self.prepare(images))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        batch, channels, height, width = x.shape
        columns = x.permute(0, 3, 1, 2).reshape(batch, width, channels * height)
        columns = self.project(columns)
        self.sequence.flatten_parameters()
        columns, _ = self.sequence(columns)
        scores = torch.einsum("bwh,dh->bdw", self.key(columns), self.queries.to(columns.dtype))
        attention = scores.float().softmax(dim=-1).to(columns.dtype)
        context = torch.einsum("bdw,bwh->bdh", attention, columns)
        logits = self.classifier(context)  # [batch, digits, classes]
        return logits.transpose(1, 2)  # [batch, classes, digits]


MODELS = ("resnet18", "resnet34", "resnet18_spatial", "captcha_resnet", "captcha_seq")


def create_model(name, image_size=(80, 260)):
    if name == "captcha_seq":
        return CAPTCHASequenceNet(image_size=image_size)
    if name == "captcha_resnet":
        return CAPTCHAResNet(image_size=image_size)
    if name == "resnet18_spatial":
        return TorchvisionResNet(depth=18, image_size=image_size, spatial_head=True)
    if name in {"resnet18", "resnet34"}:
        return TorchvisionResNet(depth=int(name.removeprefix("resnet")), image_size=image_size)
    raise ValueError(f"unknown model {name!r}; choose one of {', '.join(MODELS)}")
