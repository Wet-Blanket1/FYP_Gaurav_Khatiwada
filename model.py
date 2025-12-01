# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class YOLOHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels * 2, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels * 2)
        self.conv2 = nn.Conv2d(in_channels * 2, num_anchors * (5 + num_classes), 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return x


class YOLOv3_ResNet50(nn.Module):
    def __init__(self, num_classes, num_anchors, pretrained=False):
        super().__init__()
        resnet = resnet50(pretrained=pretrained)

        self.layer1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool, resnet.layer1
        )
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.head3 = YOLOHead(1024, num_anchors, num_classes)
        self.reduce3 = nn.Conv2d(1024, 512, 1)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.head2 = YOLOHead(768, num_anchors, num_classes)
        self.reduce2 = nn.Conv2d(768, 256, 1)

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.head1 = YOLOHead(384, num_anchors, num_classes)

        self.num_classes = num_classes

    def forward(self, x):
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        p3 = self.head3(c4)
        r3 = self.reduce3(c4)

        u2 = self.up2(r3)
        cat2 = torch.cat([u2, c3], dim=1)
        p2 = self.head2(cat2)
        r2 = self.reduce2(cat2)

        u1 = self.up1(r2)
        cat1 = torch.cat([u1, c2], dim=1)
        p1 = self.head1(cat1)

        def reshape(x):
            B, C, H, W = x.shape
            A = C // (5 + self.num_classes)
            return x.view(B, A, 5 + self.num_classes, H, W)

        return reshape(p1), reshape(p2), reshape(p3)
