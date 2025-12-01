import torch
import torch.nn as nn
from torchvision.models import resnet50

class ImprovedYOLOHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super(ImprovedYOLOHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
        )
        self.final_conv = nn.Conv2d(256, num_anchors * (5 + num_classes), 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        nn.init.normal_(self.final_conv.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.final_conv.bias, 0.0)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.final_conv(x)
        B, _, H, W = x.shape
        x = x.view(B, self.num_anchors, 5 + self.num_classes, H, W)
        return x

class SimpleMultiScaleYOLOv3(nn.Module):
    def __init__(self, num_classes=5, num_anchors=3, pretrained=True):
        super(SimpleMultiScaleYOLOv3, self).__init__()

        backbone = resnet50(pretrained=pretrained)

        # Initial layers
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # Backbone layers
        self.layer1 = backbone.layer1  # 256 channels
        self.layer2 = backbone.layer2  # 512 channels
        self.layer3 = backbone.layer3  # 1024 channels
        self.layer4 = backbone.layer4  # 2048 channels

        # Additional conv layers to adapt channels
        self.adapt_conv1 = nn.Conv2d(512, 512, 1)   # For 52x52
        self.adapt_conv2 = nn.Conv2d(1024, 1024, 1) # For 26x26
        self.adapt_conv3 = nn.Conv2d(2048, 1024, 1) # For 13x13

        # Three detection heads
        self.head_52x52 = ImprovedYOLOHead(512, num_anchors, num_classes)   # Small objects
        self.head_26x26 = ImprovedYOLOHead(1024, num_anchors, num_classes)  # Medium objects
        self.head_13x13 = ImprovedYOLOHead(1024, num_anchors, num_classes)  # Large objects

        self._freeze_batch_norm()

    def _freeze_batch_norm(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):
        # Backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)   # 104x104
        x2 = self.layer2(x1)  # 52x52
        x3 = self.layer3(x2)  # 26x26
        x4 = self.layer4(x3)  # 13x13

        # Adapt channels and create detection outputs
        p1 = self.adapt_conv1(x2)  # 52x52 - Small objects
        p2 = self.adapt_conv2(x3)  # 26x26 - Medium objects
        p3 = self.adapt_conv3(x4)  # 13x13 - Large objects

        out_52x52 = self.head_52x52(p1)
        out_26x26 = self.head_26x26(p2)
        out_13x13 = self.head_13x13(p3)

        return [out_52x52, out_26x26, out_13x13]