import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    def __init__(self, in_planes, stride=1, norm_layer=nn.BatchNorm2d):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], num_outputs=256, nc=3, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, num_Blocks[0], stride=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(BasicBlock, 128, num_Blocks[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(BasicBlock, 256, num_Blocks[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(BasicBlock, 512, num_Blocks[3], stride=2, norm_layer=norm_layer)
        self.linear = nn.Sequential(nn.Linear(512, 512), nn.ReLU(True))
        self.output = nn.Linear(512, num_outputs)

    def _make_layer(self, BasicBlock, planes, num_Blocks, stride, norm_layer):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlock(self.in_planes, stride, norm_layer=norm_layer)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, features: list=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if features is not None:
            features.append(x)
        x = self.layer2(x)
        if features is not None:
            features.append(x)
        x = self.layer3(x)
        if features is not None:
            features.append(x)
        x = self.layer4(x)
        if features is not None:
            features.append(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        if features is not None:
            features.append(x)
        return self.output(x) 