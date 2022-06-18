import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from .simple_resnet import ResNet18


class CallForwader:
    def __init__(self, to) -> None:
        self.to = to

    def __call__(self, *args, **kwargs):
        return self.to(*args, **kwargs)


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, pretrained_model: str, ngf=64, n_blocks=6, img_size=256):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        feature_encoder = ResNet18(num_outputs=128)
        feature_encoder.load_state_dict(torch.load(pretrained_model))
        feature_encoder.eval()
        for parameters in feature_encoder.parameters():
            parameters.requires_grad = False
        self.feature_encoder = CallForwader(feature_encoder)
        self.device_init = False

        DownBlock = []
        DownBlock += [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                      nn.InstanceNorm2d(ngf),
                      nn.ReLU(True)]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True)]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Class Activation Map
        ActMap = []
        ActMap += [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0, bias=False),
            ILN(128),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, ngf * mult, kernel_size=3, stride=1, padding=0, bias=False),
            ILN(ngf * mult),
            nn.ReLU(True),
            ResnetBlock(ngf * mult, use_bias=False),
            nn.Conv2d(ngf * mult, ngf * mult, kernel_size=1, stride=1, bias=True),
        ]

        # Up-Sampling
        UpBlock = []
        for i in range(n_blocks):
            UpBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock += [nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.ReflectionPad2d(1),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                         ILN(int(ngf * mult / 2)),
                         nn.ReLU(True)]

        UpBlock += [nn.ReflectionPad2d(3),
                     nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.Tanh()]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.ActMap = nn.Sequential(*ActMap)
        self.UpBlock = nn.Sequential(*UpBlock)

    def forward(self, input: torch.Tensor, heatmap: list=None):
        if not self.device_init:
            self.feature_encoder.to.to(input.device)
            self.device_init = True

        x = self.DownBlock(input)

        features = []
        self.feature_encoder(input, features=features)
        actMap = self.ActMap(features[2])
        x = x * actMap
        if heatmap:
            hm = torch.mean(actMap, dim = 1)
            heatmap.append(hm)

        out = self.UpBlock(x)
        return out


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out
