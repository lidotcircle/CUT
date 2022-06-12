# A pytorch implementation of the SPatchGAN discriminator, tested with pytorch 1.7.1.
# This file is released under the BSD 3-Clause license, see https://github.com/NetEase-GameAI/SPatchGAN/blob/main/LICENSE
# Author: shaoxuning@corp.netease.com
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class SPatchDiscriminator(nn.Module):
    """Defines an SPatchGAN discriminator"""

    def __init__(self,
                 stats: list,
                 input_nc: int = 3,
                 ch: int = 256,
                 n_downsample_init: int = 2,
                 n_scales: int = 4,
                 n_adapt: int = 2,
                 n_mix: int = 2):
        """Constructs an SPatchGAN discriminator

        Parameters:
            stats (list) -- a list of the statistical features, e.g., ['mean', 'max', 'stddev']
            input_nc (int) -- Number of channels of the input images.
            ch (int) -- Base channel number.
            n_downsample_init (int) -- Number of downsampling layers in the initial feature extraction block.
            n_scales (int) -- Number of scales in D.
            n_adapt (int) -- Number of layers in each adaptation block.
            n_mix (int) -- Number of mixing layers in each MLP.
        """
        super().__init__()

        self._ch = ch
        self._n_downsample_init = n_downsample_init
        self._n_scales = n_scales
        self._n_adapt = n_adapt
        self._n_mix = n_mix
        self._stats = stats

        in_ch = input_nc
        out_ch = self._ch
        feat_extract = []
        for i in range(self._n_downsample_init):
            feat_extract += [spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))]
            feat_extract += [nn.LeakyReLU(0.2, True)]
            in_ch = out_ch
            out_ch *= 2

        self.feat_extract = nn.Sequential(*feat_extract)

        self.downsample_blocks = nn.ModuleList()

        self.scales = nn.ModuleList()

        for i in range(self._n_scales):
            downsample_block = [spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))]
            in_ch = out_ch
            downsample_block += [nn.LeakyReLU(0.2, True)]
            self.downsample_blocks += [nn.Sequential(*downsample_block)]
            self.scales += [SPatchStats(ch=in_ch, stats=self._stats, n_adapt=self._n_adapt, n_mix=self._n_mix)]

    def forward(self, x):
        """The forward method of the SPatchGAN discriminator.
        Input: a tensor of NCHW
        Output: logits of (N, n_scales, n_stats)
        """
        logits = []
        x = self.feat_extract(x)
        for i in range(self._n_scales):
            x = self.downsample_blocks[i](x)
            logits.append(self.scales[i](x))

        logits = torch.stack(logits, dim=1)  # Stack the scales
        return logits


class SPatchStats(nn.Module):

    def __init__(self, ch, stats, n_adapt, n_mix):
        super().__init__()
        self._ch = ch
        self._stats = stats
        self._n_adapt = n_adapt
        self._n_mix = n_mix

        adapt_layers = []
        for i in range(self._n_adapt):
            adapt_layers += [spectral_norm(nn.Conv2d(self._ch, self._ch, kernel_size=1))]
            adapt_layers += [nn.LeakyReLU(0.2, True)]
        self.adapt = nn.Sequential(*adapt_layers)

        self.mean_mlp = MLP(ch=self._ch, n_mix=self._n_mix) if 'mean' in self._stats else None
        self.max_mlp = MLP(ch=self._ch, n_mix=self._n_mix) if 'max' in self._stats else None
        self.stddev_mlp = MLP(ch=self._ch, n_mix=self._n_mix) if 'stddev' in self._stats else None

    def forward(self, x):
        x = self.adapt(x)
        logits = []
        if 'mean' in self._stats:
            gap = F.adaptive_avg_pool2d(x, 1)
            gap = gap.squeeze(3).squeeze(2)
            gap_logit = self.mean_mlp(gap)
            logits.append(gap_logit)
        if 'max' in self._stats:
            gmp = F.adaptive_max_pool2d(x, output_size=1)
            gmp = gmp.squeeze(3).squeeze(2)
            gmp_logit = self.max_mlp(gmp)
            logits += [gmp_logit]
        if 'stddev' in self._stats:
            diff_square = torch.square(x - F.adaptive_avg_pool2d(x, 1))
            stddev = torch.sqrt(F.adaptive_avg_pool2d(diff_square, 1))
            stddev = stddev.squeeze(3).squeeze(2)
            stddev_logit = self.stddev_mlp(stddev)
            logits.append(stddev_logit)

        logits = torch.cat(logits, dim=1)
        return logits


class MLP(nn.Module):

    def __init__(self, ch, n_mix):
        super().__init__()
        self._ch = ch
        self._n_mix = n_mix
        net = []
        for i in range(self._n_mix):
            net += [spectral_norm(nn.Linear(self._ch, self._ch))]
            net += [nn.LeakyReLU(0.2, True)]
        net += [spectral_norm(nn.Linear(self._ch, 1))]
        self.net = nn.Sequential(*net)

    def forward(self, input):
        return self.net(input)
