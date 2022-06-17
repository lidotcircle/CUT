import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .styleformer import StyleFormer
from .cvt import CvT


class AdaIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(AdaIN, self).__init__()
        self.eps = eps
        self.num_features = num_features

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out = (input - in_mean) / torch.sqrt(in_var + self.eps)
        gamma = gamma.expand(1, 1, -1, -1).permute(2, 3, 0, 1)
        beta = beta.expand(1, 1, -1, -1).permute(2, 3, 0, 1)
        out = out * gamma.expand(input.shape[0], -1, -1, -1) + beta.expand(input.shape[0], -1, -1, -1)

        return out


def get_filter(filt_size=3):
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt

def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer

class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        self.adaptive_norm = norm_layer == AdaIN

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x, gamma = None, beta = None):
        """Forward function (with skip connections)"""
        if self.adaptive_norm:
            out = x
            for _, layer in enumerate(self.conv_block):
                if isinstance(layer, AdaIN):
                    out = layer(out, gamma, beta)
                else:
                    out = layer(out)
        else:
            out = self.conv_block(x)

        out = x + out  # add skip connections
        return out


class Transtyle(torch.nn.Module):
    def __init__(
        self,
        input_nc=3,
        output_nc=3,
        ngf=64,
        use_dropout=False,
        n_blocks=6,
        padding_type='reflect',
        no_antialias=False,
        no_antialias_up=False,
        style_extractor='cvt',
        opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(Transtyle, self).__init__()
        self.opt = opt
        use_bias = True
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]


        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if(no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(int(n_blocks / 2)):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=nn.InstanceNorm2d, use_dropout=use_dropout, use_bias=use_bias)]

        self.adaIN_layers = []
        self.num_style_outputs = 0
        self.style_start_and_len = {}
        def add_adaIN(dim):
            self.adaIN_layers.append(len(model))
            self.style_start_and_len[len(model)] = (self.num_style_outputs, dim * 2)
            self.num_style_outputs += dim * 2

        for i in range(n_blocks - int(n_blocks / 2)):
            add_adaIN(ngf * mult)
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=AdaIN, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model.append(nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias))
                add_adaIN(int(ngf * mult / 2))
                model.append(AdaIN(int(ngf * mult / 2)))
                model.append(nn.ReLU(True))
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias)]
                add_adaIN(int(ngf * mult / 2))
                model.append(AdaIN(int(ngf * mult / 2)))
                model.append(nn.ReLU(True))
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        if style_extractor == 'cvt':
            self.styleformer = CvT(num_classes=self.num_style_outputs)
        elif style_extractor == 'vit':
            self.styleformer = StyleFormer(num_outputs=self.num_style_outputs)
        else:
            raise NotImplementedError()
        
        self.style_logit_output = nn.Linear(self.num_style_outputs, 256)
        self.model = nn.Sequential(*model)

    def forward(self, input, layers=[], encode_only=False, return_style=False):
        feat = input
        feats = []
        styles = None
        style_logit = None
        for layer_id, layer in enumerate(self.model):
            # print(layer_id, layer)
            if layer_id in self.adaIN_layers and styles == None:
                styles = self.styleformer(input)
                style_logit = self.style_logit_output(styles)

            if layer_id in self.adaIN_layers:
                sstart, slen = self.style_start_and_len[layer_id]
                gamma_beta = styles[:,sstart:sstart+slen]
                gamma_beta = gamma_beta.view(gamma_beta.size(0), 2, -1)
                gamma = gamma_beta[:,0]
                beta = gamma_beta[:,1]
                feat = layer(feat, gamma, beta)
            else:
                feat = layer(feat)
            if layer_id in layers:
                # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                feats.append(feat)
            else:
                # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                pass
            if (len(layers) == 0 or layer_id == layers[-1]) and encode_only:
                # print('encoder only return features')
                if return_style:
                    return feats, style_logit  # return intermediate features alone; stop in the last layers
                else:
                    return feats

        if len(layers) > 0:
            if return_style:
                return feat, feats, style_logit  # return both output and intermediate features
            else:
                return feat, feats  # return both output and intermediate features
        else:
            if return_style:
                return feat, style_logit
            else:
                return feat


class TransDiscriminator(torch.nn.Module):
    def __init__(self):
        super(TransDiscriminator, self).__init__()
        self.encoder = StyleFormer(1024, num_layers=6)

    def forward(self, input):
        return self.encoder(input)