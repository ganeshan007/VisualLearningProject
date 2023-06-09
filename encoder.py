import torch
import functools
import torch.nn as nn
import torch.nn.functional as F

def get_norm_layer(norm_type='batch'):
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    return norm_layer

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=1.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=1.0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def define_encoder(input_nc, output_nc, ndf, model_name, norm='batch', init_type='xavier', gpu_ids=[], vaeLike=True):
    encoder = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)
    non_linearity = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    if use_gpu:
        assert(torch.cuda.is_available())
    if model_name == 'resnet_128':
        encoder = ResnetEncoder(input_nc, output_nc, ndf, norm_layer, 4, non_linearity, gpu_ids, vaeLike)
    elif model_name == 'resnet_256':
        encoder = ResnetEncoder(input_nc, output_nc, ndf, norm_layer, 5, non_linearity, gpu_ids, vaeLike)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % model_name)
    
    if use_gpu:
        encoder.cuda()
    encoder.apply(weights_init_xavier)
    return encoder


def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes,
                           kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=True)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += nn.Sequential(*[nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=True)])
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out

    
class ResnetEncoder(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ndf=64, n_blocks=4,
                 norm_layer=None, nl_layer=None, gpu_ids=[], vaeLike=False):
        super().__init__()
        self.gpu_ids = gpu_ids
        self.vaeLike = vaeLike
        max_ndf = 4
        conv_layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)
            output_ndf = ndf * min(max_ndf, n + 1)
            conv_layers += [BasicBlock(input_ndf,
                                       output_ndf, norm_layer, nl_layer)]
        conv_layers += [nl_layer(), nn.AvgPool2d(8)]
        if vaeLike:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
            self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        return output
