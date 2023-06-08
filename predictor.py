import torch
import numpy as np
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out
    

class ReflectionPadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2):
        super(ReflectionPadConv, self).__init__()
        
        # Reflection padding
        padding = int(np.floor(kernel_size/2))
        self.padding = nn.ReflectionPad2d(padding)
        
        # Convolutional layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        
    def forward(self, x):
        x = self.padding(x)
        x = self.conv(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.conv1 = ReflectionPadConv(channels,channels,kernel_size=3,stride=1)
        self.in1 = nn.InstanceNorm2d(channels,affine=True)
        self.conv2 = ReflectionPadConv(channels,channels,kernel_size=3,stride=1)
        self.in2 = nn.InstanceNorm2d(channels,affine=True)
        self.relu = nn.LeakyReLU(0.1,inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(self.relu(self.in1(x)))
        out = self.conv2(self.relu(self.in2(out)))
        out = out + residual
        return out

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample) -> None:
        super().__init__()
        self.upsample_layer = nn.Upsample(scale_factor=upsample)
        self.conv = ReflectionPadConv(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.upsample_layer(x)
        x = self.conv(x)
        return x

