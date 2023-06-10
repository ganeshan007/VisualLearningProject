import torch
import numpy as np
import torch.nn.functional as F
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

class ConditionalMotionNet(nn.Module):
    def __init__(self, nz=8, nout=2, beta=1./64.) -> None:
        super().__init__()
        c_num = 128
        ## Downsample
        self.conv1 = ReflectionPadConv(3+nz, c_num, kernel_size=5, stride=2)
        self.conv2 = ReflectionPadConv(c_num+nz, c_num*2, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(c_num*2)
        self.conv3 = ReflectionPadConv(c_num*2+nz, c_num*4, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(c_num*4)

        ## Residual Layers
        self.res1 = ResidualBlock(c_num*4)
        self.res2 = ResidualBlock(c_num*4)
        self.res3 = ResidualBlock(c_num*4)
        self.res4 = ResidualBlock(c_num*4)
        self.res5 = ResidualBlock(c_num*4)

        ## Upsample Layers
        self.deconv1 = UpsampleConvLayer(c_num*4*2, c_num*2, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(c_num*2)
        self.deconv2 = UpsampleConvLayer(c_num*2*2, c_num, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(c_num)
        self.deconv3 = UpsampleConvLayer(c_num*2, nout, kernel_size=5, stride=1, upsample=2)

        self.relu = nn.LeakyReLU(0.1,inplace=True)
        self.beta = beta

    def forward(self, x, z, frame_size=0.):
        z2d = z.view(*z.shape,1,1).expand(*z.shape,x.shape[2],x.shape[3])
        xz_concat = torch.cat((x,z2d),1)
        h1 = self.relu(self.conv1(xz_concat))
        z2d = z.view(*z.shape,1,1).expand(*z.shape,h1.shape[2],h1.shape[3])
        h1z_concat = torch.cat((h1,z2d),1)
        h2 = self.relu(self.in2(self.conv2(h1z_concat)))
        z2d = z.view(*z.shape,1,1).expand(*z.shape,h2.shape[2],h2.shape[3])
        h2z_concat = torch.cat((h2,z2d),1)
        h3 = self.relu(self.in3(self.conv3(h2z_concat)))

        h4 = self.res1(h3)
        out = self.res2(h4)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)

        out = torch.cat((out,h3),1)
        out = self.relu(self.in4(self.deconv1(out)))
        out = torch.cat((out,h2),1)
        out = self.relu(self.in5(self.deconv2(out)))
        out = torch.cat((out,h1),1)
        out = self.deconv3(out)
        
        out = F.tanh(out)*self.beta

        return out

class ConditionalAppearanceNet(nn.Module):
    def __init__(self, nz=8) -> None:
        super().__init__()
        c_num = 128
        ## Downsample
        self.conv1 = ReflectionPadConv(3+nz, c_num, kernel_size=5, stride=2)
        self.conv2 = ReflectionPadConv(c_num+nz, c_num*2, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(c_num*2)
        self.conv3 = ReflectionPadConv(c_num*2+nz, c_num*4, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(c_num*4)

        ## Residual Layers
        self.res1 = ResidualBlock(c_num*4)
        self.res2 = ResidualBlock(c_num*4)
        self.res3 = ResidualBlock(c_num*4)
        self.res4 = ResidualBlock(c_num*4)
        self.res5 = ResidualBlock(c_num*4)

        ## Upsample Layers
        self.deconv1 = UpsampleConvLayer(c_num*4*2, c_num*2, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(c_num*2)
        self.deconv2 = UpsampleConvLayer(c_num*2*2, c_num, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(c_num)
        self.deconv3 = UpsampleConvLayer(c_num*2, nz, kernel_size=5, stride=1, upsample=2)

        self.fc1 = nn.Linear(c_num*4, 6)
        self.relu = nn.LeakyReLU(0.1,inplace=True)

    def forward(self, x, z):
        z2d = z.view(*z.shape,1,1).expand(*z.shape, x.shape[2], x.shape[3])
        xz_concat = torch.cat((x,z2d),1)
        h1 = self.relu(self.conv1(xz_concat))
        h1z_concat = torch.cat((h1,z2d),1)
        h2 = self.relu(self.in2(self.conv2(h1z_concat)))
        z2d = z.view(*z.shape,1,1).expand(z.shape, h2.shape[2], h2.shape[3])
        h2z_concat = torch.cat((h2,z2d),1)
        h3 = self.relu(self.in3(self.conv3(h2z_concat)))

        h4 = self.res1(h3)
        out = self.res2(h4)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)

        out = torch.cat((out,h3),1)
        out = self.relu(self.in4(self.deconv1(out)))
        out = torch.cat((out,h2),1)
        out = self.relu(self.in5(self.deconv2(out)))
        out = torch.cat((out,h1),1)
        out = self.deconv3(out)
        a, b = out.split(3,dim=1)

        y = F.tanh(a*x + b)
        return y, a, b

        
class GramMatrix(nn.Module):

    def forward(self, x):
        batch_size, num_features, height, width = x.size()
        # Reshape the input tensor to a 2D matrix of shape (num_features, height * width)
        features = x.view(batch_size * num_features, height * width)
        # Calculate the Gram matrix by multiplying the reshaped features matrix with its transpose
        gram = torch.matmul(features, features.t())
        # Normalize the Gram matrix by dividing by the number of elements in each feature map
        gram /= (batch_size * num_features * height * width)

        return gram


class Discriminator(nn.Module):
    def __init__(self) -> None:
        self.main = nn.Sequential(
            nn.Linear(8,128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.main(x)








