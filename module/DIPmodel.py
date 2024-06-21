# coding:utf-8
import torch
import torch.nn as nn

global num_filter_parameters
num_filter_parameters = 5  # how many parameters need to learning by DNN, 5 means use defog filter


def conv_downsample(in_filters, out_filters, normalization=False):
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers


class CNN_PP(nn.Module):
    def __init__(self, inputChannel=3):   
        super(CNN_PP, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(256,256), mode='bilinear'),    # all input images will be resize d to 256 pixels
            nn.Conv2d(inputChannel, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *conv_downsample(16, 32, normalization=True),
            *conv_downsample(32, 64, normalization=True),
            *conv_downsample(64, 128, normalization=True),
            *conv_downsample(128, 128),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, num_filter_parameters, 8, padding=0),
        )

    def forward(self, img_input):
        self.Pr = self.model(img_input)
        
        return self.Pr


if __name__ == '__main__':
    net = CNN_PP(inputChannel=3) 
    x = torch.randn(2, 3, 256, 256)
    out = net(x)
    print(out.size())



