# coding:utf-8
# This code was used to generate change detection model for complex scenes. Author:Mr zhipan wang, Email:1044625113@qq.com
# this model was modified from SNUnet, at https://github.com/likyoo
import sys
sys.setrecursionlimit(5000)

import torch
import torch.nn as nn
try:
    from module.DIPmodel import CNN_PP as DIP   
    from module.imgFilter import DefogFilter, ExposureFilter, GammaFilter, ContrastFilter, UsmFilter 
except:
    from .DIPmodel import CNN_PP as DIP   
    from .imgFilter import DefogFilter, ExposureFilter, GammaFilter, ContrastFilter, UsmFilter 


class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(
            mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output


class up(nn.Module):
    def __init__(self, in_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):

        x = self.up(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels//ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels, 1, bias=False)
        self.dropout = nn.Dropout2d(0.05)   # 缓解过拟合
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.dropout(out)
        return self.sigmod(out)


class SNUNet_ECAM(nn.Module):
    # SNUNet-CD with ECAM
    def __init__(self, in_ch=3, out_ch=2):
        super(SNUNet_ECAM, self).__init__()
        torch.nn.Module.dump_patches = True      
        
                                   
        self.DIP_T1 = DIP(inputChannel=in_ch)
        self.DIP_T2 = DIP(inputChannel=in_ch)
        
        self.DefogFilter = DefogFilter(inputChannel=in_ch)
        self.ExposureFilter = ExposureFilter(inputChannel=in_ch)
        self.GammaFilter = GammaFilter(inputChannel=in_ch)
        self.ContrastFilter = ContrastFilter(inputChannel=in_ch)
        self.UsmFilter = UsmFilter(inputChannel=in_ch)
        
        self.DefogFilter_T2 = DefogFilter(inputChannel=in_ch)
        self.ExposureFilter_T2 = ExposureFilter(inputChannel=in_ch)
        self.GammaFilter_T2 = GammaFilter(inputChannel=in_ch)
        self.ContrastFilter_T2 = ContrastFilter(inputChannel=in_ch)
        self.UsmFilter_T2 = UsmFilter(inputChannel=in_ch)
        
        
        n1 = 32     
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.Up1_0 = up(filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.Up2_0 = up(filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.Up3_0 = up(filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        self.Up4_0 = up(filters[4])

        self.conv0_1 = conv_block_nested(
            filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(
            filters[1] * 2 + filters[2], filters[1], filters[1])
        self.Up1_1 = up(filters[1])
        self.conv2_1 = conv_block_nested(
            filters[2] * 2 + filters[3], filters[2], filters[2])
        self.Up2_1 = up(filters[2])
        self.conv3_1 = conv_block_nested(
            filters[3] * 2 + filters[4], filters[3], filters[3])
        self.Up3_1 = up(filters[3])

        self.conv0_2 = conv_block_nested(
            filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(
            filters[1] * 3 + filters[2], filters[1], filters[1])
        self.Up1_2 = up(filters[1])
        self.conv2_2 = conv_block_nested(
            filters[2] * 3 + filters[3], filters[2], filters[2])
        self.Up2_2 = up(filters[2])

        self.conv0_3 = conv_block_nested(
            filters[0] * 4 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(
            filters[1] * 4 + filters[2], filters[1], filters[1])
        self.Up1_3 = up(filters[1])

        self.conv0_4 = conv_block_nested(
            filters[0] * 5 + filters[1], filters[0], filters[0])

        self.ca = ChannelAttention(filters[0] * 4, ratio=16)
        self.ca1 = ChannelAttention(filters[0], ratio=16 // 4)

        self.conv_final = nn.Conv2d(filters[0] * 4, out_ch, kernel_size=1)
        

    def forward(self, x):
        xA = x[:,0:3,:,:]
        xB = x[:,3:6,:,:]
        
        
        ParameterT1 = self.DIP_T1(xA)
        ParameterT2 = self.DIP_T2(xB)
        
        
        # image filter 
        xA = self.DefogFilter(xA, torch.unsqueeze(ParameterT1[:, 0, :, :], 1))
        xA = self.ExposureFilter(xA, torch.unsqueeze(ParameterT1[:, 1, :, :], 1))
        xA = self.GammaFilter(xA, torch.unsqueeze(ParameterT1[:, 2, :, :], 1))
        xA = self.ContrastFilter(xA, torch.unsqueeze(ParameterT1[:, 3, :, :], 1))
        xA = self.UsmFilter(xA, torch.unsqueeze(ParameterT1[:, 4, :, :], 1))
        
        xB = self.DefogFilter_T2(xB, torch.unsqueeze(ParameterT2[:, 0, :, :], 1))
        xB = self.ExposureFilter_T2(xB, torch.unsqueeze(ParameterT2[:, 1, :, :], 1))
        xB = self.GammaFilter_T2(xB, torch.unsqueeze(ParameterT2[:, 2, :, :], 1))
        xB = self.ContrastFilter_T2(xB, torch.unsqueeze(ParameterT2[:, 3, :, :], 1))
        xB = self.UsmFilter_T2(xB, torch.unsqueeze(ParameterT2[:, 4, :, :], 1))
        

        '''xA'''
        x0_0A = self.conv0_0(xA)
        x1_0A = self.conv1_0(self.pool(x0_0A))
        x2_0A = self.conv2_0(self.pool(x1_0A))
        x3_0A = self.conv3_0(self.pool(x2_0A))
        # x4_0A = self.conv4_0(self.pool(x3_0A))
        '''xB'''
        x0_0B = self.conv0_0(xB)
        x1_0B = self.conv1_0(self.pool(x0_0B))
        x2_0B = self.conv2_0(self.pool(x1_0B))
        x3_0B = self.conv3_0(self.pool(x2_0B))
        x4_0B = self.conv4_0(self.pool(x3_0B))

        x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(x2_0B)], 1))
        x0_2 = self.conv0_2(
            torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))

        x2_1 = self.conv2_1(torch.cat([x2_0A, x2_0B, self.Up3_0(x3_0B)], 1))
        x1_2 = self.conv1_2(
            torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(
            torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0A, x3_0B, self.Up4_0(x4_0B)], 1))
        x2_2 = self.conv2_2(
            torch.cat([x2_0A, x2_0B, x2_1, self.Up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(
            torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(
            torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))

        out = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)

        intra = torch.sum(torch.stack((x0_1, x0_2, x0_3, x0_4)), dim=0)
        ca1 = self.ca1(intra)
        out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))
        out = self.conv_final(out)
        
        return out
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class IA_CDNet(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(IA_CDNet, self).__init__()
        self.in_channels = int(in_channels/2)
        self.model = SNUNet_ECAM(in_ch=self.in_channels, out_ch=num_classes)

    def forward(self, x):
        out = self.model(x)
        return out


if __name__ == '__main__':
    device = torch.device('cpu')

    net = IA_CDNet(num_classes=2, in_channels=6).to(device)  
    x = torch.randn(2, 6, 512, 512).to(device)
    out = net(x)
    print(out.size())
