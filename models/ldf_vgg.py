#!/usr/bin/python3
#coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('.')
from models.vgg16 import Backbone_VGG16_in3

upsample_mode = ''

def weight_init(module):
    for n, m in module.named_children():
        # print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv0 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

    def forward(self, input1, input2=[0,0,0,0]):
        out0 = F.relu(self.bn0(self.conv0(input1[0]+input2[0])), inplace=True)
        out0 = F.interpolate(out0, size=input1[1].size()[2:], mode='bilinear')
        out1 = F.relu(self.bn1(self.conv1(input1[1]+input2[1]+out0)), inplace=True) # * 2
        out1 = F.interpolate(out1, size=input1[2].size()[2:], mode='bilinear')
        out2 = F.relu(self.bn2(self.conv2(input1[2]+input2[2]+out1)), inplace=True)
        out2 = F.interpolate(out2, size=input1[3].size()[2:], mode='bilinear')
        out3 = F.relu(self.bn3(self.conv3(input1[3]+input2[3]+out2)), inplace=True)
        return out3
    
    def initialize(self):
        weight_init(self)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4   = nn.BatchNorm2d(64)

        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1b   = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2b   = nn.BatchNorm2d(64)
        self.conv3b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3b   = nn.BatchNorm2d(64)
        self.conv4b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4b   = nn.BatchNorm2d(64)

        self.conv1d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1d   = nn.BatchNorm2d(64)
        self.conv2d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2d   = nn.BatchNorm2d(64)
        self.conv3d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3d   = nn.BatchNorm2d(64)
        self.conv4d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4d   = nn.BatchNorm2d(64)

    def forward(self, out1):
        out1 = F.relu(self.bn1(self.conv1(out1)), inplace=True)
        out2 = F.max_pool2d(out1, kernel_size=2, stride=2)
        out2 = F.relu(self.bn2(self.conv2(out2)), inplace=True)
        out3 = F.max_pool2d(out2, kernel_size=2, stride=2)
        out3 = F.relu(self.bn3(self.conv3(out3)), inplace=True)
        out4 = F.max_pool2d(out3, kernel_size=2, stride=2)
        out4 = F.relu(self.bn4(self.conv4(out4)), inplace=True)

        out1b = F.relu(self.bn1b(self.conv1b(out1)), inplace=True)
        out2b = F.relu(self.bn2b(self.conv2b(out2)), inplace=True)
        out3b = F.relu(self.bn3b(self.conv3b(out3)), inplace=True)
        out4b = F.relu(self.bn4b(self.conv4b(out4)), inplace=True)

        out1d = F.relu(self.bn1d(self.conv1d(out1)), inplace=True)
        out2d = F.relu(self.bn2d(self.conv2d(out2)), inplace=True)
        out3d = F.relu(self.bn3d(self.conv3d(out3)), inplace=True)
        out4d = F.relu(self.bn4d(self.conv4d(out4)), inplace=True)
        return (out4b, out3b, out2b, out1b), (out4d, out3d, out2d, out1d)

    def initialize(self):
        weight_init(self)


class LDF_VGG(nn.Module):
    def __init__(self, cfg):
        super(LDF_VGG, self).__init__()
        self.cfg      = cfg
        self.bkbone_div1, self.bkbone_div2, self.bkbone_div4, self.bkbone_div8, self.bkbone_div16 = Backbone_VGG16_in3(cfg.MODEL.BAKCBONE_PATH)
        # self.bkbone_div1, self.bkbone_div2, self.bkbone_div4, self.bkbone_div8, self.bkbone_div16 = Backbone_VGG16_in3(None)


        b_conv_channel_list = [64, 128, 256, 512, 512]

        conv_down = []
        for i in range(4):
            in_ch = b_conv_channel_list[i + 1]
            out_ch = in_ch * 2
            conv_down.append(
                nn.Sequential(
                    nn.AvgPool2d(kernel_size = 2, stride = 2),
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride = 1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, stride = 1, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )

        # 256, 512, 1024, 2048
        self.conv2_down, self.conv3_down, self.conv4_down, self.conv5_down = conv_down

        self.conv5b   = nn.Sequential(nn.Conv2d(b_conv_channel_list[4] * 2, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4b   = nn.Sequential(nn.Conv2d(b_conv_channel_list[3] * 2, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3b   = nn.Sequential(nn.Conv2d(b_conv_channel_list[2] * 2, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2b   = nn.Sequential(nn.Conv2d(b_conv_channel_list[1] * 2, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
        self.conv5d   = nn.Sequential(nn.Conv2d(b_conv_channel_list[4] * 2, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4d   = nn.Sequential(nn.Conv2d(b_conv_channel_list[3] * 2, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3d   = nn.Sequential(nn.Conv2d(b_conv_channel_list[2] * 2, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2d   = nn.Sequential(nn.Conv2d(b_conv_channel_list[1] * 2, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.decoderb = Decoder()
        self.encoder  = Encoder()
        self.decoderd = Decoder()
        self.linearb  = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.lineard  = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linear   = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=3, padding=1))
        # self.initialize()

    def init_head(self): 
        # init head model
        print(f"random init predict head")
        module_list = [
            self.conv4b,
            self.conv5b,
            self.conv3b,
            self.conv2b,
            self.conv4d,
            self.conv5d,
            self.conv3d,
            self.conv2d,
            self.decoderb,
            self.encoder,
            self.decoderd,
            self.linearb,
            self.lineard,
            self.linear,
        ]
        for module in module_list:
            weight_init(module)

    def init_base(self, bkbone_path = None):
        print("[FIXME]: vgg already initilizaed in vgg backbone model setu")
        # self.bkbone.initialize(bkbone_path)

    def forward(self, x, shape=None):
        # out1, out2, out3, out4, out5 = self.bkbone(x)
        in_data_1 = self.bkbone_div1(x)
        in_data_2 = self.bkbone_div2(in_data_1)
        in_data_4 = self.bkbone_div4(in_data_2)
        in_data_8 = self.bkbone_div8(in_data_4)
        in_data_16 = self.bkbone_div16(in_data_8)
        out2, out3, out4, out5 = self.conv2_down(in_data_2), self.conv3_down(in_data_4), self.conv4_down(in_data_8), self.conv5_down(in_data_16)
        
        out2b, out3b, out4b, out5b   = self.conv2b(out2), self.conv3b(out3), self.conv4b(out4), self.conv5b(out5)
        out2d, out3d, out4d, out5d   = self.conv2d(out2), self.conv3d(out3), self.conv4d(out4), self.conv5d(out5)

        outb1 = self.decoderb([out5b, out4b, out3b, out2b])
        outd1 = self.decoderd([out5d, out4d, out3d, out2d])
        out1  = torch.cat([outb1, outd1], dim=1)
        outb2, outd2 = self.encoder(out1)
        outb2 = self.decoderb([out5b, out4b, out3b, out2b], outb2)
        outd2 = self.decoderd([out5d, out4d, out3d, out2d], outd2)
        out2  = torch.cat([outb2, outd2], dim=1)

        if shape is None:
            shape = x.size()[2:]
        # using interpolate to resize / upscale to target shape, used for test
        out1  = F.interpolate(self.linear(out1),   size=shape, mode='bilinear')
        outb1 = F.interpolate(self.linearb(outb1), size=shape, mode='bilinear')
        outd1 = F.interpolate(self.lineard(outd1), size=shape, mode='bilinear')

        out2  = F.interpolate(self.linear(out2),   size=shape, mode='bilinear')
        outb2 = F.interpolate(self.linearb(outb2), size=shape, mode='bilinear')
        outd2 = F.interpolate(self.lineard(outd2), size=shape, mode='bilinear')
        return outb1, outd1, out1, outb2, outd2, out2
    
    def get_parameters(self):
        base, head = [], []
        for name, param in self.named_parameters():
            # FIXME 是否把第一层的 convolution 加入优化
            if 'bkbone' in name:
                base.append(param)
            else:
                head.append(param)
        return {
            'base': base,
            'head': head
        }

    def random_init(self):
        print(f"random init all modules of model!!")
        weight_init(self)


if __name__ == "__main__":
    b, c, w, h = 1, 3, 352, 352
    input = torch.randn((b,c, w, h))
    ldf_vgg = LDF_VGG(None)
    ldf_vgg.get_parameters()
    out_list = ldf_vgg(input)
    for out in out_list:
        print(out.shape)
