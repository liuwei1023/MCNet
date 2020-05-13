from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import time 

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class CBHModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(CBHModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.nonlinear = hswish()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.nonlinear(out)

        return out 

class UPModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UPModule, self).__init__()
        self.conv = CBHModule(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
 
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = x1 + x2 
        x = self.conv(x)
        return x

class Block(nn.Module):
    def __init__(self, kernel_size, in_channels, expand_channels, out_channels, nonlinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_channels, expand_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_channels)
        self.nonlinear1 = nonlinear
        self.conv2 = nn.Conv2d(expand_channels, expand_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_channels)
        self.nonlinear2 = nonlinear
        self.conv3 = nn.Conv2d(expand_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride == 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.nonlinear1(self.bn1(self.conv1(x)))
        out = self.nonlinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out 
        return out 


class MobileNetV3(nn.Module):
    def __init__(self, out_channels):
        super(MobileNetV3, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),           # 0
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),           # 1
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),           # 2
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),   # 3
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),  # 4
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),  # 5
            Block(3, 40, 240, 80, hswish(), None, 2),                       # 6
            Block(3, 80, 200, 80, hswish(), None, 1),                       # 7
            Block(3, 80, 184, 80, hswish(), None, 1),                       # 8
            Block(3, 80, 184, 80, hswish(), None, 1),                       # 9
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),             # 10
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),            # 11
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),            # 12
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),            # 13
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),            # 14
        )

        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()

        self.conv3 = CBHModule(960, 320, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv4 = CBHModule(320, 24, kernel_size=1, stride=1, padding=0, bias=False)

        self.up1 = UPModule(24, 24)
        self.up2 = UPModule(24, 24)
        self.up3 = UPModule(24, 24)

        self.conkeep12 = CBHModule(160, 24, kernel_size=1, stride=1, padding=0, bias=False)
        self.conkeep5  = CBHModule(40,  24, kernel_size=1, stride=1, padding=0, bias=False)
        self.conkeep2  = CBHModule(24,  24, kernel_size=1, stride=1, padding=0, bias=False)

        self.outconv = CBHModule(24, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.init_params()


    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):

        out = self.hs1(self.bn1(self.conv1(x)))

        keep = {"2": None, "5": None, "12": None}
        for index, item in enumerate(self.bneck):
            out = item(out)

            if str(index) in keep:
                keep[str(index)] = out 
        
        out = self.hs2(self.bn2(self.conv2(out)))
        out = self.conv3(out)
        out = self.conv4(out)

        out = self.up1(out, self.conkeep12(keep["12"]))
        out = self.up2(out, self.conkeep5(keep["5"]))
        out = self.up3(out, self.conkeep2(keep["2"]))

        out = self.outconv(out)

        return out 


class MCNet(nn.Module):
    def __init__(self, num_class):
        super(MCNet, self).__init__()
        self.backbone = MobileNetV3(128)
        self.hm = nn.Sequential(
            CBHModule(128, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(32, num_class, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.wh = nn.Sequential(
            CBHModule(128, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.reg = nn.Sequential(
            CBHModule(128, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1, bias=False),
        )
        # self.hm = nn.Conv2d(128, num_class, kernel_size=3, stride=1, padding=1, bias=False)
        # self.wh = nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1, bias=False)
        # self.reg = nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        backbone = self.backbone(x)
        hm = _sigmoid(self.hm(backbone))
        wh = self.wh(backbone)
        reg = self.reg(backbone)

        ret = {}
        ret["hm"]  = hm
        ret["wh"]  = wh 
        ret["reg"] = reg 

        return ret 



def get_MobileNet_CenterNet_model(num_class):
    model = MCNet(num_class)
    return model

def test():
    net = MCNet(1)
    t1 = time.time()
    x = torch.randn(2,3,97,176)
    y = net(x)
    t2 =time.time()
    print(f"hm={y['hm'].shape}, wh={y['wh'].shape}, reg={y['reg'].shape},  cost_time:{(t2-t1):.8f}sec!")

test()