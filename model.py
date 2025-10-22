import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        xout = self.relu_s1(self.bn_s1(self.conv_s1(x)))

        return xout


def _upsample_like(src, tar):
    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')

    return src


class Res2Net(nn.Module):

    def __init__(self, inplanes, planes, stride=1, scale=4):
        super(Res2Net, self).__init__()

        width = planes

        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.conv_x2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width * scale, inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.LeakyRelu = nn.LeakyReLU(inplace=False)

        self.width = width

    def forward(self, x):
        residual = x
        output = self.LeakyRelu(self.bn1(self.conv1(x)))

        split = torch.split(output, self.width, 1)

        K1 = self.LeakyRelu(self.bn2(self.conv_x2(split[0])))
        K2 = self.LeakyRelu(self.bn2(self.conv_x2(K1 + split[1])))
        K3 = self.LeakyRelu(self.bn2(self.conv_x2(K2 + split[2])))
        K4 = self.LeakyRelu(self.bn2(self.conv_x2(K3 + split[3])))

        out1 = torch.cat((K1, K2), 1)
        out2 = torch.cat((out1, K3), 1)
        out3 = torch.cat((out2, K4), 1)
        out = self.bn3(self.conv3(out3))
        out += residual
        out = self.LeakyRelu(out)
        return out


class block(nn.Module):
    # Res2Block
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(block, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)

        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.resnet2 = Res2Net(mid_ch, mid_ch)

        self.rebnconv3 = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)

        hx2 = self.pool1(hx1)

        hx3 = self.resnet2(hx2)

        hx6dup = _upsample_like(hx3, hx1)

        hx7 = self.rebnconv3(torch.cat((hx6dup, hx1), 1))
        hx8 = self.rebnconv4(hx7)

        return hx8 + hxin


class BottleNeck(nn.Module):
    # Res2Block Bottleneck
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(BottleNeck, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.conv1 = REBNCONV(out_ch, mid_ch, dirate=1)

        self.resnet1 = Res2Net(mid_ch, mid_ch)

        self.conv2 = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.conv3 = REBNCONV(mid_ch, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)
        hx1 = self.conv1(hxin)

        hx2 = self.resnet1(hx1)

        hx5 = self.conv2(torch.cat((hx2, hx1), 1))
        hx6 = self.conv3(hx5)

        return hx6 + hxin


class ures2net(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(ures2net, self).__init__()

        # encoder

        self.stage1 = block(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = block(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = block(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = block(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = BottleNeck(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # bottleneck

        self.stage6 = BottleNeck(512, 256, 512)

        # decoder
        self.stage5d = BottleNeck(1024, 256, 512)
        self.stage4d = block(1024, 128, 256)
        self.stage3d = block(512, 64, 128)
        self.stage2d = block(256, 32, 64)
        self.stage1d = block(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        hx = x
        # -------------------- encoder --------------------
        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side outputs
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)