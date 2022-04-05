import torch as t
import torch.nn as nn
import torch.nn.functional as F


# 定义残差块ResBlock
class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, ResBlock):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        self.layer5 = self.make_layer(ResBlock, 1024, 2, stride=1)

    # 这个函数主要是用来，重复同一个残差块
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # 在这里，整个ResNet18的结构就很清晰了
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


def ResNet18():
    return ResNet(ResBlock)


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

        '''接下来构建RPN网络，这里的网络有分支，所以不好用一个nn.sequential直接操作'''
        self.base_conv = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)

        self.RPN_CLS = nn.Sequential(
            nn.Conv2d(256, 18, kernel_size=1),  # 输出维度为[b, 18, w, h], 18=9×2（9个anchor，每个anchor二分类，使用交叉熵损失），
            nn.ReLU(True)  # 这个必须得跟上，否者最后模型跑出的值会有负数
        )
        self.RPN_POS = nn.Sequential(
            nn.Conv2d(256, 36, kernel_size=1),  # 进行回归的卷积核通道数为36=9×4（9个anchor，每个anchor有4个位置参数）
            nn.ReLU(True)
        )

    def forward(self, extractor):
        x = self.base_conv(extractor)
        rpn_class = self.RPN_CLS(x)
        rpn_prob = F.softmax(rpn_class, dim=1)

        rpn_bbox = self.RPN_POS(x)
        return rpn_class, rpn_prob, rpn_bbox


if __name__ == "__main__":
    resnet = ResNet18()
    x = t.randn((1, 3, 64, 64))
    feature_map = resnet(x)
    rpn = RPN()
    out = rpn(feature_map)
    print(feature_map.shape)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
