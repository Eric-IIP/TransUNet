import torch
import torch.nn as nn
import math
from collections import OrderedDict

class DilatedBlock(nn.Module):
    def __init__(self, inplanes, planes, name, kernelSize = 3) -> None:
        super(DilatedBlock, self).__init__()
        self.dilated1 = DilatedBlock._hdcBlock(inplanes, planes, dilation=(1,1), name=name, kernelSize=kernelSize)
        self.dilated2 = DilatedBlock._hdcBlock(inplanes, planes, dilation=(2,2), name=name, kernelSize=kernelSize)
        self.dilated5 = DilatedBlock._hdcBlock(inplanes, planes, dilation=(5,5), name=name, kernelSize=kernelSize)
        self.dilated9 = DilatedBlock._hdcBlock(inplanes, planes, dilation=(9,9), name=name, kernelSize=kernelSize)

    @staticmethod
    def _hdcBlock(inplanes, outplanes, dilation, name, kernelSize = 3):
        return nn.Sequential(
            OrderedDict([
                    (name + '_conv1', nn.Conv2d(inplanes, outplanes, kernelSize, dilation=dilation, padding=dilation)),
                    (name + '_batchnorm1', nn.BatchNorm2d(outplanes)),
                    (name + '_relu1', nn.ReLU(inplace=True))
                ])
            )

    def forward(self, x):
        dilated1 = self.dilated1(x)
        dilated2 = self.dilated2(x)
        dilated5 = self.dilated5(x)
        dilated9 = self.dilated9(x)
        output = torch.cat((dilated1, dilated2, dilated5, dilated9), 1)
        return output

class DilatedNet(nn.Module):
    def __init__(self, inplanes, planes, stride=1) -> None:
        super(DilatedNet, self).__init__()
        planes1 = inplanes
        planes2 = 32
        planes3 = 64
        planes4 = 128
        planes5 = 256
        planes10 = 512
        planes11 = 256
        planes6 = 128
        planes7 = 64
        planes8 = 32
        planes9 = 16
        
        self.dilated1 = DilatedBlock(inplanes=planes1, planes=planes2//4, name='hdc1')
        self.dilated2 = DilatedBlock(inplanes=planes2, planes=planes3//4, name='hdc2')
        self.dilated3 = DilatedBlock(inplanes=planes3, planes=planes4//4, name='hdc3')
        self.dilated4 = DilatedBlock(inplanes=planes4, planes=planes5//4, name='hdc4')
        self.dilated5 = DilatedBlock(inplanes=planes5, planes=planes10//4, name='hdc5')
        self.dilated9 = DilatedBlock(inplanes=planes10, planes=planes11//4, name='hdc5')
        self.dilated10 = DilatedBlock(inplanes=planes11, planes=planes6//4, name='hdc5')
        self.dilated6 = DilatedBlock(inplanes=planes6, planes=planes7//4, name='hdc6')
        self.dilated7 = DilatedBlock(inplanes=planes7, planes=planes8//4, name='hdc7')
        self.dilated8 = DilatedBlock(inplanes=planes8, planes=planes9//4, name='hdc8')
        self.conv19 =  nn.Sequential(
            nn.Conv2d(planes9, planes, 1, padding='same'),
            nn.Sigmoid())

    def forward(self, x):
        dilated1 = self.dilated1(x)
        dilated2 = self.dilated2(dilated1)
        dilated3 = self.dilated3(dilated2)
        dilated4 = self.dilated4(dilated3)
        dilated5 = self.dilated5(dilated4)
        dilated9 = self.dilated9(dilated5)
        dilated10 = self.dilated10(dilated9)
        dilated6 = self.dilated6(dilated10)
        dilated7 = self.dilated7(dilated6)
        dilated8 = self.dilated8(dilated7)
        out = self.conv19(dilated8)

        return out

#net = DilatedNet(4,1)
#print(list(net.parameters()))