import torch
import torch.nn as nn
import math
from EPModule import EPModule
from collections import OrderedDict

class MaxPoolSame(nn.Module):
    def __init__(self, kernelSize = 2, stride = 2) -> None:
        super(MaxPoolSame, self).__init__()
        self.kernelSize = kernelSize
        self.stride = stride

    def _maxPool2dSame(self, x: torch.Tensor):
        _, _, inWidth, inHeight = x.size()
        outWidth = math.ceil(float(inWidth)/float(self.stride))
        outHeight = math.ceil(float(inHeight)/float(self.stride))
        padAlongWidth = max((outWidth - 1) * self.stride + self.kernelSize - inWidth,0)
        padAlongHeight = max((outHeight - 1) * self.stride + self.kernelSize - inHeight,0)
        return nn.MaxPool2d(self.kernelSize, self.stride, (padAlongHeight//2, padAlongWidth//2))(x)
    
    def forward(self, x):
        return self._maxPool2dSame(x)

class DilatedBlock(nn.Module):
    def __init__(self, inplanes, planes, name, dilations=[1,2,5,9], kernelSize = 3) -> None:
        super(DilatedBlock, self).__init__()
        self.dilatedList = nn.ModuleList([])
        self.maxPoolList = nn.ModuleList([])
        self.layNumber = len(dilations)
        self.conv1x1 = DilatedBlock._hdcBlock(planes*len(dilations), planes, dilation=1, name='conv1x1')
        for i in range(self.layNumber):
            dilation = dilations[i]
            self.dilatedList.append(DilatedBlock._hdcBlock(inplanes, 
                                                           planes, 
                                                           dilation=(dilation, dilation), 
                                                           name=name, 
                                                           kernelSize=kernelSize))
            maxPoolKernelSize = (dilations[-1]-dilations[i])*2+1
            if maxPoolKernelSize>1:
                self.maxPoolList.append(MaxPoolSame(maxPoolKernelSize, 1))

        # self.conv1x1 = nn.Conv2d(inplanes, planes, 1, padding="same")
        # self.relu = nn.ReLU(inplace=True)

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
        outputList = []
        for i in range(self.layNumber-1):
            output = self.dilatedList[i](x)
            if i < self.layNumber - 1:
                output = self.maxPoolList[i](output)
            outputList.append(output)
        outputList.append(self.dilatedList[-1](x))
        output = torch.cat(outputList, 1)
        output = self.conv1x1(output)
        #output = torch.cat((dilated1, dilated2, dilated5), 1)
        #output = torch.cat((dilated1, dilated2), 1)
        #output = self.relu(output)
        return output
    
class PyramidDilate(nn.Module):
    def __init__(self, inplanes, planes) -> None:
        super(PyramidDilate, self).__init__()
        planes1 = inplanes
        planes2 = 16
        planes3 = 32
        planes4 = 64
        planes5 = 128
        planes6 = 256
        #planes7 = 512
        #planes8 = 256
        planes9 = 128
        planes10 = 64
        planes11 = 32
        planes12 = 16

        dilations1 = [1]
        self.dilate1 = DilatedBlock(planes1, planes=planes2, name="hdc1", dilations=dilations1)

        dilations2 = [1, 2]
        #self.dilate2 = DilatedBlock(planes2*len(dilations1), planes=planes3, name="hdc2", dilations=dilations2)
        self.dilate2 = DilatedBlock(planes2, planes=planes3, name="hdc2", dilations=dilations2)
        
        dilations3 = [1, 2, 5, 9]
        #self.dilate3 = DilatedBlock(planes3*len(dilations2), planes=planes4, name="hdc3", dilations=dilations3)
        self.dilate3 = DilatedBlock(planes3, planes=planes4, name="hdc3", dilations=dilations3)
        
        dilations4 = [1, 2, 5, 9]
        #self.dilate4 = DilatedBlock(planes4*len(dilations3), planes=planes5, name="hdc4", dilations=dilations4)
        self.dilate4 = DilatedBlock(planes4, planes=planes5, name="hdc4", dilations=dilations4)
        self.epmodule1 = EPModule(planes5, planes5//2)
        
        dilations5 = [1, 2, 5, 9]
        #self.dilate5 = DilatedBlock(planes5*len(dilations4), planes=planes6, name="hdc5", dilations=dilations5)
        self.dilate5 = DilatedBlock(planes5, planes=planes6, name="hdc5", dilations=dilations5)
        self.epmodule2 = EPModule(planes6, planes6//2)
        
        #dilations6 = [1, 2, 5, 9]
        #self.dilate6 = DilatedBlock(planes6*len(dilations5), planes=planes7, name="hdc6", dilations=dilations6)
        #self.dilate6 = DilatedBlock(planes6, planes=planes7, name="hdc6", dilations=dilations6)

        #dilations7 = [1, 2, 5, 9]
        #self.dilate7 = DilatedBlock(planes7*len(dilations6), planes=planes8, name="hdc7", dilations=dilations7)
        #self.dilate7 = DilatedBlock(planes7, planes=planes8//2, name="hdc7", dilations=dilations7)
        
        dilations8 = [1, 2, 5, 9]
        #self.dilate8 = DilatedBlock(planes8*len(dilations7), planes=planes9, name="hdc8", dilations=dilations8)
        self.dilate8 = DilatedBlock(planes6, planes=planes9, name="hdc8", dilations=dilations8)
        
        dilations9 = [1, 2, 5, 9]
        #self.dilate9 = DilatedBlock(planes9*len(dilations8), planes=planes10, name="hdc9", dilations=dilations9)
        self.dilate9 = DilatedBlock(planes9, planes=planes10, name="hdc9", dilations=dilations9)
        
        dilations10 = [1, 2, 5, 9]
        #self.dilate10 = DilatedBlock(planes10*len(dilations9), planes=planes11, name="hdc10", dilations=dilations10)
        self.dilate10 = DilatedBlock(planes10, planes=planes11, name="hdc10", dilations=dilations10)
        
        dilations11 = [1, 2, 5, 9]
        #self.dilate11 = DilatedBlock(planes11*len(dilations10), planes=planes12, name="hdc11", dilations=dilations11)
        self.dilate11 = DilatedBlock(planes11, planes=planes12, name="hdc11", dilations=dilations11)

        if planes==1:
            self.conv19 =  nn.Sequential(
                nn.Conv2d(planes12, planes, 1, padding='same'),
                nn.Sigmoid())
        else:
            self.conv19 =  nn.Conv2d(planes12, planes, 1, padding='same')
        
    def forward(self, x):
        dilated1 = self.dilate1(x)
        dilated2 = self.dilate2(dilated1)
        dilated3 = self.dilate3(dilated2)
        dilated4 = self.dilate4(dilated3)
        #ep1 = self.epmodule1(dilated4)
        dilated5 = self.dilate5(dilated4)
        #ep2 = self.epmodule2(dilated5)
        #dilated6 = self.dilate6(dilated5)
        #dilated7 = self.dilate7(dilated6)
        #dilated7 = torch.concat([dilated7, ep2],1)
        dilated8 = self.dilate8(dilated5)
        #dilated8 = torch.concat([dilated8, ep1],1)
        dilated9 = self.dilate9(dilated8)
        dilated10 = self.dilate10(dilated9)
        dilated11 = self.dilate11(dilated10)
        output = self.conv19(dilated11)
        return output