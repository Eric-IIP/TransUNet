import torch
import torch.nn as nn
from collections import OrderedDict

class EPModule(nn.Module):
    def __init__(self, inplanes, planes) -> None:
        super(EPModule, self).__init__()
        self.conv1 = EPModule._convBlock(inplanes, planes, 3, '3x3')
        self.conv2 = EPModule._convBlock(inplanes, planes, 1, '1x1_1')
        self.conv3 = EPModule._convBlock(planes, planes, 1, '1x1_2')
    
    @staticmethod
    def _convBlock(inplanes, features, kernel, name):
        return nn.Sequential(
            OrderedDict([
                    (name + 'conv', nn.Conv2d(inplanes, features, kernel_size=kernel, padding='same', bias=True)),
                    (name + 'batchnorm1', nn.BatchNorm2d(features)),
                    (name + 'relu1', nn.ReLU(inplace=True))
                ])
        )

    def forward(self,x):
        output1 = self.conv1(x)
        output2 = self.conv2(x)
        output2 = torch.sub(output1, output2)
        output1 = torch.add(output1, output2)
        output = self.conv3(output1)
        return output