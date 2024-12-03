import torch
import torch.nn as nn
from collections import OrderedDict
from torchsummary import summary


class DilatedBlock(nn.Module):
    def __init__(self, inplanes, planes, dilation_rates, layer_num, name, kernelSize = 3) -> None:
        assert len(dilation_rates)==layer_num, "The number of dilation rates should be equal to layer number"

        super(DilatedBlock, self).__init__()
        self.dilatedList = nn.ModuleList([DilatedBlock._hdcBlock(inplanes, planes, dilation_rates[i], name, kernelSize) for i in range(layer_num)])

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
        layer_output = [f(x) for f in self.dilatedList]
        output = torch.cat(layer_output, 1)
        return output

class DilatedNet(nn.Module):
    def __init__(self, inplanes, planes, block_num = 10, dilation_rates= [1,2,5,9], layer_num = 4, stride=1) -> None:
        assert len(dilation_rates)==layer_num, "The number of dilation rates should be equal to layer number"
        
        super(DilatedNet, self).__init__()
       
        ini_channel = 8
        planes_down_channel = [ini_channel*2**(i) for i in range(block_num//2)]
        #planes_down_channel[0] = inplanes
        ini_channel = planes_down_channel[-1]
        planes_up_channel = [ini_channel//2**(i) for i in range(block_num//2+1)]

        #assert False, str(ini_channel)+str(planes_down_channel)+str(planes_up_channel)
        self.downblock = nn.ModuleList([DilatedBlock(inplanes=planes_down_channel[i-1]*layer_num if i>0 else inplanes, 
                                                     planes=planes_down_channel[i],
                                                     dilation_rates = dilation_rates, 
                                                     layer_num = layer_num,
                                                     name='hdc_down'+str(i)) for i in range(block_num//2)])
        self.upblock = nn.ModuleList([DilatedBlock( inplanes=planes_up_channel[i]*layer_num,
                                                    planes=planes_up_channel[i+1], 
                                                    dilation_rates = dilation_rates, 
                                                    layer_num = layer_num,
                                                    name='hdc_up'+str(i)) for i in range(block_num//2)])
        
        self.conv19 =  nn.Sequential(
            nn.Conv2d(planes_up_channel[-1]*layer_num, planes, 1, padding='same'),
            nn.Sigmoid())

    def forward(self, x):
        for d in self.downblock:
            x = d(x)
        for u in self.upblock:
            x = u(x)
        out = self.conv19(x)

        return out

if __name__=='__main__':
    net = DilatedNet(4,1, block_num = 6, dilation_rates= [1,2,3], layer_num = 3)
    summary(net.cuda(), (4, 256, 256))
    #print(list(net.parameters()))