import torch
import torch.nn as nn

from model_components.ema import EMA

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch,use_atrous=False,dilation=2):
        super(conv_block, self).__init__()
        if not use_atrous:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=dilation,dilation=dilation,bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=dilation,dilation=dilation,bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x
    

class LTE_BLock(nn.Module):
    """
    LTE Block 
    """
    def __init__(self,in_ch:int,out_ch:int,pooling=True) -> None:
        # print(f'in:{in_ch},out:{out_ch}')
        super(LTE_BLock,self).__init__()
        self.conv = conv_block(in_ch=in_ch,out_ch=out_ch)
        self.pooling = pooling
        if self.pooling:
            self.pool = nn.AvgPool2d(kernel_size=2,stride=2)
        self.attn = EMA(out_ch,factor=8)
    
    def forward(self, x):
        x = self.conv(x)
        if self.pooling:
            x = self.pool(x)
        x = self.attn(x)
        return x


class LTEN(nn.Module):
    """
    Local Terrain Feature Enhancement Network (LTEN)
    """
    def __init__(self,in_ch=3) -> None:
        super(LTEN,self).__init__()
        n1 = 8
        filters = [n1 * (2**i) for i in range(1,6)]

        self.blk1 = LTE_BLock(in_ch=in_ch,out_ch=filters[0])
        self.blk2 = LTE_BLock(in_ch=filters[0],out_ch=filters[1])
        self.blk3 = LTE_BLock(in_ch=filters[1],out_ch=filters[2])
        self.blk4 = LTE_BLock(in_ch=filters[2],out_ch=filters[3])
        self.blk5 = LTE_BLock(in_ch=filters[3],out_ch=filters[4],pooling=False)

    def forward(self,x):
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = self.blk5(x)

        return x