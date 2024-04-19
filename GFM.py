import torch
from torch import nn

class GatedFusion(nn.Module):
    """
    Gated Fusion Module
    """
    def __init__(self, input_dim:int):
        super(GatedFusion, self).__init__()
        
        self.conv1 = nn.Conv2d(input_dim,input_dim,1)
        self.conv2 = nn.Conv2d(input_dim,input_dim,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, feat1, feat2):

        B, C, H, W = feat1.size()
       
        ft1 = self.conv1(feat1)
        ft2 = self.conv2(feat2)
        
        gates = self.sigmoid(ft1 + ft2)
        
        output = gates * feat1 + (1 - gates) * feat2
        
        return output