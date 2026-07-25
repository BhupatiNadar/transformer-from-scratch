import torch
import torch.nn as nn
import math

from layer_normalization import LayerNormalization

class ResidualConnection(nn.Module):
    
    def __init__(self,droupot:float):
        super().__init__()
        self.droupout=droupot
        self.norm=LayerNormalization()
        
    def forward(self,x,sublayer):
        return x + self.droupout(sublayer(self.norm(x)))