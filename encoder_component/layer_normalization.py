import torch
import torch.nn as nn
import math


class LayerNormalization(nn.Module):
    
    def __init__(self , eps : float = 1e-6) :
        super().__init__()
        
        self.eps=eps
        
        # Learnable scale parameter (γ)
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        
        # Learnable shift parameter (β)
        self.bias= nn.Parameter(torch.zeros(1)) # Added
        
    def forward(self,x):
        # Mean of each token's embedding
        mean=x.mean(dim=-1,keepdim=True)
        
        # Compute the variance of each token's embedding
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        
         # Normalize, then scale and shift
        return self.alpha * (x - mean) / torch.sqrt(variance + self.eps) + self.bias