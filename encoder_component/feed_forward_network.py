import torch
import torch.nn as nn
import math

class FeedForwardBlock(nn.Module):
    
    def __init__(self , d_model : int , d_ff : int , dropout : float ):
        super().__init__()
        self.Linear1=nn.Linear(d_model,d_ff) # W1 and B1
        self.dropout=nn.Dropout(dropout)
        self.Linear2=nn.Linear(d_ff,d_model) # w2 and b2
        
    
    def forward(self,x):
        # ( Batch , Seq_Len , d_model ) --> ( Batch , Seq_Len , d_ff ) --> ( Batch , Seq_Len , d_model )
        return self.Linear2(self.dropout(torch.relu(self.Linear1(x))))
    