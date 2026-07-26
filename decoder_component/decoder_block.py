import torch
import torch.nn as nn
import math

from encoder_component.Multihead_attention import MultiHeadAttentionBlock
from encoder_component.feed_forward_network import FeedForwardBlock
from encoder_component.residual_connection import ResidualConnection

class DecoderBlock(nn.Module):
    
    def __init__(self,self_attention_block:MultiHeadAttentionBlock , cross_attentio_block:MultiHeadAttentionBlock , feed_forward_block:FeedForwardBlock ,droupout:float):
        super().__init__()
        self.self_attention_block=self_attention_block
        self.cross_attentio_block=cross_attentio_block
        self.feed_forward_block=feed_forward_block
        self.droupout=droupout
        self.residual_connection=nn.ModuleList([ResidualConnection(droupot=droupout) for _ in range(3)])
    
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x=self.residual_connection[0](x,lambda x : self.self_attention_block(x,x,x,tgt_mask))
        x=self.residual_connection[1](x,lambda x:self.cross_attentio_block(x,encoder_output,encoder_output,src_mask))
        x=self.residual_connection[2](x,self.feed_forward_block)
        return x