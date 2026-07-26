import torch
import torch.nn as nn
import math

from encoder_component.input_embedding import InputEmbeddings
from encoder_component.positional_encoding import PositionalEncoding
from encoder_component.layer_normalization import LayerNormalization
from encoder_component.Multihead_attention import MultiHeadAttentionBlock
from encoder_component.feed_forward_network import FeedForwardBlock
from encoder_component.encoder_block import EncoderBlock
from decoder_component.decoder_block import DecoderBlock


class Encoder(nn.Module):
    
    def __init__(self,layers: nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()
        
    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()
    
    def forward(self,x,ecoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,ecoder_output,src_mask,tgt_mask)
        return self.norm(x)
        
class ProjectionLayer(nn.Module):
    
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.proj=nn.Linear(d_model,vocab_size)
        
    def forward(self,x):
        # (Batch,Seq_len,d_model) --> (Batch,Seq_len,Vocab_size)
        return torch.log_softmax(self.proj(x),dim=-1)
    

class Transformer(nn.Module):
    
    def __init__(self,encoder:Encoder,decoder:Decoder,src_embed:InputEmbeddings,tgt_embed:InputEmbeddings,src_postion:PositionalEncoding,trg_postion:PositionalEncoding,projection_layer:ProjectionLayer):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=src_embed
        self.tgt_embed=tgt_embed
        self.src_postion=src_postion
        self.tgt_postion=trg_postion
        self.projection_layer=projection_layer
    
    def encode(self,src,src_mask):
        src=self.src_embed(src)
        src=self.src_postion(src)
        return self.encoder(src,src_mask)
    
    def decode(self,encoder_output,src_mask,tgt,tgt_mask):
        tgt=self.tgt_embed(tgt)
        tgt=self.tgt_postion(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
    
    def project(self,x):
        return self.projection_layer(x)
    

def build_transformer(src_vocab_size:int,tgt_vocab_size:int,src_seq_len:int,tgt_seq_len:int,d_model:int=512,N:int=6,h:int=8,droupout:float=0.1,d_ff:int=2048)->Transformer:
    # create the embedding layers
    src_embed=InputEmbeddings(d_model=d_model,vocab_size=src_vocab_size)
    tgt_embed=InputEmbeddings(d_model=d_model,vocab_size=tgt_vocab_size)
    
    # create the postional encoding layers
    src_pos=PositionalEncoding(d_model=d_model,seq_len=src_seq_len,dropout=droupout)
    tgt_pos=PositionalEncoding(d_model=d_model,seq_len=src_seq_len,dropout=droupout)
    
    # create the encoder blocks
    encoder_blocks=[]
    for _ in range(N):
        encoder_self_attetion_block=MultiHeadAttentionBlock(d_model=d_model,h=h,droupout=droupout)
        feed_forward_block=FeedForwardBlock(d_model=d_model,d_ff=d_ff,dropout=droupout)
        encoder_block=EncoderBlock(self_attention_block=encoder_self_attetion_block,feed_forward_block=feed_forward_block,dropout=droupout)
        encoder_blocks.append(encoder_block)
        
    # create the decoder blocks
    decoder_blocks=[]
    for _ in range(N):
        decoder_self_attention_block=MultiHeadAttentionBlock(d_model=d_model,h=h,droupout=droupout)
        decoder_cross_attention_block=MultiHeadAttentionBlock(d_model=d_model,h=h,droupout=droupout)
        decoder_forward_block=FeedForwardBlock(d_model=d_model,d_ff=d_ff,dropout=droupout)
        decoder_block=DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block,decoder_forward_block,droupout)
        decoder_blocks.append(decoder_block)
        
    # create the encoder and decoder
    ecoder=Encoder(nn.ModuleList(encoder_blocks))
    decoder=Decoder(nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer=ProjectionLayer(d_model=d_model,vocab_size=tgt_vocab_size)
    
    # Create the Transformer
    trasformer=Transformer(encoder=ecoder,decoder=decoder,src_embed=src_embed,tgt_embed=tgt_embed,src_postion=src_pos,trg_postion=tgt_pos,projection_layer=projection_layer)
    
    # Initialize the parameters
    for p in trasformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return trasformer
        