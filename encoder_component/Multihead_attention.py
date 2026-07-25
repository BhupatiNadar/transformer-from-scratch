import torch
import torch.nn as nn
import math

class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model:int , h: int , droupout:float ):
        super().__init__()
        self.d_model=d_model
        self.h=h
        self.droupout=droupout
        assert d_model % h == 0, "d_model is not divisible by h"
        
        self.d_k=d_model // h
        
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        
        self.w_o=nn.Linear(d_model,d_model)
        self.droupout=nn.Dropout(droupout)
        
    
    @staticmethod    
    def attention(query,key,value,mask,dropout : nn.Dropout ):
        d_k = query.shape[-1]
        
        # (Batch , h ,Seq_Len ,d_k) --> (Batch , h, Seq_len ,Seq_len)
        attention_score=(query @ key.transpose(-2,-1)/math.sqrt(d_k))
        
        if mask is not None:
            attention_score.masked_fill_(mask == 0 ,-1e9)
        
        attention_score=attention_score.softmax(dim=-1) # (Batch,h,seq_len,seq_len)
        
        if dropout is not None:
            attention_score=dropout(attention_score)
        
        return (attention_score @ value) , attention_score
        
        
    def forward(self,q,k,v,mask):
        query=self.w_q(q) # (Batch , Seq_len , d_model ) --> (Batch, Seq_Len , d_model )
        key=self.w_k(k)  # (Batch , Seq_len , d_model ) --> (Batch, Seq_Len , d_model )
        value=self.w_v(v)  # (Batch , Seq_len , d_model ) --> (Batch, Seq_Len , d_model )
        
        # (Batch,Seq_len,d_model) --> (Batch,Seq_len,h,d_k) --> (Batch,h,Seq_len,d_k)
        query=query.view(query.shape[0],query.shape[1],self.h, self.d_k).transpose(1,2)
        key=key.view(key.shape[0],key.shape[1],self.h, self.d_k).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],self.h, self.d_k).transpose(1,2)
        
        x,self.attention_score=MultiHeadAttentionBlock.attention(query,key,value,mask,self.droupout)
        
        # (Batch , h ,Seq_len ,d_k) --> (Batch , Seq_Len , h, d_k) --> (Batch , Seq_Len , d_model)
        x=x.transpose(1,2).contiguous().view(x.shape[0], x.shape[2], self.h * self.d_k)
        
        #(Batch,Seq_Len,d_model) -- > (Batch ,Seq_Len ,d_model)
        return self.w_o(x)

        