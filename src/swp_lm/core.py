"""Self-contained SWP-LM components for reproducible V13 student experiments."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

def make_freq_bias(counts,prior_mix=0.5):
 counts=torch.as_tensor(counts,dtype=torch.float32)
 V=len(counts)
 if counts.sum()<=0:return torch.zeros(V)
 return ((1-prior_mix)/V+prior_mix*(counts/counts.sum())).clamp_min(1e-12).log()

def decode_pq(centers,codes):
 blocks=[]
 for g in range(8):
  blocks.append(centers[0,g].index_select(0,codes[:,0,g])+centers[1,g].index_select(0,codes[:,1,g]))
 return torch.cat(blocks,-1).contiguous()

class StudentLM(nn.Module):
 def __init__(self, mode, input_table, centers, codes, output_table, bias, lowrank_init=None):
  super().__init__();self.mode=mode
  self.register_buffer('input_table',input_table)
  self.register_buffer('bias',bias)
  self.register_buffer('codes',codes.to(torch.long))
  self.rnn=nn.GRU(64,128,batch_first=True)
  self.head=nn.Linear(128,64)
  if mode in ('pq_learn','pq_frozen'):
   self.centers=nn.Parameter(centers.detach().clone(),requires_grad=(mode=='pq_learn'))
   if mode=='pq_frozen':self.register_buffer('output_table',output_table.contiguous())
  elif mode in ('rank16_learn','rank16_frozen'):
   assert lowrank_init is not None
   self.table=nn.Parameter(lowrank_init.contiguous().clone(),requires_grad=(mode=='rank16_learn'))
   self.rank_head=nn.Linear(64,16,bias=False)
  else:raise ValueError(mode)
 def out_table(self):
  if self.mode=='pq_frozen':return self.output_table
  if self.mode=='pq_learn':
   groups=[]
   for g in range(8):
    blk=F.embedding(self.codes[:,0,g],self.centers[0,g])+F.embedding(self.codes[:,1,g],self.centers[1,g])
    groups.append(blk)
   return torch.cat(groups,dim=-1)
  return None
 def forward(self,ids):
  h,_=self.rnn(F.embedding(ids,self.input_table))
  c=self.head(h)
  if self.mode.startswith('pq_'):
   return F.linear(c,self.out_table(),self.bias)
  return F.linear(self.rank_head(c),self.table,self.bias)
