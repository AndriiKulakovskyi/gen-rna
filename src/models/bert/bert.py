import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.bert.cgf import BERTConfig


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class MultiheadAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        assert dim % n_heads == 0, 'dim should be div by n_heads'
        self.head_dim = self.dim // self.n_heads
        self.in_proj = nn.Linear(dim,dim*3,bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        self.out_proj = nn.Linear(dim,dim)
        
    def forward(self,x,mask=None):
        b,t,c = x.shape
        q,k,v = self.in_proj(x).chunk(3,dim=-1)
        q = q.view(b,t,self.n_heads,self.head_dim).permute(0,2,1,3)
        k = k.view(b,t,self.n_heads,self.head_dim).permute(0,2,1,3)
        v = v.view(b,t,self.n_heads,self.head_dim).permute(0,2,1,3)
        
        qkT = torch.matmul(q,k.transpose(-1,-2)) * self.scale
        qkT = self.attn_dropout(qkT)
        
        if mask is not None:
            mask = mask.to(dtype=qkT.dtype,device=qkT.device)
            qkT = qkT.masked_fill(mask==0,float('-inf'))
              
        qkT = F.softmax(qkT,dim=-1)
        attn = torch.matmul(qkT,v)
        attn = attn.permute(0,2,1,3).contiguous().view(b,t,c)
        out = self.out_proj(attn)
        
        return out


class FeedForward(nn.Module):
    def __init__(self,dim,dropout=0.):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(dim,dim*4),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim*4,dim)
        )
        
    def forward(self, x):
        return self.feed_forward(x)


class EncoderBlock(nn.Module):
    def __init__(self, dim, n_heads, attn_dropout=0., mlp_dropout=0.):
        super().__init__()
        self.attn = MultiheadAttention(dim,n_heads,attn_dropout)
        self.ffd = FeedForward(dim,mlp_dropout)
        self.ln_1 = RMSNorm(dim)
        self.ln_2 = RMSNorm(dim)
        
    def forward(self,x,mask=None):
        x = self.ln_1(x)
        x = x + self.attn(x,mask)
        x = self.ln_2(x)
        x = x + self.ffd(x)
        return x


class Embedding(nn.Module):
    def __init__(self,vocab_size,max_len,dim):
        super().__init__()
        self.max_len = max_len
        self.class_embedding = nn.Embedding(vocab_size,dim)
        self.pos_embedding = nn.Embedding(max_len,dim)
    def forward(self,x):
        x = self.class_embedding(x)
        pos = torch.arange(0,x.size(1),device=x.device)
        x = x + self.pos_embedding(pos)
        return x


class BERT(nn.Module):
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.depth = config.depth
        self.dim = config.dim
        
        # Token embeddings and positional embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.pos_embedding = nn.Embedding(config.max_len, config.dim)
        
        # Transformer encoder blocks
        self.encoders = nn.ModuleList([
            EncoderBlock(dim=config.dim,
                         n_heads=config.n_heads,
                         attn_dropout=config.attn_dropout,
                         mlp_dropout=config.mlp_dropout)
            for _ in range(self.depth)
        ])
        
        # Final layer normalization
        self.ln_f = RMSNorm(config.dim)
        
        # MLM head for masked token prediction
        self.mlm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.token_embedding.weight = self.mlm_head.weight  # Weight tying

        # Special token IDs
        self.pad_token_id = config.pad_token_id
        self.mask_token_id = config.mask_token_id

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def create_src_mask(self, src):
        # Create a mask for padding tokens
        return (src != self.pad_token_id).unsqueeze(1).unsqueeze(2)

    def forward(self, input_ids, labels=None):
        # Positional encoding
        pos = torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0)
        embeddings = self.token_embedding(input_ids) + self.pos_embedding(pos)
        
        # Create source mask
        src_mask = self.create_src_mask(input_ids)
        
        # Pass through encoder layers
        enc_out = embeddings
        for layer in self.encoders:
            enc_out = layer(enc_out, mask=src_mask)
        
        # Final normalization
        enc_out = self.ln_f(enc_out)
        
        # Predict masked tokens
        logits = self.mlm_head(enc_out)

        if labels is not None:
            # Compute loss for training
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            return {'loss': loss, 'logits': logits}
        else:
            # Predict for all `[MASK]` tokens
            mask_indices = (input_ids == self.mask_token_id).nonzero(as_tuple=True)
            mask_logits = logits[mask_indices]
            return {'logits': logits, 'mask_predictions': mask_logits.argmax(dim=-1)}
