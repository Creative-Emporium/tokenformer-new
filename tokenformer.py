import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import math

from typing import Any
from dataclasses import dataclass

@dataclass
class TokenFormerConfiguration:
    encoding: Any = tiktoken.get_encoding('gpt2')
    heads: int = 12
    context: int = 1024
    embedding: int = 768
    vocabulary: int = encoding.n_vocab
    layers: int = 12
    parameters: int = 1024
   
class NonParametricNormalization(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.channels = dim
        self.eps = eps

    def forward(self, x):
        x = F.layer_norm(x, normalized_shape=(self.channels,), eps=self.eps)
        return x
    
class TokenformerPAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.K = nn.Parameter(torch.randn(config.parameters, config.embedding))
        self.V = nn.Parameter(torch.randn(config.parameters, config.embedding))
        self.embedding = config.embedding
        self.heads = config.heads
    
    # TODO: modified softmax?
    def forward(self, x):
        scale = self.embedding ** (-0.5)
        x = scale * x @ self.K.T
        x = F.softmax(x, dim=-1)
        x = x @ self.V
        return x
    
    def nonlinear_normalization(self, inputs, normalize_type, dim=-1):
        if normalize_type == 'softmax':
            nonlinear_outputs = torch.exp(inputs)
            norm_outputs = nonlinear_outputs / torch.norm(nonlinear_outputs, p=1, dim=dim, keepdim=True) * inputs.shape[dim]
            outputs = norm_outputs
        elif normalize_type == 'gelu_l2_norm':
            nonlinear_outputs = F.gelu(inputs)
            norm_outputs = nonlinear_outputs / torch.norm(nonlinear_outputs, p=2, dim=dim, keepdim=True) * math.sqrt(nonlinear_outputs.shape[dim])
            outputs = norm_outputs
        elif normalize_type == 'l2_norm_gelu':
            norm_outputs = inputs / torch.norm(inputs, p=2, dim=dim, keepdim=True) * math.sqrt(inputs.shape[dim])
            nonlinear_outputs = F.gelu(norm_outputs)
            outputs = nonlinear_outputs
        else:
            raise NotImplementedError
        
        return outputs

class TokenFormerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.normalization = NonParametricNormalization(config.embedding)
        
        self.qattn = TokenformerPAttention(config)
        self.kattn = TokenformerPAttention(config)
        self.vattn = TokenformerPAttention(config)
        
        self.proj = TokenformerPAttention(config)
        self.ffn = TokenformerPAttention(config)
        
        self.heads = config.heads
        
    def forward(self, x):
        x_in = x
        
        x = self.normalization(x)
        
        k = self.kattn(x)
        q = self.qattn(x)
        v = self.vattn(x)
        
        # Multi-head token-token attention
        B, T, C = k.shape
        k = k.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        q = q.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        x = self.proj(x)
        
        x = x + x_in
        x_inter = x
        
        x = self.normalization(x)
        x = self.ffn(x)
        
        return x + x_inter

class TokenFormerModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.wte = nn.Embedding(config.vocabulary, config.embedding)
        self.wpe = nn.Embedding(config.context, config.embedding)

        blocks = [ TokenFormerBlock(config) for i in range(config.layers) ]
        blocks += [
            nn.LayerNorm(config.embedding),
            nn.Linear(config.embedding, config.vocabulary)
        ]
        
        self.final = nn.Sequential(*blocks)
        self.context = config.context

    def forward(self, indices, targets=None):
        T = indices.size(1)

        positions = torch.arange(0, T, dtype=torch.long, device=indices.device)

        temb = self.wte(indices)
        pemb = self.wpe(positions)

        x = temb + pemb
        logits = self.final(x)

        loss = None
        if not targets is None:
            B, T, C = logits.shape
            lv = logits.view(B * T, C)
            tv = targets.view(B * T)
            loss = F.cross_entropy(lv, tv)

        return logits, loss

    def generate(self, tokens, N):
        for _ in range(N):
            logits, _ = self(tokens[:, -self.context:])
            logits = logits[:, -1, :]
            probabilities = F.softmax(logits, dim=-1)

            probabilities, indices = torch.topk(probabilities, 10, dim=-1)
            sampled = torch.multinomial(probabilities, 1)
            generated = torch.gather(indices, -1, sampled)

            tokens = torch.cat((tokens, generated), dim=1)

        return tokens
