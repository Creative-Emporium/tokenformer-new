import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

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
    parameters: int = 512
    
class TokenformerPAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.K = nn.Parameter(torch.randn(config.parameters, config.embedding))
        self.V = nn.Parameter(torch.randn(config.parameters, config.embedding))
        # TODO: multi-head attention
        self.embedding = config.embedding
    
    def forward(self, x):
        product = torch.matmul(x, self.K.T) * (self.embedding ** -0.5)
        product = F.softmax(product, dim=-1)
        # TODO: modified softmax
        return torch.matmul(product, self.V)

class TokenFormerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embedding)
        self.ln2 = nn.LayerNorm(config.embedding)
        
        self.qattn = TokenformerPAttention(config)
        self.kattn = TokenformerPAttention(config)
        self.vattn = TokenformerPAttention(config)
        
        self.mlp = TokenformerPAttention(config)
        self.ffn = TokenformerPAttention(config)

    def forward(self, x):
        x_in = x
        
        x = self.ln1(x)
        
        q = self.qattn(x)
        k = self.kattn(x)
        v = self.vattn(x)
        
        # Token-token attention
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = self.mlp(x)
        
        x = x + x_in
        x_inter = x
        
        x = self.ln2(x)
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
