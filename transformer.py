import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

from typing import Any
from dataclasses import dataclass

@dataclass
class TransformerConfiguration:
    encoding: Any = tiktoken.get_encoding('gpt2')
    heads: int = 12
    context: int = 1024
    embedding: int = 768
    vocabulary: int = encoding.n_vocab
    layers: int = 12

class TransformerSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.embedding % config.heads == 0
        self.attn = nn.Linear(config.embedding, 3 * config.embedding)
        self.proj = nn.Linear(config.embedding, config.embedding)
        self.embedding = config.embedding
        self.heads = config.heads

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.attn(x)
        q, k, v = qkv.split(self.embedding, dim=-1)
        k = k.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        q = q.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y

class TransformerFeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.proj1 = nn.Linear(config.embedding, 4 * config.embedding)
        self.gelu = nn.GELU(approximate='tanh')
        self.proj2 = nn.Linear(4 * config.embedding, config.embedding)

    def forward(self, x):
        x = self.proj1(x)
        x = self.gelu(x)
        x = self.proj2(x)
        return x

class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embedding)
        self.ln2 = nn.LayerNorm(config.embedding)
        self.attn = TransformerSelfAttention(config)
        self.mlp = TransformerFeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TransformerModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.wte = nn.Embedding(config.vocabulary, config.embedding)
        self.wpe = nn.Embedding(config.context, config.embedding)

        blocks = [ TransformerBlock(config) for i in range(config.layers) ]
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
