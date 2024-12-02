import os
import time
import json
import numpy as np
import torch
import tqdm

from datasets import load_dataset
from dataclasses import dataclass

from transformer import TransformerConfiguration, TransformerModel
from tokenformer import TokenFormerConfiguration, TokenFormerModel

dataset = load_dataset('uonlp/CulturaX', 'en', split='train', streaming=True)
dataset = dataset.take(5)

corpus = ''
for data in dataset:
    corpus += '\n' + data['text']

def parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

# PyTorch configuration
torch.manual_seed(0)
torch.set_float32_matmul_precision('high')

# Model configuration

# TODO: argparse with langugae and config options
# TODO: perplexity

config = TokenFormerConfiguration()
model = TokenFormerModel(config)

# config = TransformerConfiguration()
# model = TransformerModel(config)

model = model.cuda()
torch.compile(model)

encoding = config.encoding

print('Corpus')
print('  vocab size', encoding.n_vocab)
print('  tokens', len(encoding.encode(corpus)))
print('  model parameters:', parameter_count(model))

# Training set
batch_size = 2
train = encoding.encode(corpus)
train = torch.tensor(train)

rng = np.random.default_rng(0)

# TODO: cycle batches...
def generate_batch():
    ix = rng.integers(len(train) - (config.context + 1), size=batch_size)
    X = torch.stack([ train[i : i + config.context] for i in ix ], dim=0)
    y = torch.stack([ train[i + 1 : i + config.context + 1] for i in ix ], dim=0)
    return X, y

batch_index = 0

def generate_next_batch():
    global batch_index
    
    stride = config.context + 1
    start = batch_index * batch_size * stride
    end = (batch_index + 1) * batch_size * stride
    batch_index += 1
    
    if end > len(train):
        batch_index = 0
        return generate_next_batch()
    
    X = torch.stack([ train[i : i + config.context] for i in range(start, end, stride) ], dim=0)
    y = torch.stack([ train[i + 1 : i + config.context + 1] for i in range(start, end, stride) ], dim=0)
    
    return X, y

opt = torch.optim.AdamW(model.parameters(), lr=5e-4)

def generate_samples(model):
    samples = [ 'Today the weather is quite', 'Here we stand, before our' ]

    for s in samples:
        encoded = [ encoding.encode(s) ]
        encoded = torch.tensor(encoded).cuda()
        generated = model.generate(encoded, 20)
        generated = generated[0].tolist()
        print('->', encoding.decode(generated))

for epoch in range(25):
    e0 = time.time()

    X, y = generate_batch()
    # X, y = generate_next_batch()

    X = X.cuda()
    y = y.cuda()

    print(f'epoch {epoch}')
    for i in range(100 + 1):
        t0 = time.time()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(X, y)

        opt.zero_grad()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        opt.step()

        t1 = time.time()

        if i % 10 == 0:
            throughput = X.numel() / (t1 - t0)
            print(f'\tstep {i :03d} | loss = {loss.item() :01.4f} | time = {1000 * (t1 - t0) :03.2f} ms | throughput = {throughput :03.2f} tokens/sec')

    e1 = time.time()

    print(f'\ttime = {(e1 - e0) / 60 :.0f} min')
    
    generate_samples(model)
    
# Perplexity evaluation
model.eval()  # Set model to evaluation mode
total_loss = 0
total_tokens = 0

with torch.no_grad():
    dataset = load_dataset('uonlp/CulturaX', 'en', split='train', streaming=True)
    dataset = dataset.take(5)
    # TODO: shuffle?
    
    corpus = ''
    for data in dataset:
        corpus += data['text']
    
    # TODO: measure perplexity retention?
    tokens = [ encoding.encode(corpus) ]
    tokens = torch.tensor(tokens).cuda()

    for i in tqdm.trange(0, len(tokens[0]) - config.context, config.context):
        X = tokens[:, i:i + config.context]
        y = tokens[:, i + 1:i + 1 + config.context]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(X, y)

        total_loss += loss.item() * X.size(1)
        total_tokens += X.size(1)

average_loss = total_loss / total_tokens
perplexity = torch.exp(torch.tensor(average_loss)).item()

print('average loss, perplexity', average_loss, perplexity)