import os
import time
import json
import numpy as np
import torch
import tqdm
import argparse
import sys

from datasets import load_dataset
from dataclasses import dataclass

from transformer import TransformerConfiguration, TransformerModel
from tokenformer import TokenFormerConfiguration, TokenFormerModel

# Run-time configuration
# TODO: checkpoint and loss destinations
# TODO: ini files
parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, choices=[ 'transformer', 'tokenformer' ])
parser.add_argument('language', type=str, choices=[ 'en', 'fr', 'ru', 'sp' ])
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--steps_per_epoch', type=int, default=1000)
args = parser.parse_args(sys.argv[1:])

class Dataset:
    
    def __init__(self, encoder, language='en', elements=100):
        dataset = load_dataset('uonlp/CulturaX', language, split='train', streaming=True)
        dataset = dataset.take(elements)
        
        self.corpus = ''
        for data in dataset:
            self.corpus += '\n' + data['text']
            
        self.corpus = encoder.encode(self.corpus)
        self.corpus = torch.tensor(self.corpus)

        self.rng = np.random.default_rng(0)
        self.device = 'cuda:0'
        
    def generate_batch(self, batch_size):
        ix = self.rng.integers(len(self.corpus) - (config.context + 1), size=batch_size)
        X = torch.stack([ self.corpus[i : i + config.context] for i in ix ], dim=0)
        y = torch.stack([ self.corpus[i + 1 : i + config.context + 1] for i in ix ], dim=0)
        return X.to(self.device), y.to(self.device)
    
    def evaluate_perplexity(self, model):
        tokens = self.corpus.to(self.device).unsqueeze(0)

        model.eval()
        total_loss = 0
        total_tokens = 0

        for i in tqdm.trange(0, len(tokens[0]) - config.context, config.context,
                             desc='\tEvaluating model perplexity', ncols=100):
            X = tokens[:, i : i + config.context]
            y = tokens[:, i + 1 : i  + 1 + config.context]

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = model(X, y)

            total_loss += loss.item() * X.size(1)
            total_tokens += X.size(1)
            
        average_loss = total_loss / total_tokens
        
        return torch.exp(torch.tensor(average_loss)).item()

# TODO: util
def parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

# PyTorch configuration
torch.manual_seed(0)
torch.set_float32_matmul_precision('high')

# Model configuration
if args.model == 'transformer':
    config = TokenFormerConfiguration()
    model = TokenFormerModel(config)
elif args.model == 'tokenformer':
    config = TransformerConfiguration()
    model = TransformerModel(config)
else:
    raise NotImplementedError(f'Invalid model \'{args.model}\'')

model = model.cuda()
torch.compile(model)

dataset = Dataset(config.encoding)

print(50 * '-')
print('Configuration:')
print(50 * '-')
print('\tvocab size', config.encoding.n_vocab)
print('\ttokens', len(dataset.corpus))
print('\tmodel parameters:', parameter_count(model))

# TODO: args option
opt = torch.optim.AdamW(model.parameters(), lr=5e-4)

# TODO: util
def generate_samples(model, encoding):
    samples = [ 'Today the weather is quite', 'Here we stand, before our' ]
    
    print('\tText generation:')
    for s in samples:
        encoded = [ encoding.encode(s) ]
        encoded = torch.tensor(encoded).cuda()
        generated = model.generate(encoded, 20)
        generated = generated[0].tolist()
        string = encoding.decode(generated)
        print('\t->', string.replace('\n', '\n\t\t'))

t_train = time.time()

for epoch in range(args.epochs):
    t_begin = time.time()

    X, y = dataset.generate_batch(args.batch_size)

    print(50 * '-')
    print(f'Epoch #{epoch + 1}')
    print(50 * '-')
    
    bar = tqdm.tqdm(range(args.steps_per_epoch + 1),
                    desc='\tTraining model on batch',
                    ncols=100)
    
    for _ in bar:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(X, y)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        
        # Relay statistics
        bar.set_postfix({
            'loss': f'{loss.item() :01.4f}',
        })
        
    bar.close()

    t_end = time.time()
    
    print(f'\tDelta = {(t_end - t_begin) / 60 :.0f} min, Progress = {(t_end - t_train) / 60 :.0f} min')
    
    # Evaluate the model
    with torch.no_grad():
        perplexity = dataset.evaluate_perplexity(model)
        generate_samples(model, config.encoding)
        print(f'\tperplexity = {perplexity}')