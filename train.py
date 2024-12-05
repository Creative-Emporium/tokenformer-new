import os
import time
import torch
import tqdm
import argparse
import sys
import pprint
import numpy as np

from transformer import TransformerConfiguration, TransformerModel
from tokenformer import TokenFormerConfiguration, TokenFormerModel
from dataset import Dataset
from config import *
from util import *

# Run-time configuration
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--batch', type=int, default=8)
args = parser.parse_args(sys.argv[1:])

def header(text, width=50):
    print(width * '-')
    print(((width - len(text)) // 2) * ' ' + text)
    print(width * '-')

choice = args.config
if args.config is None:
    header('Select training configuration'.upper())
    for i, c in enumerate(configs):
        print(f'  [{i}] {c}')
        
    i = int(input('\n> '))
    choice = list(configs.keys())[i]
    print('')

header('Selected configuration'.upper())
config = configs[choice]
pprint.PrettyPrinter(indent=4).pprint(config)
print('')

# PyTorch configuration
torch.manual_seed(0)
torch.set_float32_matmul_precision('high')

# Model configuration
def edit_model_config(config, model_config):
    for k in config:
        if k == 'encoding':
            # For now this means nothing
            pass
        elif hasattr(model_config, k):
            x = config[k]
            if k in [ 'context', 'embedding', 'heads', 'layers', 'parameters' ]:
                x = int(x)
                
            setattr(model_config, k, x)
    
    if 'extension' in config:
        setattr(model_config, 'parameters', config['extension'])

if config['model'] == 'transformer':
    model_config = TransformerConfiguration()
    edit_model_config(config, model_config)
    
    model = TransformerModel(model_config)
elif config['model'] == 'tokenformer':
    model_config = TokenFormerConfiguration()
    edit_model_config(config, model_config)
    
    model = TokenFormerModel(model_config)
else:
    raise NotImplementedError(f'Invalid model \'{args.model}\'')

model = model.cuda()
torch.compile(model)

languages = config['languages']
datasets = { lang : Dataset(model_config, lang) for lang in languages }

training = config['training']
steps = training['steps'] * training['batch'] // args.batch + 1

opt = torch.optim.AdamW(model.parameters(), lr=5e-4)

# Training history
losses = []
perplexities = { lang: [] for lang in languages }

# Prepare language masks
masked = False
if config['model'] == 'tokenformer' \
        and 'masked' in config      \
        and config['masked']:
    assert 'extension' in config
    
    extension = config['extension']
    masked = True

# Training loop
t_train = time.time()

for index, lang in enumerate(datasets):
    dataset = datasets[lang]
    
    mask = None
    if masked:
        mask = torch.zeros((model.parameter_count, 1)).cuda()
        mask[index * extension:] = 1
    
    for epoch in range(training['epochs']):
        t_begin = time.time()

        X, y = dataset.generate_batch(args.batch)

        header(f'EPOCH #{epoch + 1} [{lang}]')
        
        bar = tqdm.tqdm(range(steps),
                        desc='\tTraining model',
                        ncols=100)

        for _ in bar:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, loss = model(X, y, mask=mask)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            # Relay statistics
            bar.set_postfix({
                'loss': f'{loss.item() :01.4f}',
            })
            
            losses.append(loss.item())

        bar.close()
        print('')

        t_end = time.time()

        # Evaluate the model
        with torch.no_grad():
            for i, l in enumerate(languages):
                mask = None
                if masked:
                    mask = torch.zeros((model.parameter_count, 1)).cuda()
                    mask[i * extension : (i + 1) * extension] = 1
                
                perplexity = datasets[l].evaluate_perplexity(model, mask=mask)
                perplexities[l].append(perplexity)
                
            print('')
            for l in languages:
                print(f'\tPerplexity [{l}] = {perplexities[l][-1]}')
                
            print(f'\n\tDelta = {(t_end - t_begin) :.0f} sec, Progress = {(t_end - t_train) / 60 :.0f} min')
            print(f'\tNumber of parameters = {parameter_count(model)}')
            generate_samples(model, dataset, model_config.encoding, mask=mask)
        
        if epoch > 0 and ('grow' in config)                             \
                and epoch + 1 < training['epochs']                      \
                and (epoch + 1) % config['grow']['frequency'] == 0:
            amount = config['grow']['amount']
            print(f'\n\t[!] Growing model by {amount} parameters')
            model.grow_parameters(amount)
            
            opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
            torch.compile(model)
    
    # Grow the model between learning new languages
    if config['model'] == 'tokenformer'     \
            and 'extension' in config       \
            and (index + 1) < len(datasets):
        amount = config['extension']
        print(f'\n\t[!] Growing model by {amount} parameters')
        model.grow_parameters(amount)
        
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
        torch.compile(model)

# Write data
os.makedirs('data', exist_ok=True)

basename = choice.replace('/', '-')
basename = os.path.join('data', basename)

data = { 'loss': np.array(losses) }
for lang in perplexities:
    key = f'perplexity[{lang}]'
    data[key] = np.array(perplexities[lang])
    
np.savez(basename + '-metrics.npz', **data)
torch.save(model, basename + '-model.tch')

# TODO: repetitions over all languages to fix the wpe, wtes and etc...