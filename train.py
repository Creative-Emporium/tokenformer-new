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

assert 'model' in config

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

dataset = Dataset(model_config, config['languages'])

training = config['training']
steps = training['steps'] * training['batch'] // args.batch + 1

opt = torch.optim.AdamW(model.parameters(), lr=5e-4)

losses = []
perplexities = []

t_train = time.time()
for epoch in range(training['epochs']):
    t_begin = time.time()

    X, y = dataset.generate_batch(args.batch)

    header(f'EPOCH #{epoch + 1}')
    
    bar = tqdm.tqdm(range(steps),
                    desc='\tTraining model',
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
        
        losses.append(loss.item())

    bar.close()

    t_end = time.time()

    # Evaluate the model
    with torch.no_grad():
        perplexity = dataset.evaluate_perplexity(model)
        print(f'\n\tPerplexity = {perplexity}')
        print(f'\tDelta = {(t_end - t_begin) / 60 :.0f} min, Progress = {(t_end - t_train) / 60 :.0f} min')
        print(f'\tNumber of parameters = {parameter_count(model)}')
        generate_samples(model, dataset, model_config.encoding)
        perplexities.append(perplexity)
    
    if epoch > 0 and 'grow' in config and  (epoch + 1) % config['grow']['frequency'] == 0:
        amount = config['grow']['amount']
        print(f'\tGrowing model by {amount} parameters')
        model.grow_parameters(amount)

# Write data
os.makedirs('data', exist_ok=True)

basename = choice.replace('/', '-')
basename = os.path.join('data', basename)

np.savez(basename + '-metrics.npz',
         loss=np.array(losses),
         perplexity=np.array(perplexities))

torch.save(model, basename + '-model.tch')