import torch
import tqdm
import os
import numpy as np

from contextlib import redirect_stdout, redirect_stderr
from datasets import load_dataset

class Dataset:

    def __init__(self, config, languages, elements=100):
        # TODO: handle multiple languages
        with open(os.devnull, 'w') as f, redirect_stderr(f):
            dataset = load_dataset('uonlp/CulturaX', languages[0],
                                   split='train', streaming=True)
  
            dataset = dataset.take(elements)

            self.text = ''
            for data in dataset:
                self.text += '\n' + data['text']

        self.corpus = config.encoding.encode(self.text)
        self.corpus = torch.tensor(self.corpus)

        self.rng = np.random.default_rng(0)
        self.device = 'cuda:0'
        self.context = config.context

    def generate_batch(self, batch_size):
        ix = self.rng.integers(len(self.corpus) - (self.context + 1), size=batch_size)
        X = torch.stack([ self.corpus[i : i + self.context] for i in ix ], dim=0)
        y = torch.stack([ self.corpus[i + 1 : i + self.context + 1] for i in ix ], dim=0)
        return X.to(self.device), y.to(self.device)

    def evaluate_perplexity(self, model):
        tokens = self.corpus.to(self.device).unsqueeze(0)

        model.eval()
        total_loss = 0
        total_tokens = 0

        for i in tqdm.trange(0, len(tokens[0]) - self.context, self.context,
                             desc='\tEvaluating perplexity', ncols=100):
            X = tokens[:, i : i + self.context]
            y = tokens[:, i + 1 : i  + 1 + self.context]

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = model(X, y)

            total_loss += loss.item() * X.size(1)
            total_tokens += X.size(1)

        average_loss = total_loss / total_tokens

        return torch.exp(torch.tensor(average_loss)).item()