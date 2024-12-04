import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme()

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'sans-serif',
    'font.sans-serif': 'Biolinum',
})

# Baseline comparison -- Transformer vs. Tokenformer
transformer_metrics = np.load('data/plain-transformer-gpt2-metrics.npz')
tokenformer_metrics = np.load('data/plain-tokenformer-gpt2-metrics.npz')

transformer_perplexities = transformer_metrics['perplexity']
tokenformer_perplexities = tokenformer_metrics['perplexity']

plt.figure(figsize=(10, 8), layout='constrained')
plt.yscale('log')
plt.plot(transformer_perplexities, label='Transformer')
plt.plot(tokenformer_perplexities, label='TokenFormer')
plt.legend()
plt.show()

# Growth -- 64 per 50 and 128 per 100
growth_64_metrics = np.load('data/growth-grow-64-metrics.npz')
growth_128_metrics = np.load('data/growth-grow-128-metrics.npz')

growth_64_perplexities = growth_64_metrics['perplexity']
growth_128_perplexities = growth_128_metrics['perplexity']

plt.figure(figsize=(10, 8), layout='constrained')
plt.yscale('log')
plt.plot(growth_64_perplexities, label='Grow 64 every 50')
plt.plot(growth_128_perplexities, label='Grow 128 every 100')
plt.plot(tokenformer_perplexities, label='Baseline (TokenFormer)')
plt.legend()
plt.show()