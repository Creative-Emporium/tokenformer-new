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
plt.xlabel('Epochs')
plt.ylabel('Perplexity')
plt.plot(transformer_perplexities, label='Transformer')
plt.plot(tokenformer_perplexities, label='TokenFormer')
plt.legend()
plt.show()

# Growth -- 64 per 50 and 128 per 100
growth_64_metrics = np.load('data/growth-grow-64-metrics.npz')
growth_128_metrics = np.load('data/growth-grow-128-metrics.npz')

growth_non_reset_64_metrics = np.load('data/growth-grow-64-metrics-non-reset.npz')
growth_non_reset_128_metrics = np.load('data/growth-grow-128-metrics-non-reset.npz')

growth_64_perplexities = growth_64_metrics['perplexity[en]']
growth_128_perplexities = growth_128_metrics['perplexity[en]']

growth_non_reset_64_perplexities = growth_non_reset_64_metrics['perplexity']
growth_non_reset_128_perplexities = growth_non_reset_128_metrics['perplexity']

plt.figure(figsize=(10, 8), layout='constrained')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Perplexity')

# plt.plot(growth_non_reset_64_perplexities, label='Grow 64 every 50')
# plt.plot(growth_non_reset_128_perplexities, label='Grow 128 every 100')

plt.plot(growth_64_perplexities, label='Grow 64 every 50')
plt.plot(growth_128_perplexities, label='Grow 128 every 100')

plt.plot(tokenformer_perplexities, label='Baseline (TokenFormer)')

plt.legend()
plt.show()

# Continual learning -- with and without masking
unmasked_metrics = np.load('data/continual-unmasked-non-incremental-metrics.npz')
masked_metrics = np.load('data/continual-masked-non-incremental-metrics.npz')

unmasked_perplexities_en = unmasked_metrics['perplexity[en]']
masked_perplexities_en = masked_metrics['perplexity[en]']

unmasked_perplexities_fr = unmasked_metrics['perplexity[fr]']
masked_perplexities_fr = masked_metrics['perplexity[fr]']

unmasked_perplexities_es = unmasked_metrics['perplexity[es]']
masked_perplexities_es = masked_metrics['perplexity[es]']

_, axs = plt.subplots(3, 1, figsize=(10, 8), layout='constrained')

axs[0].set_title('English')
axs[0].plot(unmasked_perplexities_en, label='Unmasked [en]')
axs[0].plot(masked_perplexities_en, label='Masked [en]')

axs[1].set_title('French')
axs[1].plot(unmasked_perplexities_fr, label='Unmasked [fr]')
axs[1].plot(masked_perplexities_fr, label='Masked [fr]')

axs[2].set_title('Spanish')
axs[2].plot(unmasked_perplexities_es, label='Unmasked [es]')
axs[2].plot(masked_perplexities_es, label='Masked [es]')

for ax in axs:
    ax.set_yscale('log')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Perplexity')
    ax.legend()

plt.show()