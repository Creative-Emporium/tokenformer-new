import torch

def parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def generate_samples(model, dataset, encoding, window=20):
    ix = dataset.rng.integers(0, len(dataset.corpus) - window, size=5)
    samples = [ dataset.corpus[i : i + window] for i in ix ]
    samples = [ encoding.decode(s.tolist()) for s in samples ]

    print('\n\tSample generation:')
    for s in samples:
        encoded = [ encoding.encode(s) ]
        encoded = torch.tensor(encoded).cuda()
        generated = model.generate(encoded, 20)
        generated = generated[0].tolist()
        string = encoding.decode(generated)
        print('\t->', string.replace('\n', '\n\t\t'))
