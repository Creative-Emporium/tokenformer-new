# python train.py --config plain/transformer-gpt2
# python train.py --config plain/tokenformer-gpt2
python train.py --config growth/grow-64
python train.py --config growth/grow-128
python train.py --config growth/grow-256
# python train.py --config continual/unmasked-non-incremental
# python train.py --config continual/masked-non-incremental
