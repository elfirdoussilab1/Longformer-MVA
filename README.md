# Longformer-MVA

Re-implementation / evaluation of Longformer architecture.

## Resources:

- **Original paper:** [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)

- **Official repo:** [https://github.com/allenai/longformer](https://github.com/allenai/longformer)


## Implementations:

- Using **nn.MultiHeadAttention** with masking, $O(n^2)$ time & space complexities: [transformers.py](transformer.py).
- Using **attention computation from scratch** with per-diagonal attention computation, $O(n)$ time & $O(n^2)$ space complexities: [transformers_from_scratch.py](transformers_from_scratch.py). Some complexity estimations are done but they are not conclusive: [experiments_from_scratch.ipynb](experiments_from_scratch.ipynb).
- Using the **for loop method**: $O(n)$ time & space complexities: [long_attention.py](long_attention.py)