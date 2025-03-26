# MVA LLMs project: LongFormer: The Long-Document Transforme

This is the official code repository of the project 11: LongFormer presented for the LLMs course taught on the master's program MVA 2024/2025.


## Implementations:

- Using **nn.MultiHeadAttention** with masking, $O(n^2)$ time & space complexities: [transformers.py](transformers.py).
- Using **attention computation from scratch** with per-diagonal attention computation, $O(n)$ time & $O(n^2)$ space complexities: [transformers_from_scratch.py](transformers_from_scratch.py). Some complexity estimations are done but they are not conclusive: [experiments_from_scratch.ipynb](experiments_from_scratch.ipynb).
- Using the **for loop method**: $O(n)$ time & space complexities: [long_attention.py](long_attention.py). This file also contains an implementation of a LLM using this new long attention mechanism from scratch.
- Using a custom attention mask on a Transformer for additions with character-level tokenizer : [notebooks/additions_with_custom_attention.ipynb](notebooks/additions_with_custom_attention.ipynb)

## Resources:

- **Original paper:** [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)
