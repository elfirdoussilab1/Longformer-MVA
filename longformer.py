import torch
import torch.nn as nn
import torch.nn.functional as F

class SlidingWindowSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim, "Embedding dimension mismatch"

        # Sliding window attention mask (only local window attends)
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = True

        # Apply attention with mask
        attn_output, _ = self.attention(x, x, x, attn_mask=mask.to(x.device))
        return attn_output

class LongformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, ffn_dim):
        super().__init__()
        self.attention = SlidingWindowSelfAttention(embed_dim, num_heads, window_size)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )

    def forward(self, x):
        # Apply attention + residual connection
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)

        # Feed-forward network + residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

class Longformer(nn.Module):
    """Longformer implementation.

    Positional embedding is learnt. Could be changed to sinusoidal positional encoding, but for now
    we are only interested in inference speed testing, so it does not matter. 
    """

    def __init__(self, vocab_size, embed_dim, num_heads, window_size, ffn_dim, num_layers, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        self.layers = nn.ModuleList([
            LongformerBlock(embed_dim, num_heads, window_size, ffn_dim)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, vocab_size)  # Output layer for token prediction

    def forward(self, x):
        batch_size, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        # Token + Positional embeddings
        x = self.embedding(x) + self.position_embedding(pos)

        # Pass through multiple Longformer blocks
        for layer in self.layers:
            x = layer(x)

        return self.fc(x)
