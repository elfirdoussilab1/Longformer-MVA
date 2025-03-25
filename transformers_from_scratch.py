import torch.nn as nn
from attentions_from_scratch import SelfAttentionBlock, MultiHeadSelfAttention, SliddingWindowAttention, MultiHeadSliddingWindowAttention
import torch


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, V_dim, Q_dim, attention_class, **kwargs):
        """
        A generic Transformer block that allows different attention mechanisms.

        Args:
        - embed_dim (int): Dimension of embeddings.
        - num_heads (int): Number of attention heads (used if applicable).
        - ffn_dim (int): Hidden dimension of the feed-forward network.
        - attention_module (nn.Module): Custom attention layer (e.g., MHSA, Sliding Window, etc.).
        - **kwargs: Extra parameters for the attention mechanism (e.g., window_size).
        """
        super().__init__()
        self.attention = attention_class(embed_dim, V_dim, Q_dim, num_heads, **kwargs)  # Pass custom attention layer
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

class Transformer(nn.Module):
    """Transformer implementation.

    Positional embedding is learnt. Could be changed to sinusoidal positional encoding, but for now
    we are only interested in inference speed testing, so it does not matter. 
    """

    def __init__(self, vocab_size, embed_dim, num_heads, ffn_dim, num_layers, max_len, V_dim, Q_dim, attention_class=MultiHeadSelfAttention, positional_embedding=None, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if positional_embedding is None:
            self.position_embedding = nn.Embedding(max_len, embed_dim)
        else:
            self.position_embedding = positional_embedding(embed_dim, max_len=max_len)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim, V_dim, Q_dim, attention_class, **kwargs)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, vocab_size)  # Output layer for token prediction

    def forward(self, x):
        device = next(self.parameters()).device  # Dynamically get the model's device
        
        # Token + Positional embeddings
        if isinstance(self.position_embedding, nn.Embedding):
            batch_size, seq_len = x.shape
            pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            x = self.embedding(x) + self.position_embedding(pos)
        else:
            x = self.position_embedding(self.embedding(x))

        # Pass through multiple Longformer blocks
        for layer in self.layers:
            x = layer(x)

        return self.fc(x)


class Longformer(Transformer):
    """Longformer implementation using Sliding Window Self-Attention."""
    
    def __init__(self, vocab_size, embed_dim, num_heads, ffn_dim, num_layers, max_len, V_dim, Q_dim, window_size, dilation, positional_embedding=None):
        super().__init__(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            max_len=max_len,
            V_dim=V_dim, 
            Q_dim=Q_dim,
            attention_class=MultiHeadSliddingWindowAttention,  # Default to Longformer's attention
            positional_embedding=positional_embedding,
            window_size=window_size,
            dilation=dilation
        )