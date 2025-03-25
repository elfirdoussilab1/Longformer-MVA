import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# Here we implement the SlidingWindow LongFormer from scratch

# Attention Head
class LongAttentionHead(nn.Module):
    """Slidding window self-attention."""
    def __init__(self, embed_dim, head_size, window_size):
        super(LongAttentionHead, self).__init__()
        self.window_size = window_size
        self.matrix_Q = nn.Linear(embed_dim, head_size)
        self.matrix_K = nn.Linear(embed_dim, head_size)
        self.matrix_V = nn.Linear(embed_dim, head_size)

    def forward(self, x):
        device = next(self.parameters()).device  # Retrieve device from model parameters
        n_batch, n_seq, embed_dim = x.shape
        Q = self.matrix_Q(x) # Q.shape = (n_batch, n_seq, head_dim)
        K = self.matrix_K(x)
        V = self.matrix_V(x)
        head_dim = Q.shape[-1]
        attentions = torch.zeros((n_batch, n_seq, head_dim), device= device)

        for i in range(n_seq): # iterate over all tokens
            indices = np.arange(np.maximum(i - self.window_size // 2, 0), np.minimum(i + self.window_size // 2, n_seq - 1) + 1)
            indices = torch.tensor(indices, dtype= torch.long, device= device) # shape (I,) where I <= window_size
            # Q.K
            c = torch.einsum("b i d, b i d -> b i", Q[:,[i]*len(indices),:], K[:,indices,:]) # shape (b, I)
            # normalization by embed_dim like in classical attention
            c /= embed_dim**0.5
            c = torch.softmax(c, dim = -1) # shape (B, I)
            attentions[:, i,:] = torch.sum(c.unsqueeze(-1) * V[:, indices,:], dim = 1)

        return attentions


# Multi-Head Long Attention
class MultiHeadLongAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, window_size, dropout = 0.05):
        super().__init__()
        embed_dim = num_heads * head_size
        self.heads = nn.ModuleList([LongAttentionHead(embed_dim, head_size, window_size) for _ in range(num_heads)]) # head_size * num_heads = embed_dim
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        return out

class MLP(nn.Module):
    def __init__(self, embed_dim, dropout = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.ReLU(), 
            nn.Linear(4 * embed_dim, embed_dim), # As in the paper 512 * 4 = 2048
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# LongFormer Block
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, embed_dim, num_heads, window_size):
        super().__init__()
        head_size = embed_dim // num_heads
        self.attn = MultiHeadLongAttention(num_heads, head_size, window_size)
        self.ffwd = MLP(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x)) # Residual connection & Norm
        x = x + self.ffwd(self.ln2(x))
        return x

# LongFormer Model
class LongFormer(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_heads, window_size, n_blocks, positional_encoder):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # first argument = number of tokens we have,
        # second argument = embedding dimension
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim) # Maps each word to a a vector of dimension embed_dim !
        self.positional_encoder = positional_encoder

        # LongFormer Blocks
        self.blocks = nn.Sequential(
            *[Block(embed_dim, num_heads, window_size) for _ in range(n_blocks)],
            nn.LayerNorm(embed_dim)
        )

        # Linear layer
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, sequence, targets=None):
        B, T = sequence.shape # T = sequence length
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(sequence) # (B, T, d) # d = embed_dim
        pos_emb = self.positional_encoder(tok_emb.transpose(0, 1)).transpose(0, 1) # (B, T, d)

        x = tok_emb + pos_emb # (B, T, d)
        x = self.blocks(x) # Apply head of self-attention (B, T, d)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits
        # To compute the loss, do this: 
        # B, T, d = logits.shape
        # logits = logits.view(B*T, d)
        # targets = targets.view(B*T)
        # loss = F.cross_entropy(logits, targets) 

        if targets is None:
            loss = None
        else:
            B, T, p = logits.shape
            logits = logits.view(B*T, p) # We do this because to compute Cross-Entropy loss, logits should have C (channels) as a second dimension!
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) 

        return logits, loss