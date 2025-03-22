import torch
import torch.nn as nn
import numpy as np

class SliddingWindowAttention(nn.Module):
    """Slidding window self-attention."""
    def __init__(self, embed_dim, Q_dim=64, V_dim=64, window_size=3):
        super(SliddingWindowAttention, self).__init__()
        self.window_size = window_size
        self.matrix_Q = nn.Linear(embed_dim, Q_dim)
        self.matrix_K = nn.Linear(embed_dim, Q_dim)
        self.matrix_V = nn.Linear(embed_dim, V_dim)

    def forward(self, x):
        device = next(self.parameters()).device  # Retrieve device from model parameters
        n_batch, n_seq, embed_dim = x.shape
        Q = self.matrix_Q(x) # Q.shape = (n_batch, n_seq, Q_dim)
        K = self.matrix_K(x)
        V = self.matrix_V(x)
        Q_dim = Q.shape[-1]
        #columns = torch.zeros((n_batch,n_seq,n_seq), device=device)
        attentions = torch.zeros((n_batch, n_seq, Q_dim), device= device)

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