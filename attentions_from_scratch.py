import torch
import torch.nn as nn
from utils import get_diagonal

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, Q_dim=64, V_dim=64):
        super(SelfAttentionBlock, self).__init__()
        self.matrix_Q = nn.Linear(embed_dim, Q_dim)
        self.matrix_K = nn.Linear(embed_dim, Q_dim)
        self.matrix_V = nn.Linear(embed_dim, V_dim)

    def forward(self, x):
        Q = self.matrix_Q(x)
        K = self.matrix_K(x)
        V = self.matrix_V(x)
        
        return torch.softmax(Q@K.transpose(1,2), dim=2)@V
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, Q_dim=64, V_dim=64, n_head=64):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim%n_head == 0, f'Cannot divide vector x (dim = {embed_dim}) into n_head={n_head}'

        self.n_head = n_head
        self.Q_dim = Q_dim 
        self.V_dim = V_dim

        self.matrix_Q = nn.Linear(embed_dim, n_head*Q_dim)
        self.matrix_K = nn.Linear(embed_dim, n_head*Q_dim)
        self.matrix_V = nn.Linear(embed_dim, n_head*V_dim)

        self.linear_out = nn.Linear(n_head*V_dim, embed_dim)

    def forward(self, x):
        # x.shape = (n_batch, n_seq, embed_dim)
        n_batch, n_seq, embed_dim = x.shape
        Q = self.matrix_Q(x) # Q.shape = (n_batch, n_seq, n_head*Q_dim)
        K = self.matrix_K(x)
        V = self.matrix_V(x)

        Q = Q.view((n_batch, n_seq, self.n_head, self.Q_dim)).transpose(1,2) #(n_batch, n_head, n_seq, Q_dim)
        K = K.view((n_batch, n_seq, self.n_head, self.Q_dim)).transpose(1,2) #(n_batch, n_head, n_seq, Q_dim)
        V = V.view((n_batch, n_seq, self.n_head, self.V_dim)).transpose(1,2) #(n_batch, n_head, n_seq, V_dim)

        attention_score_matrix = Q@K.transpose(-2,-1) # (n_batch, n_head, n_seq, n_seq)
        attention_score_matrix = torch.softmax(attention_score_matrix, dim=-1) # (n_batch, n_head, n_seq, n_seq)
        outputs = attention_score_matrix@V #(n_batch, n_head, n_seq, V_dim)

        outputs = outputs.transpose(1,2).contiguous().view(n_batch, n_seq, self.n_head*self.V_dim) #(n_batch, n_head, V_dim)

        return self.linear_out(outputs)


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

        columns = torch.zeros((n_batch,n_seq,n_seq), device=device)

        for i,di in enumerate(range(-abs(self.window_size//2), self.window_size//2+1)):
            indices_i, indices_j = get_diagonal(n_seq, di)
            c = torch.einsum("b i d, b i d -> b i", Q[:,indices_i,:], K[:,indices_j,:])
            columns[:,indices_i,i] = c

        return torch.softmax(columns, dim=2)@V
    

class MultiHeadSliddingWindowAttention(nn.Module):
    """Slidding window multi-head self-attention."""
    def __init__(self, embed_dim, Q_dim=64, V_dim=64, n_head=8, window_size=3):
        super(MultiHeadSliddingWindowAttention, self).__init__()
        assert embed_dim%n_head == 0, f'Cannot divide vector x (dim = {embed_dim}) into n_head={n_head}'

        self.window_size = window_size
        self.n_head = n_head
        self.Q_dim = Q_dim 
        self.V_dim = V_dim

        self.matrix_Q = nn.Linear(embed_dim, n_head*Q_dim)
        self.matrix_K = nn.Linear(embed_dim, n_head*Q_dim)
        self.matrix_V = nn.Linear(embed_dim, n_head*V_dim)

        self.linear_out = nn.Linear(n_head*V_dim, embed_dim)

    def forward(self, x):
        device = next(self.parameters()).device  # Retrieve device from model parameters
        # x.shape = (n_batch, n_seq, embed_dim)
        n_batch, n_seq, embed_dim = x.shape
        Q = self.matrix_Q(x) # Q.shape = (n_batch, n_seq, n_head*Q_dim)
        K = self.matrix_K(x)
        V = self.matrix_V(x)

        Q = Q.view((n_batch, n_seq, self.n_head, self.Q_dim)).transpose(1,2) #(n_batch, n_head, n_seq, Q_dim)
        K = K.view((n_batch, n_seq, self.n_head, self.Q_dim)).transpose(1,2) #(n_batch, n_head, n_seq, Q_dim)
        V = V.view((n_batch, n_seq, self.n_head, self.V_dim)).transpose(1,2) #(n_batch, n_head, n_seq, V_dim)

        columns = torch.zeros((n_batch,self.n_head,n_seq,n_seq), device=device)

        for i,di in enumerate(range(-abs(self.window_size//2), self.window_size//2+1)):
            indices_i, indices_j = get_diagonal(n_seq, di)
            c = torch.einsum("a b i d, a b i d -> a b i", Q[:,:,indices_i,:], K[:,:,indices_j,:])
            columns[:,:,indices_i,i] = c

        columns = torch.softmax(columns, dim=-1) # (n_batch, n_head, n_seq, n_seq)
        outputs = columns@V #(n_batch, n_head, n_seq, V_dim)

        outputs = outputs.transpose(1,2).contiguous().view(n_batch, n_seq, self.n_head*self.V_dim) #(n_batch, n_head, V_dim)

        return self.linear_out(outputs)
    

class DilatedSliddingWindowAttention(nn.Module):
    """Dilated Sliding Window Self-Attention."""
    def __init__(self, embed_dim, Q_dim=64, V_dim=64, window_size=5, dilation=1):
        super(DilatedSliddingWindowAttention, self).__init__()
        self.window_size = window_size
        self.dilation = dilation
        self.matrix_Q = nn.Linear(embed_dim, Q_dim)
        self.matrix_K = nn.Linear(embed_dim, Q_dim)
        self.matrix_V = nn.Linear(embed_dim, V_dim)

    def forward(self, x):
        device = next(self.parameters()).device
        n_batch, n_seq, embed_dim = x.shape
        Q = self.matrix_Q(x)
        K = self.matrix_K(x)
        V = self.matrix_V(x)

        
        n_fixed = len(range(-abs(self.window_size // 2), self.window_size // 2 + 1)) # this makes the space complexity O(n)
        columns = torch.zeros((n_batch, n_seq, n_fixed), device=device)
        
        for i, di in enumerate(range(-abs(self.window_size // 2), self.window_size // 2 + 1)):  
            dilated_di = di * self.dilation  # Apply dilation
            indices_i, indices_j = get_diagonal(n_seq, dilated_di)
            c = torch.einsum("b i d, b i d -> b i", Q[:, indices_i, :], K[:, indices_j, :])
            columns[:, indices_i, indices_j] = c
        
        return torch.softmax(columns, dim=2) @ V


class MultiHeadDilatedSliddingWindowAttention(nn.Module):
    """Multi-Head Dilated Sliding Window Self-Attention."""
    def __init__(self, embed_dim, Q_dim=64, V_dim=64, n_head=8, window_size=3, dilation=1):
        super(MultiHeadDilatedSliddingWindowAttention, self).__init__()
        assert embed_dim % n_head == 0, f'Cannot divide vector x (dim = {embed_dim}) into n_head={n_head}'

        self.window_size = window_size
        self.dilation = dilation
        self.n_head = n_head
        self.Q_dim = Q_dim
        self.V_dim = V_dim

        self.matrix_Q = nn.Linear(embed_dim, n_head * Q_dim)
        self.matrix_K = nn.Linear(embed_dim, n_head * Q_dim)
        self.matrix_V = nn.Linear(embed_dim, n_head * V_dim)

        self.linear_out = nn.Linear(n_head * V_dim, embed_dim)

    def forward(self, x):
        device = next(self.parameters()).device
        n_batch, n_seq, embed_dim = x.shape
        Q = self.matrix_Q(x).view(n_batch, n_seq, self.n_head, self.Q_dim).transpose(1, 2)
        K = self.matrix_K(x).view(n_batch, n_seq, self.n_head, self.Q_dim).transpose(1, 2)
        V = self.matrix_V(x).view(n_batch, n_seq, self.n_head, self.V_dim).transpose(1, 2)
        
        columns = torch.zeros((n_batch, self.n_head, n_seq, n_seq), device=device)
        
        for i, di in enumerate(range(-abs(self.window_size // 2), self.window_size // 2 + 1)):
            dilated_di = di * self.dilation  # Apply dilation
            indices_i, indices_j = get_diagonal(n_seq, dilated_di)  # Get valid diagonal indices
            c = torch.einsum("a b i d, a b i d -> a b i", Q[:, :, indices_i, :], K[:, :, indices_j, :]) # Reduction of time complexity to O(n)
            columns[:, :, indices_i, indices_j] = c

        
        columns = torch.softmax(columns, dim=-1)
        outputs = columns @ V
        
        outputs = outputs.transpose(1, 2).contiguous().view(n_batch, n_seq, self.n_head * self.V_dim)
        return self.linear_out(outputs)
