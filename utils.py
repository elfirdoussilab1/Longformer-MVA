import torch

def get_diagonal(N, offset: int):
    """Get i and j indices of diagonal.

    Params:
        N: size of the squared matrix
        offset: +1 means first diagonal to the right of the main diagonal, -1 means to the left, and so on.

    Returns:
        indices_i: i indices of the elements of the diagonal
        indices_j: j indices of the elements of the diagonal 
    """
    indices_i = torch.arange(N)
    valid_length = N - abs(offset)

    if offset < 0:  # Lower diagonals -> Remove from start
        indices_i = indices_i[-offset:]  
    else:  # Upper diagonals -> Remove from end
        indices_i = indices_i[:valid_length]
    indices_j = indices_i + offset

    return indices_i, indices_j