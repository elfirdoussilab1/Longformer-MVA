import torch

def generate_longformer_attention_mask(seq_len, window_size, tokens_with_global_attention) :
    # Initialize the attention mask
    attention_mask = torch.zeros([seq_len, seq_len])

    # Handle the tokens with global attention
    attention_mask[tokens_with_global_attention, :] = 1.0
    attention_mask[:, tokens_with_global_attention] = 1.0

    # Add the sliding window on the remaining tokens
    nb_left = nb_right = window_size // 2
    for i in range(seq_len):
        if i not in tokens_with_global_attention :
            left_token_id = max(0, i - nb_left)
            right_token_id = min(seq_len, i + nb_right)
            attention_mask[i, left_token_id:right_token_id] = 1.0

    return attention_mask


def attention_mask_for_additions(len_digits, window_size = 3, seq_len = 50):
    # Total size for addition : Size of the two numbers + '+' + '='
    total_size = 2*len_digits + 2
    attention_mask = torch.zeros([seq_len, seq_len])

    # Index of the edges of the sliding window
    nb_left = nb_right =  window_size//2

    # Indexes of '+' and '='
    idx_plus = len_digits
    idx_eq = total_size - 1

    # Global attention for '+' and '='
    attention_mask[idx_plus, :] = 1
    attention_mask[:, idx_plus] = 1
    attention_mask[idx_eq, :] = 1
    attention_mask[:, idx_eq] = 1

    for i in range(total_size):
        
        left_token_id = max(0, i-nb_left)
        right_token_id = min(total_size, i+nb_right)
        attention_mask[i, left_token_id:right_token_id+1] = 1

        if i < idx_plus :
            attention_mask[i, left_token_id+len_digits+1 : right_token_id+len_digits+2] = 1
        elif i > idx_plus and i < idx_eq :
            attention_mask[i, max(0, left_token_id - len_digits - 1): min(total_size, right_token_id - len_digits)] = 1

    return attention_mask