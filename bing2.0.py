import torch
import torch.nn as nn
import numpy as np

# Hyperparameters
input_size = 2  # coordinate (x,y)
hidden_size = 16  # hidden size of RNN
output_size = 1  # score
num_layers = 1  # number of RNN layers
batch_size = 32  # batch size
seq_len = 10  # input sequence length
max_len = seq_len + 1  # output sequence length (+1 for end token)
num_epochs = 1000  # number of epochs


# Generate random coordinates in range [0,1]
def generate_data(batch_size, seq_len):
    return torch.rand(batch_size, seq_len, input_size)


# Compute the convex hull indices for a batch of coordinates using scipy library
def convex_hull_indices(coords):
    from scipy.spatial import ConvexHull
    indices = []
    for i in range(coords.shape[0]):
        hull = ConvexHull(coords[i])
        idx = hull.vertices.tolist()
        idx.append(idx[0])  # close the loop
        indices.append(idx)
    return indices


# Pad the indices with -1 to make them fixed length sequences
def pad_sequences(indices, max_len):
    padded_indices = []
    for idx in indices:
        if len(idx) < max_len:
            idx.extend([-1] * (max_len - len(idx)))
        padded_indices.append(idx)
    return padded_indices


# One hot encode the indices
def one_hot_encode(indices, seq_len):
    one_hot_indices = []
    for idx in indices:
        one_hot_idx = []
        for i in idx:
            if i == -1:  # end token
                one_hot_i = [0] * seq_len + [1]
            else:
                one_hot_i = [0] * i + [1] + [0] * (seq_len - i) + [0]
            one_hot_idx.append(one_hot_i)
        one_hot_indices.append(one_hot_idx)
    return torch.tensor(one_hot_indices).float()


# Define encoder RNN
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.GRU(input_size, hidden_size, num_layers)

    def forward(self, x):
        batch_size = x.size(0)
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        output, (hidden_state) = self.rnn(x.permute(1, 0, 2), (hidden_state))
    # Define encoder RNN


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.GRU(input_dim, hidden_dim, num_layer)

    def forward(self, x):
        batch_sizex.size(0)
        hidde