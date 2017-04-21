import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, num_tokens, encoder_size, hidden_size, num_layers, dropout=.5, sigma=.002):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.sigma = sigma
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(num_tokens)
        self.rnn = nn.LSTM(encoder_size, hidden_size, num_layers, dropout=dropout)

        self.init_weights()

    def init_weights():
        self.embedding.weight.data = torch.randn(self.embedding.weights.data.size()) * self.sigma
        self.rnn.weight.data = torch.randn(self.rnn.weight.data.size()) * self.sigma
