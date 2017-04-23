import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, num_tokens, encoder_size, hidden_size, num_layers, dropout=.5, sigma=.002):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sigma = sigma
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(num_tokens, encoder_size)
        self.rnn = nn.LSTM(encoder_size, hidden_size, num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_size, 7*12)

        self.init_weights()

    def forward(self, x, hidden):
        embedded = self.dropout(self.embedding(x))
        rnn_out, hidden = self.rnn(embedded)
        rnn_out = self.dropout(rnn_out)
        output = self.linear(rnn_out.view(-1, self.hidden_size))
        return nn.functional.softmax(output.view(-1, x.size(1), 7, 12)), hidden

    def init_weights(self):
        self.embedding.weight.data = torch.randn(self.embedding.weight.data.size()) * self.sigma
        self.linear.weight.data = torch.randn(self.linear.weight.data.size()) * self.sigma
        self.linear.bias.data.fill_(0)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, batch_size, self.hidden_size).zero_()),
                Variable(weight.new(self.num_layers, batch_size, self.hidden_size).zero_()))
