import torch
from torch import nn


class NumberNet(nn.Module):
    def __init__(self, cat_vocab_size, max_cat_len):
        super(NumberNet, self).__init__()
        self.max_cat_len = max_cat_len

        self.linear1 = nn.Linear(cat_vocab_size, 2 * max_cat_len)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2 * max_cat_len, max_cat_len)

    def forward(self, x_material):
        x1 = self.linear1(x_material)
        x1 = self.relu(x1)
        x2 = self.linear2(x1)
        return x2
