import torch
from torch import nn
from torch.nn import functional as F, parameter
import _global

def merge(a):
    return torch.stack(a).sum(dim=0) if a!=[] else torch.zeros((1024,))

class NerNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dim = 1024
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(self.dim, 7)

    def forward(self, x,):
        return self.linear(self.dropout(x))
