import torch
from torch import nn
from torch.autograd import Function

class PoincareEmbedding(nn.Module):
    def __init__(self, N, M=3):
        super(PoincareEmbedding, self).__init__()
        self.l_embed = nn.Embedding(N, M, max_norm=1-1e-5)
        self.l_embed.weight.data[:,:] = self.l_embed.weight.data[:,:] * 1e-3
    def forward(self, x):
        return self.l_embed(x)
