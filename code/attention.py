import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = Parameter(torch.FloatTensor([1.0 / (hidden_size ** 0.5)]))

    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, value)
        return context, attention_weights
