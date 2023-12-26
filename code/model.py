import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention

class RNNAttentionLanguageModel(nn.Module):
    def __init__(self):
        super(RNNAttentionLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.rnn = nn.LSTM(n_embd, n_embd, num_layers=1, dropout=dropout, batch_first=True)
        self.attention = Attention(n_embd)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_embd, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        context, _ = self.attention(output, output, output)
        output = self.dropout(context)
        logits = self.fc(output)
        return logits, hidden

    def generate(self, start_tokens, max_new_tokens, temperature=1.0):
        model.eval()
        generated_tokens = start_tokens.clone()

        for _ in range(max_new_tokens):
            logits, _ = model(generated_tokens)
            last_logits = logits[:, -1, :] / temperature
            probabilities = F.softmax(last_logits, dim=-1)
            sampled_token = torch.multinomial(probabilities, 1)
            generated_tokens = torch.cat((generated_tokens, sampled_token), dim=1)

        return generated_tokens
