import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from time import time
from datasets import load_dataset
from model import RNNAttentionLanguageModel, Attention
from utils import get_batch, estimate_loss

# Загрузка данных
raw_datasets = load_dataset("abobster/pushkin_new")
with open('data/input.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(raw_datasets['train']['text']))

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 50
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', device)
eval_iters = 200
n_embd = 384
dropout = 0.2
# ------------

torch.manual_seed(1337)

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

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

model = RNNAttentionLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

t0 = time()
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, elapsed: {time() - t0:.1f}s")

    xb, yb = get_batch('train')
    logits, _ = model(xb)
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
