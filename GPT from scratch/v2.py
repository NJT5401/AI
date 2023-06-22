import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
torch.manual_seed(1337)

# hyper parameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# creating a list of all the chars used in the dataset 
chars = sorted(list(set(text)))
# length of the chars list
vocab_size = len(chars)

# tokenize input text - convert text into a sequence of integers
# encoder dictionary
stoi = { ch:i for i,ch in enumerate(chars) }
# decoder dictionary
itos = { i:ch for i,ch in enumerate(chars) }
# encoder function
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# starting with the simplest language model
# bigram language model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # when calling m() and passing in batches, this will go into the embedding table and return the rows at
        # the index of the given sample
        # then pytorch arranges this into a tensor of shape (Batch, Time, Channel)
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_emb
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            # reshaping logits. this needs to happen because loss is expecting channels to be the second dimension of the tensor
            # so we stretch out batch * time and move channels to second dim
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            # shape must be done for targets
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
            # focus on last step
            logits = logits[:, -1, :] # changes logits shape to (B, C)
            probs = F.softmax(logits, dim=1)
            # sampling from distro
            idx_next = torch.multinomial(probs, num_samples=1)
            # append next to running sample
            idx = torch.cat ((idx, idx_next), dim=1)
        return idx
    
model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):
    # eval loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))