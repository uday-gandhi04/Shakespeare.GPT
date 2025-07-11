import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000 # how many steps in total to train for?
eval_interval = 500 # how many steps to take between evaluations?
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 # how many steps to take for each evaluation
n_embd = 384 # the dimensionality of the character embedding vectors
n_head = 6 # the number of heads in the multi-head attention layers
n_layer = 6 # the number of layers in the transformer
dropout = 0.2 # the dropout rate for regularization
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
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

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key=nn.Linear(n_embd,head_size, bias=False)
        self.query=nn.Linear(n_embd,head_size, bias=False)
        self.value=nn.Linear(n_embd,head_size, bias=False)
        self.dropout=nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self,x):
        B,T,C=x.shape
        k=self.key(x)
        q=self.query(x)
        # compute attention scores
        wei=q@k.transpose(-2,-1)*C**-0.5
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei=F.softmax(wei,dim=-1)
        wei=self.dropout(wei)
        # perform the weighted aggregation of the values 
        v=self.value(x)
        out=wei@v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,num_head,head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_head) ])
        self.proj=nn.Linear(n_embd, n_embd)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out)) # (B, T, C)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout),
        )
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self,n_embd,n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa_head=MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1=nn.LayerNorm(n_embd)
        self.ln2=nn.LayerNorm(n_embd)

    def forward(self, x):
        x= x+self.sa_head(self.ln1(x)) # (B,T,C)
        x=x+ self.ffwd(self.ln2(x)) # (B,T,C)
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding=torch.nn.Embedding(block_size,n_embd)
        self.block=nn.Sequential(*[Block(n_embd,n_head=n_head)for _ in range(n_layer)])
        self.ln_f=nn.LayerNorm(n_embd)
        self.lm_head=nn.Linear(n_embd,vocab_size)

    def forward(self, idx, targets=None):
        B,T=idx.shape

        token_embd=self.token_embedding_table(idx)
        pos_embd=self.positional_embedding(torch.arange(T,device=device))
        x= token_embd + pos_embd
        x = self.block(x)
        x = self.ln_f(x) 
        logits = self.lm_head(x) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond=idx[:,-block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

if __name__=="__main__":
    model = BigramLanguageModel(vocab_size)
    m = model.to(device)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    torch.save(m.state_dict(), 'Shakespear_model.pth')
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    with open('Babbel.txt', 'w', encoding='utf-8') as f:
        f.write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))