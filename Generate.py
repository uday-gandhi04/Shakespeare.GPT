import torch
import torch.nn as nn
from Script import BigramLanguageModel
import time

block_size = 256

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

model = BigramLanguageModel(vocab_size)
torch.manual_seed(int(time.time()))
model.load_state_dict(torch.load('Shakespear_model.pth',weights_only=True))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

c=torch.tensor(encode("\n"))
context = c.unsqueeze(0).to(device)
with torch.no_grad():
    print(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))
