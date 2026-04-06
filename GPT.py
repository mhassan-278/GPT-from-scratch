import torch
import torch.nn as nn

with open('input.txt', 'r') as f:
    text = f.read()

# HyperParameters
batch_size = 4
block_size = 8
epochs = 5000
eval_iter = 500
num_embed = 32
head_size = 8
num_heads = 4
num_layers = 4
learning_rate = 1e-3

# Selecting CPU or GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Preprocessing 
chars = ''.join(sorted(list(set(text))))
vocab_size = len(chars)
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for s,i in stoi.items()}
encoder = lambda s: [stoi[i] for i in s]
decoder = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encoder(text))

# Splitting for Training and Testing
n = int(len(data) * 0.9)
train = data[:n]
test = data[n:]

# Function for Batch of data (B,T,C)
def get_batch(split):

    data = train if split == 'train' else test
    ix = torch.randint(len(data) - block_size,(batch_size,))
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size + 1] for i in ix])

    return x,y

class Head(nn.Module):
    "One Head of self-attention"

    def __init__(self, head_size):
        super().__init__()

        self.query = nn.Linear(num_embed, head_size, bias= False)
        self.key = nn.Linear(num_embed, head_size, bias= False)
        self.value = nn.Linear(num_embed, head_size, bias= False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):

        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)


        k_dim = k.shape[-1]
        wei = q @ k.transpose(-2,-1) * (k_dim**-0.5)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = torch.softmax(wei, dim= -1)

        out = wei @ v
        return out
    

class MultiHeadAttention(nn.Module):
    "Multiple self-attention in parallel"

    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.ly = nn.Linear(num_embed,num_embed)


    def forward(self, x):
        out = torch.concat([h(x) for h in self.heads], dim= -1)
        out = self.ly(out)

        return out
    

class FeedForward(nn.Module):
    "Neural Network with ReLU activation"

    def __init__(self, num_embed):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(num_embed,4 * num_embed),
            nn.ReLU(),
            nn.Linear(4* num_embed,num_embed)
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    "Transformer Block"

    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads,head_size)
        self.ffwd = FeedForward(num_embed)
        self.ln1 = nn.LayerNorm(num_embed)
        self.ln2 = nn.LayerNorm(num_embed)


    def forward(self, x):
        out = x + self.sa(x)
        out = out + self.ffwd(x)
        return out

class GPTModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embed = nn.Embedding(vocab_size,num_embed)
        self.positional_embed = nn.Embedding(block_size, num_embed)

        self.blocks = nn.Sequential(*[Block() for _ in range(num_layers)])
        
        self.ln_f =nn.LayerNorm(num_embed)
        self.lm_head = nn.Linear(num_embed, vocab_size)


    def forward(self,x, y = None):

        B,T = x.shape
        token_embed = self.token_embed(x) # (B,T,num_embed)
        pos_embed = self.positional_embed(torch.arange(T)) # (T,num_embed)
        x = token_embed + pos_embed # (B,T,num_embed)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if y is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C) # Should be 1 dim for loss cal [each data point]
            target =  y.view(-1) # Should be 0 dim for loss [each data point]
            loss = nn.functional.cross_entropy(logits, target)

        return logits, loss
    
    def generate(self, idx, max_tokens): # (Input: Must be 2-dim (minimum))

        for _ in range(max_tokens):

            idx_cond = idx[:, -block_size:]
            logits,loss = self(idx_cond)
            logits = logits[:,-1,:]
            pred = torch.softmax(logits,dim= -1)
            idx_next = torch.multinomial(pred, num_samples= 1)
            idx = torch.cat((idx,idx_next), dim= 1)
        
        return idx


model = GPTModel()
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr= learning_rate)

@torch.no_grad()
def estimate_loss():
    
    model.eval()

    out = {}

    for split in ['train','test']:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):

            x,y = get_batch(split)
            x,y = x.to(device), y.to(device)
            logits,loss = model(x,y)

            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()
    return out

total_loss = 0

for i in range(epochs):

    if i % eval_iter == 0:
        losses = estimate_loss()
        print(f"Step {i}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}")

    x,y = get_batch('train')
    x,y = x.to(device), y.to(device)
    logits,loss = model(x,y)
    optimizer.zero_grad(set_to_none= True)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

    # print(f'Epoch: {i+1}, Loss: {loss.item()}')

print(decoder((model.generate(torch.zeros([1,1], dtype= torch.long),1000)[0]).tolist()))