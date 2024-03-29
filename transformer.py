# move all code from transformer.ipynb to here
import token
import torch
import torch.nn as nn

# TODO: what's the best way to manage conifg?
torch.manual_seed(1337)
learning_rate = 1e-3
n_epochs= 10
n_steps = 1000
batch_size = 32
block_size = 8
device='cpu'
n_embed = 32
head_size = 8
n_attentions = 8

with open('./data/raw/input.txt', 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

encode_mapping = {c: i for i, c in enumerate(chars)}
decode_mapping = {i: c for i, c in enumerate(chars)}
encoder = lambda x: [encode_mapping[s] for s in x]
decoder = lambda x: ''.join([decode_mapping[s] for s in x])

# Not sure why but for the Bigram mps is significantly slower than cpu
data = torch.tensor(encoder(text), dtype=torch.long, device=device)

# train test split
n = int(0.9*len(data))
train = data[:n]
test = data[n:]

def get_batch(data, batch_size):
    data_slice = torch.randint(0, len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in data_slice])
    y = torch.stack([data[i+1:i+block_size+1] for i in data_slice])
    return  x, y

x, y = get_batch(train, batch_size)

@torch.no_grad()
def running_loss():
    eval_size = 50
    out={}
    model.eval()
    for source in ['train', 'eval']:
        data = train if source == 'train' else test
        losses = torch.zeros(eval_size)
        for i in range(eval_size):
            x, y = get_batch(data, batch_size)
            _, loss = model(x, y)
            losses[i] = loss
        out[source] = losses.mean()
    return out

class SelfAttention(nn.Module):
    def __init__(self, n_embed, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, device=device)
        self.query = nn.Linear(n_embed, head_size, device=device)
        self.value = nn.Linear(n_embed, head_size, device=device)
        
    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        # Weight is the key to self attention mechanism, essentially which previous key is more relevant to current k
        weight = q@k.transpose(-2,-1)* head_size**-0.5 # B,T,H @ B,H,T -> B,T,T

        # apply triangle mask, note the order of things. I.e. when to apply mask, softmax and multply by v at the end
        # TODO: the implementation here is different from Andrej's, would this screw things up downstream?
        tri = torch.tril(torch.ones(block_size,block_size, device=device))
        weight = weight.masked_fill(tri==0, float('-inf'))
        weight = torch.softmax(weight, dim=-1)
        weight = weight@v # B,T,T @ B,T,H -> B,T,H
        return weight



from torch.nn import functional as F
# Bigram is just pairwise model
# TODO: take self attention out as its own class
class Transformer(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed, device=device)
        self.pos_embedding = nn.Embedding(block_size, n_embed, device=device)
        # self.self_attention = SelfAttention(n_embed, head_size)
        self.multi_head_attention = nn.ModuleList([SelfAttention(n_embed, head_size) for i in range(n_attentions)])
        self.fc = nn.Linear(head_size*n_attentions, vocab_size, device=device)

    def forward(self, x, y=None):
        char_embedding_layer = self.token_embedding(x) # (Batch, Time, Channels) Time=word sequence, Channels=embed_size
        pos_embedding_layer = self.pos_embedding(torch.arange(block_size, device=device)) # (T, C)
        x = char_embedding_layer + pos_embedding_layer
        # TODO: separate multi headed attention to a separate class, and see why I'm not reaching 3.2 in error
        weight = torch.cat([self.multi_head_attention[i](x) for i in range(n_attentions)],dim=-1) # B,T,H*N_attentions -> B,T,H*N_attentions
        logits = self.fc(weight) # B,T,T @ B,T,H -> B,T,H
        # TODO: normalization
        
        # When generating, y is None
        if y is None:
            loss=None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            y = y.view(-1)
            # Lesson, check documentation and see if order is as intended
            loss = F.cross_entropy(logits, y)
        return logits, loss
    
    def generate(self, x, max_tokens=100):
        for i in range(max_tokens):
            idx_block = x[:,-block_size:]
            logits, loss = self(idx_block)
            # Due to weird forward logic, when generating, logits is not reshaped
            generated = logits[:,-1,:] # B,C
            probs = nn.Softmax(dim=-1)(generated)
            next_tokens = torch.multinomial(probs, 1)
            x = torch.cat([x, next_tokens], dim=1)
        return x
    
model = Transformer(vocab_size)
out, loss = model(x, y)
optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)


# TODO: understand better here, does the order matter?
print("using device", x.device)
for epoch in range(n_epochs):
    for _ in range(n_steps):
        x, y = get_batch(train, batch_size)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_eval = running_loss()
    print(f"""Epoch {epoch}, Loss: {loss.item()}, running_train_loss: {epoch_eval['train']}, running_eval_loss: {epoch_eval['eval']}""")

# add device and see difference?
print(decoder(model.generate(torch.zeros(1, 8, dtype=torch.long), max_tokens=500).tolist()[0]))

