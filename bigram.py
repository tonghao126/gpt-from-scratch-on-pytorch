# move all code from transformer.ipynb to here
import torch
import torch.nn as nn

# TODO: what's the best way to manage conifg?
torch.manual_seed(1337)
learning_rate = 0.01
n_epochs= 10
n_steps = 100
batch_size = 32
block_size = 8
device='cpu'

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
    eval_size = 10
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


from torch.nn import functional as F
# Bigram is just pairwise model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        # Not sure why he uses vocab_size as the embedding dim, maybe just convenient? It's usually multiple of 2
        self.embedding = nn.Embedding(vocab_size, vocab_size, device=device)

    def forward(self, x, y=None):
        logits = self.embedding(x) # (Batch, Time, Channels) Time=word sequence, Channels=Embedding
        # This logic seems quite confusing, is there anyway to optimize this?
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
            logits, loss = self(x)
            # Due to weird forward logic, when generating, logits is not reshaped
            generated = logits[:,-1,:] # B,C
            probs = nn.Softmax(dim=-1)(generated)
            next_tokens = torch.multinomial(probs, 1)
            x = torch.cat([x, next_tokens], dim=1)
        return x
    
model = BigramLanguageModel(vocab_size)
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
print(decoder(model.generate(torch.zeros(1, 1, dtype=torch.long), max_tokens=500).tolist()[0]))