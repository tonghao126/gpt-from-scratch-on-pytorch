# move all code from transformer.ipynb to here
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.utils.config import Config
#read in yaml file
import yaml
import IPython.core.ultratb
import sys
sys.excepthook = IPython.core.ultratb.ColorTB()
from src.models.transformer import *
from src.models.diagnostics import running_loss
from src.data.data_processing import *

config_overwrite = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
config = Config(**config_overwrite)
print(config)


with open('./data/raw/input.txt', 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

encode_mapping = {c: i for i, c in enumerate(chars)}
decode_mapping = {i: c for i, c in enumerate(chars)}
encoder = lambda x: [encode_mapping[s] for s in x]
decoder = lambda x: ''.join([decode_mapping[s] for s in x])

# Not sure why but for the Bigram mps is significantly slower than cpu
data = torch.tensor(encoder(text), dtype=torch.long, device=config.device)

model = Transformer(vocab_size, config)
optimizer = torch.optim.AdamW(model.parameters(),lr=config.learning_rate)

train, eval = train_test_split(data, 0.9)
# print(model.parameters)

# TODO: understand better here, does the order matter?
for epoch in range(config.n_epochs):
    for _ in range(config.n_steps):
        x, y = get_batch(train, config.batch_size, config.block_size)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_eval = running_loss(model, eval, config.block_size, config.batch_size)
    print(f"""Epoch {epoch}, Loss: {loss.item()}, running_train_loss: {epoch_eval['train']}, running_eval_loss: {epoch_eval['eval']}""")

# add device and see difference?
print(decoder(model.generate(torch.zeros(1, config.block_size, dtype=torch.long, device=config.device), max_tokens=500).tolist()[0]))

