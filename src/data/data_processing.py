import torch

# train test split
def train_test_split(data:torch.tensor, ratio:float=0.9):
    n = int(ratio*len(data))
    train = data[:n]
    test = data[n:]
    return train, test

def get_batch(data, batch_size, block_size):
    data_slice = torch.randint(0, len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in data_slice])
    y = torch.stack([data[i+1:i+block_size+1] for i in data_slice])
    return  x, y