import torch
from src.data.data_processing import get_batch

@torch.no_grad()
def running_loss(model, data, batch_size, block_size, eval_size=50):
    eval_size = 50
    out={}
    model.eval()
    for source in ['train', 'eval']:
        losses = torch.zeros(eval_size)
        for i in range(eval_size):
            x, y = get_batch(data, batch_size, block_size)
            _, loss = model(x, y)
            losses[i] = loss
        out[source] = losses.mean()
    return out
