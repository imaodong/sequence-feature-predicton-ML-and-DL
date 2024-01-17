
import torch
gpu_id = 4

batch_size = 64
mod = 96
embedding_dim = 7
hide_dim = 32
head_count = 4
epochs = 80
dropout_rate = 0.1
learning_rate = 1e-3
device = torch.device(f'cuda:{gpu_id}')