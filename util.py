import math
from parameter_set import *
import torch


def attention(Q, K, V, mask):
    score = torch.matmul(Q, K.permute(0, 1, 3, 2))
    score /=  (hide_dim/head_count) ** 0.5

    score = score.masked_fill_(mask, -float('inf'))
    score = torch.softmax(score, dim=-1)

    score = torch.matmul(score, V)
    score = score.permute(0, 2, 1, 3).reshape(-1, mod, hide_dim)

    return score


class MultiHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_Q = torch.nn.Linear(embedding_dim, hide_dim)
        self.fc_K = torch.nn.Linear(embedding_dim, hide_dim)
        self.fc_V = torch.nn.Linear(embedding_dim, hide_dim)
        self.out_fc = torch.nn.Linear(hide_dim, embedding_dim)
        self.norm = torch.nn.LayerNorm(normalized_shape=embedding_dim, elementwise_affine=True)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, Q, K, V, mask):
        b = Q.shape[0]
        clone_Q = Q.clone()
        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)
        K = self.fc_K(K)
        V = self.fc_V(V)
        Q = self.fc_Q(Q)
        sub_dim = int(hide_dim/head_count)
        Q = Q.reshape(b, mod, head_count, sub_dim).permute(0, 2, 1, 3)
        K = K.reshape(b, mod, head_count, sub_dim).permute(0, 2, 1, 3)
        V = V.reshape(b, mod, head_count, sub_dim).permute(0, 2, 1, 3)
        score = attention(Q, K, V, mask)
        score = self.dropout(self.out_fc(score))
        score = clone_Q + score
        return score


class PositionEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()

        def get_pe(pos, i, d_model):
            fenmu = 1e4 ** (i / d_model)
            pe = pos / fenmu

            if i % 2 == 0:
                return math.sin(pe)
            return math.cos(pe)

        pe = torch.empty(mod, embedding_dim)
        for i in range(mod):
            for j in range(embedding_dim):
                pe[i, j] = get_pe(i, j, embedding_dim)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        embed = x
        embed = embed + self.pe
        return embed


class FullyConnectedOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=embedding_dim, out_features=hide_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hide_dim, out_features=embedding_dim),
            torch.nn.Dropout(p=0.1),
        )

        self.norm = torch.nn.LayerNorm(normalized_shape=embedding_dim,
                                       elementwise_affine=True)

    def forward(self, x):
        clone_x = x.clone()
        x = self.norm(x)
        out = self.fc(x)
        out = clone_x + out
        return out
