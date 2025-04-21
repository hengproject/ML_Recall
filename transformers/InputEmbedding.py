import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        #  embedding(x): (batch_size, seq_len, d_model) --> embedding(x):(vocab_size,d_model)
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 奇数列
        pe[:, 1::2] = torch.cos(position * div_term)  # 偶数列
        pe = pe.unsqueeze(0)  # [shape] pe: (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # [shape] x: (batch_size, seq_len, d_model) pe:(1, max_len, d_model) 也就是说只需要提供奇拿
        # x.size(1) = seq_len --> pe => (1,seq_len,d_model) x=>(batch_suze,seq_len,d_model)
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=5000):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

    def forward(self, x):
        # [shape] x: (batch_size, seq_len)
        x = self.token_embedding(x)  # TokenEmbedding@x.shape → (batch, seq_len, d_model)
        x = self.pos_encoding(x)  # PositionalEncoding@x.shape → (batch, seq_len, d_model)
        return x


if __name__ =='__main__':
    vocab_size = 1000
    d_model = 512
    seq_len = 10
    batch_size = 2

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    embedder = InputEmbedding(vocab_size, d_model, max_len=seq_len)

    out = embedder(input_ids)
    print("Input IDs shape:", input_ids.shape)  # (2, 10)
    print("Output embedding shape:", out.shape)  # (2, 10, 512)
    torch.nn.TransformerEncoderLayer
    torch.nn.TransformerEncoder