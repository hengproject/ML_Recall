import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.MultiHeadAttentijon import MultiHeadAttention


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff, dropout=0.1):
        super().__init__()

        self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),  # Linear@x.shape=(batch_size, seq_len, d_model) → (batch_size, seq_len, dim_ff)
            nn.ReLU(),
            nn.Linear(dim_ff, d_model)  # Linear@x.shape=(batch_size, seq_len, dim_ff) → (batch_size, seq_len, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        # [size] x: (batch_size, seq_len, d_model)

        # Pre-LN + Attention + Dropout + Residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm,
            padding_mask=padding_mask,
            causal_mask=False
        )
        x = x + self.dropout1(attn_out)

        # Pre-LN + FFN + Dropout + Residual
        x_ffn = self.norm2(x)
        ffn_out = self.ffn(x_ffn)
        x = x + self.dropout2(ffn_out)

        return x  # (batch_size, seq_len, d_model)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, dim_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, padding_mask=None):
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)
        return x


if __name__ == '__main__':
    def test_encoder_block_pre_ln():
        batch_size = 2
        seq_len = 6
        d_model = 512
        num_heads = 8
        dim_ff = 2048

        # 模拟输入
        x = torch.randn(batch_size, seq_len, d_model)

        # 模拟 padding mask（batch 中第 2 个序列有 3 个 padding）
        padding_mask = torch.tensor([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1]
        ])  # shape: (batch_size, seq_len)

        # 创建模块并测试
        block = TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, dim_ff=dim_ff)
        out = block(x, padding_mask=padding_mask)

        print("Input shape:", x.shape)  # (2, 6, 512)
        print("Output shape:", out.shape)  # (2, 6, 512)


    test_encoder_block_pre_ln()


    def test_transformer_encoder():
        batch_size = 2
        seq_len = 6
        d_model = 512
        num_heads = 8
        dim_ff = 2048
        num_layers = 3

        x = torch.randn(batch_size, seq_len, d_model)
        padding_mask = torch.tensor([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1]
        ])

        encoder = TransformerEncoder(d_model, num_heads, dim_ff, num_layers)
        out = encoder(x, padding_mask=padding_mask)

        print("Input shape:", x.shape)
        print("Output shape:", out.shape)


    test_transformer_encoder()
