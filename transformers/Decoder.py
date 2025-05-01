import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.MultiHeadAttentijon import MultiHeadAttention

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, tgt_padding_mask=None, memory_padding_mask=None):
        """
        :param x: decoder input (batch_size, tgt_len, d_model)
        :param encoder_output: encoder output (batch_size, src_len, d_model)
        :param tgt_padding_mask: (batch_size, tgt_len)
        :param memory_padding_mask: (batch_size, src_len)
        """
        # Masked Self-Attention (Q=K=V=x)
        x_norm = self.norm1(x)
        self_attn_out, _ = self.self_attn(
            x_norm,
            query=x_norm,
            key=x_norm,
            value=x_norm,
            padding_mask=tgt_padding_mask,
            causal_mask=True
        )
        x = x + self.dropout1(self_attn_out)

        #  Cross Attention (Q = decoder, K/V = encoder)
        x_norm = self.norm2(x)
        cross_attn_out, _ = self.cross_attn(
            x_norm,
            query=x_norm,
            key=encoder_output,
            value=encoder_output,
            padding_mask=memory_padding_mask,
            causal_mask=False
        )
        x = x + self.dropout2(cross_attn_out)

        # Feed Forward Network
        x_norm = self.norm3(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout3(ffn_out)

        return x  # shape: (batch_size, tgt_len, d_model)


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)  # 最后加个输出归一化（论文原版也有）

    def forward(self, x, encoder_output, tgt_padding_mask=None, memory_padding_mask=None):
        for layer in self.layers:
            x = layer(
                x,
                encoder_output,
                tgt_padding_mask=tgt_padding_mask,
                memory_padding_mask=memory_padding_mask
            )
        return self.norm(x)  # (batch_size, tgt_len, d_model)
