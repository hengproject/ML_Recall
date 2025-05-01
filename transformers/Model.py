import torch
from torch import nn
from transformers.Encoder import  TransformerEncoder
from transformers.Decoder import TransformerDecoder
from transformers.InputEmbedding import InputEmbedding

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, dim_ff, num_layers, max_len=5000, dropout=0.1):
        super().__init__()
        # 编码器和解码器共享输入嵌入方式
        self.src_embed = InputEmbedding(vocab_size, d_model, max_len)
        self.tgt_embed = InputEmbedding(vocab_size, d_model, max_len)

        self.encoder = TransformerEncoder(d_model, num_heads, dim_ff, num_layers, dropout)
        self.decoder = TransformerDecoder(d_model, num_heads, dim_ff, num_layers, dropout)

        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, src_ids, tgt_ids, src_padding_mask=None, tgt_padding_mask=None):
        """
        src_ids: (B, src_len) - input tokens
        tgt_ids: (B, tgt_len) - target tokens (teacher forcing)
        """

        src_embed = self.src_embed(src_ids)  # (B, src_len, d_model)
        tgt_embed = self.tgt_embed(tgt_ids)  # (B, tgt_len, d_model)

        memory = self.encoder(src_embed, padding_mask=src_padding_mask)
        out = self.decoder(
            tgt_embed,
            memory,
            tgt_padding_mask=tgt_padding_mask,
            memory_padding_mask=src_padding_mask
        )

        logits = self.output_proj(out)  # (B, tgt_len, vocab_size)
        return logits

if __name__ == '__main__':
    def test_transformer_model():
        vocab_size = 1000
        d_model = 512
        num_heads = 8
        dim_ff = 2048
        num_layers = 4
        max_len = 100

        batch_size = 2
        src_len = 7
        tgt_len = 5

        # 模拟 token id（不含特殊 token）
        src_ids = torch.randint(0, vocab_size, (batch_size, src_len))  # (2, 7)
        tgt_ids = torch.randint(0, vocab_size, (batch_size, tgt_len))  # (2, 5)

        # 0 表示 token，1 表示 padding
        src_padding_mask = torch.tensor([
            [0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1]
        ])

        tgt_padding_mask = torch.tensor([
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1]
        ])

        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            dim_ff=dim_ff,
            num_layers=num_layers,
            max_len=max_len,
            dropout=0.1
        )

        logits = model(src_ids, tgt_ids, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)

        print("Src ids shape:    ", src_ids.shape)  # (2, 7)
        print("Tgt ids shape:    ", tgt_ids.shape)  # (2, 5)
        print("Logits shape:     ", logits.shape)  # (2, 5, 1000)


    test_transformer_model()