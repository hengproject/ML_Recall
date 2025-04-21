import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个 head 的维度

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, X, padding_mask=None, causal_mask=False):
        batch_size, seq_len, _ = X.size()

        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32, device=X.device))

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 1, float('-inf'))

        if causal_mask:
            causal = torch.triu(torch.ones(seq_len, seq_len, device=X.device), diagonal=1).bool()
            scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))

        alpha = F.softmax(scores, dim=-1)
        context = torch.matmul(alpha, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)

        return output, alpha


if __name__ == '__main__':
    # ============================
    # 测试函数：带 padding + causal mask
    # ============================
    alpha = None
    def test_mha_with_masks():
        global alpha
        batch_size = 2
        seq_len = 5
        d_model = 512
        num_heads = 8

        mha = MultiHeadAttention(d_model, num_heads)
        X = torch.randn(batch_size, seq_len, d_model)

        padding_mask = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1]
        ])

        output, alpha = mha(X, padding_mask=padding_mask, causal_mask=True)

        print("Output shape:", output.shape)
        print("Attention shape:", alpha.shape)
        print("Alpha (head 0, batch 0):")
        print(alpha[0, 0])


    test_mha_with_masks()
