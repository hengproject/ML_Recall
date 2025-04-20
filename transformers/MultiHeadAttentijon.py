import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个 head 的维度

        self.W_q = nn.Linear(d_model, d_model)  # d_model = h * d_k ,一次性生成所有的头，再reshape成多头
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)  # concat后的那一步

    def forward(self, X: torch.Tensor):
        batch_size, n, _ = X.size()
        # (batch_size, seq_len, d_model) @ (d_model,d_model) ->  (batch_size, seq_len, d_model) == (batch_size, n, d_model)
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        # reshape and split heads: (batch, n, d_model) -> (batch, n, num_heads, d_k) -> (batch, num_heads, n, d_k)
        Q = Q.view(batch_size, n, self.num_heads, self.d_k).transpose(1, 2)  # (B, h, n, d_k)
        K = K.view(batch_size, n, self.num_heads, self.d_k).transpose(1, 2)  # (B, h, n, d_k)
        V = V.view(batch_size, n, self.num_heads, self.d_k).transpose(1, 2)  # (B, h, n, d_k)

        # (B, h, n, d_k) @ (B, h, d_k,n) = (B, h), (n, d_k) @ (d_k, n)=> (B,h,n,n)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        alpha = F.softmax(scores, dim=-1)  # 注意力权重: (batch, h, n, n) [第一个是Q的n,第二个是K的n]
        context = alpha @ V  # (batch, h, n, n) @ (B, h, n, d_k) = (B, h, n, d_k)
        # (B, h, n, d_k) -> (B,n,h,d_k) --contiguous()保证内存是连续的,才可以用view() -->(batch, n, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, n, self.d_model)
        output = self.W_o(context)  # 输出线性层: (batch, n, d_model)@(d_model,d_model) ->(batch, n, d_model)

        return output, alpha


def test_multihead_attention():
    global alpha
    batch_size = 2
    seq_len = 5  # n
    d_model = 512
    num_heads = 8

    mha = MultiHeadAttention(d_model, num_heads)
    X = torch.randn(batch_size, seq_len, d_model)  # 模拟输入

    output, alpha = mha(X)

    print("Input shape:", X.shape)  # (2, 5, 512)
    print("Output shape:", output.shape)  # (2, 5, 512)
    print("Attention shape:", alpha.shape)  # (2, 8, 5, 5) 每个 head 都是 (n x n)



def show_attention_heatmap(alpha, batch_idx=0, head_idx=0):
    # 提取对应的注意力权重矩阵
    attn = alpha[batch_idx, head_idx].detach().cpu().numpy()  # (n_q, n_k)

    plt.figure(figsize=(6, 5))
    sns.heatmap(attn, annot=True, fmt=".2f", cmap="Blues", cbar=True)
    plt.title(f"Attention Heatmap (Batch {batch_idx}, Head {head_idx})")
    plt.xlabel("Key Position (n_k)")
    plt.ylabel("Query Position (n_q)")
    plt.show()



if __name__ =='__main__':
    alpha = None
    test_multihead_attention()
    show_attention_heatmap(alpha, 0)
    show_attention_heatmap(alpha, 1)