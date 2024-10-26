import torch.nn as nn
import torch


class RMSNorm(nn.Module):
    def __init__(self, embedding_nums, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.rand(embedding_nums), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        rms_x = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = x / rms_x
        out = out * self.weight
        return out


class RopeCoding(nn.Module):
    def __init__(self, embedding_nums, max_seq_len=10000):
        super(RopeCoding, self).__init__()
        self.cos, self.sin = self.build_theta_sin_cos(embedding_nums, max_seq_len)

    def forward(self, x):
        seq_lens = x.shape[1]
        cos_ = self.cos.to(x.device)[:seq_lens]
        sin_ = self.sin.to(x.device)[:seq_lens]
        return x * cos_ + self.reverse_half(x) * sin_

    def build_theta_sin_cos(self, embedding_nums, seq_len):
        base = 10000.0
        d = int(embedding_nums / 2)
        B = base ** (1/d)
        _2i = torch.arange(0, d)
        theta_base = 1.0 / (B ** _2i)
        seq_len_tensor = torch.arange(seq_len)


        out = torch.outer(seq_len_tensor, theta_base)
        out = torch.cat((out, out), dim=-1)
        return torch.cos(out), torch.sin(out)

    def reverse_half(self, x):
        embedding_nums = x.shape[-1]
        U = x[:, :, :embedding_nums // 2]
        V = x[:, :, embedding_nums // 2:]
        return torch.cat((-V, U), dim=-1)


class FeedForward(nn.Module):
    def __init__(self, embedding_nums, hidden_dim=None, multiple_of=256, ffn_dim_multiplier=None):
        super().__init__()
        hidden_dim = int(embedding_nums*4)
        hidden_dim = int(2*hidden_dim/3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier*hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(embedding_nums, hidden_dim, bias=False)
        # down_projection
        self.w2 = nn.Linear(hidden_dim, embedding_nums, bias=False)
        # up_projection
        self.w3 = nn.Linear(embedding_nums, hidden_dim)

        self.selu = nn.SELU()

    def forward(self, x):
        x_selu = self.w1(self.selu(x))
        x_w3 = self.w3(x)
        return self.w2(x_selu * x_w3)


class MultHeadSelfAttentionDecoder(nn.Module):
    def __init__(self, embedding_nums, head_nums):
        super().__init__()
        self.W = nn.Linear(embedding_nums, embedding_nums, bias=False)
        assert embedding_nums % head_nums == 0, "embedding维度不能整除多头数"
        head_embedding_nums = int(embedding_nums / head_nums)
        self.mult_head_self_attention = nn.ModuleList([SelfAttentionDecoder(embedding_nums, head_embedding_nums)
                                                       for _ in range(head_nums)])

    def forward(self, x, padding_pask=None):
        mult_head_out_list = []
        for single_head_self_attention in self.mult_head_self_attention:
            mult_head_out_list.append(single_head_self_attention(x, padding_pask))
        mult_head_out = torch.cat(mult_head_out_list, dim=-1)
        out = self.W(mult_head_out)
        return out


class SelfAttentionDecoder(nn.Module):
    def __init__(self, embedding_nums, head_embedding_nums):
        super().__init__()
        self.W_Q = nn.Linear(embedding_nums, head_embedding_nums, bias=False)
        self.W_K = nn.Linear(embedding_nums, head_embedding_nums, bias=False)
        self.W_V = nn.Linear(embedding_nums, head_embedding_nums, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])
        self.rope_coding = RopeCoding(head_embedding_nums)

    def forward(self, x, padding_mask=None):
        if self.training:
            q = self.W_Q(x)
            k = self.W_K(x)
            v = self.W_V(x)
            q = self.rope_coding(q)
            k = self.rope_coding(k)
            score = (q @ k.transpose(-1, 1)) / torch.sqrt(torch.tensor(q.shape[-1]))

            if padding_mask is not None:
                score = fill_padding_mask(score, padding_mask)
            score = fill_look_ahead_mask(score)
            score_norm = self.softmax(score)
            x_attention = score_norm @ v
            return x_attention

        else:
            if len(self.k_cache) > 0:
                x = x[:, -1:, :]
                q = self.W_Q(x)
                k = self.W_K(x)
                v = self.W_V(x)

                q = q.expand(q.shape[0], x.shape[1], q.shape[-1])
                k = q.expand(k.shape[0], x.shape[1], k.shape[-1])
                q = self.rope_coding(q)[:, -1:, :]
                k = self.rope_coding(k)[:, -1:, :]
                self.k_cache = torch.cat((self.k_cache, k), dim=1)
                self.v_cache = torch.cat((self.v_cache, v), dim=1)

            else:
                q = self.W_Q(x)
                k = self.W_K(x)
                v = self.W_V(x)
                q = self.rope_coding(q)
                k = self.rope_coding(k)
                self.k_cache = k
                self.v_cache = v

            score = (q @ self.k_cache.transpose(-1, 1)) / torch.sqrt(torch.tensor(q.shape[-1]))
            score_norm = self.softmax(score)
            try:
                x_attention = score_norm @ self.v_cache
            except:
                pass
            return x_attention



class DecoderBlock(nn.Module):
    def __init__(self, embedding_nums, head_nums):
        super().__init__()
        self.multi_head_attention = MultHeadSelfAttentionDecoder(embedding_nums, head_nums)
        self.feed_forward = FeedForward(embedding_nums)
        self.attention_norm = RMSNorm(embedding_nums)
        self.ffn_norm = RMSNorm(embedding_nums)


    def forward(self, x, padding_mask=None):
        attention_norm_x = self.attention_norm(x)
        attention_x = self.multi_head_attention(x)
        residual_x1 = x + attention_x

        ffn_norm_x = self.ffn_norm(residual_x1)
        ffn_x = self.feed_forward(ffn_norm_x)
        residual_x2 = residual_x1 + ffn_x
        return residual_x2







class LlamaModel(nn.Module):
    def __init__(self, embedding_nums, token_nums, max_lens, head_nums, block_nums):
        super().__init__()
        self.token_embedding = nn.Embedding(token_nums, embedding_nums)
        self.norm = RMSNorm(embedding_nums)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(embedding_nums, head_nums) for _ in range(block_nums)])
        self.classifier_linear = nn.Linear(embedding_nums, token_nums)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, x_idx, y_idx):
        x_emb = self.token_embedding(x_idx)
        for decode_block in self.decoder_blocks:
            x_emb = decode_block(x_emb, x_idx>0)
        x_norm = self.norm(x_emb)
        pre = self.classifier_linear(x_norm)
        loss = self.loss_func(pre.reshape(pre.shape[0]*pre.shape[1], pre.shape[-1]), y_idx.reshape(-1))
        return loss

    def generate(self, x_token, temperature, topK):
        x_emb = self.token_embedding(x_token)

        while True:
            x_emb_copy = x_emb.clone()

            for decode_block in self.decoder_blocks:
                x_emb_copy = decode_block(x_emb_copy)
            pre = self.classifier_linear(x_emb_copy[0, -1])
            pre_pro = torch.softmax(pre, dim=0) / temperature
            pre_pro = torch.softmax(pre_pro, dim=0)
            pre_pro_tok, pre_pro_tok_idx = torch.topk(pre_pro, topK)
            pre_pro_tok_pro = torch.softmax(pre_pro_tok, dim=0)
            next_token = pre_pro_tok_idx[int(torch.multinomial(pre_pro_tok_pro, num_samples=1)[0])]
            if next_token == 2 or x_emb.shape[1] >= 100:
                break
            else:
                next_token_tensor = torch.tensor([[next_token]]).long().to(x_token.device)
                next_token_emb = self.token_embedding(next_token_tensor)
                x_emb = torch.cat((x_emb, next_token_emb), dim=1)
                yield next_token_tensor

def fill_padding_mask(score, padding_mask):
    padding_mask_sum = torch.sum(padding_mask, dim=-1)
    for i in range(padding_mask_sum.shape[0]):
        score[i, :, padding_mask_sum[i]:] = -torch.inf
    return score


def fill_look_ahead_mask(score):
    sentence_lens = score.shape[-1]
    for i in range(sentence_lens):
        score[:, i, i+1:] = -torch.inf
    return score


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand(4, 30, 512).to(device)
    model = RMSNorm(512).to(device)
    res = model(x)
    ropeCoding = RopeCoding(512).to(device)
    res = ropeCoding(x)
    pass
