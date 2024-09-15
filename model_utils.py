import copy
import math
import warnings

import torch.nn as nn
import torch.nn.functional as F
import torch


class LayerNorm(nn.Module):
    """Construct a layer norm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h    # attention head
        self.linear = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(dropout)
        """这里不用正则化 2head的效果比1head的要好一点"""

    def forward(self, query, key, mask=None):
        # original mask = batch * 1 * maxlength; q,k = batch * maxlength * dim
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)    # batch * 1 * 1 * query.size(1)
        n_batches = query.size()[0]
        query, key = [l(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2)    # b*h*l*d
                      for l, x in zip(self.linear, (query, key))]
        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn


def attention(query, key, mask, dropout):
    d_K = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_K)   # q = Q*Wq, k = K*Wk

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return p_attn


class AA_Attn(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(AA_Attn, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h    # attention head
        self.linear = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, mask=None):
        # original mask = batch * 1 * maxlength; q,k = batch * maxlength * dim
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)    # batch * 1 * 1 * query.size(1)
        n_batches = query.size()[0]
        query, key = [l(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2)    # b*h*l*d
                      for l, x in zip(self.linear, (query, key))]
        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn


def AA_GCN(input_embed, adj, weight):
    Ax_adj = adj.bmm(input_embed)
    AxW_adj = weight(Ax_adj)
    denom_adj = adj.sum(2).unsqueeze(2) + 1
    AxW_adj = AxW_adj / denom_adj
    gAxW_adj = F.relu(AxW_adj)
    return gAxW_adj


class CosAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    @staticmethod
    def forward(feature, aspect_v, d_mask):
        Q = aspect_v   # b*d   batch*dim=8*200
        Q = Q.unsqueeze(1)   # b*1*d
        score = F.cosine_similarity(feature, Q, dim=-1)   # b*l*d,b*1*d 对从句中所有的词与aspect的词向量计算相似度
        # attn_score = F.softmax(score, dim=-1)
        d_mask = d_mask.unsqueeze(2)  # (B, L, 1)

        attention_weight = feature * score.unsqueeze(-1) * d_mask   # 乘以余弦相似度再乘以mask

        return attention_weight


class RelationAttention(nn.Module):
    def __init__(self, in_dim=300, hidden_dim=64):
        # in_dim: the dimension fo query vector
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, feature, dep_tags_v, d_mask):
        """
        (feature, dep_feature, src_mask)

        C feature/context [N, L, D]
        Q dep_tags_v          [N, L, D]
        mask d_mask          [N, L]
        """
        Q = self.fc1(dep_tags_v)    # b*l*300 -> b*l*64
        Q = self.relu(Q)    # b*l*64
        Q = self.fc2(Q)  # (N, L, 1)    # b*l*1 length个一维（依赖关系）
        Q = Q.squeeze(2)   # 16*69
        Q = F.softmax(mask_logits(Q, d_mask), dim=1)   # 只剩下src部分的embed

        Q = Q.unsqueeze(2)    # 16*69*1
        out = torch.bmm(feature.transpose(1, 2), Q)     # feature.transpose(1, 2) = 16*69*600.transpose(1, 2) = 16*600*69 * 16*69*1= 16*600*1
        out = out.squeeze(2)   # 16*600
        # out = F.sigmoid(out)
        return out  # ([N, L])


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)


def average_adj(head, adj, mask):
    adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(adj, 1, dim=1)]

    # average multi head
    temp = torch.zeros_like(adj_list[0])
    h = adj.size()[1]
    for i in range(h):
        if temp is None:
            temp = adj_list[i]
        else:
            temp += adj_list[i]
    attn_adj = temp / head  # attention combined

    # keep diag always 1
    for j in range(attn_adj.size(0)):
        attn_adj[j] -= torch.diag(torch.diag(attn_adj[j]))
        attn_adj[j] += torch.eye(attn_adj[j].size(0)).cuda()
    adj_ag = mask.unsqueeze(1).transpose(1, 2) * attn_adj

    return adj_ag


def select(matrix, top_num):
    batch = matrix.size(0)
    l = matrix.size(1)
    matrix = matrix.reshape(batch, -1)   # 8*(length*length)
    max_k, _ = torch.topk(matrix, top_num, dim=1)

    for i in range(batch):
        matrix[i] = (matrix[i] >= max_k[i][-1])
    matrix = matrix.reshape(batch, l, l)
    matrix = matrix + matrix.transpose(-2, -1)

    # self loop
    for i in range(batch):
        matrix[i].fill_diagonal_(1)

    return matrix


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def sequence_mask(lengths, max_len=None):
    """
    create a boolean mask from sequence length `[batch_size,  seq_len]`
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) >= (lengths.unsqueeze(1))


class MultiHead_Attention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1, k_dim=None, v_dim=None):
        super(MultiHead_Attention, self).__init__()
        self.embed_dim = embed_dim
        self.k_dim = k_dim if k_dim is not None else embed_dim
        self.v_dim = v_dim if v_dim is not None else embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.k_dim)
        self.v_proj = nn.Linear(embed_dim, self.v_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=True):

        L, B, D = query.size()
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = self.embed_dim // self.num_heads
        assert head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"   # 必须整除
        scaling = float(head_dim) ** -0.5

        q = self.q_proj(query)   # q=Wq
        k = self.k_proj(key)    # k=Wk
        v = self.v_proj(value)    # v=Wv
        q = q * scaling

        # check attn_mask
        if attn_mask is not None:
            assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
                   attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or \
                   attn_mask.dtype == torch.bool, 'Only float, byte, and bool types are supported for attn_mask, not {}'.format(
                    attn_mask.dtype)
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [B * self.num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
            # attn_mask's dim is 3 now.

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = q.contiguous().view(L, B * self.num_heads, head_dim).transpose(0, 1)    # 48*30*100
        if k is not None:
            k = k.contiguous().view(-1, B * self.num_heads, head_dim).transpose(0, 1)    # 48*30*100
        if v is not None:
            v = v.contiguous().view(-1, B * self.num_heads, head_dim).transpose(0, 1)    # 48*30*100

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == B
            assert key_padding_mask.size(1) == src_len

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))  # [B*num_heads,L,D] * [B*num_heads,D,L] -->[B*num_heads,L,L]
        assert list(attn_output_weights.size()) == [B * self.num_heads, L, src_len]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(B, self.num_heads, L, src_len)    # 24*2*30*30
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # [B,N,L,L]->[B,1,1,L]
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(B * self.num_heads, L, src_len)   # 48*30*30

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v)  # [B,N,L,L] [B,N,L,D]
        assert list(attn_output.size()) == [B * self.num_heads, L, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(L, B, self.embed_dim)    # 30*24*200 两个head合并
        attn_output = self.out_proj(attn_output)    # 最后乘以Weight

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(B, self.num_heads, L, src_len)     # 24*2*30*30
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output, None

class CapsuleLayer(nn.Module):
    def __init__(self, input_dim_capsule, num_capsule, dim_capsule, routings, activation='default'):
        super(CapsuleLayer, self).__init__()
        self.num_capsule = num_capsule    # 胶囊的个数
        self.dim_capsule = dim_capsule     # 胶囊的维度
        self.routings = routings     # routing=5
        self.t_epsilon = 1e-7    # 计算squash需要的参数
        if activation == 'default':
            self.activation = self.squash
        else:
            self.activation = nn.ReLU(inplace=True)

        # 只需要这一个 weight=(1, 968, 16*10)
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, input_dim_capsule, num_capsule * dim_capsule)))

    def forward(self, x):
        u_hat_vecs = torch.matmul(x, self.W)    # x=b*968   W=1*968*160 = b*160

        batch_size = x.size(0)
        input_num_capsule = x.size(1)
        u_hat_vecs = u_hat_vecs.view((batch_size, 1, self.num_capsule, self.dim_capsule))    # b*1*16*10
        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3)   # b*16*1*10
        b = torch.zeros_like(u_hat_vecs[:, :, :, 0])

        outputs = 0
        for i in range(self.routings):
            b = b.permute(0, 2, 1)
            c = F.softmax(b, dim=2)
            c = c.permute(0, 2, 1)
            b = b.permute(0, 2, 1)
            outputs = self.activation(torch.einsum('bij,bijk->bik', (c, u_hat_vecs)))
            if i < self.routings - 1:
                b = torch.einsum('bik,bijk->bij', (outputs, u_hat_vecs))
        return outputs

    def squash(self, x, axis=-1):
        s_squared_norm = (x ** 2).sum(axis, keepdim=True)
        scale = torch.sqrt(s_squared_norm + self.t_epsilon)
        return x / scale

