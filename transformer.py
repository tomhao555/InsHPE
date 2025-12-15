import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

class Transformer(nn.Module):
    def __init__(self,  dim=512, depth=2, num_heads=4, mlp_ratio=2.):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio))


    def forward(self, query, key):
        output = query
        for i, layer in enumerate(self.layers):
            output = layer(query=output, key=key)

        return output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.3):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, query, key, value):
        B, N1, C = query.shape
        _, N2, _ = key.shape
        query = query.reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        key = key.reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        value = value.reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        #元素乘法
        # attn = self.sigmoid(query * key)
        # x = (value * attn).transpose(1, 2).reshape(B, N1, C)
        x = torch.matmul(attn, value).transpose(1, 2).reshape(B, N1, C)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.channels = dim

        self.encode_value = nn.Linear(dim, dim)
        self.encode_query = nn.Linear(dim, dim)
        self.encode_key = nn.Linear(dim, dim)

        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)


    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, query, key):
        b, q_n, c = query.shape
        _, k_n, c = key.shape

        # 1D位置编码
        q_embedding = nn.Parameter(torch.randn(1, q_n, c)).cuda()
        k_embedding = nn.Parameter(torch.randn(1, k_n, c)).cuda()

        query_embed = repeat(q_embedding, '() n  d -> b n  d', b = b)
        key_embed = repeat(k_embedding, '() n  d -> b n  d', b = b)

        q_embed = self.with_pos_embed(query, query_embed)
        k_embed = self.with_pos_embed(key, key_embed)

        # 编码特征
        v = self.encode_value(key)
        q = self.encode_query(q_embed)
        k = self.encode_key(k_embed)

        query = query + self.attn(query=q, key=k, value=v)
 
        query = query + self.mlp(self.norm2(query))

        return query
