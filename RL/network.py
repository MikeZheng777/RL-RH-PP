import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import NamedTuple


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.embed_dim = self.kdim = self.vdim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, attn_mask=None):
        B, T, embed_dim = x.shape
        H, head_dim = self.num_heads, self.head_dim
        q, k, v = [y.view(B, T, H, head_dim) for y in F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)]
        attn = torch.einsum('bthd,bThd->bhtT', q, k) / math.sqrt(head_dim)
        if attn_mask is not None:
            attn = attn + attn_mask.view(B, 1, 1, T)
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhtT,bThd->bthd', attn, v)
        return self.out_proj(out.reshape(B, T, embed_dim))


class MultiheadCrossAttention(MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__(embed_dim, num_heads, dropout=dropout)

    def forward(self, x, y, attn_mask=None):
        B, T_x, embed_dim = x.shape
        B, T_y, embed_dim = y.shape
        H, head_dim = self.num_heads, self.head_dim
        q = F.linear(x, self.in_proj_weight[:embed_dim], self.in_proj_bias[:embed_dim]).view(B, T_x, H, head_dim)
        k, v = [y.view(B, T_y, H, head_dim) for y in F.linear(y, self.in_proj_weight[embed_dim:], self.in_proj_bias[embed_dim:]).chunk(2, dim=-1)]
        attn = torch.einsum('bthd,bThd->bhtT', q, k) / math.sqrt(head_dim)
        if attn_mask is not None:
            attn = attn + attn_mask.view(B, 1, 1, T_y)
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhtT,bThd->bthd', attn, v)
        return self.out_proj(out.reshape(B, T_x, embed_dim))


class MHAEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, feedforward_factor, dropout=0, activation=F.leaky_relu, layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.dim_feedforward = feedforward_factor * d_model
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        if self.dim_feedforward > 0:
            self.linear1 = nn.Linear(d_model, self.dim_feedforward)
            self.linear2 = nn.Linear(self.dim_feedforward, d_model)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.activation = activation

    def forward(self, x, attn_mask=None):
        return self.feedforward_block(self.self_attn_block(x, attn_mask=attn_mask))

    def self_attn_block(self, x, attn_mask=None):
        return x + self.self_attn(self.norm1(x), attn_mask=attn_mask)

    def feedforward_block(self, x):
        return x + self.linear2(self.activation(self.linear1(self.norm2(x)))) if self.dim_feedforward > 0 else x


class GNNEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, feedforward_factor, n_nodes, dropout=0, activation=F.leaky_relu, layer_norm_eps=1e-5):
        super().__init__()
        self.n_nodes = n_nodes
        self.self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.dim_feedforward = feedforward_factor * d_model
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        if self.dim_feedforward > 0:
            self.linear1 = nn.Linear(d_model, self.dim_feedforward)
            self.linear2 = nn.Linear(self.dim_feedforward, d_model)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.activation = activation

    def forward(self, x, attn_mask=None):
        return self.feedforward_block(self.self_attn_block(x, attn_mask=attn_mask))

    def self_attn_block(self, x, attn_mask=None):
        return x + self.self_attn(self.norm1(x), attn_mask=attn_mask)

    def feedforward_block(self, x):
        return x + self.linear2(self.activation(self.linear1(self.norm2(x)))) if self.dim_feedforward > 0 else x


class CrossMHAEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, feedforward_factor, dropout=0, activation=F.leaky_relu, layer_norm_eps=1e-5):
        super().__init__()
        self.cross_atn = MultiheadCrossAttention(d_model, n_head, dropout=dropout)
        self.dim_feedforward = feedforward_factor * d_model
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        if self.dim_feedforward > 0:
            self.linear1 = nn.Linear(d_model, self.dim_feedforward)
            self.linear2 = nn.Linear(self.dim_feedforward, d_model)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.activation = activation

    def forward(self, x, y, attn_mask=None):
        return self.feedforward_block(self.cross_attn_block(x, y, attn_mask=attn_mask))

    def cross_attn_block(self, x, y, attn_mask=None):
        return x + self.cross_atn(self.norm1(x), y, attn_mask=attn_mask)

    def feedforward_block(self, x):
        return x + self.linear2(self.activation(self.linear1(self.norm2(x)))) if self.dim_feedforward > 0 else x


class MAPFEncoder_GNN(nn.Module):
    def __init__(self, opts):
        super().__init__()

        self.hidden_dims = opts.hidden_dims
        self.num_heads = opts.num_heads
        self.ff_factor = opts.ff_factor
        self.T = opts.observation_window
        self.rows = opts.map_rows
        self.cols = opts.map_cols
        self.G_size = self.rows * self.cols
        self.embed_dim = self.hidden_dims[0]
        self.agent_num = opts.agent_num
        self.location_embedding = nn.Embedding(self.G_size + 1, self.embed_dim, padding_idx=self.G_size)
        self.n_GNN = opts.n_GNN

        self.positional_encoding = self.get_positional_encoding(self.T, self.embed_dim).to(opts.device)

        self.temporal_encoder = nn.ModuleList([
            MHAEncoderLayer(self.hidden_dims[1], self.num_heads, self.ff_factor)
            for _ in range(self.n_GNN)
        ])
        self.spatial_encoder = nn.ModuleList([
            GNNEncoderLayer(self.hidden_dims[1], self.num_heads, self.ff_factor, self.agent_num)
            for _ in range(self.n_GNN)
        ])

        self.init_parameters()

    def get_positional_encoding(self, T, emb_dim):
        pe = torch.zeros(T, emb_dim)
        position = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, emb_dim, 2).float() / emb_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, x_in, map_in):
        path_embedding = self.location_embedding(x_in)  # B, N, T, dim
        B, N_agent, T, emb_dim = path_embedding.size()

        path_embedding = path_embedding.view(B * N_agent, T, emb_dim)
        path_embedding = path_embedding + self.positional_encoding

        inner = path_embedding

        for i, temp_linear, spatial_atten in zip(range(self.n_GNN), self.temporal_encoder, self.spatial_encoder):
            inner = inner + temp_linear(inner)
            inner = spatial_atten(inner.view(B, N_agent, T, emb_dim).permute(0, 2, 1, 3).reshape(B * T, N_agent, emb_dim))
            inner = inner.view(B, T, N_agent, emb_dim).permute(0, 2, 1, 3).reshape(B * N_agent, T, emb_dim)

        inner = inner.view(B, N_agent, T, emb_dim)
        agent_embed = inner[:, :, 0]

        return agent_embed


class DecoderFixed(NamedTuple):
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return DecoderFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],
            glimpse_val=self.glimpse_val[:, key],
            logit_key=self.logit_key[key]
        )


class NCODecoder(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.embedding_dim = embedding_dim = opts.hidden_dims[-1]
        self.n_heads = opts.num_heads
        step_context_dim = embedding_dim
        self.mask_inner = opts.mask_inner
        self.mask_logits = opts.mask_logits
        self.tanh_clipping = opts.tanh_clipping

        self.W_placeholder = nn.Parameter(torch.Tensor(embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % self.n_heads == 0
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)
        )

    def _precompute(self, embeddings, num_steps=1):
        graph_embed = embeddings.mean(1)
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return DecoderFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, None, None, :].expand_as(compatibility)] = -math.inf

        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        final_Q = glimpse
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask.unsqueeze(1)] = -math.inf
        log_p = F.log_softmax(logits, dim=-1)

        return log_p, glimpse.squeeze(-2)

    def forward(self, agent_embd, old_action=None):
        B, A, dim = agent_embd.size()
        fixed = self._precompute(agent_embd)
        actions = torch.zeros(B, A, dtype=torch.long, device=agent_embd.device)
        logps = torch.zeros(B, A, dtype=torch.float, device=agent_embd.device)
        mask = torch.zeros(B, A, dtype=torch.bool, device=agent_embd.device)

        for i in range(A):
            step_context = self.W_placeholder.view(1, 1, -1).expand(B, 1, dim) if i == 0 \
                else agent_embd.gather(1, actions.clone()[:, i-1:i].view(B, 1, 1).expand(B, 1, dim))
            query = fixed.context_node_projected + self.project_step_context(step_context)

            glimpse_K, glimpse_V, logit_K = fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

            if i > 0:
                mask = mask.scatter(1, actions[:, i-1:i], True)

            log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)
            probs = log_p.exp()[:, 0, :]
            log_p = log_p[:, 0, :]

            if old_action is not None:
                action_selected = old_action[:, i].view(-1, 1)
            else:
                action_selected = probs.multinomial(1)
                while mask.gather(1, action_selected).any():
                    print('Sampled bad values, resampling!')
                    action_selected = probs.multinomial(1)

            log_p_selected = log_p.gather(1, action_selected)
            actions[:, i] = action_selected.squeeze(1)
            logps[:, i] = log_p_selected.squeeze(1)

        return actions, logps.sum(1)
