import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size

from utils import init_weights

class STEdgeEncoder(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 num_att_layers: int = 2,
                 dropout: float = 0.1,
                 rotate: bool = True) -> None:
        super(STEdgeEncoder, self).__init__()
        self.embed_dim = embed_dim

        self.edge_layers = nn.ModuleList(
            [EdgeAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
             for _ in range(num_att_layers)])
        self.norm = nn.LayerNorm(embed_dim * 2)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
        self.apply(init_weights)

    def forward(self,
                data,
                node_embed: torch.Tensor,
                edge_index: Adj,
                size: Size = None) -> torch.Tensor:

        node_pair = torch.cat((node_embed[edge_index[0]], node_embed[edge_index[1]]), dim=1)
        node_pair = node_pair[data.v2x_mask]
        edge_embed = self.mlp(self.norm(node_pair))
        for layer in self.edge_layers:
            edge_embed = layer(edge_embed)

        return edge_embed

class CSEdgeEncoder(nn.Module):

    def __init__(self,
                 edge_dim: int,
                 embed_dim: int) -> None:
        super(CSEdgeEncoder, self).__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(edge_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim))
        self.apply(init_weights)

    def forward(self,
                rel_pos: torch.Tensor,
                rotate_mat: torch.Tensor) -> torch.Tensor:

        edge_embed = self.edge_mlp(torch.bmm(rel_pos.unsqueeze(-2), rotate_mat).squeeze(-2))
        
        return edge_embed


class EdgeAttention(MessagePassing):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 **kwargs) -> None:
        super(EdgeAttention, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
        self.apply(init_weights)

    def forward(self,
                edge_embed: torch.Tensor,
                size: Size = None) -> torch.Tensor:
        edge_embed = edge_embed + self._mha_block(self.norm1(edge_embed), size)
        edge_embed = edge_embed + self._ff_block(self.norm2(edge_embed))
        return edge_embed

    def _mha_block(self,
                   edge_embed: torch.Tensor,
                   size: Size) -> torch.Tensor:
        
        query = self.lin_q(edge_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = self.lin_k(edge_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(edge_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = torch.softmax(input=(query * key).sum(dim=-1) / scale, dim=0)
        alpha = self.attn_drop(alpha)

        update = (value * alpha.unsqueeze(-1)).view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(update) + self.lin_hh(edge_embed))
        update = self.out_proj(update + gate * (self.lin_self(edge_embed) - update))
        
        return self.proj_drop(update)

    def _ff_block(self, edge_embed: torch.Tensor) -> torch.Tensor:
        return self.mlp(edge_embed)
