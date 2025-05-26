from typing import Optional, Tuple

import torch.nn.functional as F

import torch
import torch.nn as nn
from utils import init_weights
from utils import TemporalData
from utils import MLPEmbedding

class NodeEncoder(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 num_st_layers: int = 2,
                 num_motion_layers: int = 4,
                 dropout: float = 0.1) -> None:
        super(NodeEncoder, self).__init__()
        self.historical_steps = historical_steps

        self.st_node_encoder = STNodeEncoder(historical_steps=historical_steps,
                                            node_dim=node_dim,
                                            embed_dim=embed_dim,
                                            num_heads=num_heads,
                                            dropout=dropout,
                                            num_layers=num_st_layers)

        self.motion_node_encoder = MotionNodeEncoder(historical_steps=historical_steps,
                                                node_dim=node_dim,
                                                embed_dim=embed_dim,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                num_layers=num_motion_layers)

    def forward(self,
                data: TemporalData) -> torch.Tensor:

        st_embed = self.st_node_encoder(positions=data.positions[:, : self.historical_steps],
                                        padding_mask=data['padding_mask'][:, : self.historical_steps])

        motion_embed = self.motion_node_encoder(x=data.x,
                                                bos_mask=data['bos_mask'],
                                                padding_mask=data['padding_mask'][:, : self.historical_steps],
                                                rotate_mat=data['rotate_mat'])

        return st_embed, motion_embed


class STNodeEncoder(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.1) -> None:
        super(STNodeEncoder, self).__init__()
        self.historical_steps = historical_steps
        encoder_layer = TemporalEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers,
                                                         norm=nn.LayerNorm(embed_dim))
        self.center_embed = MLPEmbedding(in_channel=node_dim, out_channel=embed_dim)
        self.padding_token = nn.Parameter(torch.Tensor(1, historical_steps, embed_dim))
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.Tensor(historical_steps + 1, 1, embed_dim))
        nn.init.normal_(self.padding_token, mean=0., std=.02)
        nn.init.normal_(self.cls_token, mean=0., std=.02)
        nn.init.normal_(self.pos_embed, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                positions: torch.Tensor,
                padding_mask: torch.Tensor,
                ) -> torch.Tensor:

        x = torch.where(padding_mask.unsqueeze(-1),
                        self.padding_token.expand(positions.shape[0], -1, -1),
                        self.center_embed(positions)).transpose(0, 1)
        expand_cls_token = self.cls_token.expand(-1, x.shape[1], -1)
        x = torch.cat((x, expand_cls_token), dim=0) # [T+1, N, D]
        x = x + self.pos_embed

        channel_mask = padding_mask
        channel_mask_cls = torch.zeros(x.shape[1], 1, dtype=torch.bool, device=x.device)
        channel_mask = torch.cat((channel_mask, channel_mask_cls), dim=1)

        out = self.transformer_encoder(src=x, mask=None, src_key_padding_mask=channel_mask)
        return out[-1]  # [N, D]

    @staticmethod
    def generate_square_subsequent_mask(seq_len: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class MotionNodeEncoder(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1) -> None:
        super(MotionNodeEncoder, self).__init__()
        self.historical_steps = historical_steps
        self.center_embed = MLPEmbedding(in_channel=node_dim, out_channel=embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
        self.bos_token = nn.Parameter(torch.Tensor(historical_steps, embed_dim))
        nn.init.normal_(self.bos_token, mean=0., std=.02)
        encoder_layer = TemporalEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers,
                                                         norm=nn.LayerNorm(embed_dim))
        self.padding_token = nn.Parameter(torch.Tensor(historical_steps, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.Tensor(historical_steps + 1, 1, embed_dim))
        attn_mask = self.generate_square_subsequent_mask(historical_steps + 1)
        self.register_buffer('attn_mask', attn_mask)
        nn.init.normal_(self.padding_token, mean=0., std=.02)
        nn.init.normal_(self.cls_token, mean=0., std=.02)
        nn.init.normal_(self.pos_embed, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: torch.Tensor,
                bos_mask: torch.Tensor,
                padding_mask: torch.Tensor,
                rotate_mat: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        
        center_embed = self.center_embed(torch.matmul(x.unsqueeze(-2), rotate_mat.unsqueeze(1))).squeeze(-2)
        center_embed = torch.where(bos_mask.unsqueeze(-1), self.bos_token, center_embed)
        center_embed = center_embed + self.mlp(self.norm(center_embed))
        
        x = torch.where(padding_mask.t().unsqueeze(-1), self.padding_token, center_embed.transpose(0, 1))
        expand_cls_token = self.cls_token.expand(-1, x.shape[1], -1)
        x = torch.cat((x, expand_cls_token), dim=0) # [T+1, N, D]
        x = x + self.pos_embed
        
        out = self.transformer_encoder(src=x, mask=self.attn_mask, src_key_padding_mask=None)
        return out[-1]  # [N, D]

    @staticmethod
    def generate_square_subsequent_mask(seq_len: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-1e9')).masked_fill(mask == 1, float(0.0))
        return mask

class LaneNodeEncoder(nn.Module):

    def __init__(self,
                 node_dim: int,
                 embed_dim: int) -> None:
        super(LaneNodeEncoder, self).__init__()
        self.lane_mlp = nn.Sequential(nn.Linear(node_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim))
        self.apply(init_weights)

    def forward(self,
                lane_vector: torch.Tensor,
                rotate_mat: torch.Tensor) -> torch.Tensor:
        
        lane_embed = self.lane_mlp(torch.bmm(lane_vector.unsqueeze(-2), rotate_mat).squeeze(-2))
        
        return lane_embed

class TemporalEncoderLayer(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1) -> None:
        super(TemporalEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self,
                  x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor],
                  key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.relu_(self.linear1(x))))
        return self.dropout2(x)