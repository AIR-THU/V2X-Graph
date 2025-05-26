from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_weights

class Decoder(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 embed_dim: int,
                 future_steps: int,
                 num_modes: int,
                 uncertain: bool = True,
                 min_scale: float = 1e-3) -> None:
        super(Decoder, self).__init__()
        self.historical_steps = historical_steps
        self.input_size = embed_dim
        self.hidden_size = embed_dim
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.uncertain = uncertain
        self.min_scale = min_scale

        self.aggr_embed = nn.Sequential(
            nn.Linear(self.input_size * 5, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True))
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.future_steps * 2))
        if uncertain:
            self.scale = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.future_steps * 2))
        self.pi = nn.Sequential(
            nn.Linear(self.input_size * 5, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))
        self.st_proj = nn.Linear(self.input_size, self.input_size)
        self.motion_proj = nn.Linear(self.input_size, self.input_size)
        self.mfg_proj = nn.Linear(self.input_size, self.input_size)
        self.alg_multihead_proj = nn.Linear(self.input_size, num_modes * self.input_size)
        self.cig_multihead_proj = nn.Linear(self.input_size, num_modes * self.input_size)
        self.padding_token = nn.Parameter(torch.Tensor(1, self.hidden_size))
        nn.init.normal_(self.padding_token, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                st_embed: torch.Tensor,
                motion_embed: torch.Tensor,
                mfg_embed: torch.Tensor,
                alg_embed: torch.Tensor,
                cig_embed: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        st_embed = self.st_proj(st_embed).expand(self.num_modes, *motion_embed.shape)
        motion_embed = self.motion_proj(motion_embed).expand(self.num_modes, *motion_embed.shape)
        mfg_embed = self.mfg_proj(mfg_embed).expand(self.num_modes, *mfg_embed.shape)
        alg_embed = self.alg_multihead_proj(alg_embed).view(-1, self.num_modes, self.input_size)  # [N, F, D]
        alg_embed = alg_embed.transpose(0, 1)  # [F, N, D]
        cig_embed = self.cig_multihead_proj(cig_embed).view(-1, self.num_modes, self.input_size)  # [N, F, D]
        cig_embed = cig_embed.transpose(0, 1)  # [F, N, D]

        pi = self.pi(torch.cat((st_embed,
                                motion_embed,
                                mfg_embed,
                                alg_embed,
                                cig_embed
                                ), dim=-1)).squeeze(-1).t()
        out = self.aggr_embed(
            torch.cat((cig_embed, alg_embed, mfg_embed, motion_embed, st_embed), dim=-1))
        loc = self.loc(out).view(self.num_modes, -1, self.future_steps, 2)  # [F, N, H, 2]
        if self.uncertain:
            scale = F.elu_(self.scale(out), alpha=1.0).view(self.num_modes, -1, self.future_steps, 2) + 1.0
            scale = scale + self.min_scale  # [F, N, H, 2]
            return torch.cat((loc, scale), dim=-1), pi  # [F, N, H, 4], [N, F]
        else:
            return loc, pi  # [F, N, H, 2], [N, F]

