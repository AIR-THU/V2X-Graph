import torch
import torch.nn as nn

from models import InterpretableAssociation
from models import MotionFusionSubGraph
from models import AgentLaneSubGraph
from models import CooperativeInteractionSubGraph

from utils import TemporalData


class InterpretableGraph(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 embed_dim: int,
                 node_dim: int,
                 edge_dim: int,
                 num_modes: int = 6,
                 num_heads: int = 8,
                 num_st_att_layers: int = 2,
                 num_mfg_layers: int = 3,
                 num_cig_layers: int = 3,
                 local_radius: int = 50,
                 dropout: float = 0.1) -> None:
        super(InterpretableGraph, self).__init__()
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_modes = num_modes
        
        self.interpretable_association = InterpretableAssociation(historical_steps=historical_steps,
                                            embed_dim=embed_dim,
                                            num_heads=num_heads,
                                            dropout=dropout,
                                            num_att_layers=num_st_att_layers)
        
        self.motion_fusion_subgraph = MotionFusionSubGraph(historical_steps=historical_steps,
                                                  embed_dim=embed_dim,
                                                  num_heads=num_heads,
                                                  num_layers=num_mfg_layers,
                                                  dropout=dropout)
        
        self.agent_lane_subgraph = AgentLaneSubGraph(node_dim=node_dim,
                                    edge_dim=edge_dim,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    dropout=dropout,
                                    local_radius=local_radius)
        
        self.cooperative_interaction_subgraph = CooperativeInteractionSubGraph(historical_steps=historical_steps,
                                                  embed_dim=embed_dim,
                                                  edge_dim=edge_dim,
                                                  num_modes=num_modes,
                                                  num_heads=num_heads,
                                                  num_layers=num_cig_layers,
                                                  dropout=dropout)

    def forward(self,
                data: TemporalData,
                st_embed: torch.Tensor,
                motion_embed: torch.Tensor
                ) -> torch.Tensor:

        edge_pred, st_edge_embed = self.interpretable_association(data=data, st_embed=st_embed)
        
        mfg_embed = self.motion_fusion_subgraph(data=data, x=motion_embed, edge_embed=st_edge_embed)

        alg_embed = self.agent_lane_subgraph(data=data, mfg_embed=mfg_embed)

        cig_embed = self.cooperative_interaction_subgraph(data=data, alg_embed=alg_embed)   # [F, N, D]
        
        return edge_pred, mfg_embed, alg_embed, cig_embed