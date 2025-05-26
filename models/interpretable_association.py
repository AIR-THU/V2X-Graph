import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics import Precision, Recall

from models.edge_encoder import STEdgeEncoder

from utils import TemporalData
from utils import init_weights


class InterpretableAssociation(pl.LightningModule):
    def __init__(self,
                 historical_steps: int,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 num_att_layers: int,
                 **kwargs) -> None:
        super(InterpretableAssociation, self).__init__()
        self.save_hyperparameters()

        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        
        self.edge_encoder = STEdgeEncoder(embed_dim=embed_dim,
                                            num_heads=num_heads,
                                            num_att_layers=num_att_layers,
                                            dropout=dropout)
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        self.classifier = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                        nn.ReLU(),
                                        nn.Linear(embed_dim, 2))
        
        self.cls_loss = torch.nn.CrossEntropyLoss()

        self.binary_precision = Precision(task='binary', average='macro', num_classes=2)
        self.recall = Recall(task='binary', average='macro', num_classes=2)

        self.apply(init_weights)


    def forward(self,
                data: TemporalData,
                st_embed):
        
        edge_index = data.edge_index
        st_edge_embed = self.edge_encoder(data, st_embed, edge_index)
        cls_out = self.classifier(self.layer_norm(st_edge_embed))
        edge_conf = F.softmax(input=cls_out, dim=1)[:, 1]
        edge_mask = edge_conf > 0.5
        edge_pred = data.edge_index[:, data.v2x_mask][:, edge_mask]
        data.edge_asso = edge_pred
        data.edge_conf = edge_conf
        data.edge_mask = edge_mask
        
        return cls_out, st_edge_embed
