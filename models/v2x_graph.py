import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import LaplaceNLLLoss
from losses import SoftTargetCrossEntropyLoss
from torch.nn import CrossEntropyLoss
from metrics import ADE
from metrics import FDE
from metrics import MR
from torchmetrics import Precision, Recall
from models import NodeEncoder
from models import InterpretableGraph
from models import Decoder
from utils import TemporalData

class V2XGraph(pl.LightningModule):

    def __init__(self,
                 historical_steps: int,
                 future_steps: int,
                 num_modes: int,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 num_st_layers: int,
                 num_st_att_layers: int,
                 num_motion_layers: int,
                 num_mfg_layers: int,
                 num_cig_layers: int,
                 local_radius: float,
                 parallel: bool,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 cooperation: str,
                 **kwargs) -> None:
        super(V2XGraph, self).__init__()
        self.save_hyperparameters()
        self.cooperation = cooperation
        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.parallel = parallel
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max

        self.node_encoder = NodeEncoder(historical_steps=historical_steps,
                                        node_dim=node_dim,
                                        embed_dim=embed_dim,
                                        num_heads=num_heads,
                                        dropout=dropout,
                                        num_st_layers=num_st_layers,
                                        num_motion_layers=num_motion_layers)

        self.interpretable_graph = InterpretableGraph(historical_steps=historical_steps,
                                                      embed_dim=embed_dim,
                                                      node_dim=node_dim,
                                                      edge_dim=edge_dim,
                                                      num_modes=num_modes,
                                                      num_heads=num_heads,
                                                      num_st_att_layers=num_st_att_layers,
                                                      num_mfg_layers=num_mfg_layers,
                                                      num_cig_layers=num_cig_layers,
                                                      local_radius=local_radius,
                                                      dropout=dropout)

        self.decoder = Decoder(historical_steps=historical_steps,
                               embed_dim=embed_dim,
                               future_steps=future_steps,
                               num_modes=num_modes,
                               uncertain=True)

        self.reg_loss = LaplaceNLLLoss(reduction='mean')
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')
        self.edge_loss = CrossEntropyLoss()

        self.minADE = ADE()
        self.minFDE = FDE()
        self.minMR = MR()
        
        self.binary_precision = Precision(task='binary', average='macro', num_classes=2)
        self.recall = Recall(task='binary', average='macro', num_classes=2)
        
    def forward(self, data: TemporalData):
        rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
        sin_vals = torch.sin(data['rotate_angles'])
        cos_vals = torch.cos(data['rotate_angles'])
        rotate_mat[:, 0, 0] = cos_vals
        rotate_mat[:, 0, 1] = -sin_vals
        rotate_mat[:, 1, 0] = sin_vals
        rotate_mat[:, 1, 1] = cos_vals
        if data.y is not None:
            data.y = torch.bmm(data.y, rotate_mat)
        data['rotate_mat'] = rotate_mat # [N, 2, 2]
        
        if self.cooperation == 'ego':
            data.source_mask = data.ego_mask
            data.interact_mask = data.interact_ego_mask
        elif self.cooperation == 'v2i':
            data.source_mask = data.ego_mask | data.road_mask
            data.interact_mask = data.interact_ego_mask | data.interact_road_mask
        elif self.cooperation == 'v2v':
            data.source_mask = data.ego_mask | data.veh_mask
            data.interact_mask = data.interact_ego_mask | data.interact_veh_mask
        elif self.cooperation == 'v2x':
            data.source_mask = data.ego_mask | data.veh_mask | data.road_mask
            data.interact_mask = data.interact_ego_mask | data.interact_road_mask | data.interact_veh_mask
        else:
            raise ValueError(f'Invalid cooperation')

        st_embed, motion_embed = self.node_encoder(data=data)

        edge_out, mfg_embed, alg_embed, cig_embed = self.interpretable_graph(data=data,
                                                                             st_embed=st_embed,
                                                                             motion_embed=motion_embed)

        y_hat, pi = self.decoder(
            st_embed=st_embed, motion_embed=motion_embed, mfg_embed=mfg_embed, alg_embed=alg_embed, cig_embed=cig_embed)
        #return y_hat, pi    #[F, N, H, 2], [N, F]
        return y_hat, pi, edge_out

    def training_step(self, data):
        
        y_hat, pi, edge_out = self(data)

        #reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        reg_mask = ~data['padding_mask'][:, self.historical_steps:] & data.source_mask.unsqueeze(-1).repeat(1, self.historical_steps)
        valid_steps = reg_mask.sum(dim=-1)
        cls_mask = valid_steps > 0
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        soft_target = F.softmax(-l2_norm[:, cls_mask] / valid_steps[cls_mask], dim=0).t().detach()
        cls_loss = self.cls_loss(pi[cls_mask], soft_target)
        
        #filter_mask = data.v2x_aa_mask & data.v2x_type_mask
        filter_mask = data.v2x_ins_mask & data.v2x_type_mask
        filter_mask = filter_mask[data.v2x_mask]
        
        edge_out = edge_out[filter_mask, :]
        
        edge_label = torch.ones(data.edge_index.shape[1], dtype=torch.int64, device=data.edge_index.device)
        edge_label = (edge_label & data.v2x_pesudo_mask)[data.v2x_mask][filter_mask]

        edge_loss = self.edge_loss(edge_out, edge_label.to(torch.int64))
        loss = reg_loss + cls_loss + edge_loss
        
        self.log('train_reg_loss', reg_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('hard_loss', reg_loss + cls_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('soft_cls_loss', edge_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        
        return loss

    def validation_step(self, data, batch_idx):
        
        y_hat, pi, edge_out = self(data)
        
        #reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        reg_mask = ~data['padding_mask'][:, self.historical_steps:] & data.source_mask.unsqueeze(-1).repeat(1, self.historical_steps)
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        self.log('val_reg_loss', reg_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

        y_hat_agent = y_hat[:, data['agent_index'], :, : 2]
        y_agent = data.y[data['agent_index']]
        fde_agent = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
        best_mode_agent = fde_agent.argmin(dim=0)
        y_hat_best_agent = y_hat_agent[best_mode_agent, torch.arange(data.num_graphs)]

        self.minADE.update(y_hat_best_agent, y_agent)
        self.minFDE.update(y_hat_best_agent, y_agent)
        self.minMR.update(y_hat_best_agent, y_agent)
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log('val_minMR', self.minMR, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        
        filter_mask = data.v2x_ins_mask & data.v2x_type_mask
        filter_mask = filter_mask[data.v2x_mask]

        edge_out = edge_out[filter_mask, :].argmax(dim=1)
        edge_label = torch.ones(data.edge_index.shape[1], dtype=torch.int64, device=data.edge_index.device)
        edge_label = (edge_label & data.v2x_pesudo_mask)[data.v2x_mask][filter_mask]

        self.binary_precision.update(edge_out, edge_label)
        self.recall.update(edge_out, edge_label)
        self.log('Precision', self.binary_precision, prog_bar=True, on_step=False, on_epoch=True, batch_size=data.num_graphs)
        self.log('Recall', self.recall, prog_bar=True, on_step=False, on_epoch=True, batch_size=data.num_graphs)
        
    def predict_step(self, data, batch_idx):
        
        y_hat, pi, edge_out = self(data)
        
        y_hat_agent = y_hat[:, data.agent_index, :, : 2]
        y_agent = data.y[data.agent_index]
        fde_agent = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
        best_mode_agent = fde_agent.argmin(dim=0)
        y_hat_best_agent = y_hat_agent[best_mode_agent, torch.arange(data.num_graphs)]
 
        metrics = {
            'minADE': self.minADE(y_hat_best_agent, y_agent).item(),
            'minFDE': self.minFDE(y_hat_best_agent, y_agent).item(),
            'minMR': self.minMR(y_hat_best_agent, y_agent).item()
        }
        
        #save_output(data, y_hat, metrics, batch_idx)

        return self(data)

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)

        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('V2XGraph')
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--rotate', type=bool, default=True)
        parser.add_argument('--node_dim', type=int, default=2)
        parser.add_argument('--edge_dim', type=int, default=2)
        parser.add_argument('--embed_dim', type=int, default=128)
        parser.add_argument('--num_heads', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--num_st_layers', type=int, default=2)
        parser.add_argument('--num_st_att_layers', type=int, default=2)
        parser.add_argument('--num_motion_layers', type=int, default=4)
        parser.add_argument('--num_mfg_layers', type=int, default=3)
        parser.add_argument('--num_cig_layers', type=int, default=3)
        parser.add_argument('--local_radius', type=float, default=50)
        parser.add_argument('--parallel', type=bool, default=False)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        return parent_parser
