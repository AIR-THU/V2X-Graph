import os

from itertools import permutations
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data.makedirs import makedirs
from tqdm import tqdm

from utils import TemporalData
from utils import intersection, aabox_overlap
from utils import rowwise_in

import cv2

from dair_map_api import DAIRV2XMap

class TFDDataset(Dataset):

    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 local_radius: float = 50) -> None:
        self._split = split
        self._local_radius = local_radius
        if split == 'sample':
            self._directory = 'forecasting_sample'
        elif split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test_obs'
        else:
            raise ValueError(split + ' is not valid')
        self.root = root
        self._raw_file_names = os.listdir(self.raw_dir)
        self._processed_file_names = [os.path.splitext(f)[0] + '.pt' for f in self.raw_file_names]
        makedirs(self.processed_dir)
        self._resume_file_names = list(set(self._processed_file_names).difference(set(os.listdir(self.processed_dir))))
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]
        self._resume_paths = [os.path.join(self.processed_dir, f) for f in self._resume_file_names]
        super(TFDDataset, self).__init__(root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'v2x_fusion', self._directory, 'data')

    @property
    def road_raw_dir(self) -> str:
        return os.path.join(self.root, 'infrastructure-trajectories', self._directory, 'data')
    
    @property
    def traj_match_labels_dir(self) -> str:
        return os.path.join(self.root, 'traj_match_labels', self._directory, 'data')
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'v2x_fusion', self._directory, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def resume_file_names(self) -> List[str]:
        return [os.path.splitext(f)[0] + '.csv' for f in self._resume_file_names]

    @property
    def resume_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        downloading."""
        files = self.resume_file_names
        return [os.path.join(self.raw_dir, f) for f in files]

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def process(self) -> None:

        dm = DAIRV2XMap()
        for raw_path in tqdm(self.resume_paths):

            df_name = os.path.split(raw_path)[1]
            road_path = os.path.join(self.road_raw_dir, df_name)
            asso_path = os.path.join(self.traj_match_labels_dir, df_name)

            kwargs = process_argoverse(self._split, raw_path, dm, self._local_radius, road_path, asso_path)

            data = TemporalData(**kwargs)
            torch.save(data, os.path.join(self.processed_dir, str(kwargs['seq_id']) + '.pt'))

    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx) -> Data:
        return torch.load(self.processed_paths[idx])

def get_traj_match_labels(asso_path, actor_ids):
    asso_df = pd.read_csv(asso_path)
    asso_df.drop(asso_df.loc[asso_df.tag == 'AV'].index, inplace=True)
    timestamps = list(np.sort(asso_df['timestamp'].unique()))
    historical_timestamps = timestamps[: 50]
    asso_df = asso_df[asso_df['timestamp'].isin(historical_timestamps)]
    v2x_ids = list(asso_df.id.unique())

    relation_ls = []

    for id in v2x_ids:

        df_fusion = asso_df.loc[asso_df.id == id, ['id', 'ego_side_id', 'coop_side_id']]
        df_fusion.drop_duplicates(inplace=True)

        ego_id = df_fusion.ego_side_id.drop_duplicates(inplace=False).to_list()
        if -1 in ego_id:
            ego_id.remove(-1)
        coop_id = df_fusion.coop_side_id.drop_duplicates(inplace=False).to_list()
        if -1 in coop_id:
            coop_id.remove(-1)

        ego_side_ids = [actor_ids.index(iter) for iter in ego_id]
        coop_side_ids = [actor_ids.index(iter) for iter in coop_id]

        relation = list(product(coop_side_ids, ego_side_ids))
        relation.extend(list(product(ego_side_ids, coop_side_ids)))

        relation_ls += relation

    relation_ls = list(set(relation_ls))

    return relation_ls

def process_argoverse(split: str,
                      raw_path: str,
                      dm: DAIRV2XMap,
                      radius: float,
                      road_path: Optional[str] = None,
                      asso_path: Optional[str] = None) -> Dict:
    seq_id = os.path.splitext(os.path.basename(raw_path))[0]

    ego_df = pd.read_csv(raw_path)
    road_df = pd.read_csv(road_path)
    v2x_df = pd.concat([ego_df, road_df], ignore_index=True)

    timestamps = list(np.sort(v2x_df['timestamp'].unique()))
    historical_timestamps = timestamps[: 50]
    # filter out actors that are unseen during the historical time steps
    ego_historical_df = ego_df[ego_df['timestamp'].isin(historical_timestamps)]
    ego_df_id = ego_historical_df.id.drop_duplicates(inplace=False)
    
    road_historical_df = road_df[road_df['timestamp'].isin(historical_timestamps)]
    road_df_id = road_historical_df.id.drop_duplicates(inplace=False)
    
    vic_historical_df = v2x_df[v2x_df['timestamp'].isin(historical_timestamps)]
    actor_ids = list(vic_historical_df['id'].unique())
    v2x_df = v2x_df[v2x_df['id'].isin(actor_ids)]
    num_nodes = len(actor_ids)

    assert len(ego_df_id) + len(road_df_id) == num_nodes, seq_id
    assert ego_df_id.values[-1] < road_df_id.values[0], seq_id

    av_df = v2x_df[v2x_df['tag'] == 'AV'].iloc
    av_index = actor_ids.index(av_df[0]['id'])
    agent_df = v2x_df[v2x_df['tag'] == 'TARGET_AGENT'].iloc
    agent_index = actor_ids.index(agent_df[0]['id'])
    city = v2x_df['city'].values[0]

    # make the scene centered at AV
    origin = torch.tensor([av_df[49]['x'], av_df[49]['y']], dtype=torch.float64)
    theta = torch.tensor(av_df[49]['theta'], dtype=torch.float64)
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]], dtype=torch.float64)
    # initialization
    x = torch.zeros(num_nodes, 100, 2, dtype=torch.float64)
    last_positions = torch.zeros(num_nodes, 2, dtype=torch.float64)
    edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
    padding_mask = torch.ones(num_nodes, 100, dtype=torch.bool) # False means valid
    bos_mask = torch.zeros(num_nodes, 50, dtype=torch.bool)
    rotate_angles = torch.zeros(num_nodes, dtype=torch.float64)
    actor_type = torch.ones(num_nodes, dtype=torch.uint8)
    type_dict = {'None': 0, 'PEDESTRIAN': 1, 'BICYCLE': 2, 'VEHICLE': 3}

    ego_mask = torch.zeros(num_nodes, dtype=torch.bool)
    road_mask = torch.zeros(num_nodes, dtype=torch.bool)
    ego_mask[:len(ego_df_id)] = True
    road_mask[len(ego_df_id):] = True

    min_box_ls = [None] * num_nodes
    
    for actor_id, actor_df in v2x_df.groupby('id'):
        actor_hist_df = actor_df[actor_df['timestamp'].isin(historical_timestamps)]
        node_idx = actor_ids.index(actor_id)
        node_steps = [timestamps.index(timestamp) for timestamp in actor_df['timestamp']]
        padding_mask[node_idx, node_steps] = False  # has frame at this timestamp

        xy = torch.from_numpy(np.stack([actor_df['x'].values, actor_df['y'].values], axis=-1)).double()
        
        x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat)
        node_historical_steps = list(filter(lambda node_step: node_step < 50, node_steps))
        if len(node_historical_steps) > 1:  # calculate the heading of the actor (approximately)
            rotate_angles[node_idx] = actor_hist_df['theta'].values[-1]
        else:  # make no predictions for the actor if the number of valid time steps is less than 5
            padding_mask[node_idx, 50:] = True
        if len(actor_hist_df.type.values):
            actor_type[node_idx] = type_dict[actor_hist_df.type.values[-1]]
        else:
            actor_type[node_idx] = type_dict['None']
        
        last_positions[node_idx] = torch.from_numpy(
            np.stack([actor_hist_df['x'].values[-1], actor_hist_df['y'].values[-1]], axis=-1)).float()
        
        min_box_ls[node_idx] = cv2.boxPoints(cv2.minAreaRect(np.float32(x[node_idx, :50, :][~padding_mask[node_idx, :50]])))

    # bos_mask is True if time step t is valid and time step t-1 is invalid
    bos_mask[:, 0] = ~padding_mask[:, 0]
    bos_mask[:, 1: 50] = padding_mask[:, : 49] & ~padding_mask[:, 1: 50]

    positions = x.clone()
    
    x[:, 50:] = x[:, 50:] - x[:, 49].unsqueeze(-2) #vectorize?
    x[:, 1: 50] = torch.where((padding_mask[:, : 49] | padding_mask[:, 1: 50]).unsqueeze(-1),
                              torch.zeros(num_nodes, 49, 2),
                              x[:, 1: 50] - x[:, : 49]) # difference
    x[:, 0] = torch.zeros(num_nodes, 2)

    # get lane features at the current time step
    (lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index,
     lane_actor_vectors) = get_lane_features(
        dm,
        range(num_nodes),
        last_positions,
        origin, rotate_mat, city, radius)

    v2x_edge_index = get_traj_match_labels(asso_path, actor_ids)

    y = None if split == 'test' else x[:, 50:]

    v2x_all_index = list(product(range(len(ego_df_id)), range(len(ego_df_id), num_nodes)))
    v2x_all_index.extend(list(product(range(len(ego_df_id), num_nodes), range(len(ego_df_id)))))
    edge_aa_v2x_mask = torch.zeros(len(v2x_all_index), dtype=torch.bool).t().contiguous()
    edge_ins_v2x_mask = torch.zeros(len(v2x_all_index), dtype=torch.bool).t().contiguous()
    edge_type_v2x_mask = torch.zeros(len(v2x_all_index), dtype=torch.bool).t().contiguous()

    for idx, pair in enumerate(v2x_all_index):

        if actor_type[pair[0]] == actor_type[pair[1]]:
            edge_type_v2x_mask[idx] = True
        else:
            continue
        aa_iou = aabox_overlap(x[pair[0], :50, :][~padding_mask[pair[0], :50]],
                      x[pair[1], :50, :][~padding_mask[pair[1], :50]])
        if aa_iou > 0:
            edge_aa_v2x_mask[idx] = True
        else:
            continue

        ins_iou = intersection(min_box_ls[pair[0]], min_box_ls[pair[1]])
        if ins_iou > 0:
            edge_ins_v2x_mask[idx] = True

    v2x_all_index = torch.LongTensor(v2x_all_index).t().contiguous()
    
    if len(v2x_all_index.shape) > 1:
        edge_aa_index = v2x_all_index[:, edge_aa_v2x_mask]
        edge_ins_index = v2x_all_index[:, edge_ins_v2x_mask]
        edge_type_index = v2x_all_index[:, edge_type_v2x_mask]
    else:
        edge_aa_index = v2x_all_index
        edge_ins_index = v2x_all_index
        edge_type_index = v2x_all_index

    v2x_edge_index = torch.LongTensor(v2x_edge_index).t().contiguous()

    edge_ego = torch.LongTensor(list(permutations(range(len(ego_df_id)), 2))).t().contiguous()
    edge_road = torch.LongTensor(list(permutations(range(len(ego_df_id), num_nodes), 2))).t().contiguous()

    if len(v2x_all_index.shape) > 1:
        v2x_mask = rowwise_in(edge_index.t(), v2x_all_index.t())
        v2x_aa_mask = rowwise_in(edge_index.t(), edge_aa_index.t())
        v2x_ins_mask = rowwise_in(edge_index.t(), edge_ins_index.t())
        v2x_type_mask = rowwise_in(edge_index.t(), edge_type_index.t())
    else:
        v2x_mask = torch.zeros(len(edge_index.t()), dtype=torch.bool).t().contiguous()
        v2x_aa_mask = torch.zeros(len(edge_index.t()), dtype=torch.bool).t().contiguous()
        v2x_ins_mask = torch.zeros(len(edge_index.t()), dtype=torch.bool).t().contiguous()
        v2x_type_mask = torch.zeros(len(edge_index.t()), dtype=torch.bool).t().contiguous()
    
    if len(v2x_edge_index.shape) > 1:
        v2x_pesudo_mask = rowwise_in(edge_index.t(), v2x_edge_index.t())
    else:
        v2x_pesudo_mask = torch.zeros(len(edge_index.t()), dtype=torch.bool).t().contiguous()
    
    if len(v2x_edge_index.shape) > 1:
        v2x_pesudo_mask = rowwise_in(edge_index.t(), v2x_edge_index.t())
    else:
        v2x_pesudo_mask = torch.zeros(len(edge_index.t()), dtype=torch.bool).t().contiguous()
    
    if len(edge_ego.shape) > 1:
        interact_ego_mask = rowwise_in(edge_index.t(), edge_ego.t())
    else:
        interact_ego_mask = torch.zeros(len(edge_index.t()), dtype=torch.bool).t().contiguous()
    if len(edge_road.shape) > 1:
        interact_road_mask = rowwise_in(edge_index.t(), edge_road.t())
    else:
        interact_road_mask = torch.zeros(len(edge_index.t()), dtype=torch.bool).t().contiguous()
    
    return {
        'x': x[:, : 50].float(),  # [N, 50, 2]
        'positions': positions.float(),  # [N, 100, 2]
        'edge_index': edge_index,  # [2, N x N - 1]
        'v2x_edge_index': v2x_edge_index,
        'v2x_pesudo_mask': v2x_pesudo_mask,
        'v2x_mask': v2x_mask,
        'v2x_aa_mask': v2x_aa_mask,
        'v2x_ins_mask': v2x_ins_mask,
        'v2x_type_mask': v2x_type_mask,
        'interact_ego_mask': interact_ego_mask,
        'interact_road_mask': interact_road_mask,
        'actor_type': actor_type,
        'y': y.float(),  # [N, 50, 2]
        'num_nodes': num_nodes,
        'num_car_actors': len(ego_df_id),
        'num_road_actors': len(road_df_id),
        'ego_mask': ego_mask,
        'road_mask': road_mask,
        'padding_mask': padding_mask,  # [N, 100]
        'bos_mask': bos_mask,  # [N, 50]
        'rotate_angles': rotate_angles.float(),  # [N]
        'lane_vectors': lane_vectors.float(),  # [L, 2]
        'is_intersections': is_intersections,  # [L]
        'turn_directions': turn_directions,  # [L]
        'traffic_controls': traffic_controls,  # [L]
        'lane_actor_index': lane_actor_index,  # [2, E_{A-L}]
        'lane_actor_vectors': lane_actor_vectors.float(),  # [E_{A-L}, 2]
        'seq_id': int(seq_id),
        'av_index': av_index,
        'agent_index': agent_index,
        'city': city,
        'origin': origin.unsqueeze(0).float(),
        'theta': theta.float(),
        'last_positions': last_positions.double(),
    }


def get_lane_features(dm: DAIRV2XMap,
                      node_inds: List[int],
                      node_positions: torch.Tensor,
                      origin: torch.Tensor,
                      rotate_mat: torch.Tensor,
                      city: str,
                      radius: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                              torch.Tensor]:
    lane_positions, lane_vectors, is_intersections, turn_directions, traffic_controls = [], [], [], [], []
    lane_ids = set()
    for node_position in node_positions:
        lane_ids.update(dm.get_lane_ids_in_xy_bbox(node_position[0], node_position[1], city, radius))
    node_positions = torch.matmul(node_positions - origin, rotate_mat).double()
    for lane_id in lane_ids:
        lane_centerline = torch.from_numpy(dm.get_lane_segment_centerline(lane_id, city)[:, : 2]).double()
        lane_centerline = torch.matmul(lane_centerline - origin, rotate_mat)
        is_intersection = dm.lane_is_in_intersection(lane_id, city)
        turn_direction = dm.get_lane_turn_direction(lane_id, city)
        traffic_control = dm.lane_has_traffic_control_measure(lane_id, city)
        lane_positions.append(lane_centerline[:-1])
        lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])
        count = len(lane_centerline) - 1
        is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))
        if turn_direction == 'NONE':
            turn_direction = 0
        elif turn_direction == 'LEFT':
            turn_direction = 1
        elif turn_direction == 'RIGHT':
            turn_direction = 2
        elif turn_direction == 'UTURN':
            turn_direction = 3
        else:
            raise ValueError('turn direction is not valid')
        turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
        traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))
    lane_positions = torch.cat(lane_positions, dim=0)
    lane_vectors = torch.cat(lane_vectors, dim=0)
    is_intersections = torch.cat(is_intersections, dim=0)
    turn_directions = torch.cat(turn_directions, dim=0)
    traffic_controls = torch.cat(traffic_controls, dim=0)

    lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()
    lane_actor_vectors = \
        lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1)
    mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius
    lane_actor_index = lane_actor_index[:, mask]
    lane_actor_vectors = lane_actor_vectors[mask]

    return lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index, lane_actor_vectors
