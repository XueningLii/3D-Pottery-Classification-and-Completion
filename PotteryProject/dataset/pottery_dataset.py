import os
import random
import pandas as pd
import trimesh
import numpy as np
import torch
from torch.utils.data import Dataset
import utils.utils as utils
from utils.utils import distance_squre

class PotteryDataset(Dataset):
    def __init__(self, dataset_dir, labels_file, use_PF_Net = False, transform = None):
        self.dataset_dir = dataset_dir
        self.labels_df = pd.read_csv(labels_file, header=None, names=['file_name', 'label'])
        self.use_PF_Net = use_PF_Net
        self.transform = transform

        self.file_paths = {}
        for idx, row in self.labels_df.iterrows():
            file_name = row['file_name']
            broken_path = os.path.join(self.dataset_dir + 'broken', file_name)
            complete_path = os.path.join(self.dataset_dir + 'complete', file_name)
            if self.use_PF_Net:
              # add split path for PF_Net data
              split_path = os.path.join(self.dataset_dir,'split')
              self.file_paths[file_name] = {'broken': broken_path, 'complete': complete_path, 'split': split_path}
            else:
              self.file_paths[file_name] = {'broken': broken_path, 'complete': complete_path}

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 1000

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.labels_df.iloc[idx, 0]
        label = self.labels_df.iloc[idx, 1]
        label = torch.tensor(np.array([label]).astype(np.int32))

        paths = self.file_paths[file_name]

        broken_mesh = trimesh.load(paths['broken'])
        complete_mesh = trimesh.load(paths['complete'])

        broken_points = torch.tensor(broken_mesh.vertices, dtype=torch.float32)
        complete_points = torch.tensor(complete_mesh.vertices, dtype=torch.float32)

        # normalize
        # broken_points = self.normalize_to_unit_sphere(broken_points)
        # broken_points = self.min_max_normalize(broken_points)
        # complete_points = self.normalize_to_unit_sphere(complete_points)
        # complete_points = self.min_max_normalize(complete_points)

        broken_points = (broken_points - broken_points.min(dim=0)[0]) / (broken_points.max(dim=0)[0] - broken_points.min(dim=0)[0])
        complete_points = (complete_points - complete_points.min(dim=0)[0]) / (complete_points.max(dim=0)[0] - complete_points.min(dim=0)[0])
        
        if self.use_PF_Net:

            if idx in self.cache:
              print('load from cache! time saved!')
              real_center, real_center_key1, real_center_key2, input_cropped, input_cropped1, input_cropped2 = self.cache[idx]
            else:
              pt_file_name = file_name[:-4] + '.pt'
              real_center = torch.load(os.path.join(paths['split'], 'real_center', pt_file_name))
              real_center_key1 = torch.load(os.path.join(paths['split'], 'real_center_key1', pt_file_name))
              real_center_key2 = torch.load(os.path.join(paths['split'], 'real_center_key2', pt_file_name))
              input_cropped = torch.load(os.path.join(paths['split'], 'input_cropped', pt_file_name))
              input_cropped1 = torch.load(os.path.join(paths['split'], 'input_cropped1', pt_file_name))
              input_cropped2 = torch.load(os.path.join(paths['split'], 'input_cropped2', pt_file_name))

              if len(self.cache) < self.cache_size:
                  self.cache[idx] = (real_center, real_center_key1, real_center_key2, input_cropped, input_cropped1, input_cropped2)

            return real_center, real_center_key1, real_center_key2, input_cropped, input_cropped1, input_cropped2, label[0]
        else:
            return broken_points, complete_points, label[0]

    def transform(pointcloud):
        angle = torch.rand(1) * 360
        rotation_matrix = torch.tensor([
            [torch.cos(angle), -torch.sin(angle), 0],
            [torch.sin(angle), torch.cos(angle), 0],
            [0, 0, 1]
        ])
        pointcloud = torch.matmul(pointcloud, rotation_matrix)

        noise = torch.randn_like(pointcloud) * 0.01
        pointcloud += noise

        return pointcloud

    def normalize_to_unit_sphere(pc):
        centroid = pc.mean(dim=0)
        pc = pc - centroid
        m = torch.max(torch.sqrt(torch.sum(pc**2, dim=1)))
        pc = pc / m
        return pc

    def min_max_normalize(pc):
        pc_min = pc.min(dim=0)[0]
        pc_max = pc.max(dim=0)[0]
        pc = (pc - pc_min) / (pc_max - pc_min)
        return pc