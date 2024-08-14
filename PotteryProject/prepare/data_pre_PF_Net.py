import os
import torch
import random
import trimesh
from pathlib import Path
import utils.utils as utils
from utils.utils import distance_squre

def data_pre(input_folder): 
    crop_point_num = 512

    split_folder = os.path.join(input_folder, 'split')
    real_center_folder = os.path.join(split_folder, 'real_center')
    real_center_key1_folder = os.path.join(split_folder, 'real_center_key1')
    real_center_key2_folder = os.path.join(split_folder, 'real_center_key2')
    input_cropped_folder = os.path.join(split_folder, 'input_cropped')
    input_cropped1_folder = os.path.join(split_folder, 'input_cropped1')
    input_cropped2_folder = os.path.join(split_folder, 'input_cropped2')

    os.makedirs(split_folder, exist_ok=True)
    os.makedirs(real_center_folder, exist_ok=True)
    os.makedirs(real_center_key1_folder, exist_ok=True)
    os.makedirs(real_center_key2_folder, exist_ok=True)
    os.makedirs(input_cropped_folder, exist_ok=True)
    os.makedirs(input_cropped1_folder, exist_ok=True)
    os.makedirs(input_cropped2_folder, exist_ok=True)

    ply_files = Path(os.path.join(input_folder, 'complete')).rglob('*.ply')

    for ply_file in ply_files:
        
        complete_mesh = trimesh.load(ply_file)
        complete_points = torch.tensor(complete_mesh.vertices, dtype=torch.float32)
        complete_points = (complete_points - complete_points.min(dim=0)[0]) / (complete_points.max(dim=0)[0] - complete_points.min(dim=0)[0])
        
        # real_center and input_cropped
        real_center = torch.zeros(1, crop_point_num, 3)  
        input_cropped = complete_points.clone()  
        
        # Add random view cropping  
        choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]), torch.Tensor([-1, 0, 0]), torch.Tensor([-1, 1, 0])]
        index = random.sample(choice, 1)  
        distance_list = []
        p_center = index[0]
        
        # Calculate the distance from each point to the view
        distance_list = torch.norm(complete_points - p_center, dim=1)
        # Sort by distance
        distance_order = torch.argsort(distance_list)
        
        # input_cropped and real_center got cropped 
        for sp in range(crop_point_num):
            input_cropped[distance_order[sp]] = torch.zeros(3)  
            real_center[0, sp] = complete_points[distance_order[sp]]  

        real_center = torch.squeeze(real_center, 0)
        
        # print(real_center.shape)
        # print(input_cropped.shape)
        
        # sample for real_center
        real_center_key1_idx = utils.farthest_point_sample_pc(real_center, 64, RAN=False)
        real_center_key1 = utils.index_points_pc(real_center, real_center_key1_idx)
        real_center_key2_idx = utils.farthest_point_sample_pc(real_center, 128, RAN=True)
        real_center_key2 = utils.index_points_pc(real_center, real_center_key2_idx)
        
        # sample for input_cropped
        input_cropped1_idx = utils.farthest_point_sample_pc(input_cropped, 1024, RAN=True)  
        input_cropped1 = utils.index_points_pc(input_cropped, input_cropped1_idx)
        input_cropped2_idx = utils.farthest_point_sample_pc(input_cropped, 512, RAN=False)  
        input_cropped2 = utils.index_points_pc(input_cropped, input_cropped2_idx)

        # get file name
        file_name = os.path.splitext(os.path.basename(ply_file))[0]

        # save path
        output_folder = split_folder

        # save files
        torch.save(real_center, os.path.join(output_folder, 'real_center', f'{file_name}.pt'))
        torch.save(real_center_key1, os.path.join(output_folder, 'real_center_key1', f'{file_name}.pt'))
        torch.save(real_center_key2, os.path.join(output_folder, 'real_center_key2', f'{file_name}.pt'))
        torch.save(input_cropped, os.path.join(output_folder, 'input_cropped', f'{file_name}.pt'))
        torch.save(input_cropped1, os.path.join(output_folder, 'input_cropped1', f'{file_name}.pt'))
        torch.save(input_cropped2, os.path.join(output_folder, 'input_cropped2', f'{file_name}.pt'))

        print(f'Processed: {file_name}')