import os
import pdb
import csv
import numpy as np
import cv2
import torch
import random
# import pandas as pd
from tqdm import tqdm
import scipy.io as sio
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from spconv.pytorch.utils import PointToVoxel


if __name__ == '__main__':
    data_path = r'/yourpath//DVS-Lip/test'
    save_path = r'/yourpath/DVS-Lip-Voxel'
    device = torch.device("cuda:0")

    class_dir = os.listdir(data_path)
    for classID in range(len(class_dir)):
        cls = class_dir[classID]
        label = classID
        cls_name = cls
        fileLIST = os.listdir(os.path.join(data_path, cls))
        if not os.path.exists(os.path.join(save_path,'test', cls_name)):
            os.makedirs(os.path.join(save_path,'test',  cls_name))
        for FileID in tqdm(range(len(fileLIST))):
            file_Name = fileLIST[FileID]
            save_name = file_Name.split('.')[0]
            if os.path.exists(os.path.join(save_path,'t', cls_name, '{}.mat'.format(save_name))):   
                continue
            read_path = os.path.join(data_path, cls, file_Name)
            data = np.load(read_path)
            data = data[np.where(
                (data['x'] >= 16) & (data['x'] < 112) & (data['y'] >= 16) & (
                        data['y'] < 112))]          
            data['x'] -= 16
            data['y'] -= 16
            t, x, y, p = data['t'], data['x'], data['y'], data['p']
            p[p == 0] = -1      
            time_length = t[-1]-t[0]       
            t = ((t-t[0]) / time_length) * 60.0      
            data = np.stack([t, x, y, p], axis=-1)
            data = np.array([[a1, a2, a3, a4] for a1, a2, a3, a4 in data])
            data = torch.from_numpy(data).float()
            voxel_generator = PointToVoxel(
                vsize_xyz=[1, 4, 4],  
                coors_range_xyz=[0, 0, 0, 60, 96, 96], 
                max_num_points_per_voxel=30,  
                max_num_voxels=60000, 
                num_point_features=4
            )
            voxels, coordinates, num_points = voxel_generator(data)
            features = torch.zeros(60, 20, 30)
            coor = torch.zeros(60,20,3)
            for i in range(60):
                index_i = torch.where(coordinates[:, 2] == i)[0]
                if len(index_i) > 0:
                    voxels_i = voxels[index_i]
                    coor_i = coordinates[index_i]
                    num_points_i = num_points[index_i]
                    sorted_indices = torch.argsort(num_points_i, descending=True)[:20]
                    top_voxels = voxels_i[sorted_indices]
                    top_coor = coor_i[sorted_indices]
                    actual_num_voxels = top_voxels.shape[0]
                    features[i, :actual_num_voxels] = top_voxels[:,:,3]
                    coor[i,:actual_num_voxels] = top_coor
                          
            coor[:, [0, 1, 2]] = coor[:, [2, 1, 0]]
            coor = coor.cpu()
            features = features.cpu()
            coor = coor.numpy()
            features = features.numpy()
            sio.savemat(os.path.join(save_path , 'test', cls_name,  '{}.mat'.format(save_name)), mdict={'coor': coor, 'features': features})


            
# ######### P2V ##############
# Select voxel size [1,4,4], which divides into [602424] intervals. Each interval can contain up to 30 points (node feature dimension in the graph). Use a function to partition the voxel.
# Obtain voxel [N,30,4]: We have obtained N voxel boxes with points
# Each voxel box contains 30 points with coordinates [t,x,y,p] (padding with 0 if necessary). Based on the variable num_points, obtain the voxel box index voxels_idx for the top 20 voxel boxes with the most points (graph node count).
# For the top 20 voxel boxes with most points, read their p values as features, resulting in features [20,30] (retain the original number if the obtained voxel boxes are less than 20).
# Obtain the 3D coordinates coor of the voxel boxes (in the voxel coordinates [60,24,24]).
# Save coor and features.