###################################################

import os
import pdb
import csv
import numpy as np
import random
import cv2
import torch
# import pandas as pd
from tqdm import tqdm
import scipy.io as sio
from torch_geometric.data import Data
from spconv.pytorch.utils import PointToVoxel
import math
def calculate_edges(coor, r=5):
    d = 32
    alpha = 1
    beta = 1
    data_size = coor.shape[0]
    edges = np.zeros([100000, 2])
    points = coor[:, 0:3]
    row_num = 0
    edge_sum=0
    for i in range(data_size - 1):
        count = 0
        distance_matrix = points[i + 1 : data_size + 1, 0:3]
        distance_matrix[:, 1:3] = distance_matrix[:, 1:3] - points[i, 1:3]
        distance_matrix[:, 0] = distance_matrix[:, 0] - points[i, 0]
        distance_matrix = np.square(distance_matrix)
        distance_matrix[:, 0] *= alpha
        distance_matrix[:, 1:3] *= beta
        distance = np.sqrt(np.sum(distance_matrix, axis=1))
        index = np.where(distance <= r)
        if index:
            index = index[0].tolist()
            for id in index:
                edges[row_num, 0] = i
                edges[row_num + 1, 1] = i
                edges[row_num, 1] = int(id) + i + 1
                edges[row_num + 1, 0] = int(id) + i + 1
                row_num = row_num + 2
                count = count + 1
                edge_sum+=2
                if count > d:
                    break
        if edge_sum>40000:
            break
    edges = edges[~np.all(edges == 0, axis=1)]
    edges = np.transpose(edges)
    return edges

def generate_graph(features,coor):
    position = np.copy(coor)
    edges = calculate_edges(position, 3)
    attr = np.linalg.norm((coor[edges[0].astype(int)] - coor[edges[1].astype(int)]), axis=1)
    attr = attr.reshape(-1,1)
    return features,coor,edges,attr


if __name__ == '__main__':
    data_path = '/yourpath/DVS-Lip-Voxel/test'
    save_path = '/yourpath/DVS-Lip-Voxelgraphlist'
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda:0")

    class_dir = os.listdir(data_path)
    class_dir.sort()
  
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
            if not os.path.exists(os.path.join(save_path,'test', cls_name)):
                os.makedirs(os.path.join(save_path,'test', cls_name))
            if os.path.exists(os.path.join(save_path,'test', cls_name, '{}.pt'.format(save_name))):
                continue
            read_path = os.path.join(data_path, cls,'{}.mat'.format(save_name))
            data = sio.loadmat(read_path)
            graph = []
            for i in range (30):
                feature_i = np.concatenate((data['features'][3*i], data['features'][3*i+1], data['features'][3*i+2]), axis=0)    
                coor_i = np.concatenate((data['coor'][3*i], data['coor'][3*i+1], data['coor'][3*i+2]), axis=0)            
                feature_i, position_i, edges_i, attr_i = generate_graph(feature_i, coor_i)   
                
                feature_i = torch.tensor(feature_i)[:, :].float()
                edge_index_i = torch.tensor(np.array(edges_i).astype(np.int32), dtype=torch.long)
                pos_i = torch.tensor(np.array(position_i), dtype=torch.float32)
                edges_attr_i = torch.tensor(np.array(attr_i), dtype=torch.float32)
                label_idx = torch.tensor(int(label), dtype=torch.long)   
                
                graph_i = Data(x=feature_i, edge_index=edge_index_i, pos=pos_i, y=label_idx.unsqueeze(0), edge_attr=edges_attr_i)
                graph.append(graph_i)

            G_save_path =os.path.join(os.path.join(save_path,'test', cls_name, '{}.pt'.format(save_name)))
            torch.save(graph, G_save_path)
        
        ########### V2G ######################
        # eg: 读取data内含coor[1000,3],包含1000个体素的t,x,y(在100，24，24的体素坐标系上）
        #                features[1000,100],内含1000个体素的，每个体素内100个点p
        #将每个体素视作图的结点
        #计算边：计算每个体素坐标和其他体素坐标的距离，得到距离矩阵distance[999,1](如下：)
        #    选取distance中小于5的部分有（38）个，将他们连成边，对所有体素做上述操作
        #    计算所有边的距离，将距离作为边的属性
        #    将图的结点特征，边，边属性保存
