import glob
import json
import os
import torch
from torch.utils.data import Dataset
from event_utils.cvtransforms import *


# https://github.com/uzh-rpg/rpg_e2vid/blob/d0a7c005f460f2422f2a4bf605f70820ea7a1e5f/utils/inference_utils.py#L480
def events_to_voxel_grid_pytorch(events, num_bins, width, height, device):
   

    assert (events.shape[1] == 4)
    assert (num_bins > 0)
    assert (width > 0)
    assert (height > 0)

    with torch.no_grad():
        events_torch = torch.from_numpy(events).float()
        events_torch = events_torch.to(device)
        

        voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32, device=device)

        
        voxel_grid = voxel_grid.flatten()
       
        last_stamp = events_torch[-1, 0]
        first_stamp = events_torch[0, 0]
       
        deltaT = float(last_stamp - first_stamp)

        if deltaT == 0:
            deltaT = 1.0

        events_torch[:, 0] = (num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
        ts = events_torch[:, 0]
       
        xs = events_torch[:, 1].long()
        ys = events_torch[:, 2].long()
        pols = events_torch[:, 3].float()
       
        pols[pols == 0] = -1  # polarity should be +1 / -1
       

        tis = torch.floor(ts)
        
        tis_long = tis.long()
        dts = ts - tis
        

        vals_left = pols * (1.0 - dts.float())
        vals_right = pols * dts.float()

        valid_indices = tis < num_bins
        valid_indices &= tis >= 0
        voxel_grid.index_add_(dim=0,
                              index=xs[valid_indices] + ys[valid_indices] * width + tis_long[
                                  valid_indices] * width * height,
                              source=vals_left[valid_indices])

        valid_indices = (tis + 1) < num_bins
        valid_indices &= tis >= 0

        voxel_grid.index_add_(dim=0,
                              index=xs[valid_indices] + ys[valid_indices] * width + (
                                          tis_long[valid_indices] + 1) * width * height,
                              source=vals_right[valid_indices])

        voxel_grid = voxel_grid.view(num_bins, height, width)

    return voxel_grid


def events_to_voxel_all(events, frame_nums, seq_len, num_bins, width, height, device):
    
    voxel_len = min(seq_len, frame_nums) * num_bins

    voxel_grid_all = np.zeros((num_bins * seq_len, 1, height, width))

    voxel_grid = events_to_voxel_grid_pytorch(events, voxel_len, width, height, device)
    voxel_grid = voxel_grid.unsqueeze(1).cpu().numpy()
    
    voxel_grid_all[:voxel_len] = voxel_grid
    
    return voxel_grid_all


class DVS_Lip(Dataset):
    def __init__(self, phase, event_args, mode='train'):        
        self.labels = sorted(os.listdir(os.path.join(event_args.event_root, phase)))
        self.length = event_args.seq_len
        self.phase = phase
        self.mode = mode
        self.args = event_args
        self.speech_speed_var = self.args.speech_speed_var
        self.net_type = self.args.net_type
        
        
       
        self.file_list = sorted(glob.glob(os.path.join(event_args.event_root, phase, '*', '*.npy')))
        self.file_list = [file.replace('\\', '/') for file in self.file_list]
        
        
         
        self.voxel_graph_list = sorted(glob.glob(os.path.join(event_args.voxel_graph_root, phase, '*', '*.pt')))
        
        self.voxel_graph_list = [file.replace('\\', '/') for file in self.voxel_graph_list]
        
        
        with open('/'.join(event_args.event_root.split('/')[:-1]) + '/frame_nums.json', 'r') as f:
            self.frame_nums = json.load(f)

    def __getitem__(self, index):
        # load timestamps

        word = self.file_list[index].split('/')[-2]
        person = self.file_list[index].split('/')[-1][:-4]
       

        frame_num = self.frame_nums[self.phase][word][int(person)]  
        voxel_graph = torch.load(self.voxel_graph_list[index])
        # load events

        events_input = np.load(self.file_list[index])
       
        events_input = events_input[np.where(
            (events_input['x'] >= 16) & (events_input['x'] < 112) & (events_input['y'] >= 16) & (
                        events_input['y'] < 112))]
        events_input['x'] -= 16
        events_input['y'] -= 16

        
        t, x, y, p = events_input['t'], events_input['x'], events_input['y'], events_input['p']
        events_input = np.stack([t, x, y, p], axis=-1)

        event_frame = events_to_voxel_all(events_input, frame_num, self.length, self.args.num_bins, 96, 96, device='cpu')  # (30*num_bins, 96, 96)

        if self.mode == 'train':
            event_frame = RandomCrop(event_frame, (88, 88))
            event_frame = HorizontalFlip(event_frame)
        else:
            event_frame = CenterCrop(event_frame, (88, 88))

        result = {
            'event_frame': torch.FloatTensor(event_frame),
            'label': self.labels.index(word),
        }
        return result,voxel_graph

    def __len__(self):
        return len(self.file_list)



