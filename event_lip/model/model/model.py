import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GMMConv,global_mean_pool

from .single import MTGA as single

def conv1x3x3(in_planes, out_planes, stride=1):
    """1x3x3 convolution with padding"""
    return torch.nn.Conv3d(in_planes, out_planes,
                     kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution with padding"""
    return torch.nn.Conv3d(in_planes, out_planes, kernel_size=(1, 1, 1), stride=(1, stride, stride), bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResidualBlock, self).__init__()
        self.left_conv1 = GMMConv(in_channel, out_channel, dim=1, kernel_size=5)
        self.left_bn1 = torch.nn.SyncBatchNorm(out_channel)
        self.left_conv2 = GMMConv(out_channel, out_channel, dim=1, kernel_size=5)
        self.left_bn2 = torch.nn.SyncBatchNorm(out_channel)

        self.shortcut_conv = GMMConv(in_channel, out_channel, dim=1, kernel_size=1)
        self.shortcut_bn = torch.nn.SyncBatchNorm(out_channel)

    def forward(self, data):
        data.x = F.elu(self.left_bn2(
            self.left_conv2(F.elu(self.left_bn1(self.left_conv1(data.x, data.edge_index, data.edge_attr))),
                            data.edge_index, data.edge_attr)) + self.shortcut_bn(
            self.shortcut_conv(data.x, data.edge_index, data.edge_attr)))

        return data

######################################################################################################
class FusionTransformer(nn.Module):
    def __init__(self, frame_feature_size, voxel_feature_size, hidden_size, num_heads, num_layers):
        super(FusionTransformer, self).__init__()

        self.frame_encoder = torch.nn.Linear(frame_feature_size, hidden_size)
        self.voxel_encoder = torch.nn.Linear(voxel_feature_size, hidden_size)
        self.transformer = torch.nn.Transformer(hidden_size, num_heads, num_layers)

    def forward(self, frame_features, voxel_features):
        # Encode frame and voxel features
        encoded_frame = self.frame_encoder(frame_features)
        encoded_voxel = self.voxel_encoder(voxel_features)

        # Reshape encoded features for transformer input
        encoded_frame = encoded_frame.permute(1, 0, 2)  # (sequence_length, batch_size, hidden_size)
        encoded_voxel = encoded_voxel.permute(1, 0, 2)  # (sequence_length, batch_size, hidden_size)

        # Apply transformer fusion
        fused_features = self.transformer(encoded_frame, encoded_voxel)

        # Reshape fused features back to original shape
        fused_features = fused_features.permute(1, 0, 2)  # (batch_size, sequence_length, hidden_size)

        return fused_features

        


class MTGA(nn.Module):
    def __init__(self, args):
        super(MTGA, self).__init__()
        self.args = args

        self.net = single(args)


    def forward(self, data,voxel):
        return self.net(data,voxel)


