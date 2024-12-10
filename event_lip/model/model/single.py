import torch
import torch.nn as nn
import math
from torch.cuda.amp import autocast
import torch.nn.functional as F
from model.tcn import MultibranchTemporalConvNet
from torch_geometric.nn import GMMConv,global_mean_pool
from model.model.attention import SelfAttention, positional_encoding, SelfAttentionLayer
import matplotlib.pyplot as plt


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
    
class VoxelBranch(nn.Module):
    def __init__(self, args):
        super(VoxelBranch, self).__init__()
        self.args = args
        # self.conv_0 = GMMConv(20, 32, dim=1, kernel_size=5)
        self.conv_0 = GMMConv(30, 32, dim=1, kernel_size=5)
        self.bn_0 = torch.nn.SyncBatchNorm(32)
        self.conv_node = GMMConv(3, 32, dim=1, kernel_size=5)
        self.bn_node = torch.nn.SyncBatchNorm(32)
        self.fc1 = torch.nn.Linear(1024,512)
        self.bn = torch.nn.SyncBatchNorm(512)
        self.drop_out = torch.nn.Dropout()
        self.fc2 = torch.nn.Linear(512, 100)
        self.block1 = ResidualBlock(32,64)
        self.block2 = ResidualBlock(64,128)
        self.block3 = ResidualBlock(128,256)
        self.block4 = ResidualBlock(256,512)
        # self.fusiontransformer = FusionTransformer(512,512,1024,4,8)
        self.gru = torch.nn.GRU(512, 1024, 3, batch_first=True, bidirectional=True, dropout=0.2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.v_cls = torch.nn.Linear(1024, self.args.n_class)
    # def forward(self, frame, voxel_graph):
    def forward(self, voxel_graph_list):
        
        num_frame = len(voxel_graph_list)
        batch_size = self.args.batch_size                                        #[B,30,1,96,96]
        num_voxels_total = voxel_graph_list[0].x.size(0)                        
        num_voxels_each = int(num_voxels_total/batch_size)
        voxel_graph_feat_list = torch.zeros((num_frame, self.args.batch_size, num_voxels_each, 512)).cuda()
        node_pos_feat_list = torch.zeros((num_frame, self.args.batch_size,32)).cuda()
        laterals=[]
        processed_features_l1= []
        processed_features_l2= []
        processed_features_l3= []
        processed_features_l4= []
        base_channals=64

        for i in range (num_frame):
            voxel_graph = voxel_graph_list[i]        
            voxel_graph.x = voxel_graph.x.cuda()                        #[total_num_voxels,max_points]            
            voxel_graph.edge_attr = voxel_graph.edge_attr.cuda()        #[2,num_edges(all)]
            voxel_graph.edge_index = voxel_graph.edge_index.cuda()      #[num_deges(all),1]
            voxel_graph.pos = voxel_graph.pos.cuda() 
            ####计算结点的绝对位置信息###
            node_pos_feat = F.elu(self.bn_node(self.conv_node(voxel_graph.pos, voxel_graph.edge_index, voxel_graph.edge_attr)))   #[total_num_voxels,32]
            node_pos_feat = node_pos_feat.reshape(batch_size, num_voxels_each ,32) 
            node_pos_feat = torch.mean(node_pos_feat, 1).squeeze(-1)
            node_pos_feat_list[i] = node_pos_feat       #[T,b,512]
            ####

            ###    
            voxel_graph.x= F.elu(self.bn_0(self.conv_0(voxel_graph.x, voxel_graph.edge_index, voxel_graph.edge_attr)))         #[total_num_voxels,64]
            
            voxel_graph = self.block1(voxel_graph)                      #[total_num_voxels,64]
            
            voxel_graph_single_feat_1 =voxel_graph.x.reshape(batch_size, num_voxels_each ,base_channals)    #[b,num_nodes,64]
            pooled = F.adaptive_avg_pool1d(voxel_graph_single_feat_1.permute(0, 2, 1), 1).squeeze(-1)
            processed_features_l1.append(pooled)
            # processed_features_l1.append(voxel_graph_single_feat_1)

            voxel_graph = self.block2(voxel_graph)                     #[total_num_voxels,256]
            
            voxel_graph_single_feat_2 =voxel_graph.x.reshape(batch_size, num_voxels_each ,base_channals*2)
            pooled = F.adaptive_avg_pool1d(voxel_graph_single_feat_2.permute(0, 2, 1), 1).squeeze(-1)
            processed_features_l2.append(pooled)
            # processed_features_l2.append(voxel_graph_single_feat_2)         ##[b,num_nodes,128]
        
            voxel_graph = self.block3(voxel_graph)                     #[total_num_voxels,512]
            
            voxel_graph_single_feat_3 =voxel_graph.x.reshape(batch_size, num_voxels_each ,base_channals*4)
            pooled = F.adaptive_avg_pool1d(voxel_graph_single_feat_3.permute(0, 2, 1), 1).squeeze(-1)
            processed_features_l3.append(pooled)
            # processed_features_l3.append(voxel_graph_single_feat_3)     ##[b,num_nodes,256]
            
            
            voxel_graph = self.block4(voxel_graph)                     #[total_num_voxels,512]
            voxel_graph_single_feat_4 =voxel_graph.x.reshape(batch_size, num_voxels_each ,base_channals*8)
            pooled = F.adaptive_avg_pool1d(voxel_graph_single_feat_4.permute(0, 2, 1), 1).squeeze(-1)     
            processed_features_l4.append(pooled)
            # processed_features_l4.append(voxel_graph_single_feat_4)     ###[b,num_nodes,512]
            
            
            
            voxel_graph_feat = voxel_graph.x.reshape(batch_size, num_voxels_each ,512)                  #[batch_size, num_voxels_each, 512]
            # voxel_graph_feat = global_mean_pool(voxel_graph.x,voxel_graph.batch)                      #[batchsize, 512]
            voxel_graph_feat_list[i] = voxel_graph_feat                                                 #[num_frame, batch_size, num_voxels_each, 512]
        
        final_features_l1 = torch.stack(processed_features_l1, dim=1)
        final_features_l2 = torch.stack(processed_features_l2, dim=1)
        final_features_l3 = torch.stack(processed_features_l3, dim=1)
        final_features_l4 = torch.stack(processed_features_l4, dim=1)
        
        laterals.append(final_features_l1)
        laterals.append(final_features_l2)
        laterals.append(final_features_l3)
        laterals.append(final_features_l4)
        
        voxel_graph_feat_list=voxel_graph_feat_list.transpose(0, 1)     
        node_pos_feat_list = node_pos_feat_list.transpose(0,1)
            
       
        x=voxel_graph.x
       
        return voxel_graph_feat_list,laterals,node_pos_feat_list
    
def conv1x3x3(in_planes, out_planes, stride=1):
    """1x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes,
                     kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 1, 1), stride=(1, stride, stride), bias=False)

class MFM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MFM, self).__init__()
        self.layer1 = nn.Sequential(
            conv1x1x1(in_channel, out_channel),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.local_att_layer = nn.Sequential(
            conv1x1x1(out_channel, out_channel//4),
            nn.BatchNorm3d(out_channel//4),
            nn.ReLU(inplace=True),
            conv1x1x1(out_channel//4, out_channel),
            nn.BatchNorm3d(out_channel)
        )
        self.global_att_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1x1(out_channel, out_channel//4),
            nn.BatchNorm3d(out_channel//4),
            nn.ReLU(inplace=True),
            conv1x1x1(out_channel//4, out_channel),
            nn.BatchNorm3d(out_channel)
        )
        self.sigmoid = nn.Sigmoid()
        self.adjust_channels = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        
        

    def forward(self, voxel_feat, frame_feat):
        
        H=frame_feat.shape[3]
        W=frame_feat.shape[3]
        C=frame_feat.shape[1]
        
        voxel_feat_expanded = voxel_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, W)
        
       
        b, t, c, h, w = voxel_feat_expanded.shape
        voxel_feat_adjusted = self.adjust_channels(voxel_feat_expanded.view(-1, c, h, w))
        voxel_feat_adjusted = voxel_feat_adjusted.view(b, C,t, h, w)
        
        
        fused_feat = frame_feat * 0.5 + voxel_feat_adjusted * 0.5  

        return fused_feat
    
    



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.se = se

        if self.se:
            self.gap = nn.AdaptiveAvgPool3d(1)
            self.conv3 = conv1x1x1(planes, planes//16)
            self.conv4 = conv1x1x1(planes//16, planes)


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # print(out.shape)# torch.Size([16, 16, 210, 22, 22])



        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        if self.se:
            w = self.gap(out)
            w = self.conv3(w)
            w = self.relu(w)
            w = self.conv4(w).sigmoid()

            out = out * w

        out = out + residual
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, block, layers, se=False, **kwargs):
        super(ResNet18, self).__init__()
        in_channels = kwargs['in_channels']

        # print(self.low_rate)

        self.base_channel = kwargs['base_channel']
        # self.inplanes = (self.base_channel + self.base_channel//self.alpha*self.t2s_mul) if self.low_rate else self.base_channel // self.alpha
        self.inplanes = 64
        # print(self.inplanes)
        self.conv1 = nn.Conv3d(in_channels, self.base_channel,
                               kernel_size=(5, 7, 7),
                               stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(self.base_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.se = se
        self.layer1 = self._make_layer(block, self.base_channel, layers[0])
        self.layer2 = self._make_layer(block, 2 * self.base_channel, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * self.base_channel, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * self.base_channel, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)


        self.bn2 = nn.BatchNorm1d(8 * self.base_channel)
        # if self.low_rate:
        #     self.bn2 = nn.BatchNorm1d(8*self.base_channel + 8*self.base_channel//self.alpha*self.t2s_mul)
        # elif self.t2s_mul == 0:
        #     self.bn2 = nn.BatchNorm1d(16*self.base_channel//self.alpha)
        self.init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        # print(self.inplanes)
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, se=self.se))
        self.inplanes = planes * block.expansion
        # print(planes)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, se=self.se))

        # self.inplanes += self.low_rate * block.expansion * planes // self.alpha * self.t2s_mul

        return nn.Sequential(*layers)

    

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


#
#
class LowRateBranch(ResNet18):
    def __init__(self, block, layers, se, n_frame=30, **kargs):
        super().__init__(block, layers, se, n_frame=n_frame, **kargs)
        self.base_channel = kargs['base_channel']
        self.init_params()
        self.mfm1 = MFM(in_channel=128,
                        out_channel=64)
        self.mfm2 = MFM(in_channel=256,
                        out_channel=128)
        self.mfm3 = MFM(in_channel=512,
                        out_channel=256)
        self.mfm4 = MFM(in_channel=4 * self.base_channel,
                        out_channel=4*self.base_channel)
        
        self.fvf1= FVFusionNetwork(voxel_channels=64,
                                   frame_channels=64,
                                   output_channels=64)
        self.fvf2= FVFusionNetwork(voxel_channels=128,
                                   frame_channels=128,
                                   output_channels=128)
        self.fvf3= FVFusionNetwork(voxel_channels=256,
                                   frame_channels=256,
                                   output_channels=256)
        self.fvf4= FVFusionNetwork(voxel_channels=512,
                                   frame_channels=512,
                                   output_channels=512)
    def forward(self, x,laterals):
        x=x.cuda()
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.shape)# torch.Size([32, 64, 30, 22, 22])
        # print(laterals[0].shape)# torch.Size([32, 32, 30, 22, 22])
        
        # x:b 64 30 22 22
        # voxel : b 30 128
        
        
        
        # x = torch.cat([x, laterals[0]], dim=1)
        # print(x.shape)# torch.Size([32, 96, 30, 22, 22])
        x = self.layer1(x) # (b, 64, 30, 22, 22)
        
        # laterals[0]  b 30 20 128
        x= self.fvf1(laterals[0], x)
        # x = self.mfm1(laterals[0], x)  # torch.Size([32, 64, 30, 22, 22])


       
        # x = torch.cat([x, laterals[1]], dim=1) # (b, 80, 30, 22, 22)
        x = self.layer2(x) # (b, 128, 30, 11, 11)
        
        x= self.fvf2(laterals[1], x)
        # x = self.mfm2(laterals[1], x)
        
        
        # b 128 30 11 11
        x = self.layer3(x) # (b, 256, 30, 6, 6)
        
        x= self.fvf3(laterals[2], x)
        # x = self.mfm3(laterals[2], x)
        # x = torch.cat([x, laterals[2]], dim=1) # (b, 160, 30, 11, 11)
        

        # x = self.mfm4(laterals[3], x)
        # x = torch.cat([x, laterals[3]], dim=1) # (b, 320, 30, 6, 6)
        x = self.layer4(x) # (b, 512, 30, 3, 3)
        
        x= self.fvf4(laterals[3], x)

        # x = torch.cat([x, laterals[4]], dim=1) # (b, 640, 30, 3, 3)
        x = self.avgpool(x) # (b, 640, 30, 1, 1)

        x = x.transpose(1, 2).contiguous() # (b, 30, 640, 1, 1)
        x = x.view(-1, x.size(2)) # (b*30, 640)
        x = self.bn2(x)

        return x
#

class MultiBranchNet(nn.Module):
    def __init__(self, args):
        super(MultiBranchNet, self).__init__()
        self.args = args
        self.low_rate_branch = LowRateBranch(block=BasicBlock,
                                             layers=[2, 2, 2, 2],
                                             se=args.se,
                                             in_channels=1,
                                             low_rate=1,
                                             alpha=args.alpha,
                                             t2s_mul=args.t2s_mul,
                                             beta=args.beta,
                                             base_channel=args.base_channel)
        self.voxel_branch = VoxelBranch(args=args)

    def forward(self,data,voxel):
        x=data
        b = x.size(0)
        laterals=[]
        voxel_feat, laterals, node_pos_feat_list = self.voxel_branch(voxel)  #voxel_feat 1 60 20 512        #laterals[i] [4,30,20,64]
        x = self.low_rate_branch(x,laterals)
        x = x.view(b, -1, 8 * self.args.base_channel)       #[b,t,c]
       
        return x,node_pos_feat_list
        


class MTGA(nn.Module):
    def __init__(self, args):
        super(MTGA, self).__init__()
        self.args = args
        self.mbranch = MultiBranchNet(args)
        self.add_channel = 1 if self.args.word_boundary else 0

        if self.args.back_type == 'TCN':
            tcn_options = {'num_layers': 4,
                           'kernel_size': [3, 5, 7],
                           'dropout': 0.2,
                           'dwpw': False,
                           'width_mult': 1
                           }
            self.TCN = MultibranchTemporalConvNet(num_inputs=512 + self.add_channel, num_channels=[768, 768, 768, 768], tcn_options=tcn_options)
            self.v_cls = nn.Linear(768, self.args.n_class)
        elif self.args.back_type == 'GRU':
            # self.gru = nn.GRU(512 + self.add_channel+ 32, 1024, 3, batch_first=True, bidirectional=True, dropout=0.2)
            self.gru = nn.GRU(512, 1024, 3, batch_first=True, bidirectional=True, dropout=0.2)
            self.v_cls = nn.Linear(1024 * 2+32, self.args.n_class)

        self.dropout = nn.Dropout(p=0.5)
        # args.base_channel = 64, self.args.alpha = 4, self.args.t2s_mul = 2
        # in_dim = 8 * args.base_channel + 8 * args.base_channel // self.args.alpha * self.args.t2s_mul
        # in_dim = 8 * args.base_channel
        self.attention = SelfAttention(feature_dim=2048+32)
        
        
        self.self_attention_layer = SelfAttentionLayer(2048+32)

        # self.accumulated_features = []  # 用于存储特征
        # self.accumulated_labels = []    # 用于存储标签
        
    def forward(self, data,voxel):
        event_frame = data['event_frame']
        if self.training:
            with autocast():
                feat,voxel_feat = self.mbranch(event_frame,voxel) 

                # feat 
                feat = self.dropout(feat)
                feat = feat.float()
        else:
            feat,voxel_feat = self.mbranch(event_frame,voxel)
            feat = feat.float()

    
        self.gru.flatten_parameters()
        feat, _ = self.gru(feat)
        
        feat = torch.cat([feat, voxel_feat], dim=2)
        

        feat=self.attention(feat)  #[B T 2048+32]
        logit = self.v_cls(self.dropout(feat)).mean(1)

        return logit



class FVFusionNetwork(nn.Module):
    def __init__(self, voxel_channels, frame_channels, output_channels):
        super(FVFusionNetwork, self).__init__()
        
        self.adjust_channels = nn.Conv2d(voxel_channels, frame_channels, kernel_size=1)
        self.attention_net = nn.Sequential(
            nn.Conv2d(frame_channels * 2, frame_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(frame_channels, 1, kernel_size=1)
        )
        self.output_conv = nn.Conv2d(frame_channels, output_channels, kernel_size=1)
        self.fusion_conv = nn.Conv2d(frame_channels * 2, frame_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(frame_channels)
        self.relu = nn.ReLU()
        self.residual_conv = nn.Conv2d(frame_channels * 2, frame_channels, kernel_size=1, bias=False)
        
    def forward(self,  voxel_feat,frame_feat):
        B, C, H, W =frame_feat.shape[0], frame_feat.shape[1], frame_feat.shape[-2], frame_feat.shape[-1]
        T=frame_feat.shape[-3]
        voxel_feat_expanded = voxel_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, W)   # [b,t,c,H,W]
        b, t, c, h, w = voxel_feat_expanded.shape
        voxel_feat_expanded = voxel_feat_expanded.view(b, C,t, h, w) 
        output = []
        for t in range(T): 
            x1 = frame_feat[:, :,t, :, :]
            x2 = voxel_feat_expanded[:, :, t, :, :] 
            combined = torch.cat((x1, x2), dim=1)
            residual = self.residual_conv(combined)  
            fused_frame = self.fusion_conv(combined)
            fused_frame = self.bn(fused_frame)
            fused_frame = self.relu(fused_frame)
            fused_frame = fused_frame + residual
            output.append(fused_frame.unsqueeze(1))
        output = torch.cat(output, dim=1)
        output = output.permute(0, 2, 1,3,4)
        return output
    
    
        

        
