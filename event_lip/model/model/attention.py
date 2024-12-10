import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def positional_encoding(nodes, max_values):
   
    n, d = nodes.size()
    
    max_values = torch.tensor(max_values).cuda()
   
    normalized_nodes = 2 * (nodes / max_values) - 1
    
   
    encoded_nodes = torch.cat([torch.sin(normalized_nodes),
                               torch.cos(normalized_nodes)], dim=1)
    
    return encoded_nodes


class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttention, self).__init__()
        self.feature_dim = feature_dim
        self.query_weight = nn.Linear(feature_dim, feature_dim)
        self.key_weight = nn.Linear(feature_dim, feature_dim)
        self.value_weight = nn.Linear(feature_dim, feature_dim)
        
        
    def forward(self, x):
        # x: (batch_size, seq_len, feature_dim)
        query = self.query_weight(x)
        key = self.key_weight(x)
        value = self.value_weight(x)
        
       
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.feature_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        
        attended_values = torch.matmul(attention_weights, value)
        return attended_values
    


class SelfAttentionLayer(torch.nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttentionLayer, self).__init__()
        self.feature_dim = feature_dim
        self.query_weight = nn.Linear(feature_dim, feature_dim)
        self.key_weight = nn.Linear(feature_dim, feature_dim)
        self.value_weight = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, feature_dim)
        query = self.query_weight(x)
        key = self.key_weight(x)
        value = self.value_weight(x)
        
       
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.feature_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        
        
       
        attn_distribution = torch.mean(attention_weights, dim=2)
        return attn_distribution