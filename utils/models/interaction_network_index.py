"""
Manual implementation of the (aggr=add, flow=source_to_target) torch_geometric.nn.conv.MessagePassing InteractioNetwork
"""
import os 
import time
import sys
import numpy as np 
import itertools

import torch 
import torch.nn as nn
from torch_scatter import scatter

#%%
 
class InteractionNetwork(nn.Module): # 'MP' for message-passing
    def __init__(self):
        
        super().__init__()
        #self.R1 = RelationalModel(10, 4, 40)
        #self.O = ObjectModel(7, 3, 40)
        #self.R2 = RelationalModel(10, 1, 40)
        
        self.R1 = nn.Sequential(
            nn.Linear(10, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Linear(40, 4),
        )
        
        self.O = nn.Sequential(
            nn.Linear(7, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Linear(40, 3),
        )
        
        self.R2 = nn.Sequential(
            nn.Linear(10, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Linear(40, 1),
        )
        
        self.p=3
        self.q=4
        self.node_dim=-2
        self.aggr="add"
        
    def message(self, x_i, x_j, edge_attr):      
        m1 = torch.cat([x_i, x_j, edge_attr], dim=1)
        self.E = self.R1(m1)
        return self.E
    
    def aggregate(self, inputs, index, ptr = None, dim_size = None):
        out = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)
        return out

    def update(self, aggr_out, x):
        c = torch.cat([x, aggr_out], dim=1)
        return self.O(c) 
    
    def forward(self, data):#x, edge_index, edge_attr):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        # Message
        x_i, x_j = x[edge_index[1]], x[edge_index[0]]
        msg_out = self.message(x_i, x_j, edge_attr) # self.message(edge_index, edge_attr)
        
        # Aggregate
        index = edge_index[1,:]
        ptr = None
        dim_size = x.shape[0]
        aggr_out = self.aggregate(msg_out, index, ptr, dim_size)
        
        # Update
        update_out = self.update(aggr_out, x)
        x_tilde = update_out
        
        #return update_out
        m2 = torch.cat([x_tilde[edge_index[1]],
                        x_tilde[edge_index[0]],
                        self.E], dim=1)
        return torch.sigmoid(self.R2(m2))


