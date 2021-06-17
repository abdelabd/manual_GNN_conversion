#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 19:14:35 2021

@author: abdel
"""
import os
import numpy as np
import itertools
from torch_geometric.data import Data, DataLoader

main_dir = "/home/abdel/IRIS_HEP"

from .fix_graph_size import pad_nodes_and_edges


#%%

def load_sample(dataset, n_max=112, m_max=148):
    
    got_one = False
    idx = 0
    while not got_one:
        
        if idx>=len(dataset)-1:
            got_one=True
            out = "somethings wrong"
            
        else: 
            data = dataset[idx]
            n = data.x.shape[0]
            m = data.edge_attr.shape[0]
            print(f"n: {n}")
            print(f"m: {m}")
            
            if n<=n_max and m<=m_max:
                out = pad_nodes_and_edges(data.x, data.edge_attr, data.edge_index, n_want=n_max, m_want=m_max)
            else:
                out = None
                
            if out==None:
                idx += 1
            else:
                got_one = True
    
    Rn, Re, edge_index = out

    return Rn, Re, edge_index, data.y


                    
                
    
    
    
