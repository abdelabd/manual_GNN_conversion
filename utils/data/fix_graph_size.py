#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 19:12:47 2021

@author: abdel
"""

import os 
import numpy as np 
import itertools
from itertools import combinations
import torch 
from torch_geometric.data import DataLoader

#%%

def pad_nodes_and_edges(x, edge_attr, edge_index, n_want=112, m_want=148):
    n = x.shape[0]
    m = edge_attr.shape[0]
    
    n_diff = n_want - n
    m_diff = m_want - m
    
    x_prime = torch.cat([x, torch.zeros(n_diff, x.shape[1])], dim=0)
    edge_attr_prime = torch.cat([edge_attr, torch.zeros(m_diff, edge_attr.shape[1])], dim=0)
    
    edge_index_appendage = torch.zeros(2, m_diff, dtype=torch.int64)
    dummy_nodes = np.arange(n, n_want, 1)
    all_possible_dummy_indeces = [i for i in combinations(dummy_nodes, 2)]
    
    if len(all_possible_dummy_indeces)>=m_diff:
        dummy_indeces = all_possible_dummy_indeces[:m_diff]
        for i in range(m_diff):
            edge_index_appendage[0,i] = dummy_indeces[i][0]
            edge_index_appendage[1,i] = dummy_indeces[i][1]
        edge_index_prime = torch.cat([edge_index, edge_index_appendage], dim=1)        
        return x_prime, edge_attr_prime, edge_index_prime

    else: 
        print('bad graph')
        return None
