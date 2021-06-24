#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 19:14:35 2021

@author: abdel
"""

from .fix_graph_size import pad_nodes_and_edges

def load_sample(dataset, n_node_max=112, n_edge_max=148):
    
    got_one = False
    idx = 0
    while not got_one:

        data = dataset[idx]
        node_attr, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        if (node_attr.shape[0]<n_node_max) and (edge_attr.shape[0]<n_edge_max):
            node_attr_prime, edge_attr_prime, edge_index_prime = pad_nodes_and_edges(node_attr, edge_attr, edge_index, n_node_max=n_node_max, n_edge_max=n_edge_max)
            got_one = True

        elif idx >= len(dataset)-1:
            node_attr_prime, edge_attr_prime, edge_index_prime = None, None, None
            got_one = True

        else:
            idx += 1

    return node_attr_prime, edge_attr_prime, edge_index_prime, data.y


                    
                
    
    
    
