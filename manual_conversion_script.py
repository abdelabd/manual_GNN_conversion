#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 13:46:22 2021

@author: abdel
"""

import os 
import torch

import hls4ml
from hls4ml.model.hls_model import HLSModel, HLSModel_GNN
from hls4ml.utils.config import create_vivado_config
from hls4ml.writer.vivado_writer import VivadoWriter, VivadoWriter_GNN
from hls4ml.converters.pyg_to_hls import PygModelReader

# locals
main_dir = "/home/abdel/IRIS_HEP/GNN_hls_layers" #change to repo directory 
os.chdir(main_dir)

# Model
from utils.models.interaction_network_pyg import InteractionNetwork as InteractionNetwork_pyg
model = InteractionNetwork_pyg()
model_dict = torch.load("IN_pyg_small_state_dict.pt")
model.load_state_dict(model_dict)
del model_dict 


#%%
def main():
    
    n = 112 # num_nodes
    m = 148 # num_edges
    p = 3 # node_dim
    q = 4 # edge_dim
    r = 4 # effect_dim

    config = {
    "output_dir": main_dir+"/hls_output",
    "project_name": "myproject",
    "fpga_part": 'xcku115-flvb2104-2-i',
    "clock_period": 5,
    "io_type": "io_parallel"
    }
    config = create_vivado_config(**config)
    config['PytorchModel'] = model
    config['n_node'] = n
    config['n_edge'] = m
    config['n_features'] = p
    config['e_features'] = q

    model_config = {
    'Precision': 'ap_fixed<16,6>',
    'ReuseFactor': 1,
    'Strategy': 'Latency'
    }

    config['HLSConfig']['Model'] = model_config

    layer_list = []
    reader = PygModelReader(config)
    input_shapes = reader.input_shapes
    output_shapes = {}

    EdgeAttr_layer = {
    'name': 'Re',
    'class_name': 'InputLayer',
    'input_shape': input_shapes['EdgeAttr'],
    'inputs': 'input'
    }
    layer_list.append(EdgeAttr_layer)

    NodeAttr_layer = {
    'name': 'Rn',
    'class_name': 'InputLayer',
    'input_shape': input_shapes['NodeAttr'],
    'inputs': 'input'
    }
    layer_list.append(NodeAttr_layer)

    EdgeIndex_layer = {
    'name': 'edge_index',
    'class_name': 'InputLayer',
    'input_shape': input_shapes['EdgeIndex'],
    'inputs': 'input'
    }
    layer_list.append(EdgeIndex_layer)

    R1_layer = {
    'name': 'R1',
    'class_name': 'EdgeBlock',
    'n_node': n,
    'n_edge': m,
    'n_features': p,
    'e_features': q,
    'out_features': q,
    'inputs': ['Re', 'Rn', 'edge_index'],
    'outputs': ["layer4_out_L", "layer4_out_Q"]
    }
    layer_list.append(R1_layer)

    O_layer = {
    'name': 'O',
    'class_name': 'NodeBlock',
    'n_node': n,
    'n_edge': m,
    'n_features': p,
    'e_features': q,
    'out_features': p,
    'inputs': ['Rn', "layer4_out_Q"],
    'outputs': ["layer5_out_P"]
    }
    layer_list.append(O_layer)

    R2_layer = {
    'name': 'R2',
    'class_name': 'EdgeBlock',
    'n_node': n,
    'n_edge': m,
    'n_features': p,
    'e_features': q,
    'out_features': 1,
    'inputs': ['layer4_out_L', 'layer5_out_P', 'edge_index'],
    'outputs': ['layer6_out_L', 'layer6_out_Q']
    }
    layer_list.append(R2_layer)
    
    hls_model = HLSModel_GNN(config, reader, layer_list)
    hls_model.inputs = ['Re', 'Rn', 'edge_index']
    hls_model.outputs = ['layer6_out_L']

    print("")
    print('Compiling')
    hls_model.compile()
    
if __name__ == '__main__':
    main()