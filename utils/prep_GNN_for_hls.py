#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 14:48:38 2021

@author: abdel
"""

from hls4ml.utils.config import create_vivado_config
from hls4ml.converters.pyg_to_hls import PygModelReader

def prep_GNN_for_hls(model):
    n = 112 # num_nodes
    m = 148 # num_edges
    p = 3 # node_dim
    q = 4 # edge_dim
    r = 4 # effect_dim

    config = {
    "output_dir": "/home/abdel/IRIS_HEP/GNN_hls_layers/hls_output",
    "project_name": "myproject",
    "fpga_part": 'xcku115-flvb2104-2-i',
    "clock_period": 5,
    "io_type": "io_parallel"
    }
    config = create_vivado_config(**config)
    config['PytorchModel'] = model
    config['n_nodes'] = n
    config['n_edges'] = m
    config['node_dim'] = p
    config['edge_dim'] = q

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
    'n_nodes': n,
    'n_edges': m,
    'node_dim': p,
    'edge_dim': q,
    'out_features': q,
    'inputs': ['Re', 'Rn', 'edge_index'],
    'outputs': ["layer4_out_L", "layer4_out_Q"]
    }
    layer_list.append(R1_layer)

    O_layer = {
    'name': 'O',
    'class_name': 'NodeBlock',
    'n_nodes': n,
    'n_edges': m,
    'node_dim': p,
    'edge_dim': q,
    'out_features': p,
    'inputs': ['Rn', "layer4_out_Q"],
    'outputs': ["layer5_out_P"]
    }
    layer_list.append(O_layer)
    
    R2_layer = {
    'name': 'R2',
    'class_name': 'EdgeBlock',
    'n_nodes': n,
    'n_edges': m,
    'node_dim': p,
    'edge_dim': q,
    'out_features': 1,
    'inputs': ['layer4_out_L', 'layer5_out_P', 'edge_index'],
    'outputs': ['layer6_out_L', 'layer6_out_Q']
    }
    layer_list.append(R2_layer)

    return config, reader, layer_list
