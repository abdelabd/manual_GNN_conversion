#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 13:55:41 2021

@author: abdel
"""

import os 
import numpy as np 
import torch 
import hls4ml
from hls4ml.model.hls_model import HLSModel

from hls4ml.model.hls_layers import Layer, layer_map, register_layer
from hls4ml.templates import templates

# Locals (for development)
main_dir = "/home/abdel/IRIS_HEP/GNN_hls_layers"
os.chdir(main_dir)
from examples import layer_list, hls_model
graph = hls_model.graph

#%%
n = 112
m = 148
p = 3
q = 4
r = 4

accum_t = hls4ml.model.hls_layers.FixedPrecisionType()
class_name = "InputLayer"

Rn_layer_dict = {'name': 'Rn',
                 'class_name': class_name,
                 'accum_t': accum_t,
                 'input_shape': [n, p]}

Re_layer_dict = {'name': 'Re',
                  'class_name': class_name,
                  'accum_t': accum_t,
                  'input_shape': [m,q]}

edge_index_layer_dict = {'name': 'edge_index',
                         'class_name': class_name,
                         'accum_t': accum_t,
                         'input_shape': [2,m]}

