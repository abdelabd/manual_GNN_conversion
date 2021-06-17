#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 19:13:24 2021

@author: abdel
"""

import os 
import re

import ctypes
import numpy.ctypeslib as npc

from collections import OrderedDict
import torch
import numpy as np
import yaml

import hls4ml
from hls4ml.model.hls_model import HLSModel, HLSModel_GNN
from hls4ml.utils.config import create_vivado_config
from hls4ml.writer.vivado_writer import VivadoWriter, VivadoWriter_GNN
from hls4ml.converters.pyg_to_hls import PygModelReader

# for locals
main_dir = "/home/abdel/IRIS_HEP/GNN_hls_layers"
os.chdir(main_dir)
from utils.prep_GNN_for_hls import prep_GNN_for_hls
from utils.data.dataset_pyg import GraphDataset

# model
from utils.models.interaction_network_pyg import InteractionNetwork as InteractionNetwork_pyg
model = InteractionNetwork_pyg()
model_dict = torch.load("/home/abdel/IRIS_HEP/interaction_networks/trained_models/IN_pyg_small_state_dict.pt")
model.load_state_dict(model_dict)

# data
indir = "/home/abdel/IRIS_HEP/trackml_data/processed_plus_pyg_small"
graph_files = np.array(os.listdir(indir))
graph_files = np.array([os.path.join(indir, graph_file)
                            for graph_file in graph_files])
graph_paths = [os.path.join(indir, filename)
                   for filename in graph_files]
n_graphs = len(graph_files)
IDs = np.arange(n_graphs)
np.random.seed(7)
np.random.shuffle(IDs)
params = {'batch_size': 1, 'shuffle': True, 'num_workers': 6}
dataset = GraphDataset(graph_files=graph_files[IDs])
del indir, graph_files, graph_paths, n_graphs, IDs, params

#%% hls_model

config, reader, layer_list = prep_GNN_for_hls(model)
hls_model = HLSModel_GNN(config, reader, layer_list)
hls_model.inputs = ['Re', 'Rn', 'edge_index']
hls_model.outputs = ['layer6_out_L']
hls_model.compile()

#%% top function

top_func_lib = hls_model._top_function_lib
top_func = getattr(top_func_lib, 'myproject_float')

#%% torch data

from utils.data.load_sample import load_sample
class data_wrapper(object):
    def __init__(self, Rn, Re, edge_index):
        self.x = Rn
        self.edge_attr = Re
        self.edge_index = edge_index
        
Rn_T, Re_T, edge_index_T, target_T = load_sample(dataset)
torch_pred_T = model(data_wrapper(Rn_T, Re_T, edge_index_T))
torch_pred = torch_pred_T.detach().cpu().numpy()
torch_pred_round = np.reshape(np.round(torch_pred)[:len(target_T)], newshape=(len(target_T),))
print("match:", sum(torch_pred_round==target_T.detach().cpu().numpy()))

#%% numpy, hls_model data

Rn, Re, edge_index, target = Rn_T.detach().cpu().numpy(), Re_T.detach().cpu().numpy(), edge_index_T.detach().cpu().numpy(), target_T.detach().cpu().numpy()

def mat_to_vec(mat):
    nrows, ncols = mat.shape
    out = np.zeros(shape=(nrows*ncols))
    i = 0
    for r in range(nrows):
        for c in range(ncols):
            out[i] = mat[r][c]
            i += 1
            
    return out

Rn_1D_test = mat_to_vec(Rn)

Re_1D = np.reshape(Re, newshape = (Re.shape[0]*Re.shape[1]))
Rn_1D = np.reshape(Rn, newshape = (Rn.shape[0]*Rn.shape[1]))
edge_index_1D = np.reshape(edge_index, newshape=(edge_index.shape[0]*edge_index.shape[1])).astype(np.int32)
hls_pred = np.zeros(shape=(torch_pred.shape[0],)).astype(np.float32)

def vec_to_mat(vec, n_rows, n_cols):
    i = 0
    out = np.zeros(shape=(n_rows, n_cols))
    for r in range(n_rows):
        for c in range(n_cols):
            out[r,c] = vec[i]
            i += 1
            
    return out

Re_2D_test = vec_to_mat(Re_1D, 148, 4)

#%% ctypes

Re_ctype = ctypes.c_float
Rn_ctype = ctypes.c_float
edge_index_ctype = ctypes.c_int
hls_pred_ctype = ctypes.c_float

#%% top function

top_func.restype = None
top_func.argtypes = [npc.ndpointer(Re_ctype), npc.ndpointer(Rn_ctype), npc.ndpointer(edge_index_ctype ), npc.ndpointer(hls_pred_ctype),
                     ctypes.POINTER(ctypes.c_ushort), ctypes.POINTER(ctypes.c_ushort), ctypes.POINTER(ctypes.c_ushort), ctypes.POINTER(ctypes.c_ushort)]

#%% test
edge_index = edge_index.astype(np.int32)

#%%
os.chdir(hls_model.config.get_output_dir() + '/firmware')
top_func(Re, Rn, edge_index, hls_pred, ctypes.byref(ctypes.c_ushort()), ctypes.byref(ctypes.c_ushort()), ctypes.byref(ctypes.c_ushort()), ctypes.byref(ctypes.c_ushort()))

#%%
def sigmoid(x):
    return 1/(1+np.exp(-x))

out = sigmoid(hls_pred)
    



