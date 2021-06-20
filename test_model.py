#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 22:14:23 2021

@author: abdel
"""
import os 
import yaml 
import argparse


import ctypes
import numpy.ctypeslib as npc
import numpy as np
import torch
import hls4ml

from hls4ml.model.hls_model import HLSModel_GNN
from hls4ml.utils.config import create_vivado_config
from hls4ml.writer.vivado_writer import VivadoWriter_GNN
from hls4ml.converters.pyg_to_hls import PygModelReader

# locals
from utils.prep_GNN_for_hls import prep_GNN_for_hls
from utils.data.dataset_pyg import GraphDataset
from utils.models.interaction_network_pyg import InteractionNetwork as InteractionNetwork_pyg
from utils.data.load_sample import load_sample

#%%

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='test_config.yaml')
    return parser.parse_args()
    
def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)
    
    # data
    graph_indir = config['graph_indir'] 
    graph_files = np.array(os.listdir(graph_indir))
    graph_files = np.array([os.path.join(graph_indir, graph_file)
                            for graph_file in graph_files])
    graph_paths = [os.path.join(graph_indir, filename)
                   for filename in graph_files]
    n_graphs = len(graph_files)
    IDs = np.arange(n_graphs)
    params = {'batch_size': 1, 'shuffle': False, 'num_workers': 6}
    dataset = GraphDataset(graph_files=graph_files[IDs])
    
    
    # torch model
    model = InteractionNetwork_pyg()
    model_dict_dir = config['model_dict_dir']
    model_dict = torch.load(model_dict_dir)
    model.load_state_dict(model_dict)
    
    
    # hls model
    config, reader, layer_list = prep_GNN_for_hls(model)
    hls_model = HLSModel_GNN(config, reader, layer_list)
    hls_model.inputs = ['edge_attr', 'node_attr', 'edge_index']
    hls_model.outputs = ['layer6_out_L']
    hls_model.compile()
    
    
    # sample data (torch.Tensor)
    Rn_T, Re_T, edge_index_T, target_T = load_sample(dataset)
    class data_wrapper(object):
        def __init__(self, Rn, Re, edge_index):
            self.x = Rn
            self.edge_attr = Re
            self.edge_index = edge_index
    torch_pred_T = model(data_wrapper(Rn_T, Re_T, edge_index_T))
    
    # test torch model inference
    torch_pred = torch_pred_T.detach().cpu().numpy()
    torch_pred_round = np.reshape(np.round(torch_pred)[:len(target_T)], newshape=(len(target_T),))
    print("torch match:", sum(torch_pred_round==target_T.detach().cpu().numpy()))
    
    # sample data (np.array)
    Rn, Re, edge_index, target = Rn_T.detach().cpu().numpy(), Re_T.detach().cpu().numpy(), edge_index_T.detach().cpu().numpy(), target_T.detach().cpu().numpy()
    edge_index = edge_index.astype(np.int32)
    
    # sample data (np.1D-array)
    Re_1D = np.reshape(Re, newshape = (Re.shape[0]*Re.shape[1]))
    Rn_1D = np.reshape(Rn, newshape = (Rn.shape[0]*Rn.shape[1]))
    edge_index_1D = np.reshape(edge_index, newshape=(edge_index.shape[0]*edge_index.shape[1])).astype(np.int32)
    hls_pred_noact = np.zeros(shape=(torch_pred.shape[0],)).astype(np.float32) # <--output of hls_model sent here, noact = noactivation
    
    # define ctypes
    Re_ctype = ctypes.POINTER(ctypes.ARRAY(ctypes.c_float,len(Re_1D)))
    Rn_ctype = ctypes.POINTER(ctypes.ARRAY(ctypes.c_float,len(Rn_1D)))
    edge_index_ctype = ctypes.POINTER(ctypes.ARRAY(ctypes.c_float,len(edge_index_1D)))
    hls_pred_ctype = ctypes.POINTER(ctypes.ARRAY(ctypes.c_float,len(hls_pred_noact)))
    
    # sample data (C-arrays)
    Re_c = Re_1D.ctypes.data_as(Re_ctype)
    Rn_c = Rn_1D.ctypes.data_as(Rn_ctype)
    edge_index_c = edge_index_1D.ctypes.data_as(edge_index_ctype)
    hls_pred_c = hls_pred_noact.ctypes.data_as(hls_pred_ctype)
    
    # get top function, set up ctypes
    libpath = 'hls_output/firmware/myproject-WITH_SAVE.so'
    top_func_lib = ctypes.cdll.LoadLibrary(libpath)
    top_func = getattr(top_func_lib, 'myproject_float')
    top_func.restype = None
    top_func.argtypes = [Re_ctype, Rn_ctype, edge_index_ctype, hls_pred_ctype,
                         ctypes.POINTER(ctypes.c_ushort), ctypes.POINTER(ctypes.c_ushort), ctypes.POINTER(ctypes.c_ushort), ctypes.POINTER(ctypes.c_ushort)]

    # call top function
    main_dir = os.getcwd()
    os.chdir(hls_model.config.get_output_dir() + '/firmware')
    top_func(Re_c, Rn_c, edge_index_c, hls_pred_c, ctypes.byref(ctypes.c_ushort()), ctypes.byref(ctypes.c_ushort()), ctypes.byref(ctypes.c_ushort()), ctypes.byref(ctypes.c_ushort()))
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    hls_pred = sigmoid(hls_pred_noact)
    
    print(hls_pred)
    print(torch_pred)
    #save outputs 
    os.chdir(main_dir)
    np.savetxt('target.csv', target, delimiter=',')
    np.savetxt('torch_pred.csv', torch_pred, delimiter=',')
    np.savetxt('hls_pred_nosigmoid.csv', hls_pred_noact, delimiter=',')
    np.savetxt('hls_pred.csv', hls_pred, delimiter=',')

#%% 
if __name__=='__main__':
    main()


            
    
    
    
    
    
    
    
    
