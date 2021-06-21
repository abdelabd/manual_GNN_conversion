#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 17:20:17 2021

@author: abdel
"""

import os 
import yaml 
import argparse
import shutil

import ctypes
import numpy.ctypeslib as npc
import numpy as np
import pandas as pd
import torch

import hls4ml
from hls4ml.model.hls_model import HLSModel_GNN
from hls4ml.utils.config import create_vivado_config
from hls4ml.writer.vivado_writer import VivadoWriter_GNN
from hls4ml.converters.pyg_to_hls import PygModelReader
from hls4ml.model.hls_layers import HLSType, IntegerPrecisionType

# locals
from utils.prep_GNN_for_hls import prep_GNN_for_hls
from utils.data.dataset_pyg import GraphDataset
from utils.models.interaction_network_pyg import InteractionNetwork as InteractionNetwork_pyg
from utils.data.load_sample import load_sample

#%% helper classes

class data_wrapper():
        def __init__(self, x, edge_attr, edge_index):
            self.x = x
            self.edge_attr = edge_attr
            self.edge_index = edge_index
            
class model_wrapper():
    def __init__(self, model):
        self.model = model
    
    def EdgeBlock_forward(self, block, Re, Rn, edge_index, out_dim):
        L = torch.zeros((Re.shape[0], out_dim))
        Q = torch.zeros((Rn.shape[0], out_dim))              
        for i in range(Re.shape[0]):
            r = edge_index[1][i]
            s = edge_index[0][i]
            
            x_i = Rn[r]
            x_j = Rn[s]
            edge_attr = Re[i]
            
            l_logits = torch.cat((x_i, x_j), dim=0)
            l = torch.cat((l_logits, edge_attr), dim=0)
            L[i] = block(l)
            for j in range(out_dim):
                Q[r,j] += L[i,j]
        return L, Q
        
    def NodeBlock_forward(self, block, Rn, Q, out_dim):
        P = torch.zeros((Rn.shape[0], out_dim))
        for i in range(Rn.shape[0]):
            p = torch.cat((Rn[i], Q[i]))
            P[i] = block(p)
        return P
        
    def forward(self, data):
        Rn = data.x
        Re = data.edge_attr
        edge_index = data.edge_index
        
        L, Q = self.EdgeBlock_forward(self.model.R1, Re, Rn, edge_index, out_dim=4) #IN_edge_module_1
        P = self.NodeBlock_forward(self.model.O, Rn, Q, out_dim=3) #IN_node_module
        L_out, Q_out = self.EdgeBlock_forward(self.model.R2, L, P, edge_index, out_dim=1) #IN_edge_module
        out = torch.sigmoid(L_out)
        return L, Q, P, L_out, Q_out, out

#%% helper functions

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='test_bench_config.yaml')
    return parser.parse_args()

def sigmoid(x):
    return 1/(1+np.exp(-x))

def predict_with_save(hls_model, data, out_dim, save_dir): #hls_model must be compiled already
    # torch data
    Rn_T, Re_T, edge_index_T = data.x, data.edge_attr, data.edge_index
    
    # np.array data
    Rn, Re, edge_index = Rn_T.detach().cpu().numpy(), Re_T.detach().cpu().numpy(), edge_index_T.detach().cpu().numpy()
    
    # np.1D-array data
    Re_1D = np.reshape(Re, newshape = (Re.shape[0]*Re.shape[1]))
    Rn_1D = np.reshape(Rn, newshape = (Rn.shape[0]*Rn.shape[1]))
    edge_index_1D = np.reshape(edge_index, newshape=(edge_index.shape[0]*edge_index.shape[1])).astype(np.float32)
    hls_pred_noact = np.zeros(shape=(Re.shape[0],)).astype(np.float32)
    
    # C data
    Re_ctype = ctypes.POINTER(ctypes.ARRAY(ctypes.c_float,len(Re_1D)))
    Rn_ctype = ctypes.POINTER(ctypes.ARRAY(ctypes.c_float, len(Rn_1D)))
    edge_index_ctype = ctypes.POINTER(ctypes.ARRAY(ctypes.c_uint32, len(edge_index_1D)))
    hls_pred_ctype = ctypes.POINTER(ctypes.ARRAY(ctypes.c_float, Re.shape[0]))
    
    Re_c = Re_1D.ctypes.data_as(Re_ctype)
    Rn_c = Rn_1D.ctypes.data_as(Rn_ctype)
    edge_index_c = edge_index_1D.ctypes.data_as(edge_index_ctype)
    hls_pred_noact_c = hls_pred_noact.ctypes.data_as(hls_pred_ctype)
    
    # get hls_model top function    
    shutil.copyfile(os.getcwd()+"/myproject_with_save.cpp", hls_model.config.get_output_dir()+"/firmware/myproject_with_save.cpp")
    shutil.copyfile(os.getcwd()+"/build_lib_with_save.sh", hls_model.config.get_output_dir()+"/build_lib_with_save.sh")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    main_dir = os.getcwd()
    os.chdir(hls_model.config.get_output_dir())
    ret_val = os.system("bash build_lib_with_save.sh")
    lib_path = hls_model.config.get_output_dir() + "/firmware/myproject-WITH_SAVE.so"
    top_func_lib = ctypes.cdll.LoadLibrary(lib_path)
    top_func = getattr(top_func_lib, 'myproject_float')
    
    # set top function ctypes
    top_func.restype = None
    top_func.argtypes = [Re_ctype, Rn_ctype, edge_index_ctype, hls_pred_ctype,
                     ctypes.POINTER(ctypes.c_ushort), ctypes.POINTER(ctypes.c_ushort), ctypes.POINTER(ctypes.c_ushort), ctypes.POINTER(ctypes.c_ushort)]
    
    # call top_function
    os.chdir(hls_model.config.get_output_dir()+"/firmware")
    top_func(Re_c, Rn_c, edge_index_c, hls_pred_noact_c, ctypes.byref(ctypes.c_ushort()), ctypes.byref(ctypes.c_ushort()), ctypes.byref(ctypes.c_ushort()), ctypes.byref(ctypes.c_ushort()))
        
    # get+save intermediates and outputs
    L_1D = pd.read_csv("edge_update_1.csv", header=None).to_numpy()
    L = np.reshape(L_1D, newshape=Re.shape)
    os.remove("edge_update_1.csv")
    np.savetxt(save_dir+"/hls_"+"edge_update_1.csv", L, delimiter=',')
    
    Q_1D = pd.read_csv("edge_update_aggr_1.csv", header=None).to_numpy()
    Q = np.reshape(Q_1D, newshape=(Rn.shape[0], Re.shape[1]))
    os.remove("edge_update_aggr_1.csv")
    np.savetxt(save_dir+"/hls_"+"edge_update_aggr_1.csv", Q, delimiter=',')
    
    P_1D = pd.read_csv("node_update.csv", header=None).to_numpy()
    P = np.reshape(P_1D, newshape=Rn.shape)
    os.remove("node_update.csv")
    np.savetxt(save_dir+"/hls_"+"node_update.csv", P, delimiter=',')
    
    L_out_1D = pd.read_csv("edge_update_2.csv", header=None).to_numpy()
    L_out = np.reshape(L_out_1D, newshape=(Re.shape[0], out_dim))
    os.remove("edge_update_2.csv")
    np.savetxt(save_dir+"/hls_"+"edge_update_2.csv", L_out, delimiter=',')
    
    Q_out_1D = pd.read_csv("edge_update_aggr_2.csv", header=None).to_numpy()
    Q_out = np.reshape(Q_out_1D, newshape=(Rn.shape[0], out_dim))
    os.remove("edge_update_aggr_2.csv")
    np.savetxt(save_dir+"/hls_"+"edge_update_aggr_2.csv", Q_out, delimiter=',')
    
    out_1D = sigmoid(L_out_1D)
    out = np.reshape(out_1D, newshape=L_out.shape)
    np.savetxt(save_dir+"/hls_"+"out.csv", out, delimiter=',')
    
    os.remove("edge_attr.csv")
    np.savetxt(save_dir+"/input_"+"edge_attr.csv", Re, delimiter=",")
    
    os.remove("edge_index.csv")
    np.savetxt(save_dir+"/input_"+"edge_index.csv", edge_index, delimiter=",")
    
    os.remove("node_attr.csv")
    np.savetxt(save_dir+"/input_"+"node_attr.csv", Rn, delimiter=",")


#%%
def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)
    graph_indir = config['graph_indir']
    save_dir = config['save_dir']
    
    graph_files = np.array(os.listdir(graph_indir))
    graph_files = np.array([os.path.join(graph_indir, graph_file)
                            for graph_file in graph_files])
    graph_paths = [os.path.join(graph_indir, filename)
                   for filename in graph_files]
    n_graphs = len(graph_files)
    IDs = np.arange(n_graphs)
    np.random.seed(9)
    np.random.shuffle(IDs)
    params = {'batch_size': 1, 'shuffle': True, 'num_workers': 6}
    dataset = GraphDataset(graph_files=graph_files[IDs])

    # sample data (torch)
    Rn_T, Re_T, edge_index_T, target_T = load_sample(dataset)
    data = data_wrapper(Rn_T, Re_T, edge_index_T)

    # torch model
    model = InteractionNetwork_pyg()
    model_dict = torch.load("IN_pyg_small_state_dict.pt")
    model.load_state_dict(model_dict)
    del model_dict

    # torch model wrapper
    torch_model_TB = model_wrapper(model)

    # hls model
    config, reader, layer_list = prep_GNN_for_hls(model)
    hls_model = HLSModel_GNN(config, reader, layer_list)
    hls_model.inputs = ['edge_attr', 'node_attr', 'edge_index']
    hls_model.outputs = ['layer6_out_L']
    hls_model.graph['edge_index'].precision['input3_t'] = HLSType('input3_t', IntegerPrecisionType(width=32, signed=False))
    hls_model.compile()
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # normal torch prediction
    model_pred = model(data).detach().cpu().numpy()

    # wrapper torch prediction 
    L, Q, P, L_out, Q_out, out = torch_model_TB.forward(data)
    L = L.detach().cpu().numpy()
    Q = Q.detach().cpu().numpy()
    P = P.detach().cpu().numpy()
    L_out = L_out.detach().cpu().numpy()
    Q_out = Q_out.detach().cpu().numpy()
    out = out.detach().cpu().numpy()
    np.savetxt(save_dir+"/torch_" + "edge_update_1.csv", L, delimiter=",")
    np.savetxt(save_dir+"/torch_" + "edge_update_aggr_1.csv", Q, delimiter=",")
    np.savetxt(save_dir+"/torch_" + "node_update.csv", P, delimiter=",")
    np.savetxt(save_dir+"/torch_" + "edge_update_2.csv", L_out, delimiter=",")
    np.savetxt(save_dir+"/torch_" + "edge_update_aggr_2.csv", Q_out, delimiter=",")
    np.savetxt(save_dir+"/torch_" + "out.csv", out, delimiter=",")

    # hls_model prediction
    predict_with_save(hls_model, data, out_dim=1, save_dir = save_dir)
    
if __name__=="__main__":
    main()