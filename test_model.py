#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 22:14:23 2021

@author: abdel
"""
import os 
import yaml 
import argparse
import numpy as np
import torch

from hls4ml.model.hls_model import HLSModel_GNN
from hls4ml.converters.pyg_to_hls import pyg_to_hls
from hls4ml.model.hls_layers import HLSType, IntegerPrecisionType

# locals
from utils.data.dataset_pyg import GraphDataset
from utils.models.interaction_network_pyg import InteractionNetwork as InteractionNetwork_pyg
from utils.data.load_sample import load_sample

#%%
class data_wrapper(object):
    def __init__(self, Rn, Re, edge_index):
        self.x = Rn
        self.edge_attr = Re
        self.edge_index = edge_index

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='test_config.yaml')
    return parser.parse_args()
    
def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)

    # torch model
    model = InteractionNetwork_pyg()
    model_dict_dir = config['model_dict_dir']
    model_dict = torch.load(model_dict_dir)
    model.load_state_dict(model_dict)

    # hls model
    graph_dims = {
        "n_node_max": 112,
        "n_edge_max": 148,
        "node_dim": 3,
        "edge_dim": 4
    }
    hls_model_config, reader, layer_list = pyg_to_hls(model, graph_dims)
    hls_model = HLSModel_GNN(hls_model_config, reader, layer_list)
    hls_model.inputs = ['node_attr', 'edge_attr',  'edge_index']
    hls_model.outputs = ['layer6_out_L']
    hls_model.graph['edge_index'].precision['input3_t'] = HLSType('input3_t',
                                                                  IntegerPrecisionType(width=32, signed=False))
    hls_model.compile()
    print("")

    # dataset
    graph_indir = config['graph_indir']
    graph_files = np.array(os.listdir(graph_indir))
    graph_files = np.array([os.path.join(graph_indir, graph_file)
                            for graph_file in graph_files])
    n_graphs = len(graph_files)
    IDs = np.arange(n_graphs)
    dataset = GraphDataset(graph_files=graph_files[IDs])
    Rn_T, Re_T, edge_index_T, target_T = load_sample(dataset)  # sample data (torch.Tensor)
    Rn, Re, edge_index, target = Rn_T.detach().cpu().numpy(), Re_T.detach().cpu().numpy(), edge_index_T.detach().cpu().numpy().astype(
        np.int32), target_T.detach().cpu().numpy()  # sample data (np.array)
    print(f"num edges: {target_T.shape[0]}")

    # torch model inference
    torch_pred = model(data_wrapper(Rn_T, Re_T, edge_index_T)).detach().cpu().numpy()
    torch_pred = np.reshape(torch_pred[:target_T.shape[0]], newshape=(target_T.shape[0],))
    print("torch match: ", sum(np.round(torch_pred) == target_T.detach().cpu().numpy()))

    # hls model inference
    hls_pred = sigmoid(hls_model.predict(Rn, Re, edge_index))
    hls_pred = np.reshape(hls_pred[:target.shape[0]], newshape=(target.shape[0],))
    print("hls match: ", sum(np.round(hls_pred)== target_T.detach().cpu().numpy()))

    # save testbench data
    #os.makedirs('tb_data', exist_ok=True)
    #np.savetxt('tb_data/input_edge_data.dat', Re_1D.reshape(1, -1), fmt='%f', delimiter=' ')
    #np.savetxt('tb_data/input_node_data.dat', Rn_1D.reshape(1, -1), fmt='%f', delimiter=' ')
    #np.savetxt('tb_data/input_edge_index.dat', edge_index_1D.reshape(1, -1), fmt='%f', delimiter=' ')
    #np.savetxt('tb_data/output_predictions.dat', torch_pred.reshape(1, -1), fmt='%f', delimiter=' ')

    #print/save outputs
    np.savetxt('target.csv', target, delimiter=',')
    np.savetxt('torch_pred.csv', torch_pred, delimiter=',')
    np.savetxt('hls_pred.csv', hls_pred, delimiter=',')
    err = np.mean(np.abs(torch_pred-hls_pred))
    print(f"torch/hls mean absolute difference: {err}")

#%% 
if __name__=='__main__':
    main()


            
    
    
    
    
    
    
    
    
