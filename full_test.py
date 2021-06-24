import os
import yaml
import argparse
import glob
import time

import ctypes
import numpy.ctypeslib as npc
import numpy as np
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
from utils.data.fix_graph_size import fix_graph_size

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
    main_dir = os.getcwd()
    args = parse_args()
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)

    # dataset
    graph_indir = config['graph_indir']
    graph_files = np.array(os.listdir(graph_indir))
    graph_files = np.array([os.path.join(graph_indir, graph_file)
                            for graph_file in graph_files])
    n_graphs = len(graph_files)
    IDs = np.arange(n_graphs)
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
    hls_model.graph['edge_index'].precision['input3_t'] = HLSType('input3_t',
                                                                  IntegerPrecisionType(width=32, signed=False))
    hls_model.compile()

    # get hls_model top function
    list_of_so_files = glob.glob(os.path.join(hls_model.config.get_output_dir(), 'firmware/myproject-*.so'))
    libpath = max(list_of_so_files, key=os.path.getctime)  # get latest *.so file
    print('loading shared library', libpath)
    top_func_lib = ctypes.cdll.LoadLibrary(libpath)
    top_func = getattr(top_func_lib, 'myproject_float')

    # define top function ctypes
    edge_attr_ctype = ctypes.POINTER(ctypes.ARRAY(ctypes.c_float, hls_model.reader.n_edge * hls_model.reader.edge_dim))
    node_attr_ctype = ctypes.POINTER(ctypes.ARRAY(ctypes.c_float, hls_model.reader.n_node * hls_model.reader.node_dim))
    edge_index_ctype = ctypes.POINTER(ctypes.ARRAY(ctypes.c_uint32, 2 * hls_model.reader.n_edge))
    hls_pred_ctype = ctypes.POINTER(ctypes.ARRAY(ctypes.c_float, hls_model.reader.n_edge))
    top_func.restype = None
    top_func.argtypes = [edge_attr_ctype, node_attr_ctype, edge_index_ctype, hls_pred_ctype,
                         ctypes.POINTER(ctypes.c_ushort), ctypes.POINTER(ctypes.c_ushort),
                         ctypes.POINTER(ctypes.c_ushort), ctypes.POINTER(ctypes.c_ushort)]

    all_error = []
    n_graphs = 0
    for data in dataset[:10000]:
        node_attr_T, edge_attr_T, edge_index_T, bad_graph = fix_graph_size(data.x, data.edge_attr, data.edge_index, n_node_max=112, n_edge_max=148)
        if bad_graph:
            continue

        n_graphs += 1
        torch_pred = model(data_wrapper(node_attr_T, edge_attr_T, edge_index_T)).detach().cpu().numpy()
        torch_pred = torch_pred[:data.edge_attr.shape[0]].reshape(data.edge_attr.shape[0],) #drop dummy edges, reshape to 1D

        # data -->  np.array, np.1D-array
        node_attr, edge_attr, edge_index = node_attr_T.detach().cpu().numpy(), edge_attr_T.detach().cpu().numpy(), edge_index_T.detach().cpu().numpy().astype(np.int32)
        node_attr_1D = np.reshape(node_attr, newshape=(node_attr.shape[0] * node_attr.shape[1]))
        edge_attr_1D = np.reshape(edge_attr, newshape=(edge_attr.shape[0] * edge_attr.shape[1]))
        edge_index_1D = np.reshape(edge_index, newshape=(edge_index.shape[0] * edge_index.shape[1])).astype(np.float32)
        hls_pred_noact = np.zeros(shape=(edge_attr.shape[0],)).astype(np.float32)  # <--output of hls_model sent here, noact = noactivation

        # data --> C-array
        edge_attr_c = edge_attr_1D.ctypes.data_as(edge_attr_ctype)
        node_attr_c = node_attr_1D.ctypes.data_as(node_attr_ctype)
        edge_index_c = edge_index_1D.ctypes.data_as(edge_index_ctype)
        hls_pred_c = hls_pred_noact.ctypes.data_as(hls_pred_ctype)

        # hls_model inference
        os.chdir(os.path.join(hls_model.config.get_output_dir(), 'firmware'))
        top_func(edge_attr_c, node_attr_c, edge_index_c, hls_pred_c, ctypes.byref(ctypes.c_ushort()), ctypes.byref(ctypes.c_ushort()),
                 ctypes.byref(ctypes.c_ushort()), ctypes.byref(ctypes.c_ushort()))
        hls_pred = sigmoid(hls_pred_noact)
        hls_pred = hls_pred[:data.edge_attr.shape[0]] #drop dummy edges
        os.chdir(main_dir)

        err = np.mean(np.abs(torch_pred-hls_pred))
        all_error.append(err)

    np.savetxt("all_error.csv", np.array(all_error), delimiter=",", fmt='%.3e')
    print(f"n_graphs: {n_graphs}")
    print(f"mean error: {np.mean(all_error)}")

if __name__=="__main__":
    main()


