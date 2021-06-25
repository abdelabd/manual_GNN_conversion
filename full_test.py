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
from utils.data.fix_graph_size import fix_graph_size

class data_wrapper(object):
    def __init__(self, node_attr, edge_attr, edge_index):
        self.x = node_attr
        self.edge_attr = edge_attr
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
    hls_model.inputs = ['node_attr', 'edge_attr', 'edge_index']
    hls_model.outputs = ['layer6_out_L']
    hls_model.graph['edge_index'].precision['input3_t'] = HLSType('input3_t',
                                                                  IntegerPrecisionType(width=32, signed=False))
    hls_model.compile()

    # dataset
    graph_indir = config['graph_indir']
    graph_files = np.array(os.listdir(graph_indir))
    graph_files = np.array([os.path.join(graph_indir, graph_file)
                            for graph_file in graph_files])
    n_graphs = len(graph_files)
    IDs = np.arange(n_graphs)
    dataset = GraphDataset(graph_files=graph_files[IDs])

    all_error = []
    n_graphs = 0
    for data in dataset[:10000]:
        node_attr_T, edge_attr_T, edge_index_T, bad_graph = fix_graph_size(data.x, data.edge_attr, data.edge_index, n_node_max=112, n_edge_max=148)
        if bad_graph:
            continue

        n_graphs += 1
        num_edges = data.edge_attr.shape[0]

        node_attr, edge_attr, edge_index = node_attr_T.detach().cpu().numpy(), edge_attr_T.detach().cpu().numpy(), edge_index_T.detach().cpu().numpy().astype(np.int32)  # np.array data

        # torch prediction
        torch_pred = model(data_wrapper(node_attr_T, edge_attr_T, edge_index_T)).detach().cpu().numpy()
        torch_pred = np.reshape(torch_pred[:num_edges], newshape=(num_edges,))

        # hls prediction
        hls_pred = sigmoid(hls_model.predict(node_attr, edge_attr, edge_index))
        hls_pred = np.reshape(hls_pred[:num_edges], newshape=(num_edges,))

        # compare outputs
        err = np.mean(np.abs(torch_pred-hls_pred))
        all_error.append(err)

    np.savetxt("all_error.csv", np.array(all_error), delimiter=",", fmt='%.3e')
    print(f"n_graphs: {n_graphs}")
    print(f"torch<-->hls mean absolute difference: {np.mean(all_error)}")

if __name__=="__main__":
    main()


