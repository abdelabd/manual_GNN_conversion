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
from utils.models.interaction_network_pyg_add import InteractionNetwork as InteractionNetwork_add
from utils.models.interaction_network_pyg_mean import InteractionNetwork as InteractionNetwork_mean
from utils.models.interaction_network_pyg_max import InteractionNetwork as InteractionNetwork_max
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
    add_arg('--n-graphs', type=int, default=1)
    add_arg('--aggregation-method', type=str, default='add')
    add_arg('--save-errors', action='store_true')
    return parser.parse_args()

def load_models(trained_model_dir, graph_dims, aggr='add'): #aggr = aggregation_method: ['add', 'mean', 'max']
    # get torch model
    if aggr=='mean':
        model = InteractionNetwork_mean()
    elif aggr=='max':
        model = InteractionNetwork_max()
    else:
        model = InteractionNetwork_add()
    model_dict = torch.load(trained_model_dir + "//IN_pyg_small_" + aggr + "_state_dict.pt")
    model.load_state_dict(model_dict)

    # get hls model
    hls_model_config, reader, layer_list = pyg_to_hls(model, graph_dims)
    hls_model_config['OutputDir'] = hls_model_config['OutputDir'] + "/%s"%aggr
    hls_model = HLSModel_GNN(hls_model_config, reader, layer_list)
    hls_model.inputs = ['node_attr', 'edge_attr', 'edge_index']
    hls_model.outputs = ['layer6_out_L']
    hls_model.graph['edge_index'].precision['input3_t'] = HLSType('input3_t', IntegerPrecisionType(width=32, signed=False))
    hls_model.compile()
    return model, hls_model

def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)

    if args.aggregation_method=='all':
        aggr = ['add', 'mean']
    elif args.aggregation_method=='mean':
        aggr = ['mean']
    elif args.aggregation_method=='max':
        aggr = ['max']
    else:
        aggr = ['add']

    # dataset
    graph_indir = config['graph_indir']
    graph_files = np.array(os.listdir(graph_indir))
    graph_files = np.array([os.path.join(graph_indir, graph_file)
                                for graph_file in graph_files])
    n_graphs = len(graph_files)
    IDs = np.arange(n_graphs)
    dataset = GraphDataset(graph_files=graph_files[IDs])

    # define graph dimensions for hls model
    graph_dims = {
        "n_node_max": 112,
        "n_edge_max": 148,
        "node_dim": 3,
        "edge_dim": 4
    }

    for a in aggr:
        model, hls_model = load_models(config['trained_model_dir'], graph_dims, aggr=a)

        all_torch_error = []
        all_hls_error = []
        all_torch_hls_diff = []
        n_graphs = 0
        for data in dataset[:args.n_graphs]:
            node_attr_T, edge_attr_T, edge_index_T, bad_graph = fix_graph_size(data.x, data.edge_attr, data.edge_index, n_node_max=graph_dims['n_node_max'], n_edge_max=graph_dims['n_edge_max'])
            if bad_graph:
                continue

            n_graphs += 1
            target = np.reshape(data.y.detach().cpu().numpy(), newshape=(data.y.shape[0],))

            node_attr, edge_attr, edge_index = node_attr_T.detach().cpu().numpy(), edge_attr_T.detach().cpu().numpy(), edge_index_T.detach().cpu().numpy().astype(np.int32)  # np.array data

            # torch prediction
            torch_pred = model(data_wrapper(node_attr_T, edge_attr_T, edge_index_T.transpose(0,1))).detach().cpu().numpy()
            torch_pred = np.reshape(torch_pred[:target.shape[0]], newshape=(target.shape[0],)) #drop dummy edges

            # hls prediction
            hls_pred = sigmoid(hls_model.predict(node_attr, edge_attr, edge_index))
            hls_pred = np.reshape(hls_pred[:target.shape[0]], newshape=(target.shape[0],)) #drop dummy edges

            # compare outputs
            torch_err = np.mean(np.abs(target-torch_pred))
            hls_err = np.mean(np.abs(target-hls_pred))
            torch_hls_diff = np.mean(np.abs(torch_pred-hls_pred))
            all_torch_error.append(torch_err)
            all_hls_error.append(hls_err)
            all_torch_hls_diff.append(torch_hls_diff)

        if args.save_errors:
            np.savetxt(hls_model.config.get_output_dir()+"/all_torch_error.csv", np.array(all_torch_error), delimiter=",")
            np.savetxt(hls_model.config.get_output_dir() + "/all_hls_error.csv", np.array(all_hls_error), delimiter=",")
            np.savetxt(hls_model.config.get_output_dir() + "/all_torch_hls_diff.csv", np.array(all_torch_hls_diff), delimiter=",")

        print(f"With aggregation method = {a}")
        print(f"     n_graphs: {n_graphs}")
        print(f"     mean absolute torch error: {np.mean(all_torch_error)}")
        print(f"     mean absolute hls error: {np.mean(all_hls_error)}")
        print(f"     torch<-->hls mean absolute difference: {np.mean(all_torch_hls_diff)}")
        print("")

if __name__=="__main__":
    main()


