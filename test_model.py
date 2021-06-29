import os
import yaml
import argparse
import numpy as np
import torch

from hls4ml.model.hls_model import HLSModel_GNN
from hls4ml.converters.pyg_to_hls import pyg_to_hls
from hls4ml.model.hls_layers import HLSType, IntegerPrecisionType, FixedPrecisionType

# locals
from utils.data.dataset_pyg import GraphDataset
from utils.models.interaction_network_pyg_add import InteractionNetwork as InteractionNetwork_add
from utils.models.interaction_network_pyg_mean import InteractionNetwork as InteractionNetwork_mean
from utils.models.interaction_network_pyg_max import InteractionNetwork as InteractionNetwork_max
from utils.data.fix_graph_size import fix_graph_size
from model_wrappers import model_add_wrapper, model_mean_wrapper, model_max_wrapper

class data_wrapper(object):
    def __init__(self, node_attr, edge_attr, edge_index):
        self.x = node_attr
        self.edge_attr = edge_attr
        self.edge_index = edge_index

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# error criteria
def MAE(target, pred):
    inside_sum = np.abs(target-pred)
    return np.mean(inside_sum)
def MSE(target, pred):
    inside_sum = (target - pred)**2
    return np.mean(inside_sum)
def RMSE(target, pred):
    return np.sqrt(MSE(target, pred))
def mean_cross_entropy(target, pred, epsilon=1e-8): #keeps dividing by zero :(
    pred_prime = np.clip(pred, epsilon, 1 - epsilon).astype(np.float64)
    #print(min(pred_prime))
    inside_sum = target*np.log(pred_prime)+(1-target)*np.log(1-pred_prime)
    return -np.sum(inside_sum)/len(target)

def load_models(trained_model_dir, graph_dims, aggr='add', save_intermediates=False): #aggr = aggregation_method: ['add', 'mean', 'max']
    # get torch model
    torch_map = {
        'add': InteractionNetwork_add,
        'mean': InteractionNetwork_mean,
        'max': InteractionNetwork_max
    }
    torch_model = torch_map[aggr]()
    torch_model_dict = torch.load(trained_model_dir + "//IN_pyg_small_" + aggr + "_state_dict.pt")
    torch_model.load_state_dict(torch_model_dict)

    # get hls model
    hls_model_config, reader, layer_list = pyg_to_hls(torch_model, graph_dims, save_intermediates=save_intermediates)
    hls_model_config['OutputDir'] = hls_model_config['OutputDir'] + "/%s"%aggr
    hls_model = HLSModel_GNN(hls_model_config, reader, layer_list)
    hls_model.inputs = ['node_attr', 'edge_attr', 'edge_index']
    hls_model.outputs = ['layer6_out_L']
    hls_model.compile()

    # get torch wrapper
    wrapper_map = {
        'add': model_add_wrapper,
        'mean': model_mean_wrapper,
        'max': model_max_wrapper
    }
    if save_intermediates:
        torch_wrapper = wrapper_map[aggr](torch_model, save_dir=hls_model.config.get_output_dir())
    else: torch_wrapper = wrapper_map[aggr](torch_model)

    return torch_model, hls_model, torch_wrapper

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='test_config.yaml')
    add_arg('--n-graphs', type=int, default=1)
    add_arg('--aggregation-method', type=str, default='add')
    add_arg('--save-errors', action='store_true')
    add_arg('--save-intermediates', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)

    if args.aggregation_method=='all':
        aggr = ['add', 'mean', 'max']
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
        torch_model, hls_model, torch_wrapper = load_models(config['trained_model_dir'], graph_dims, aggr=a, save_intermediates=args.save_intermediates)

        all_torch_error = {
            "MAE": [],
            "MSE": [],
            "RMSE": [],
        }
        all_hls_error = {
            "MAE": [],
            "MSE": [],
            "RMSE": [],
        }
        all_torch_hls_diff = {
            "MAE": [],
            "MSE": [],
            "RMSE": [],
        }
        n_graphs = 0
        for i, data in enumerate(dataset[:args.n_graphs]):
            node_attr_T, edge_attr_T, edge_index_T, bad_graph = fix_graph_size(data.x, data.edge_attr, data.edge_index, n_node_max=graph_dims['n_node_max'], n_edge_max=graph_dims['n_edge_max'])
            if bad_graph:
                continue

            n_graphs += 1
            target = np.reshape(data.y.detach().cpu().numpy(), newshape=(data.y.shape[0],))

            node_attr, edge_attr, edge_index = node_attr_T.detach().cpu().numpy(), edge_attr_T.detach().cpu().numpy(), edge_index_T.detach().cpu().numpy().astype(np.int32)  # np.array data

            # torch prediction
            torch_pred = torch_model(data_wrapper(node_attr_T, edge_attr_T, edge_index_T.transpose(0,1))).detach().cpu().numpy()
            torch_pred = np.reshape(torch_pred[:target.shape[0]], newshape=(target.shape[0],)) #drop dummy edges

            # hls prediction
            hls_pred = sigmoid(hls_model.predict(node_attr, edge_attr, edge_index))
            hls_pred = np.reshape(hls_pred[:target.shape[0]], newshape=(target.shape[0],)) #drop dummy edges

            # get errors
            all_torch_error["MAE"].append(MAE(target, torch_pred))
            all_torch_error["MSE"].append(MSE(target, torch_pred))
            all_torch_error["RMSE"].append(RMSE(target, torch_pred))

            all_hls_error["MAE"].append(MAE(target, hls_pred))
            all_hls_error["MSE"].append(MSE(target, hls_pred))
            all_hls_error["RMSE"].append(RMSE(target, hls_pred))

            all_torch_hls_diff["MAE"].append(MAE(torch_pred, hls_pred))
            all_torch_hls_diff["MSE"].append(MSE(torch_pred, hls_pred))
            all_torch_hls_diff["RMSE"].append(RMSE(torch_pred, hls_pred))

            if i==args.n_graphs-1:
                wrapper_pred = torch_wrapper.forward(data_wrapper(node_attr_T, edge_attr_T, edge_index_T.transpose(0,1))) #saves intermediates
                wrapper_pred = wrapper_pred.detach().cpu().numpy()
                wrapper_pred = np.reshape(wrapper_pred[:target.shape[0]], newshape=(target.shape[0],)) #drop dummy edges
                wrapper_MAE = MAE(torch_pred, wrapper_pred)

        print(f"n_graphs: {n_graphs}")
        print(f"With aggregation method = {a}:")
        print(f"     single-graph wrapper-->torch MAE: {wrapper_MAE}")
        print("")
        for err_type in ["MAE", "MSE", "RMSE"]:#, "MCE"]:
            print(f"     with error criteria = {err_type}:")
            print(f"          mean torch error: %s" %np.mean(all_torch_error["%s" %err_type]))
            print(f"          mean hls error: %s" %np.mean(all_hls_error["%s" %err_type]))
            print(f"          mean hls-->torch error: %s" %np.mean(all_torch_hls_diff["%s" %err_type]))
            print("")

if __name__=="__main__":
    main()


