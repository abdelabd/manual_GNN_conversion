import os
import yaml
import argparse
import numpy as np
import torch
import time

# locals
from utils.models.interaction_network_pyg import InteractionNetwork
from model_wrappers import model_wrapper
from utils.data.dataset_pyg import GraphDataset
from utils.data.fix_graph_size import fix_graph_size

class data_wrapper(object):
    def __init__(self, node_attr, edge_attr, edge_index, target):
        self.x = node_attr
        self.edge_attr = edge_attr
        self.edge_index = edge_index.transpose(0,1)
        self.target = target

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

def load_models(trained_model_dir, aggr='add', flow='source_to_target', save_intermediates=False): #aggr = aggregation_method: ['add', 'mean', 'max']
    # get torch model
    torch_model = InteractionNetwork(aggr=aggr, flow=flow)
    torch_model_dict = torch.load(trained_model_dir + "//IN_pyg_small" + f"_{aggr}" + f"_{flow}" + "_state_dict.pt")
    torch_model.load_state_dict(torch_model_dict)

    # get torch wrapper
    torch_wrapper = model_wrapper(torch_model)

    return torch_model, torch_wrapper

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='test_config.yaml')
    add_arg('--n-graphs', type=int, default=1)
    add_arg('--aggregation-method', type=str, default='add', help='[add, mean, max, all]')
    add_arg('--flow', type=str, default='source_to_target', help='[source_to_target, target_to_source, all]')
    add_arg('--complexity', type=str, default='normal')
    add_arg('--save-errors', action='store_true')
    add_arg('--save-intermediates', action='store_true')
    return parser.parse_args()

def main():
    tic = time.time()
    args = parse_args()
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)

    if args.aggregation_method=='all':
        aggregations = ['add', 'mean', 'max']
    else: aggregations = [args.aggregation_method]

    if args.flow == 'all':
        flows = ['source_to_target', 'target_to_source']
    else: flows = [args.flow]

    if args.complexity == 'all':
        complexities = ['simple', 'complex', 'double_complex', 'triple_complex']
    else: complexities = [args.complexity]

    # dataset
    graph_indir = config['graph_indir']
    graph_files = np.array(os.listdir(graph_indir))
    graph_files = np.array([os.path.join(graph_indir, graph_file)
                                for graph_file in graph_files])
    n_graphs = len(graph_files)
    IDs = np.arange(n_graphs)
    dataset = GraphDataset(graph_files=graph_files[IDs])

    # fix graph sizes
    graph_dims = {
        "n_node_max": 112,
        "n_edge_max": 148,
        "node_dim": 3,
        "edge_dim": 4
    }
    graphs = []
    n_graphs = 0
    for i, data in enumerate(dataset[:args.n_graphs]):
        target = data.y
        node_attr, edge_attr, edge_index, bad_graph = fix_graph_size(data.x, data.edge_attr, data.edge_index,
                                                                           n_node_max=graph_dims['n_node_max'],
                                                                           n_edge_max=graph_dims['n_edge_max'])
        if not bad_graph:
            n_graphs += 1
            graphs.append(data_wrapper(node_attr, edge_attr, edge_index, target))
    print(f"n_graphs: {n_graphs}")

    for a in aggregations:
        for f in flows:
            torch_model, torch_wrapper = load_models(config['trained_model_dir'], aggr=a, flow=f)
            for c in complexities:
                all_torch_error = {
                    "MAE": [],
                    "MSE": [],
                    "RMSE": [],
                    }
                all_wrapper_error = {
                    "MAE": [],
                    "MSE": [],
                    "RMSE": [],
                }
                all_torch_wrapper_diff = {
                    "MAE": [],
                    "MSE": [],
                    "RMSE": [],
                }
                for i, data in enumerate(graphs):
                    target = np.reshape(data.target.detach().cpu().numpy(), newshape=(data.target.shape[0],))

                    # torch prediction
                    torch_pred = torch_model(data).detach().cpu().numpy()
                    torch_pred = np.reshape(torch_pred[:target.shape[0]], newshape=(target.shape[0],)) #drop dummy edges

                    # wrapper prediction
                    wrapper_pred = torch_wrapper.forward(data, complexity=c)
                    wrapper_pred = wrapper_pred.detach().cpu().numpy()
                    wrapper_pred = np.reshape(wrapper_pred[:target.shape[0]], newshape=(target.shape[0],))  # drop dummy edges

                    # get errors
                    all_torch_error["MAE"].append(MAE(target, torch_pred))
                    all_torch_error["MSE"].append(MSE(target, torch_pred))
                    all_torch_error["RMSE"].append(RMSE(target, torch_pred))

                    all_wrapper_error["MAE"].append(MAE(target, wrapper_pred))
                    all_wrapper_error["MSE"].append(MSE(target, wrapper_pred))
                    all_wrapper_error["RMSE"].append(RMSE(target, wrapper_pred))

                    all_torch_wrapper_diff["MAE"].append(MAE(torch_pred, wrapper_pred))
                    all_torch_wrapper_diff["MSE"].append(MSE(torch_pred, wrapper_pred))
                    all_torch_wrapper_diff["RMSE"].append(RMSE(torch_pred, wrapper_pred))

                print(f"With aggregation={a}, flow={f}, complexity={c}")
                for err_type in ["MAE", "MSE", "RMSE"]:
                    print(f"     with error criteria = {err_type}:")
                    print(f"          mean torch error: %s" %np.mean(all_torch_error["%s" %err_type]))
                    print(f"          mean wrapper error: %s" %np.mean(all_wrapper_error["%s" %err_type]))
                    print(f"          mean wrapper->torch error: %s" %np.mean(all_torch_wrapper_diff["%s" %err_type]))
                    print("")

    duration = time.time() - tic
    print(f"Duration: {duration//60} minutes, {duration%60} seconds")

if __name__=="__main__":
    main()