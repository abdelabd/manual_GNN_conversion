import os
import yaml
import argparse
import numpy as np
import torch

from collections import OrderedDict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error

# locals
#from utils.models.interaction_network_pyg import InteractionNetwork
from utils.models.interaction_network_brevitas import InteractionNetwork
from model_wrappers import model_wrapper
from utils.data.dataset_pyg import GraphDataset
from utils.data.fix_graph_size import fix_graph_size

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='test_config.yaml')

    add_arg('--n-graphs', type=int, default=100)
    add_arg('--exclude-bad-graphs', action='store_true',
            help='if false, truncated and padded-but-not-separate graphs are included in the performance assessment')

    add_arg('--bit-width', type=int, default=6)

    add_arg('--max-nodes', type=int, default=113, help='max number of nodes')
    add_arg('--max-edges', type=int, default=196, help='max number of edges')

    return parser.parse_args()

class data_wrapper(object):
    def __init__(self, node_attr, edge_attr, edge_index, target):
        self.x = node_attr
        self.edge_attr = edge_attr
        self.edge_index = edge_index.transpose(0,1)

        node_attr, edge_attr, edge_index = self.x.detach().cpu().numpy(), self.edge_attr.detach().cpu().numpy(), self.edge_index.transpose(0, 1).detach().cpu().numpy().astype(np.float32)
        node_attr, edge_attr, edge_index = np.ascontiguousarray(node_attr), np.ascontiguousarray(edge_attr), np.ascontiguousarray(edge_index)
        self.hls_data = [node_attr, edge_attr, edge_index]

        self.target = target
        self.np_target = np.reshape(target.detach().cpu().numpy(), newshape=(target.shape[0],))

def load_graphs(graph_indir, graph_dims, n_graphs, exclude_bad_graphs=False):
    graph_files = np.array(os.listdir(graph_indir))
    graph_files = np.array([os.path.join(graph_indir, graph_file)
                            for graph_file in graph_files])
    n_graphs_total = len(graph_files)
    IDs = np.arange(n_graphs_total)
    dataset = GraphDataset(graph_files=graph_files[IDs])

    graphs = []
    for data in dataset[:n_graphs]:
        node_attr, edge_attr, edge_index, target, bad_graph = fix_graph_size(data.x, data.edge_attr, data.edge_index,
                                                                             data.y,
                                                                             n_node_max=graph_dims['n_node'],
                                                                             n_edge_max=graph_dims['n_edge'])
        if exclude_bad_graphs:
            if not bad_graph:
                graphs.append(data_wrapper(node_attr, edge_attr, edge_index, target))
        else:
            graphs.append(data_wrapper(node_attr, edge_attr, edge_index, target))

    print(f"n_graphs: {len(graphs)}")

    print("writing test bench data for 1st graph")
    data = graphs[0]
    node_attr, edge_attr, edge_index = data.x.detach().cpu().numpy(), data.edge_attr.detach().cpu().numpy(), data.edge_index.transpose(
        0, 1).detach().cpu().numpy().astype(np.int32)
    os.makedirs('tb_data', exist_ok=True)
    input_data = np.concatenate([node_attr.reshape(1, -1), edge_attr.reshape(1, -1), edge_index.reshape(1, -1)], axis=1)
    np.savetxt('tb_data/input_data.dat', input_data, fmt='%f', delimiter=' ')

    return graphs

def load_models(model_config, graph_dims,
                precision='ap_fixed<16,8>', reuse=1, resource_limit=False, par_factor=16,
                output_dir=""):

    aggr, flow, hidden_size = model_config["aggr"], model_config["flow"], model_config["n_neurons"]
    model_dict = model_config["state_dict"]

    # get torch model
    torch_model = InteractionNetwork(aggr=aggr, flow=flow, hidden_size=hidden_size)
    try:
        torch_model_dict = torch.load(model_dict)
    except RuntimeError:
        torch_model_dict = torch.load(model_dict, map_location=torch.device('cpu'))
    torch_model.load_state_dict(torch_model_dict)

    return torch_model

def reshape_pred(target, pred):

    # first, reshape to 1D
    pred = pred.flatten()
    target = target.flatten()

    # pad or truncate, if necessary
    if len(pred)<len(target): #pad with zeros
        n_diff = len(target) - len(pred)
        pred_appendage = np.zeros((n_diff,))
        pred_prime = np.concatenate((pred, pred_appendage), axis=0)
    elif len(pred)>len(target): #truncate
        pred_prime = pred[:len(target)]
    else: #neither
        pred_prime = pred

    return pred_prime

def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)

    # dataset
    graph_indir = config['graph_indir']
    graph_dims = {
        "n_node": args.max_nodes,
        "n_edge": args.max_edges,
        "node_dim": 3,
        "edge_dim": 4
    }
    graphs = load_graphs(graph_indir, graph_dims, args.n_graphs, args.exclude_bad_graphs)

    # model parameters
    torch_model = load_models(config['model'], graph_dims)
    all_torch_error = {
        "MAE": [],
        "MSE": [],
        "RMSE": [],
        'Accuracy': [],
        "f1": [],
        "AUC": []
    }
    all_hls_error = {
        "MAE": [],
        "MSE": [],
        "RMSE": [],
        'Accuracy': [],
        "f1": [],
        "AUC": []
    }
    all_torch_hls_diff = {
        "MAE": [],
        "MSE": [],
        "RMSE": [],
        "Accuracy": [],
        "f1": [],
        "AUC": []
    }
    for i, data in enumerate(graphs):
        target = data.np_target

        # torch prediction
        torch_pred = torch_model(data).detach().cpu().numpy()
        torch_pred = reshape_pred(target, torch_pred)
        if i==0: np.savetxt('tb_data/output_predictions.dat', torch_pred.reshape(1, -1), fmt='%f', delimiter=' ')

        # get errors
        all_torch_error["MAE"].append(mean_absolute_error(target, torch_pred))
        all_torch_error["MSE"].append(mean_squared_error(target, torch_pred))
        all_torch_error["RMSE"].append(mean_squared_error(target, torch_pred, squared=False))
        all_torch_error["Accuracy"].append(accuracy_score(target, np.round(torch_pred)))
        all_torch_error["f1"].append(f1_score(target, np.round(torch_pred)))
        try:
            all_torch_error["AUC"].append(roc_auc_score(target, torch_pred))
        except ValueError:
            all_torch_error["AUC"].append(0.5) #0.5=random number generator


    print(f"With aggregation={config['model']['aggr']}, flow={config['model']['flow']}, n_neurons={config['model']['n_neurons']}")
    print("")
    for err_type in ["MAE", "MSE", "RMSE"]:#, "Accuracy", "f1"]:#, "MCE"]:
        print(f"     with error criteria = {err_type}:")
        print(f"          mean torch error: %s" %np.mean(all_torch_error["%s" %err_type]))
        print("")
    for score_type in ["Accuracy", "f1", "AUC"]:
        print(f"     with score criteria = {score_type}:")
        print(f"          mean torch score: %s" %np.mean(all_torch_error["%s"%score_type]))
        print("")

if __name__=="__main__":
    main()

