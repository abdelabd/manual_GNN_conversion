import os
import yaml
import argparse
import numpy as np
import torch

from hls4ml.model.hls_model import HLSModel_GNN
from hls4ml.converters.pyg_to_hls import pyg_to_hls
from hls4ml.model.hls_layers import HLSType, IntegerPrecisionType, FixedPrecisionType
from collections import OrderedDict

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

# scores
def true_pos(target, pred):
    tp = 0
    for i in range(len(target)):
        if target[i]==1 and pred[i]==1:
            tp += 1
    return tp
def true_neg(target, pred):
    tn = 0
    for i in range(len(target)):
        if target[i]==0 and pred[i]==0:
            tn += 1
    return tn
def false_pos(target, pred):
    fp = 0
    for i in range(len(target)):
        if target[i]==0 and pred[i]==1:
            fp += 1
    return fp
def false_neg(target, pred):
    fn = 0
    for i in range(len(target)):
        if target[i]==1 and pred[i]==0:
            fn += 1
    return fn
def accuracy(target, pred):
    num = true_pos(target, pred)+true_neg(target, pred)
    den = num + false_pos(target, pred) + false_neg(target, pred)
    return num/den
def f1(target, pred):
    tp = true_pos(target, pred)
    fp = false_pos(target, pred)
    fn = false_neg(target, pred)

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*precision*recall/(precision+recall)
    return f1

def load_models(trained_model_dir, graph_dims, aggr='add', flow='source_to_target', n_neurons=40): #aggr = aggregation_method: ['add', 'mean', 'max']
    # get torch model
    torch_model = InteractionNetwork(aggr=aggr, flow=flow, hidden_size=n_neurons)
    torch_model_dict = torch.load(trained_model_dir + "//IN_pyg_small" + f"_{aggr}" + f"_{flow}" + f"_{n_neurons}"+ "_state_dict.pt")
    torch_model.load_state_dict(torch_model_dict)

    # get hls model
    forward_dict = OrderedDict()
    forward_dict["R1"] = "EdgeBlock"
    forward_dict["O"] = "NodeBlock"
    forward_dict["R2"] = "EdgeBlock"

    hls_model_config, reader, layer_list = pyg_to_hls(torch_model, forward_dict, graph_dims)
    hls_model_config['OutputDir'] = hls_model_config['OutputDir'] + "/%s"%aggr + "/%s"%flow + "/neurons_%s"%n_neurons
    hls_model = HLSModel_GNN(hls_model_config, reader, layer_list)
    hls_model.inputs = ['node_attr', 'edge_attr', 'edge_index']
    hls_model.outputs = ['layer6_out_L']
    hls_model.compile()

    # get torch wrapper
    torch_wrapper = model_wrapper(torch_model)

    return torch_model, hls_model, torch_wrapper

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='test_config.yaml')
    add_arg('--n-graphs', type=int, default=1)
    add_arg('--aggregation-method', type=str, default='add', help='[add, mean, max, all]')
    add_arg('--flow', type=str, default='source_to_target', help='[source_to_target, target_to_source, all]')
    add_arg('--max-nodes', type=int, default=112)
    add_arg('--max-edges', type=int, default=148)
    add_arg('--n-neurons', type=int, default=40)
    add_arg('--save-errors', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)

    if args.aggregation_method=='all':
        aggregations = ['add', 'mean', 'max']
    else: aggregations = [args.aggregation_method]

    if args.flow == 'all':
        flows = ['source_to_target', 'target_to_source']
    else: flows = [args.flow]

    if args.n_neurons == 'all':
        n_neurons = [8,40]
    else: n_neurons = [args.n_neurons]

    # dataset
    graph_indir = config['graph_indir']
    graph_files = np.array(os.listdir(graph_indir))
    graph_files = np.array([os.path.join(graph_indir, graph_file)
                                for graph_file in graph_files])
    n_graphs = len(graph_files)
    IDs = np.arange(n_graphs)
    dataset = GraphDataset(graph_files=graph_files[IDs])

    # get fixed_size graphs
    graph_dims = {
        "n_node_max": args.max_nodes,
        "n_edge_max": args.max_edges,
        "node_dim": 3,
        "edge_dim": 4
    }
    graphs = []
    for data in dataset[:args.n_graphs]:
        node_attr, edge_attr, edge_index, bad_graph = fix_graph_size(data.x, data.edge_attr, data.edge_index,
                                                                           n_node_max=graph_dims['n_node_max'],
                                                                           n_edge_max=graph_dims['n_edge_max'])
        if not bad_graph:
            target = data.y
            graphs.append(data_wrapper(node_attr, edge_attr, edge_index, target))
    print(f"n_graphs: {len(graphs)}")

    for a in aggregations:
        for f in flows:
            for nn in n_neurons:
                torch_model, hls_model, torch_wrapper = load_models(config['trained_model_dir'], graph_dims, aggr=a, flow=f, n_neurons=nn)

                all_torch_error = {
                    "MAE": [],
                    "MSE": [],
                    "RMSE": [],
                    'Accuracy': [],
                    "f1": []
                }
                all_hls_error = {
                    "MAE": [],
                    "MSE": [],
                    "RMSE": [],
                    'Accuracy': [],
                    "f1": []
                }
                all_torch_hls_diff = {
                    "MAE": [],
                    "MSE": [],
                    "RMSE": [],
                    "Accuracy": [],
                    "f1": []
                }
                for i, data in enumerate(graphs):
                    target = np.reshape(data.target.detach().cpu().numpy(), newshape=(data.target.shape[0],))

                    # torch prediction
                    torch_pred = torch_model(data).detach().cpu().numpy()
                    torch_pred = np.reshape(torch_pred[:target.shape[0]], newshape=(target.shape[0],)) #drop dummy edges

                    # hls prediction
                    node_attr, edge_attr, edge_index = data.x.detach().cpu().numpy(), data.edge_attr.detach().cpu().numpy(), data.edge_index.transpose(0,1).detach().cpu().numpy().astype(np.int32)  # np.array data
                    hls_pred = sigmoid(hls_model.predict(node_attr, edge_attr, edge_index))
                    hls_pred = np.reshape(hls_pred[:target.shape[0]], newshape=(target.shape[0],)) #drop dummy edges

                    # get errors
                    all_torch_error["MAE"].append(MAE(target, torch_pred))
                    all_torch_error["MSE"].append(MSE(target, torch_pred))
                    all_torch_error["RMSE"].append(RMSE(target, torch_pred))
                    all_torch_error["Accuracy"].append(accuracy(target, np.round(torch_pred)))
                    all_torch_error["f1"].append(f1(target, np.round(torch_pred)))

                    all_hls_error["MAE"].append(MAE(target, hls_pred))
                    all_hls_error["MSE"].append(MSE(target, hls_pred))
                    all_hls_error["RMSE"].append(RMSE(target, hls_pred))
                    all_hls_error["Accuracy"].append(accuracy(target, np.round(hls_pred)))
                    all_hls_error["f1"].append(f1(target, np.round(hls_pred)))

                    all_torch_hls_diff["MAE"].append(MAE(torch_pred, hls_pred))
                    all_torch_hls_diff["MSE"].append(MSE(torch_pred, hls_pred))
                    all_torch_hls_diff["RMSE"].append(RMSE(torch_pred, hls_pred))
                    all_torch_hls_diff["Accuracy"].append(accuracy(np.round(torch_pred), np.round(hls_pred)))
                    all_torch_hls_diff["f1"].append(f1(np.round(torch_pred), np.round(hls_pred)))

                    if i==len(graphs)-1:
                        wrapper_pred = torch_wrapper.forward(data) #saves intermediates
                        wrapper_pred = wrapper_pred.detach().cpu().numpy()
                        wrapper_pred = np.reshape(wrapper_pred[:target.shape[0]], newshape=(target.shape[0],)) #drop dummy edges
                        wrapper_MAE = MAE(torch_pred, wrapper_pred)

                print(f"With aggregation={torch_model.aggr}, flow={torch_model.flow}, n_neurons={nn}")
                print(f"     single-graph wrapper-->torch MAE: {wrapper_MAE}")
                print("")
                for err_type in ["MAE", "MSE", "RMSE"]:#, "Accuracy", "f1"]:#, "MCE"]:
                    print(f"     with error criteria = {err_type}:")
                    print(f"          mean torch error: %s" %np.mean(all_torch_error["%s" %err_type]))
                    print(f"          mean hls error: %s" %np.mean(all_hls_error["%s" %err_type]))
                    print(f"          mean hls-->torch error: %s" %np.mean(all_torch_hls_diff["%s" %err_type]))
                    print("")
                for score_type in ["Accuracy", "f1"]:
                    print(f"     with score criteria = {score_type}:")
                    print(f"          mean torch score: %s" %np.mean(all_torch_error["%s"%score_type]))
                    print(f"          mean hls score: %s" %np.mean(all_hls_error["%s"%score_type]))
                    print(f"          mean hls-->torch score: %s" % np.mean(all_torch_hls_diff["%s" % score_type]))
                    print("")


if __name__=="__main__":
    main()

