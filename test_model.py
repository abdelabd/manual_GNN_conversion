import os
import yaml
import argparse
import numpy as np
import torch

from hls4ml.model.hls_model import HLSModel_GNN
from hls4ml.converters.pyg_to_hls import pyg_to_hls
from hls4ml.model.hls_layers import HLSType, IntegerPrecisionType, FixedPrecisionType
from collections import OrderedDict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error

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

def load_models(trained_model_dir, graph_dims, aggr='add', flow='source_to_target', n_neurons=40, fp_bits=16): #aggr = aggregation_method: ['add', 'mean', 'max']
    # get torch model
    torch_model = InteractionNetwork(aggr=aggr, flow=flow, hidden_size=n_neurons)
    torch_model_dict = torch.load(trained_model_dir + "//IN_pyg_small" + f"_{aggr}" + f"_{flow}" + f"_{n_neurons}"+ "_state_dict.pt")
    torch_model.load_state_dict(torch_model_dict)

    # get hls model
    forward_dict = OrderedDict()
    forward_dict["R1"] = "EdgeBlock"
    forward_dict["O"] = "NodeBlock"
    forward_dict["R2"] = "EdgeBlock"
    if fp_bits==16:
        fpib=6
    else: fpib=int(fp_bits/2)
    hls_model = pyg_to_hls(torch_model, forward_dict, graph_dims,
                           activate_final='sigmoid',
                           output_dir="/%s"%aggr + "/%s"%flow + "/neurons_%s"%n_neurons,
                           fixed_precision_bits=fp_bits,
                           fixed_precision_int_bits=fpib)
    # get torch wrapper
    torch_wrapper = model_wrapper(torch_model)

    return torch_model, hls_model, torch_wrapper

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='test_config.yaml')
    add_arg('--n-graphs', type=int, default=1)
    add_arg('--aggregation', type=str, default='add', choices =['add', 'mean', 'max', 'all'], help='[add, mean, max, all]')
    add_arg('--flow', type=str, default='source_to_target', choices = ['source_to_target', 'target_to_source', 'all'], help='[source_to_target, target_to_source, all]')
    add_arg('--fp-bits', type=int, default=16, help='number of fixed point bits')
    add_arg('--max-nodes', type=int, default=112, help='max number of nodes')
    add_arg('--max-edges', type=int, default=148, help='max number of edges')
    add_arg('--n-neurons', type=int, default=40, choices=[8,40], help='number of neurons')
    add_arg('--save-errors', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)

    if args.aggregation=='all':
        aggregations = ['add', 'mean', 'max']
    else: aggregations = [args.aggregation]

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

    print("writing test bench data for 1st graph")
    data = graphs[0]
    node_attr, edge_attr, edge_index = data.x.detach().cpu().numpy(), data.edge_attr.detach().cpu().numpy(), data.edge_index.transpose(0,1).detach().cpu().numpy().astype(np.int32)
    os.makedirs('tb_data',exist_ok=True)
    input_data = np.concatenate([node_attr.reshape(1, -1), edge_attr.reshape(1, -1), edge_index.reshape(1, -1)], axis=1)
    np.savetxt('tb_data/input_data.dat', input_data, fmt='%f', delimiter=' ')

    for a in aggregations:
        for f in flows:
            for nn in n_neurons:
                torch_model, hls_model, torch_wrapper = load_models(config['trained_model_dir'], graph_dims, aggr=a, flow=f, n_neurons=nn, fp_bits=args.fp_bits)

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
                    target = np.reshape(data.target.detach().cpu().numpy(), newshape=(data.target.shape[0],))

                    # torch prediction
                    torch_pred = torch_model(data).detach().cpu().numpy()
                    torch_pred = np.reshape(torch_pred[:target.shape[0]], newshape=(target.shape[0],)) #drop dummy edges
                    if i==0: np.savetxt('tb_data/output_predictions.dat', torch_pred.reshape(1, -1), fmt='%f', delimiter=' ')
    

                    # hls prediction
                    node_attr, edge_attr, edge_index = data.x.detach().cpu().numpy(), data.edge_attr.detach().cpu().numpy(), data.edge_index.transpose(0,1).detach().cpu().numpy().astype(np.int32)  # np.array data

                    if i==0: hls_model.compile()
                    hls_pred = hls_model.predict(node_attr, edge_attr, edge_index)
                    hls_pred = np.reshape(hls_pred[:target.shape[0]], newshape=(target.shape[0],)) #drop dummy edges

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

                    all_hls_error["MAE"].append(mean_absolute_error(target, hls_pred))
                    all_hls_error["MSE"].append(mean_squared_error(target, hls_pred))
                    all_hls_error["RMSE"].append(mean_squared_error(target, hls_pred, squared=False))
                    all_hls_error["Accuracy"].append(accuracy_score(target, np.round(hls_pred)))
                    all_hls_error["f1"].append(f1_score(target, np.round(hls_pred)))
                    try:
                        all_hls_error["AUC"].append(roc_auc_score(target, hls_pred))
                    except:
                        all_hls_error["AUC"].append(0.5)

                    all_torch_hls_diff["MAE"].append(mean_absolute_error(torch_pred, hls_pred))
                    all_torch_hls_diff["MSE"].append(mean_squared_error(torch_pred, hls_pred))
                    all_torch_hls_diff["RMSE"].append(mean_squared_error(torch_pred, hls_pred, squared=False))
                    all_torch_hls_diff["Accuracy"].append(accuracy_score(np.round(torch_pred), np.round(hls_pred)))
                    all_torch_hls_diff["f1"].append(f1_score(np.round(torch_pred), np.round(hls_pred)))
                    try:
                        all_torch_hls_diff["AUC"].append(roc_auc_score(np.round(torch_pred), hls_pred))
                    except ValueError:
                        all_torch_hls_diff["AUC"].append(0.5)

                    if i==len(graphs)-1:
                        wrapper_pred = torch_wrapper.forward(data) #saves intermediates
                        wrapper_pred = wrapper_pred.detach().cpu().numpy()
                        wrapper_pred = np.reshape(wrapper_pred[:target.shape[0]], newshape=(target.shape[0],)) #drop dummy edges
                        wrapper_MAE = mean_absolute_error(torch_pred, wrapper_pred)

                print(f"With aggregation={torch_model.aggr}, flow={torch_model.flow}, n_neurons={nn}")
                print(f"     single-graph wrapper-->torch MAE: {wrapper_MAE}")
                print("")
                for err_type in ["MAE", "MSE", "RMSE"]:#, "Accuracy", "f1"]:#, "MCE"]:
                    print(f"     with error criteria = {err_type}:")
                    print(f"          mean torch error: %s" %np.mean(all_torch_error["%s" %err_type]))
                    print(f"          mean hls error: %s" %np.mean(all_hls_error["%s" %err_type]))
                    print(f"          mean hls-->torch error: %s" %np.mean(all_torch_hls_diff["%s" %err_type]))
                    print("")
                for score_type in ["Accuracy", "f1", "AUC"]:
                    print(f"     with score criteria = {score_type}:")
                    print(f"          mean torch score: %s" %np.mean(all_torch_error["%s"%score_type]))
                    print(f"          mean hls score: %s" %np.mean(all_hls_error["%s"%score_type]))
                    print(f"          mean hls-->torch score: %s" % np.mean(all_torch_hls_diff["%s" % score_type]))
                    print("")


if __name__=="__main__":
    main()

