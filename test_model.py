import os
import yaml
import argparse
import numpy as np
import torch

from hls4ml.utils.config import config_from_pyg_model
from hls4ml.converters import convert_from_pyg_model
from collections import OrderedDict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error

# locals
from utils.models.interaction_network_pyg import InteractionNetwork
from model_wrappers import model_wrapper
from utils.data.dataset_pyg import GraphDataset
from utils.data.fix_graph_size import fix_graph_size

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='test_config.yaml')
    add_arg('--max-nodes', type=int, default=112, help='max number of nodes')
    add_arg('--max-edges', type=int, default=204, help='max number of edges')
    add_arg('--n-neurons', type=int, default=8, choices=[8, 40], help='number of neurons')
    add_arg('--n-graphs', type=int, default=100)
    add_arg('--aggregation', type=str, default='add', choices =['add', 'mean', 'max', 'all'], help='[add, mean, max, all]')
    add_arg('--flow', type=str, default='source_to_target', choices = ['source_to_target', 'target_to_source', 'all'], help='[source_to_target, target_to_source, all]')
    add_arg('--precision', type=str, default='ap_fixed<16,8>', help='precision to use')
    add_arg('--reuse', type=int, default=1, help="reuse factor")
    add_arg('--resource-limit', action='store_true', help='if true, then dataflow version implemented, otherwise pipeline version')
    add_arg('--output-dir', type=str, default="", help='output directory')
    add_arg('--synth',action='store_true', help='whether to synthesize')

    args = parser.parse_args()
    if args.aggregation=='all':
        args.aggregation = ['add', 'mean', 'max']
    else: args.aggregation = [args.aggregation]

    if args.flow == 'all':
        args.flow = ['source_to_target', 'target_to_source']
    else: args.flow = [args.flow]

    if args.n_neurons == 'all':
        args.n_neurons = [8,40]
    else: args.n_neurons = [args.n_neurons]

    return args

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

def load_graphs(graph_indir, graph_dims, n_graphs):
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
        if not bad_graph:
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

def load_models(trained_model_dir, graph_dims, aggr='add', flow='source_to_target', n_neurons=40,
                precision='ap_fixed<16,8>', output_dir="", reuse=1, resource_limit=False):
    # get torch model
    torch_model = InteractionNetwork(aggr=aggr, flow=flow, hidden_size=n_neurons)
    torch_model_dict = torch.load(trained_model_dir + "//IN_pyg_small" + f"_{aggr}" + f"_{flow}" + f"_{n_neurons}"+ "_state_dict.pt")
    torch_model.load_state_dict(torch_model_dict)

    # forward_dict: defines the order in which graph-blocks are called in the model's 'forward()' method
    forward_dict = OrderedDict()
    forward_dict["R1"] = "EdgeBlock"
    forward_dict["O"] = "NodeBlock"
    forward_dict["R2"] = "EdgeBlock"

    # get hls model
    if output_dir == "":
        output_dir = "hls_output/%s"%aggr + "/%s"%flow + "/neurons_%s"%n_neurons
    config = config_from_pyg_model(torch_model,
                                   default_precision=precision,
                                   default_index_precision='ap_uint<16>', 
                                   default_reuse_factor=reuse)
    hls_model = convert_from_pyg_model(torch_model,
                                       forward_dictionary=forward_dict,
                                       **graph_dims,
                                       activate_final="sigmoid",
                                       output_dir=output_dir,
                                       hls_config=config,
                                       fpga_part='xcvu9p-flga2104-2L-e',
                                       resource_limit=resource_limit
                                       )

    hls_model.compile()
    print("Model compiled at: ", hls_model.config.get_output_dir())
    model_config = f"aggregation: {aggr} \nflow: {flow} \nn_neurons: {n_neurons} \nprecision: {precision} \ngraph_dims: {graph_dims} \nreuse_factor: {reuse} \nresource_limit: {resource_limit}"
    with open(hls_model.config.get_output_dir() + "//model_config.txt", "w") as file:
        file.write(model_config)

    # get torch wrapper
    torch_wrapper = model_wrapper(torch_model)

    return torch_model, hls_model, torch_wrapper

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
    graphs = load_graphs(graph_indir, graph_dims, args.n_graphs)

    for a in args.aggregation:
        for f in args.flow:
            for nn in args.n_neurons:
                torch_model, hls_model, torch_wrapper = load_models(config['trained_model_dir'], graph_dims, aggr=a,
                                                                    flow=f, n_neurons=nn, precision=args.precision,
                                                                    output_dir=args.output_dir, reuse=args.reuse,
                                                                    resource_limit=args.resource_limit)
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
                    torch_pred = np.reshape(torch_pred[:target.shape[0]], newshape=(target.shape[0],)) #drop dummy edges
                    if i==0: np.savetxt('tb_data/output_predictions.dat', torch_pred.reshape(1, -1), fmt='%f', delimiter=' ')

                    # hls prediction
                    hls_pred = hls_model.predict(data.hls_data)
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


                if args.synth:
                    hls_model.build(csim=False,synth=True)

if __name__=="__main__":
    main()

