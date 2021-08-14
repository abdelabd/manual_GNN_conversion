import os
import yaml
import argparse

import numpy as np
import pandas as pd
import torch

from hls4ml.utils.config import config_from_pyg_model
from hls4ml.converters import convert_from_pyg_model
from hls4ml.model.hls_model import HLSModel
from collections import OrderedDict

# locals
from utils.models.interaction_network_pyg import InteractionNetwork
from utils.data.dataset_pyg import GraphDataset
from utils.data.fix_graph_size import fix_graph_size

# helpers
def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='test_config.yaml')
    add_arg('--max-nodes', type=int, default=112, help='max number of nodes')
    add_arg('--max-edges', type=int, default=148, help='max number of edges')
    add_arg('--n-neurons', type=int, default=40, choices=[8, 40], help='number of neurons')
    add_arg('--n-graphs', type=int, default=100)
    add_arg('--aggregation', type=str, default='add', choices =['add', 'mean', 'max', 'all'], help='[add, mean, max, all]')
    add_arg('--flow', type=str, default='source_to_target', choices = ['source_to_target', 'target_to_source', 'all'], help='[source_to_target, target_to_source, all]')
    add_arg('--reuse', type=int, default=1, help="reuse factor")
    add_arg('--output-dir', type=str, default="", help='output directory')

    args = parser.parse_args()
    if args.aggregation == 'all':
        args.aggregation = ['add', 'mean', 'max']
    else:
        args.aggregation = [args.aggregation]

    if args.flow == 'all':
        args.flow = ['source_to_target', 'target_to_source']
    else:
        args.flow = [args.flow]

    if args.n_neurons == 'all':
        args.n_neurons = [8, 40]
    else:
        args.n_neurons = [args.n_neurons]

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

def get_hls_model(torch_model, graph_dims, precision='ap_fixed<16,8>', reuse=1, output_dir=""):
    # forward_dict: defines the order in which graph-blocks are called in the model's 'forward()' method
    forward_dict = OrderedDict()
    forward_dict["R1"] = "EdgeBlock"
    forward_dict["O"] = "NodeBlock"
    forward_dict["R2"] = "EdgeBlock"

    if output_dir == "":
        output_dir = "hls_output/%s"%torch_model.aggr + "/%s"%torch_model.flow + "/neurons_%s"%torch_model.n_neurons + "/%s"%precision
    config = config_from_pyg_model(torch_model,
                                   default_precision=precision,
                                   default_index_precision='ap_uint<16>',
                                   default_reuse_factor=reuse)
    hls_model = convert_from_pyg_model(torch_model,
                                       n_edge=graph_dims['n_edge'],
                                       n_node=graph_dims['n_node'],
                                       edge_dim=graph_dims['edge_dim'],
                                       node_dim=graph_dims['node_dim'],
                                       forward_dictionary=forward_dict,
                                       activate_final='sigmoid',
                                       output_dir=output_dir,
                                       hls_config=config)

    hls_model.compile()
    print("Model compiled at: ", hls_model.config.get_output_dir())
    model_config = f"aggregation: {torch_model.aggr} \nflow: {torch_model.flow} \nn_neurons: {torch_model.n_neurons} \nprecision: {precision} \ngraph_dims: {graph_dims} \nreuse_factor: {reuse}"
    with open(hls_model.config.get_output_dir() + "//model_config.txt", "w") as file:
        file.write(model_config)

    return hls_model

def load_graphs(graph_indir, graph_dims, n_graphs):
    graph_files = np.array(os.listdir(graph_indir))
    graph_files = np.array([os.path.join(graph_indir, graph_file)
                            for graph_file in graph_files])
    n_graphs_total = len(graph_files)
    IDs = np.arange(n_graphs_total)
    dataset = GraphDataset(graph_files=graph_files[IDs])

    graphs = []
    for data in dataset[:n_graphs]:
        node_attr, edge_attr, edge_index, bad_graph = fix_graph_size(data.x, data.edge_attr, data.edge_index,
                                                                     n_node_max=graph_dims['n_node'],
                                                                     n_edge_max=graph_dims['n_edge'])
        if not bad_graph:
            target = data.y
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

def get_single_ROC(target, pred, density):
    discriminants = np.linspace(0, 1, density)
    tpr = np.zeros(len(discriminants))
    fpr = np.zeros(len(discriminants))

    for i, disc in enumerate(discriminants):
        new_pred = np.zeros(len(pred))
        new_pred[pred>=disc]=1

        tp = true_pos(target, new_pred)
        tn = true_neg(target, new_pred)
        fp = false_pos(target, new_pred)
        fn = false_neg(target, new_pred)
        tpr[i] = true_pos_rate(target, new_pred)
        fpr[i] = false_pos_rate(target, new_pred)

    return tpr, fpr

def test_model(model, graphs, targets=None):
    density=100
    TPR = np.zeros(density)
    FPR = np.zeros(density)
    predictions = []
    if targets is None:
        targets = [data.np_target for data in graphs]

    for i, data in enumerate(graphs):
        target_i = targets[i]

        if isinstance(model, torch.nn.Module):
            pred = model(data).detach().cpu().numpy()
            if i == 0: np.savetxt('tb_data/output_predictions.dat', pred.reshape(1, -1), fmt='%f', delimiter=' ')
        elif isinstance(model, HLSModel):
            pred = model.predict(data.hls_data)

        pred = np.reshape(pred[:target_i.shape[0]], newshape=(target_i.shape[0],))  # drop dummy edges
        predictions.append(pred)

        tpr_i, fpr_i = get_single_ROC(target_i, pred, density)
        TPR += tpr_i
        FPR += fpr_i

    TPR *= 1/len(graphs)
    FPR *= 1/len(graphs)

    return TPR, FPR, predictions

# stat helpers (the sklearn functions require the estimator as input, but this doesn't work so well with an hls model)
def true_pos(target, pred):
    n = 0
    for i in range(len(target)):
       if target[i]==1 and pred[i]==1:
           n+=1
    return n

def false_pos(target, pred):
    n = 0
    for i in range(len(target)):
        if target[i]==0 and pred[i]==1:
            n+=1
    return n

def true_neg(target, pred):
    n=0
    for i in range(len(target)):
        if target[i]==0 and pred[i]==0:
            n+=1
    return n

def false_neg(target, pred):
    n=0
    for i in range(len(target)):
        if target[i]==1 and pred[i]==0:
            n+=1
    return n

def true_pos_rate(target, pred):
    tp = true_pos(target, pred)
    fn = false_neg(target, pred)
    if tp==0:
        return 0
    else:
        return tp/(tp+fn)

def false_pos_rate(target, pred):
    fp = false_pos(target, pred)
    tn = true_neg(target, pred)
    if fp==0:
        return 0
    else:
        return fp/(fp+tn)

def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)

    # load graphs and target-data, write first graph to testbench folder
    graph_indir = config['graph_indir']
    graph_dims = {
        "n_node": args.max_nodes,
        "n_edge": args.max_edges,
        "node_dim": 3,
        "edge_dim": 4
    }
    graphs = load_graphs(graph_indir, graph_dims, args.n_graphs)

    #fp_bits = [16]
    fp_bits = np.arange(10, 34, 2)
    FPR = []
    TPR = []
    precision_labels = []
    for a in args.aggregation:
        for f in args.flow:
            for nn in args.n_neurons:

                # first get torch predictions
                torch_model = InteractionNetwork(aggr=a, flow=f, hidden_size=nn)
                torch_model_dict = torch.load(config['trained_model_dir'] + "//IN_pyg_small" + f"_{a}" + f"_{f}" + f"_{nn}" + "_state_dict.pt")
                torch_model.load_state_dict(torch_model_dict)

                torch_TPR, torch_FPR, torch_predictions = test_model(torch_model, graphs)
                torch_precision_labels = ["torch" for i in range(len(torch_FPR))]
                FPR.extend(torch_FPR)
                TPR.extend(torch_TPR)
                precision_labels.extend(torch_precision_labels)

                # iterate through precisions and get hls predictions
                for fpb in fp_bits:
                    if fpb<16:
                        fpib=6
                    else:
                        fpib=8
                    precision = f"ap_fixed<{fpb}, {fpib}>"
                    hls_model = get_hls_model(torch_model, graph_dims, precision=precision, reuse=args.reuse, output_dir=args.output_dir)

                    hls_TPR, hls_FPR, hls_predictions = test_model(hls_model, graphs)
                    hls_precision_labels = [f"hls: {precision}" for i in range(len(hls_FPR))]
                    FPR.extend(hls_FPR)
                    TPR.extend(hls_TPR)
                    precision_labels.extend(hls_precision_labels)

                    hls_to_torch_TPR, hls_to_torch_FPR, hls_predictions = test_model(hls_model, graphs, targets=[np.round(i) for i in torch_predictions])
                    hls_to_torch_precision_labels = [f"hls-->torch: {precision}" for i in range(len(hls_to_torch_FPR))]
                    FPR.extend(hls_to_torch_FPR)
                    TPR.extend(hls_to_torch_TPR)
                    precision_labels.extend(hls_to_torch_precision_labels)

                os.makedirs(f"numbers_for_paper/{a}/{f}/neurons_{nn}", exist_ok=True)
                df = pd.DataFrame({"false_pos_rate": FPR, "true_pos_rate": TPR, "precision": precision_labels})
                df.to_csv(f"numbers_for_paper/{a}/{f}/neurons_{nn}/ROC_data.csv", index=False)

if __name__=="__main__":
    main()


