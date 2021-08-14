import os
import yaml
import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import seaborn as sns
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from hls4ml.utils.config import config_from_pyg_model
from hls4ml.converters import convert_from_pyg_model

# locals
from utils.models.interaction_network_pyg import InteractionNetwork
from utils.data.dataset_pyg import GraphDataset
from utils.data.fix_graph_size import fix_graph_size

# stat helpers
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

def sensitivity(tp, fn):
    if tp==0:
        return 0
    else:
        return tp/(tp+fn)

def specificity(tn, fp):
    if tn==0:
        return 0
    else:
        return tn/(tn+fp)

def get_sens_spec(target, pred, discriminants):
    sensit = np.zeros(len(discriminants))
    specif = np.zeros(len(discriminants))

    for i, disc in enumerate(discriminants):
        new_pred = np.zeros(len(pred))
        new_pred[pred>=disc]=1

        tp = true_pos(target, new_pred)
        tn = true_neg(target, new_pred)
        fp = false_pos(target, new_pred)
        fn = false_neg(target, new_pred)
        sensit[i] = sensitivity(tp, fn)
        specif[i] = specificity(tn, fp)

    return sensit, specif

# other helpers
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

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='test_config.yaml')
    add_arg('--output-dir', type=str, default="")
    add_arg('--n-graphs', type=int, default=100)
    add_arg('--max-nodes', type=int, default=112)
    add_arg('--max-edges', type=int, default=148)

    add_arg('--aggregation', type=str, default='add', help='[add, mean, max, all]')
    add_arg('--n-neurons', type=str, default="40")

    args = parser.parse_args()

    if args.aggregation=='all':
        args.aggregation = ['add', 'mean', 'max']
    else: args.aggregation = [args.aggregation]

    if args.n_neurons == "all":
        args.n_neurons = [8,40]
    else: args.n_neurons = [int(args.n_neurons)]

    return args

def main():
    args = parse_args()
    with open(args.config) as file:
        config = yaml.load(file, yaml.FullLoader)

    # dataset
    graph_indir = config['graph_indir']
    graph_dims = {
        "n_node": args.max_nodes,
        "n_edge": args.max_edges,
        "node_dim": 3,
        "edge_dim": 4
    }
    graphs = load_graphs(graph_indir, graph_dims, args.n_graphs)

    #fp_bits = [10,12,14, 16, 18, 20]
    fp_bits = np.arange(10, 32, 2)
    discriminants = np.linspace(0,1,100)
    for a in args.aggregation:
        for nn in args.n_neurons:

            # load torch model, predict and get ROC
            torch_model = InteractionNetwork(aggr=a, flow="source_to_target", hidden_size=nn)
            torch_model_dict = torch.load(config['trained_model_dir'] + f"//IN_pyg_small_{a}_source_to_target_{nn}_state_dict.pt")
            torch_model.load_state_dict(torch_model_dict)
            torch_sensitivity = np.zeros(len(discriminants))
            torch_specificity = np.zeros(len(discriminants))
            for i, data in enumerate(graphs):
                target = data.np_target
                torch_pred = torch_model(data).detach().cpu().numpy()
                torch_pred = np.reshape(torch_pred[:target.shape[0]],newshape=(target.shape[0],))  # drop dummy edges
                tsens_i, tspec_i = get_sens_spec(target, torch_pred, discriminants)

                torch_sensitivity += tsens_i
                torch_specificity += tspec_i

            torch_sensitivity *= 1 / len(graphs)
            torch_specificity *= 1 / len(graphs)

            df = pd.DataFrame(columns=["sensitivity", "specificity", "precision"])
            for fpb in fp_bits:
                if fpb<16:
                    fpib=6
                else: fpib=8
                precision = f"ap_fixed<{fpb}, {fpib}>"
                hls_model = get_hls_model(torch_model, graph_dims, precision=precision)

                hls_sensitivity = np.zeros(len(discriminants))
                hls_specificity = np.zeros(len(discriminants))
                for i, data in enumerate(graphs):
                    target = np.reshape(data.target.detach().cpu().numpy(), newshape=(data.target.shape[0],))

                    # hls prediction
                    hls_pred = hls_model.predict(data.hls_data)
                    hls_pred = np.reshape(hls_pred[:target.shape[0]], newshape=(target.shape[0],)) #drop dummy edges

                    hsens_i, hspec_i = get_sens_spec(target, hls_pred, discriminants)
                    hls_sensitivity += hsens_i
                    hls_specificity += hspec_i

                hls_sensitivity *= 1 / len(graphs)
                hls_specificity *= 1 / len(graphs)
                df_i = pd.DataFrame({"sensitivity": hls_sensitivity, "specificity": hls_specificity, "precision": [f"ap_fixed<{fpb}, {fpib}>" for i in range(len(discriminants))]})
                df = df.append(df_i, ignore_index=True)

            df["false_pos"] = 1-df["specificity"].copy()
            df["true_pos"] = df["sensitivity"].copy()
            plt.figure()
            plt.plot(1-torch_specificity, torch_sensitivity, "r", label="torch")
            sns.lineplot(x="false_pos", y="true_pos", hue="precision", palette="crest", data=df)
            plt.title(f"aggr={a}, hidden neurons={nn}")
            plt.savefig(f"numbers_for_paper/{a}/{torch_model.flow}/neurons_{nn}/ROC.jpg")
            plt.close()

            df_i = pd.DataFrame(
                {"sensitivity": torch_sensitivity, "specificity": torch_specificity, "false_pos": 1 - torch_specificity,
                 "true_pos": torch_sensitivity, "precision": ["torch" for i in range(len(torch_sensitivity))]})
            df = df.append(df_i, ignore_index=True)
            os.makedirs(f"numbers_for_paper/{a}/{torch_model.flow}/neurons_{nn}", exist_ok=True)
            df.to_csv(f"numbers_for_paper/{a}/{torch_model.flow}/neurons_{nn}/ROC_data.csv", index=False)

if __name__=="__main__":
    main()

