import os
import yaml
import argparse
from collections import OrderedDict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

import mplhep as hep

plt.style.use(hep.style.ROOT)
from tqdm import tqdm

# locals
from utils.models.interaction_network_pyg import InteractionNetwork as InteractionNetwork
from utils.models.interaction_network_brevitas import InteractionNetwork as InteractionNetwork_quantized
from utils.data.dataset_pyg import GraphDataset
from utils.data.fix_graph_size import fix_graph_size
from test_model import data_wrapper, reshape_pred#, load_graphs as load_fixed_graphs
from make_roc import get_hls_model

def load_fixed_graphs(graph_indir, graph_dims, n_graphs, exclude_bad_graphs=False):
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
                graphs.append(data_wrapper(node_attr, edge_attr, edge_index, data.y))
        else:
            graphs.append(data_wrapper(node_attr, edge_attr, edge_index, data.y))

    print(f"n_graphs: {len(graphs)}")

    print("writing test bench data for 1st graph")
    data = graphs[0]
    node_attr, edge_attr, edge_index = data.x.detach().cpu().numpy(), data.edge_attr.detach().cpu().numpy(), data.edge_index.transpose(
        0, 1).detach().cpu().numpy().astype(np.int32)
    os.makedirs('tb_data', exist_ok=True)
    input_data = np.concatenate([node_attr.reshape(1, -1), edge_attr.reshape(1, -1), edge_index.reshape(1, -1)], axis=1)
    np.savetxt('tb_data/input_data.dat', input_data, fmt='%f', delimiter=' ')

    return graphs

def load_unfixed_graphs(graph_indir, graph_dims, n_graphs, exclude_bad_graphs=False):
    graph_files = np.array(os.listdir(graph_indir))
    graph_files = np.array([os.path.join(graph_indir, graph_file)
                            for graph_file in graph_files])
    n_graphs_total = len(graph_files)
    IDs = np.arange(n_graphs_total)
    dataset = GraphDataset(graph_files=graph_files[IDs])

    graphs = []
    for data in dataset[:n_graphs]:
        _, _, _, _, bad_graph = fix_graph_size(data.x, data.edge_attr, data.edge_index,
                                                                             data.y,
                                                                             n_node_max=graph_dims['n_node'],
                                                                             n_edge_max=graph_dims['n_edge'])

        if exclude_bad_graphs:
            if not bad_graph:
                graphs.append(data_wrapper(data.x, data.edge_attr, data.edge_index, data.y))
        else:
            graphs.append(data_wrapper(data.x, data.edge_attr, data.edge_index, data.y))

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
    add_arg('config', nargs='?', default='roc_pqt_qat_config.yaml')

    add_arg('--n-graphs', type=int, default=100)
    add_arg('--exclude-bad-graphs', action='store_true',
            help='if false, truncated and padded-but-not-separate graphs are included in the performance assessment')

    add_arg('--max-nodes', type=int, default=113)
    add_arg('--max-edges', type=int, default=196)

    add_arg('--output-dir', type=str, default='roc_plots')

    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as file:
        config = yaml.load(file, yaml.FullLoader)

    aggr = config['model']['aggr']
    flow = config['model']['flow']
    n_neurons = config['model']['n_neurons']
    top_dir = args.output_dir
    os.makedirs(top_dir, exist_ok=True)

    # datasets
    graph_indir = config['graph_indir']
    graph_dims_small = {
        "n_node": 113,
        "n_edge": 196,
        "node_dim": 3,
        "edge_dim": 4
    }
    graph_dims_large = {
        "n_node": 448,
        "n_edge": 896,
        "node_dim": 3,
        "edge_dim": 4
    }

    graphs_small = load_fixed_graphs(graph_indir, graph_dims_small, args.n_graphs, exclude_bad_graphs=False)
    graphs_large = load_fixed_graphs(graph_indir, graph_dims_large, args.n_graphs, exclude_bad_graphs=True)

    # load floating point torch model
    torch_model = InteractionNetwork(aggr=aggr, flow=flow, hidden_size=n_neurons)
    try:
        torch_model_dict = torch.load(config['model']['torch_state_dict'])
    except RuntimeError:
        torch_model_dict = torch.load(config['model']['torch_state_dict'], map_location=torch.device('cpu'))
    torch_model.load_state_dict(torch_model_dict)

    # get target and Torch predictions: 1x(n_graph*n_edge)
    target_small = [] # small graphs
    torch_pred_small = []  # small graphs
    for i, data in tqdm(enumerate(graphs_small), total=len(graphs_small)):
        target = data.np_target
        target_small.append(target)
        torch_pred = torch_model(data).detach().cpu().numpy()
        torch_pred = reshape_pred(target, torch_pred)
        torch_pred_small.append(torch_pred)
    target_small = np.concatenate(target_small)
    torch_pred_small = np.concatenate(torch_pred_small)

    target_large = [] # large graphs
    torch_pred_large = []  # large graphs
    for i, data in tqdm(enumerate(graphs_large), total=len(graphs_large)):
        target = data.np_target
        target_large.append(target)
        torch_pred = torch_model(data).detach().cpu().numpy()
        torch_pred = reshape_pred(target, torch_pred)
        torch_pred_large.append(torch_pred)
    target_large = np.concatenate(target_large)
    torch_pred_large = np.concatenate(torch_pred_large)


    # get HLS4ML post-training-quantization (PQT) predictions: 1x(n_graph*n_edge_max)
    fp_int_bits = np.arange(1, 19, 1)
    precisions = [f"ap_fixed<18, {X}>" for X in fp_int_bits]
    hls_pred_small = {}
    hls_pred_large = {}
    for precision in precisions:
        # Get hls model
        hls_model_small = get_hls_model(torch_model, graph_dims_small, precision=precision)
        hls_model_large = get_hls_model(torch_model, graph_dims_large, precision=precision)

        hls_pred_small[precision] = [] # small graphs
        for i, data in tqdm(enumerate(graphs_small), total=len(graphs_small)):
            target = data.np_target
            hls_pred = hls_model_small.predict(data.hls_data)
            hls_pred = reshape_pred(target, hls_pred)
            hls_pred_small[precision].append(hls_pred)
        hls_pred_small[precision] = np.concatenate(hls_pred_small[precision])

        hls_pred_large[precision] = []  # "good" graphs
        for i, data in tqdm(enumerate(graphs_large), total=len(graphs_large)):
            target = data.np_target
            hls_pred = hls_model_large.predict(data.hls_data)
            hls_pred = reshape_pred(target, hls_pred)
            hls_pred_large[precision].append(hls_pred)
        hls_pred_large[precision] = np.concatenate(hls_pred_large[precision])

    # get torch performance on small graphs
    fpr_torch_small, tpr_torch_small, _ = roc_curve(target_small, torch_pred_small)
    auc_torch_small = auc(fpr_torch_small, tpr_torch_small) * 100.

    # get torch performance on large graphs
    fpr_torch_large, tpr_torch_large, _ = roc_curve(target_large, torch_pred_large)
    auc_torch_large = auc(fpr_torch_large, tpr_torch_large) * 100.

    # get HLS4ML performance on all graphs
    fpr_hls_small, tpr_hls_small, auc_hls_small = {}, {}, {}
    for precision in precisions:
        fpr_hls_small[precision], tpr_hls_small[precision], _ = roc_curve(target_small, hls_pred_small[precision])
        auc_hls_small[precision] = auc(fpr_hls_small[precision], tpr_hls_small[precision]) * 100.

    # get HLS4ML performance on large graphs
    fpr_hls_large, tpr_hls_large, auc_hls_large = {}, {}, {}
    for precision in precisions:
        fpr_hls_large[precision], tpr_hls_large[precision], _ = roc_curve(target_large, hls_pred_large[precision])
        auc_hls_large[precision] = auc(fpr_hls_large[precision], tpr_hls_large[precision]) * 100.

    # Plot
    plt.figure()
    plt.plot(fp_int_bits, [auc_hls_small[precision] for precision in precisions], color='blue', linestyle=':', label="HLS4ML (PTQ), 113x196", lw=4, ms=10)#, marker='o')
    plt.plot(fp_int_bits, [auc_hls_large[precision] for precision in precisions], color='blue', linestyle='-', label="HLS4ML (PTQ), 448x896", lw=4, ms=10)#, marker='o')
    plt.plot(fp_int_bits, [auc_torch_small for precision in precisions], color='gray', linestyle=':', label="Torch (floating-point), 113x196", lw=4, ms=10) #linestyle='--', color='gray', )
    plt.plot(fp_int_bits, [auc_torch_large for precision in precisions], color='gray', linestyle='-', label="Torch (floating-point), 448x896", lw=4, ms=10)#linestyle='--', color='gray', )
    plt.xlabel('ap_fixed<18,X>')
    plt.ylabel('AUC [%]')
    plt.legend(
        title=f"{config['phi_sections']} $\phi$ sectors, {config['eta_sections']} $\eta$ sectors")
    plt.tight_layout()
    plt.savefig(os.path.join(top_dir, "AUC.png"))
    plt.savefig(os.path.join(top_dir, "AUC.pdf"))
    plt.close()

if __name__ == "__main__":
    main()

