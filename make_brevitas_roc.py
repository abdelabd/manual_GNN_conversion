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
from utils.models.interaction_network_pyg import InteractionNetwork as InteractionNetwork_float
from utils.models.interaction_network_brevitas import InteractionNetwork as InteractionNetwork_quantized
from test_model import data_wrapper, load_graphs, reshape_pred

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='roc_brevitas_config.yaml')

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

    # dataset
    graph_indir = config['graph_indir']
    graph_dims = {
        "n_node": args.max_nodes,
        "n_edge": args.max_edges,
        "node_dim": 3,
        "edge_dim": 4
    }
    graphs = load_graphs(graph_indir, graph_dims, args.n_graphs, exclude_bad_graphs=args.exclude_bad_graphs)

    # load floating point torch model, predict and get ROC
    float_model = InteractionNetwork_float(aggr=aggr, flow=flow, hidden_size=n_neurons)
    try:
        float_model_dict = torch.load(config['model']['float_state_dict'])
    except RuntimeError:
        torch_model_dict = torch.load(config['model']['float_state_dict'], map_location=torch.device('cpu'))
    float_model.load_state_dict(float_model_dict)

    target_all = []
    float_pred_all = []
    for i, data in tqdm(enumerate(graphs), total=len(graphs)):
        target = data.np_target
        float_pred = float_model(data).detach().cpu().numpy()
        float_pred = reshape_pred(target, float_pred)
        target_all.append(target)
        float_pred_all.append(float_pred)

    target_all = np.concatenate(target_all)
    float_pred_all = np.concatenate(float_pred_all)

    bit_widths = np.arange(2, 16, 2) #20, 2)
    quantized_pred_all = {}
    for bw in bit_widths:

        # Get quantized model
        quantized_model = InteractionNetwork_quantized(aggr=aggr, flow=flow, hidden_size=n_neurons, bit_width=int(bw))
        quantized_model_dict = os.path.join(config['model']['quantized_state_dict_dir'], f"PyG_geometric_2GeV_{bw}b.pt")
        try:
            quantized_model_dict = torch.load(quantized_model_dict)
        except RuntimeError:
            quantized_model_dict = torch.load(quantized_model_dict, map_location=torch.device('cpu'))
        quantized_model.load_state_dict(quantized_model_dict)

        quantized_pred_all[bw] = []
        for i, data in tqdm(enumerate(graphs), total=len(graphs)):
            target = data.np_target

            # quantized prediction
            quantized_pred = quantized_model(data).detach().cpu().numpy()
            quantized_pred = reshape_pred(target, quantized_pred)
            quantized_pred_all[bw].append(quantized_pred)

        quantized_pred_all[bw] = np.concatenate(quantized_pred_all[bw])

    fpr_float, tpr_float, _ = roc_curve(target_all, float_pred_all)
    auc_float = auc(fpr_float, tpr_float) * 100.
    plt.figure()
    plt.plot(fpr_float, tpr_float, "r", label=f"Full floating-point, AUC = {auc_float:.1f}%", lw=4)

    fpr_quantized, tpr_quantized, auc_quantized= {}, {}, {}
    linestyles = ['dotted', 'dashed', 'dashdot', (0, (1, 10)), (0, (5, 10)), (0, (3, 10, 1, 10)),
                  (0, (3, 10, 1, 10, 1, 10)), (0, (3, 1, 1, 1, 1, 1))]
    for bw, linestyle in zip(bit_widths, linestyles):
        fpr_quantized[bw], tpr_quantized[bw], _ = roc_curve(target_all, quantized_pred_all[bw])
        auc_quantized[bw] = auc(fpr_quantized[bw], tpr_quantized[bw]) * 100.
        #bw_label = precision.replace('ap_fixed', '')
        plt.plot(fpr_quantized[bw], tpr_quantized[bw], label=f'{bw} bits, AUC = {auc_quantized[bw]:.1f}%',
                 linestyle=linestyle, lw=4)
    plt.legend(
        title=f"{args.max_nodes} nodes, {args.max_edges} edges\n{config['phi_sections']} $\phi$ sectors, {config['eta_sections']} $\eta$ sectors")
    plt.tight_layout()
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.savefig(os.path.join(top_dir, "ROC.png"))
    plt.savefig(os.path.join(top_dir, "ROC.pdf"))
    plt.semilogx()
    plt.tight_layout()
    plt.savefig(os.path.join(top_dir, "ROC_logx.png"))
    plt.savefig(os.path.join(top_dir, "ROC_logx.pdf"))
    plt.close()

    plt.figure()
    plt.plot(bit_widths, [auc_quantized[bw] for bw in bit_widths], label='hls4ml', lw=4, ms=10, marker='o')
    plt.plot(bit_widths, [auc_float for bw in bit_widths], label='PyG (expected)', linestyle='--', color='gray', lw=4,
             ms=10)
    plt.xlabel('Total bits')
    plt.ylabel('AUC [%]')
    plt.legend(
        title=f"{args.max_nodes} nodes, {args.max_edges} edges\n{config['phi_sections']} $\phi$ sectors, {config['eta_sections']} $\eta$ sectors")
    plt.tight_layout()
    plt.savefig(os.path.join(top_dir, "AUC.png"))
    plt.savefig(os.path.join(top_dir, "AUC.pdf"))
    plt.close()


if __name__ == "__main__":
    main()

