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

from hls4ml.utils.config import config_from_pyg_model
from hls4ml.converters import convert_from_pyg_model

from tqdm import tqdm

# locals
from utils.models.interaction_network_pyg import InteractionNetwork
from utils.data.dataset_pyg import GraphDataset
from utils.data.fix_graph_size import fix_graph_size
from test_model import data_wrapper, load_graphs

# other helpers
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

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='roc_config.yaml')
    add_arg('--max-nodes', type=int, default=112)
    add_arg('--max-edges', type=int, default=204)
    add_arg('--n-graphs', type=int, default=100)
    add_arg('--exclude-bad-graphs', action='store_true', help='if false, truncated and padded-but-not-separate graphs are included in the performance assessment')

    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config) as file:
        config = yaml.load(file, yaml.FullLoader)

    aggr = config['model']['aggr']
    flow = config['model']['flow']
    n_neurons = config['model']['n_neurons']
    top_dir = config['output_dir']
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

    fp_bits = np.arange(6, 20, 2)
    precisions = [f"ap_fixed<{fpb}, {int(fpb/2)}>" for fpb in fp_bits]

    # load torch model, predict and get ROC
    torch_model = InteractionNetwork(aggr=aggr, flow=flow, hidden_size=n_neurons)
    try:
        torch_model_dict = torch.load(config['model']['state_dict'])
    except RuntimeError:
        torch_model_dict = torch.load(config['model']['state_dict'], map_location=torch.device('cpu'))
    torch_model.load_state_dict(torch_model_dict)
            
    target_all = []
    torch_pred_all = []
    for i, data in tqdm(enumerate(graphs), total=len(graphs)):
        target = data.np_target
        torch_pred = torch_model(data).detach().cpu().numpy()
        torch_pred = np.reshape(torch_pred[:target.shape[0]],newshape=(target.shape[0],))  # drop dummy edges
        target_all.append(target)
        torch_pred_all.append(torch_pred)

    target_all = np.concatenate(target_all)
    torch_pred_all = np.concatenate(torch_pred_all)

    hls_pred_all = {}
    for precision in precisions:
        hls_pred_all[precision] = []
        hls_model = get_hls_model(torch_model, graph_dims, precision=precision)

        for i, data in tqdm(enumerate(graphs), total=len(graphs)):
            target = np.reshape(data.target.detach().cpu().numpy(), newshape=(data.target.shape[0],))

            # hls prediction
            hls_pred = hls_model.predict(data.hls_data)
            hls_pred = np.reshape(hls_pred[:target.shape[0]], newshape=(target.shape[0],)) #drop dummy edges
            hls_pred_all[precision].append(hls_pred)

        hls_pred_all[precision] = np.concatenate(hls_pred_all[precision])

    fpr_torch, tpr_torch, _ = roc_curve(target_all, torch_pred_all)
    auc_torch = auc(fpr_torch, tpr_torch)*100.
    plt.figure()
    plt.plot(fpr_torch, tpr_torch, "r", label=f"PyG, AUC = {auc_torch:.1f}%", lw=4)
    fpr_hls, tpr_hls, auc_hls = {}, {}, {}
    linestyles = ['dotted', 'dashed', 'dashdot', (0, (1, 10)), (0, (5, 10)), (0, (3, 10, 1, 10)), (0, (3, 10, 1, 10, 1, 10)), (0, (3, 1, 1, 1, 1, 1))]            
    for precision, linestyle in zip(precisions, linestyles):
        fpr_hls[precision], tpr_hls[precision], _ = roc_curve(target_all, hls_pred_all[precision])
        auc_hls[precision] = auc(fpr_hls[precision], tpr_hls[precision])*100.
        precision_label = precision.replace('ap_fixed','')
        plt.plot(fpr_hls[precision], tpr_hls[precision], label=f'{precision_label}, AUC = {auc_hls[precision]:.1f}%', linestyle=linestyle, lw=4)
    plt.legend(title=f"{args.max_nodes} nodes, {args.max_edges} edges\n{config['phi_sections']} $\phi$ sectors, {config['eta_sections']} $\eta$ sectors")
    plt.tight_layout()
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.savefig(os.path.join(top_dir,"ROC.png"))
    plt.savefig(os.path.join(top_dir,"ROC.pdf"))
    plt.semilogx()
    plt.tight_layout()
    plt.savefig(os.path.join(top_dir,"ROC_logx.png"))
    plt.savefig(os.path.join(top_dir,"ROC_logx.pdf"))
    plt.close()

    plt.figure()
    plt.plot(fp_bits, [auc_hls[precision] for precision in precisions], label='hls4ml', lw=4, ms=10, marker='o')
    plt.plot(fp_bits, [auc_torch for fp_bit in fp_bits], label='PyG (expected)',linestyle='--', color='gray', lw=4, ms=10)
    plt.xlabel('Total bits')
    plt.ylabel('AUC [%]')
    plt.legend(title=f"{args.max_nodes} nodes, {args.max_edges} edges\n{config['phi_sections']} $\phi$ sectors, {config['eta_sections']} $\eta$ sectors")
    plt.tight_layout()
    plt.savefig(os.path.join(top_dir,"AUC.png"))
    plt.savefig(os.path.join(top_dir,"AUC.pdf"))
    plt.close()

if __name__=="__main__":
    main()

