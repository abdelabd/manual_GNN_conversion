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
    add_arg('--max-nodes', type=int, default=28, help='max number of nodes')
    add_arg('--max-edges', type=int, default=37, help='max number of edges')
    add_arg('--n-neurons', type=int, default=8, choices=[8, 40], help='number of neurons')
    add_arg('--aggregation', type=str, default='add', choices =['add', 'mean', 'max', 'all'], help='[add, mean, max, all]')
    add_arg('--flow', type=str, default='source_to_target', choices = ['source_to_target', 'target_to_source', 'all'], help='[source_to_target, target_to_source, all]')
    add_arg('--reuse', type=int, default=1, help="reuse factor")

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

def get_hls_model(torch_model, graph_dims, precision='ap_fixed<16,8>', reuse=1, output_dir=""):
    # forward_dict: defines the order in which graph-blocks are called in the model's 'forward()' method
    forward_dict = OrderedDict()
    forward_dict["R1"] = "EdgeBlock"
    forward_dict["O"] = "NodeBlock"
    forward_dict["R2"] = "EdgeBlock"

    if output_dir == "":
        precision_str = precision.replace("<", "_")
        precision_str = precision_str.replace(", ", "_")
        precision_str = precision_str.replace(">", "")

        output_dir = "hls_output/" + "%s"%torch_model.aggr + "/%s"%torch_model.flow + "/neurons_%s"%torch_model.n_neurons + "/%s"%precision_str
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

    return hls_model, output_dir

def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)

    build_template = "python build_hls.py --directory '{output_dir}'"
    graph_dims = {
        "n_node": args.max_nodes,
        "n_edge": args.max_edges,
        "node_dim": 3,
        "edge_dim": 4
    }

    fp_bits = np.arange(10, 34, 2)
    for a in args.aggregation:
        for f in args.flow:
            for nn in args.n_neurons:
                # get torch model
                torch_model = InteractionNetwork(aggr=a, flow=f, hidden_size=nn)
                torch_model_dict = torch.load(config['trained_model_dir'] + "//IN_pyg_small" + f"_{a}" + f"_{f}" + f"_{nn}" + "_state_dict.pt")
                torch_model.load_state_dict(torch_model_dict)

                # get hls model
                for fpb in fp_bits:
                    if fpb<16:
                        fpib=6
                    else:
                        fpib=8
                    precision = f"ap_fixed<{fpb}, {fpib}>"
                    hls_model, output_dir = get_hls_model(torch_model, graph_dims, precision=precision, reuse=args.reuse)
                    output_dir = output_dir.replace("hls_output/", "")


                    print(f"precision: {precision}")
                    print(f"output_dir: {output_dir}")
                    print("")
                    build_command = build_template.format(output_dir=output_dir)
                    os.system(build_command)

if __name__=="__main__":
    main()

