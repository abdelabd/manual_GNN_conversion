import os
import yaml
import argparse
from multiprocessing import Pool
import subprocess

import numpy as np
import pandas as pd
import torch

from hls4ml.utils.config import config_from_pyg_model
from hls4ml.converters import convert_from_pyg_model
from collections import OrderedDict

# locals
from utils.models.interaction_network_pyg import InteractionNetwork

# helpers
def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='test_config.yaml')

    # graph-size parameters
    add_arg('--max-nodes', type=int, default=28, help='max number of nodes')
    add_arg('--max-edges', type=int, default=56, help='max number of edges')

    # hardware parameters
    add_arg('--reuse', type=int, default=1, help="reuse factor")
    add_arg('--resource-limit', action='store_true', help='if true, then dataflow version implemented, otherwise pipeline version')
    add_arg('--par-factor', type=int, default=16, help='parallelization factor')

    # build-machine parameters
    add_arg('--ssh', action='store_true', help='runs the vivado-build through ssh instead of local machine (must provide ssh details in "build_hls_config.yml"')
    add_arg('--n-jobs', type=int, default=8, help='number of jobs/scripts that can be run on the ssh in parallel')

    add_arg('--output-dir', type=str, default='')

    return parser.parse_args()

def get_hls_model(torch_model, graph_dims,
                  precision='ap_fixed<16,8>', reuse=1, resource_limit=False, par_factor=16, output_dir=""):
    # forward_dict: defines the order in which graph-blocks are called in the model's 'forward()' method
    forward_dict = OrderedDict()
    forward_dict["R1"] = "EdgeBlock"
    forward_dict["O"] = "NodeBlock"
    forward_dict["R2"] = "EdgeBlock"

    precision_str = precision.replace("<", "_")
    precision_str = precision_str.replace(", ", "_")
    precision_str = precision_str.replace(",", "_")
    precision_str = precision_str.replace(">", "")

    if output_dir=="":
        if resource_limit:
            output_dir = f"hls_output/n{graph_dims['n_node']}xe{graph_dims['n_edge']}_dataflow/{precision_str}"
        else:
            output_dir = f"hls_output/n{graph_dims['n_node']}xe{graph_dims['n_edge']}_pipeline/{precision_str}"
    else:
        output_dir = os.path.join(output_dir, precision_str)

    config = config_from_pyg_model(torch_model,
                                   default_precision=precision,
                                   default_index_precision='ap_uint<16>',
                                   default_reuse_factor=reuse)
    hls_model = convert_from_pyg_model(torch_model,
                                       forward_dictionary=forward_dict,
                                       activate_final='sigmoid',
                                       output_dir=output_dir,
                                       hls_config=config,
                                       part='xcvu9p-flga2104-2L-e',
                                       resource_limit=resource_limit,
                                       par_factor=par_factor,
                                       **graph_dims)

    hls_model.compile()
    print("Model compiled at: ", hls_model.config.get_output_dir())
    print("")
    model_config = f"aggregation: {torch_model.aggr} \nflow: {torch_model.flow} \nn_neurons: {torch_model.n_neurons} \nprecision: {precision} \ngraph_dims: {graph_dims} \nreuse_factor: {reuse} \nresource_limit: {resource_limit}"
    with open(hls_model.config.get_output_dir() + "//model_config.txt", "w") as file:
        file.write(model_config)

    return hls_model, output_dir

def build_command(output_dir):
    build_template = "python build_hls.py --directory '{output_dir}'"
    command = build_template.format(output_dir=output_dir)
    os.system(command)

def chunkify(list, n): #converts a list into a list-of-lists, each of size <=n
    list_out = []
    idx_start = 0
    all_members_accounted = False
    while not all_members_accounted:
        idx_stop = min([idx_start+n, len(list)])
        list_i = list[idx_start:idx_stop]
        list_out.append(list_i)

        if idx_stop >= len(list):
            all_members_accounted = True
        else:
            idx_start += n
    return list_out

def main():
    # user-arguments
    args = parse_args()
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)

    # torch-model parameters
    model_config = config['model']
    aggr, flow, hidden_size = model_config['aggr'], model_config['flow'], model_config['n_neurons']

    # graph-size parameters
    graph_dims = {
        "n_node": args.max_nodes,
        "n_edge": args.max_edges,
        "node_dim": 3,
        "edge_dim": 4
    }

    # get torch model
    torch_model_dict = model_config['state_dict']
    try:
        torch_model_dict = torch.load(torch_model_dict)
    except RuntimeError:
        torch_model_dict = torch.load(torch_model_dict, map_location=torch.device('cpu'))
    torch_model = InteractionNetwork(aggr=aggr, flow=flow, hidden_size=hidden_size)
    torch_model.load_state_dict(torch_model_dict)

    # compile all the HLS models, build each model locally if args.ssh==False
    all_output_dirs = []
    #fp_bits = np.arange(6, 20, 2)
    fp_integer_bits = np.arange(2,10,2)
    for fpib in fp_integer_bits:
        precision = f"ap_fixed<10, {fpib}>"
        hls_model, output_dir = get_hls_model(torch_model, graph_dims,
                                              precision=precision, reuse=args.reuse,
                                              resource_limit=args.resource_limit, par_factor=args.par_factor,
                                              output_dir=args.output_dir)
        all_output_dirs.append(output_dir)
        if not args.ssh:
            hls_model.build(csim=False, synth=True, vsynth=True)

    return args, all_output_dirs

if __name__=="__main__":
    args, all_output_dirs = main()
    # if args.ssh==True, build the models remotely and in parallel through ssh (max of n_jobs at a time)
    if args.ssh:
        project_chunks = chunkify(all_output_dirs, args.n_jobs)
        for chunk in project_chunks:
            pool = Pool(args.n_jobs)
            pool.map(build_command, chunk)
            pool.close()
            pool.join()