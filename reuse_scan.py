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
    add_arg('--max-nodes', type=int, default=28, help='max number of nodes')
    add_arg('--max-edges', type=int, default=37, help='max number of edges')
    add_arg('--n-neurons', type=int, default=8, choices=[8, 40], help='number of neurons')
    add_arg('--aggregation', type=str, default='add', choices =['add', 'mean', 'max', 'all'], help='[add, mean, max, all]')
    add_arg('--flow', type=str, default='source_to_target', choices = ['source_to_target', 'target_to_source', 'all'], help='[source_to_target, target_to_source, all]')
    add_arg('--precision', type=str, default='ap_fixed<16,8>', help='fixed-point precision')
    add_arg('--ssh', action='store_true', help='runs the vivado-build through ssh instead of local machine (must provide ssh details in "build_hls_config.yml"')
    add_arg('--n-jobs', type=int, default=8, help='number of jobs/scripts that can be run on the ssh in parallel')

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

def get_hls_model(torch_model, graph_dims, precision='ap_fixed<16,8>', reuse=1):
    # forward_dict: defines the order in which graph-blocks are called in the model's 'forward()' method
    forward_dict = OrderedDict()
    forward_dict["R1"] = "EdgeBlock"
    forward_dict["O"] = "NodeBlock"
    forward_dict["R2"] = "EdgeBlock"

    output_dir = "hls_output/" + f"n{graph_dims['n_node']}xe{graph_dims['n_edge']}" + "/rf%s"%reuse
    config = config_from_pyg_model(torch_model,
                                   default_precision=precision,
                                   default_index_precision='ap_uint<16>',
                                   default_reuse_factor=reuse)
    hls_model = convert_from_pyg_model(torch_model,
                                       forward_dictionary=forward_dict,
                                       activate_final='sigmoid',
                                       output_dir=output_dir,
                                       hls_config=config,
                                       fpga_part='xcvu9p-flga2104-2L-e',
                                       **graph_dims)

    hls_model.compile()
    print("Model compiled at: ", hls_model.config.get_output_dir())
    print("")
    model_config = f"aggregation: {torch_model.aggr} \nflow: {torch_model.flow} \nn_neurons: {torch_model.n_neurons} \nprecision: {precision} \ngraph_dims: {graph_dims} \nreuse_factor: {reuse}"
    with open(hls_model.config.get_output_dir() + "//model_config.txt", "w") as file:
        file.write(model_config)

    return hls_model, output_dir

def build_command(output_dir):
    build_template = "python build_hls.py --directory '{output_dir}'"
    command = build_template.format(output_dir=output_dir)
    subprocess.Popen(command, shell=True)

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
    args = parse_args()
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)

    graph_dims = {
        "n_node": args.max_nodes,
        "n_edge": args.max_edges,
        "node_dim": 3,
        "edge_dim": 4
    }

    # compile all the models, build each model locally if args.ssh==False
    all_output_dirs = []
    reuse_factors = [8, 16, 24, 32, 40, 48, 56, 64]
    for a in args.aggregation:
        for f in args.flow:
            for nn in args.n_neurons:
                # get torch model
                torch_model = InteractionNetwork(aggr=a, flow=f, hidden_size=nn)
                torch_model_dict = torch.load(config['trained_model_dir'] + "//IN_pyg_small" + f"_{a}" + f"_{f}" + f"_{nn}" + "_state_dict.pt")
                torch_model.load_state_dict(torch_model_dict)

                # get hls model
                for rf in reuse_factors:
                    hls_model, output_dir = get_hls_model(torch_model, graph_dims, precision=args.precision, reuse=rf)
                    all_output_dirs.append(output_dir)
                    if not args.ssh:
                        hls_model.build(csim=False, synth=True, vsynth=True)

    return args, all_output_dirs

if __name__=="__main__":
    args, all_output_dirs = main()
    # if args.ssh==True, build the models remotely and in parallel through ssh (max of n_jobs at a time)
    if args.ssh:
        project_chunks = chunkify(all_output_dirs, args.n_jobs)
        pool = Pool(args.n_jobs)
        for project in project_chunks:
            pool.map(build_command, project)
            pool.join()

        pool.close()
