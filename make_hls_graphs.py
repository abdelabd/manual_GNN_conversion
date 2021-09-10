import os
import yaml
import argparse
import numpy as np
import torch

from utils.data.dataset_pyg import GraphDataset
from utils.data.fix_graph_size import fix_graph_size

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='hls_graph_config.yaml')
    add_arg('--max-nodes', type=int, default=112, help='max number of nodes')
    add_arg('--max-edges', type=int, default=204, help='max number of edges')

    return parser.parse_args()

class data_wrapper(object):
    def __init__(self, node_attr, edge_attr, edge_index, target):
        self.x = node_attr
        self.edge_attr = edge_attr
        self.edge_index = edge_index.transpose(0,1)
        self.target = target

def load_graphs(graph_indir, graph_dims):
    graph_files = np.array(os.listdir(graph_indir))
    graph_files = np.array([os.path.join(graph_indir, graph_file)
                            for graph_file in graph_files])
    n_graphs_total = len(graph_files)
    IDs = np.arange(n_graphs_total)
    dataset = GraphDataset(graph_files=graph_files[IDs])

    graphs = []
    for i, data in enumerate(dataset):
        node_attr, edge_attr, edge_index, target, bad_graph = fix_graph_size(data.x, data.edge_attr, data.edge_index,
                                                                             data.y,
                                                                             n_node_max=graph_dims['n_node'],
                                                                             n_edge_max=graph_dims['n_edge'])
        if not bad_graph:
            graphs.append(data_wrapper(node_attr, edge_attr, edge_index, target))

        if i%1000 == 0:
            print(f"Checked {i} graphs, loaded {len(graphs)}")

    print(f"Total: {len(graphs)} graphs loaded")
    return graphs

def save_graphs(graphs, out_dir):

    os.makedirs(out_dir, exist_ok=True)
    for i, data in enumerate(graphs):
        os.makedirs(out_dir + "//graph%s" % i, exist_ok=True)

        node_attr, edge_attr, edge_index = data.x.detach().cpu().numpy(), data.edge_attr.detach().cpu().numpy(), data.edge_index.transpose(
            0, 1).detach().cpu().numpy().astype(np.int32)
        input_data = np.concatenate([node_attr.reshape(1, -1), edge_attr.reshape(1, -1), edge_index.reshape(1, -1)],
                                    axis=1)
        np.savetxt(out_dir+"//graph%s//input_data.dat"%i, input_data, fmt='%f', delimiter=' ')

        target = data.target.detach().cpu().numpy()
        np.savetxt(out_dir+"//graph%s//target_data.dat"%i, target.reshape(1, -1), fmt='%f', delimiter=' ')

        if i%1000==0:
            print(f"Saved {i} graphs")
    print(f"Total: {i} graphs saved")

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
    graphs = load_graphs(graph_indir, graph_dims)

    # save hls dataset
    print("")
    save_graphs(graphs, config['graph_outdir'])

if __name__=="__main__":
    main()