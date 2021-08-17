import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)
import numpy as np
import glob
from tqdm import tqdm
import os
from utils.data.dataset_pyg import GraphDataset

if __name__=="__main__":

    n_edges = []
    n_nodes = []

    graph_indir = 'trackml_data/processed_plus_pyg_small'

    graph_files = np.array(os.listdir(graph_indir))
    graph_files = np.array([os.path.join(graph_indir, graph_file)
                            for graph_file in graph_files])
    n_graphs_total = len(graph_files)
    IDs = np.arange(n_graphs_total)
    dataset = GraphDataset(graph_files=graph_files[IDs])

    for data in tqdm(dataset, total=len(dataset)):
        n_node = data.x.shape[0]
        n_edge = data.edge_index.shape[0]
        n_edges.append(n_edge)
        n_nodes.append(n_node)

    plt.figure()
    plt.hist(n_edges,weights=np.ones_like(n_edges)/len(n_edges),bins=np.linspace(0,400,41), label='95th percentile: {:d}'.format(int(np.quantile(n_edges, 0.95))))
    plt.xlabel('Edges')
    plt.ylabel('Probability')
    plt.legend()
    plt.savefig('n_edges.png')
    plt.savefig('n_edges.pdf')

    plt.figure()
    plt.hist(n_nodes,weights=np.ones_like(n_nodes)/len(n_nodes),bins=np.linspace(0,200,41), label='95th percentile: {:d}'.format(int(np.quantile(n_nodes, 0.95))))
    plt.xlabel('Nodes')
    plt.ylabel('Probability')
    plt.legend()
    plt.savefig('n_nodes.png')
    plt.savefig('n_nodes.pdf')

    print("mean nodes", np.mean(n_nodes))
    print("mean edges", np.mean(n_edges))
    
    print("std nodes", np.std(n_nodes))
    print("std edges", np.std(n_edges))
    
    print("95% nodes", np.quantile(n_nodes, 0.95))
    print("95% edges", np.quantile(n_edges, 0.95))
