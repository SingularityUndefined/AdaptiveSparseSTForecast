import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse.csgraph import laplacian
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def draw_graph_from_adj(adj, title=None):
    G = nx.from_numpy_array(adj.numpy())
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    edge_labels = {(u, v): f'{d["weight"]:.1f}' for (u, v, d) in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    if title is not None:
        plt.title(title)
    plt.show()

def generate_graph_from_edges(num_nodes, edges, weights=None):
    if weights is None:
        weights = torch.ones(len(edges))
    adj = torch.zeros((num_nodes, num_nodes))
    adj[edges[:, 0], edges[:, 1]] = weights
    adj = adj + adj.t()
    adj.fill_diagonal_(0)
    return adj

def generate_y(num_nodes, sigma, L, n):
    L = torch.tensor(L, dtype=torch.float32)
    cov = sigma**2 * torch.eye(num_nodes) + torch.pinverse(L)
    cov = (cov + cov.t()) / 2
    y = torch.distributions.MultivariateNormal(
        loc=torch.zeros(num_nodes),
        covariance_matrix=cov
    ).sample((n,))
    return y

def generate_and_save_data(seed=0, sigma=0.1, save_dir='data'):
    # Set seed
    set_seed(seed)
    
    # Create data directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Graph parameters
    edges = [[0,1],[0,2],[1,2],[2,3],[3,4],[4,5],[3,5],[5,6],[6,7], [4,7]]
    edges = torch.tensor(edges)
    weights = torch.tensor([0.6]*len(edges))
    num_nodes = 8
    
    # Generate graph
    adj = generate_graph_from_edges(num_nodes, edges, weights)
    L = laplacian(adj, normed=False)
    print("Laplacian:\n", L)
    
    # Generate data
    n = 128
    y = generate_y(num_nodes, sigma, L, n)
    
    # Save data
    data_dict = {
        'adjacency': adj.numpy(),
        'laplacian': L,
        'data': y.numpy(),
        'edges': edges.numpy(),
        'weights': weights.numpy(),
        'parameters': {
            'num_nodes': num_nodes,
            'sigma': sigma,
            'n_samples': n,
            'seed': seed
        }
    }
    
    save_path = os.path.join(save_dir, f'graph_data_seed_{seed}_sigma_{sigma}.npz')
    np.savez(save_path, **data_dict)
    
    return data_dict

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # Generate data with different seeds
    seeds = [0, 42, 123]
    for seed in seeds:
        for sigma in [0.1, 0.5]:
            data = generate_and_save_data(seed, sigma)
            print(f"Data generated and saved for seed {seed} and sigma {sigma}")
            
            # Visualize the graph
            # draw_graph_from_adj(torch.tensor(data['adjacency']), f"Graph (seed={seed}, sigma={sigma})")
        
        
        # Visualize the graph
        # draw_graph_from_adj(torch.tensor(data['adjacency']), f"Graph (seed={seed})")