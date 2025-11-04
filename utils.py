import torch
import networkx as nx
import matplotlib.pyplot as plt

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
    adj = adj + adj.t()  # make it symmetric
    adj.fill_diagonal_(0)  # remove self-loops
    
    # For visualization only (using scipy sparse)
    draw_graph_from_adj(adj)
    
    return adj

def generate_y(num_nodes, sigma, L, n):
    # Convert L to tensor if it's not already
    L = torch.tensor(L, dtype=torch.float32)
    
    cov = sigma**2 * torch.eye(num_nodes) + torch.pinverse(L)
    # For generating multivariate normal, we need to make sure covariance is symmetric
    cov = (cov + cov.t()) / 2
    
    # Generate multivariate normal samples
    y = torch.distributions.MultivariateNormal(
        loc=torch.zeros(num_nodes),
        covariance_matrix=cov
    ).sample((n,))
    
    return y

def Laplacian_from_adj(adj):
    D = torch.diag(adj.sum(1))
    L = D - adj
    return L

def GLR(x, L):
    # x in shape (B, N), L in shape (N, N)
    return torch.trace(x @ L @ x.t()) / x.size(0) # scalar, mean of GLR over batch