import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def draw_graph(neighbor_list, weights, n_row, title=None):
    num_nodes = neighbor_list.size(0)
    G = nx.Graph()
    for i in range(num_nodes):
        for j, neighbor in enumerate(neighbor_list[i]):
            if neighbor.item() != -1:
                G.add_edge(i, neighbor.item(), weight=weights[i, j].item())
            
    pos = {i: (i % n_row, -(i // n_row)) for i in range(num_nodes)}
    weights = nx.get_edge_attributes(G, "weight")
    edge_weights = [weights[e] for e in G.edges()]
    w = np.array(edge_weights)
    linewidths = 2.0 + 3.0 * (w - w.min()) / (w.max() - w.min() + 1e-8)
    norm = plt.Normalize(vmin=w.min(), vmax=w.max())
    mapped = np.sqrt((w - w.min()) / (w.max() - w.min()))
    cmap = plt.cm.YlGnBu
    edge_colors = cmap(mapped)


    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color=edge_colors, width=linewidths)
    edge_labels = {(u, v): f'{d["weight"]:.2f}' for (u, v, d) in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label="edge weight")

    plt.axis("equal")
    plt.axis("off")

    if title is not None:
        plt.title(title)
    plt.show()


def generate_kNN_from_grid(n_row, kernel, k):
    num_nodes = n_row * n_row
    neighbor_list = []
    for i in range(num_nodes):
        x, y = i % n_row, i // n_row
        neighbors = []
        lrange = kernel // 2
        for dx in range(-lrange, lrange + 1):
            for dy in range(-lrange, lrange + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < n_row and 0 <= ny < n_row:
                    if i == 2:
                        print(f"Adding neighbor ({nx}, {ny}) for node ({x}, {y}), offset ({dx}, {dy})")
                    neighbors.append(ny * n_row + nx)
        neighbors = sorted(neighbors, key=lambda idx: ( (idx % n_row - x) ** 2 + (idx // n_row - y) ** 2 ))
        if k <= len(neighbors):
            neighbor_list.append(neighbors[:k])
        else:
            neighbor_list.append(neighbors + [-1] * (k - len(neighbors)))  # pad with -1 if not enough neighbors
    return torch.tensor(neighbor_list)

def generate_grid_neighbors(n_row, kernel):
    num_nodes = n_row * n_row
    neighbor_list = []
    for i in range(num_nodes):
        x, y = i % n_row, i // n_row
        neighbors = []
        lrange = kernel // 2
        for dx in range(-lrange, lrange + 1):
            for dy in range(-lrange, lrange + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < n_row and 0 <= ny < n_row:
                    neighbors.append(ny * n_row + nx)
        neighbor_list.append(neighbors)
        print(neighbor_list)
    return torch.tensor(neighbor_list)


# Example usage
n_row = 5
kernel = 3
k = 9
feature_dim = 3

num_nodes = n_row * n_row
# signal 
features = torch.randn(num_nodes, feature_dim) * 0.1
# compute weight matrix
nearest_neighbors =  generate_grid_neighbors(n_row, kernel) # generate_kNN_from_grid(n_row, kernel, k)
edge_weights = features[nearest_neighbors.view(-1)].reshape(num_nodes, kernel**2 - 1, feature_dim) - features.unsqueeze(1)  # (N, k, feature_dim)
edge_weights = torch.exp(- (edge_weights ** 2).sum(-1))  # (N, k)

print(nearest_neighbors, edge_weights)
draw_graph(nearest_neighbors, edge_weights, n_row, title=f"{n_row}x{n_row} Grid k-NN Graph with k={k}")