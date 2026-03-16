import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
def draw_graph(neighbor_list, weights, n_row, title=None, alpha=0.85):
    import matplotlib.pyplot as plt
    import numpy as np
    import networkx as nx

    num_nodes = neighbor_list.size(0)
    G = nx.Graph()

    # 构建图
    for i in range(num_nodes):
        for j, neighbor in enumerate(neighbor_list[i]):
            if neighbor.item() > i:  # 避免重复边
                G.add_edge(i, neighbor.item(), weight=weights[i, j].item())

    # 布局
    pos = {i: (i % n_row, -(i // n_row)) for i in range(num_nodes)}

    # 获取边权重
    edge_attr = nx.get_edge_attributes(G, "weight")
    edge_weights = np.array([edge_attr[e] for e in G.edges()])

    # colormap 映射
    norm = plt.Normalize(vmin=0, vmax=1)
    cmap = plt.cm.YlGnBu

    # 🔑 创建 figure / axes
    fig, ax = plt.subplots(figsize=(6, 6))

    # 1️⃣ 按权重排序绘制边（小权重先，大权重后）
    edges_sorted = sorted(G.edges(data=True), key=lambda e: e[2]["weight"])
    for e in edges_sorted:
        u, v, d = e
        w = d["weight"]
        color = cmap(np.sqrt(w))
        lc = nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            width=2.0 + 4.0 * w,
            edge_color=[color],
            alpha=alpha,
            ax=ax
        )
        lc.set_zorder(w)  # 权重大 → 在上面显示

    # 2️⃣ 绘制节点
    lc = nx.draw_networkx_nodes(
        G, pos,
        ax=ax,
        node_color='lightgrey',
        edgecolors='grey',  # 节点轮廓
        linewidths=1.5,
    )
    lc.set_zorder(edge_weights.max() + 1)

    nx.draw_networkx_labels(
        G, pos,
        ax=ax,
        # font_size=10,
        font_color='black', 
    )


    # 3️⃣ 可选边标签
    # edge_labels = {(u, v): f'{d["weight"]:.2f}' for (u, v, d) in G.edges(data=True)}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    # 4️⃣ colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # 必须设置
    cbar = fig.colorbar(sm, ax=ax, label="edge weight (sqrt scale)", shrink=0.8)
    # 可选：colorbar刻度显示原始weight而不是sqrt
    tick_locs = np.linspace(0, np.sqrt(edge_weights.max()), 5)
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([f"{t**2:.2f}" for t in tick_locs])

    if title is not None:
        ax.set_title(title, fontsize=14, pad=12)

    ax.set_aspect("equal")
    ax.axis("off")
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
                    neighbors.append(ny * n_row + nx)
        neighbors = sorted(neighbors, key=lambda idx: ( (idx % n_row - x) ** 2 + (idx // n_row - y) ** 2 ))
        if k <= len(neighbors):
            neighbor_list.append(neighbors[:k])
        else:
            neighbor_list.append(neighbors + [-1] * (k - len(neighbors)))  # pad with -1 if not enough neighbors
    return torch.tensor(neighbor_list)

def generate_grid_neigbhors(n_row):
    num_nodes = n_row * n_row
    neighbor_list = []
    for i in range(num_nodes):
        x, y = i % n_row, i // n_row
        neighbors = []
        for pos in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
            nx, ny = x + pos[0], y + pos[1]
            if 0 <= nx < n_row and 0 <= ny < n_row:
                neighbors.append(ny * n_row + nx)
        if len(neighbors) < 4:
            neighbors += [-1] * (4 - len(neighbors))
        neighbor_list.append(neighbors)
    return torch.tensor(neighbor_list)

def generate_grid_kNN_mask(n_row, kernel, k):
    grid_neighbors = generate_grid_neigbhors(n_row)
    neighbor_list = generate_kNN_from_grid(n_row, kernel, k)
    mask = (neighbor_list != -1) & (neighbor_list.unsqueeze(2) == grid_neighbors.unsqueeze(1)).any(dim=2)
    return mask

def generate_y_from_grid(n_row, sigma, n, edge_weights=0.6):
    num_nodes = n_row * n_row
    neighbor_list = generate_grid_neigbhors(n_row)
    neighbor_mask = (neighbor_list != -1)

    # if edge weights is a constant, fill all

    edge_weights = edge_weights if torch.is_tensor(edge_weights) else torch.ones((num_nodes, 4)) * edge_weights
    edge_weights = edge_weights * neighbor_mask.float()
    # print(neighbor_list, edge_weights)

    # compute Laplacian F-norm
    degrees = edge_weights.sum(dim=1)  # (N, )
    fro_norm = torch.sqrt((degrees ** 2).sum() + (edge_weights ** 2).sum())

    print('Laplacian F-norm:', fro_norm.item())


    # generate sparse laplacian matrix
    L = torch.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j, neighbor in enumerate(neighbor_list[i]):
            if neighbor.item() != -1:
                L[i, neighbor.item()] = -edge_weights[i, j]
        L[i, i] = -L[i].sum()
    
    assert torch.allclose(L, L.t()), "Laplacian matrix is not symmetric"

    # plot the graph
    draw_graph(neighbor_list, edge_weights, n_row, title=f"{n_row}x{n_row} Grid Graph")

    J = torch.ones((num_nodes,num_nodes), device=L.device) / num_nodes
    L_pinv = torch.linalg.inv(L + J) - J
    # print(L_pinv)

    cov = sigma**2 * torch.eye(num_nodes) + L_pinv
    cov = (cov + cov.t()) / 2
    
    y = torch.distributions.MultivariateNormal(
        loc=torch.zeros(num_nodes),
        covariance_matrix=cov
    ).sample((n,))
    
    return y


# Example usage
if __name__ == "__main__":
    n_row = 5
    kernel = 3
    k = 8
    feature_dim = 3
    num_nodes = n_row * n_row
    # signal 
    features = torch.randn(num_nodes, feature_dim) * 0.4
    # compute weight matrix
    # nearest_neighbors = generate_grid_neigbhors(n_row) #  generate_kNN_from_grid(n_row, kernel, k)
    # edge_weights = features[nearest_neighbors.view(-1)].reshape(num_nodes, 4, feature_dim) - features.unsqueeze(1)  # (N, 4, feature_dim)
    # edge_weights = torch.exp(- (edge_weights ** 2).sum(-1))  # (N, 4)


    nearest_neighbors = generate_kNN_from_grid(n_row, kernel, k)
    edge_weights = features[nearest_neighbors.view(-1)].reshape(num_nodes, k, feature_dim) - features.unsqueeze(1)  # (N, k, feature_dim)
    edge_weights = torch.exp(- (edge_weights ** 2).sum(-1))  # (N, k)

    print(nearest_neighbors, edge_weights)
    draw_graph(nearest_neighbors, edge_weights, n_row, title=f"{n_row}x{n_row} Grid k-NN Graph with k={k}")


    y = generate_y_from_grid(n_row, sigma=0.1, n=128)
    print(y.shape)  # (128, num_nodes)