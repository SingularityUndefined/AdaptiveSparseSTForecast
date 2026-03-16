import torch

def draw_graph(neighbor_list:torch.Tensor, weights, n_row, title=None, alpha=0.85, filename=None):
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
    fig, ax = plt.subplots(figsize=(16, 16))

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
        node_size=100
    )
    lc.set_zorder(edge_weights.max() + 1)

    # nx.draw_networkx_labels(
    #     G, pos,
    #     ax=ax,
    #     # font_size=10,
    #     font_color='black', 
    # )


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
    plt.savefig(filename, bbox_inches='tight', pad_inches=0) if filename is not None else plt.show()
    plt.close()