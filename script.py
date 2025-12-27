import torch

neighbor_list = torch.tensor([
    [1, 2, 3],
    [0, 2, 4],
    [0, 1, 5],
    [0, 4, 5],
    [1, 3, 5],
    [2, 3, 4]
])  # example neighbor list for 6 nodes with 3 neighbors each   


adj = torch.randn((6, 3))  # example sparse adjacency matrix in (N, k) format

edge_indices = neighbor_list.view(-1)#  + (torch.arange(6).unsqueeze(1) * 6)

e = torch.eye(6)

edge_diffs = e.repeat_interleave(3, dim=0) - e[edge_indices.view(-1)]  

print(edge_indices)  # print edge indices in (N, k) form

print(edge_diffs.shape)  # print edge difference vectors