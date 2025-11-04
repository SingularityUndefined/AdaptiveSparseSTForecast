from GEM_module import GEM
from utils import *
from scipy.sparse.csgraph import laplacian

# seed
torch.manual_seed(42)
torch.random.manual_seed(42)

# generate synthetic data
edges = [[0,1],[0,2],[1,2],[2,3],[3,4],[4,5],[3,5]]
edges = torch.tensor(edges)
weights = torch.tensor([0.6]*len(edges))
num_nodes = 6
adj = generate_graph_from_edges(num_nodes, edges, weights)
print(adj.norm()**2)
L = laplacian(adj, normed=False)
print(L)
# generate data
sigma = 0.1
mu = sigma ** 2
n = 128
y = generate_y(num_nodes, sigma, L, n)

# torch.manual_seed(42)
gem = GEM(num_nodes, mu=0.01, gamma=0.4, step_size=0.01, c=5, scale=True)
# initialize adjacency and S
# adj_init = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)  # all-one matrix with zero diagonal
S_init = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)  # all-one matrix with zero diagonal
# run GEM
# with torch.no_grad():
    # draw_graph_from_adj(adj_init, title='Initial Graph')
x_final, adj_final, S_final = gem(y, S_init, num_iters=5)
W_final = adj_final * S_final
print("Final learned adjacency matrix:")
print(W_final, W_final.norm()**2)