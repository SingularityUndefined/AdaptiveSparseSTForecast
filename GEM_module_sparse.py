import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphLearningModule(nn.Module):
    # generate weight matrix from node embeddings
    def __init__(self, num_nodes, num_neighbors, neighbor_list, emb_dim=6, feature_dim=3, c=8, theta=0.5):
        super(GraphLearningModule, self).__init__()
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors
        self.neighbor_list = neighbor_list # in (N, k)
        self.emb_dim = emb_dim
        self.feature_dim = feature_dim

        # embedding vectors
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, emb_dim))  
        self.fc = nn.Linear(emb_dim, feature_dim)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.c = c
        self.theta = theta

    def forward(self, x, params=None):
        """
        x: (B, N)
        params: dict[str, tensor] (可选，用于 MAML/unrolled optimization)
        """
        if params is None:
            node_embeddings = self.node_embeddings
            fc_weight = self.fc.weight
            fc_bias = self.fc.bias
        else:
            node_embeddings = params["node_embeddings"]
            fc_weight = params["fc.weight"]
            fc_bias = params["fc.bias"]

        B = x.size(0)

        # 1. embed each node (add)
        e = x.unsqueeze(-1) + node_embeddings.unsqueeze(0)  # (B, N, emb_dim)

        # 2. Feature generation: linear + activation
        f = F.linear(e, fc_weight, fc_bias)   # (B, N, feature_dim)
        f = self.leakyrelu(f) # in (B, N, feature_dim)

        # 3. pairwise difference
        df = f.unsqueeze(2) - f[:, self.neighbor_list.view(-1)].reshape(B, self.num_nodes, self.num_neighbors, self.feature_dim)  # (B, N, k, feature_dim)

        # 4. adjacency by RBF kernel
        adj = torch.exp(-(df ** 2).sum(-1) / (2 * self.theta)).mean(0)  # (N, k)

        return adj # (N, k)
    
class Generalized_EM(nn.Module):
    def __init__(self, num_nodes, num_neighbors, neighbor_list, mu, gamma, step_size, emb_dim=6, feature_dim=3, c=8, theta=0.5, PGD_iters=100, PGD_step_size=0.01, use_block_coordinate=False, scale=True):
        super(Generalized_EM, self).__init__()
        self.glm = GraphLearningModule(num_nodes, num_neighbors, neighbor_list, emb_dim, feature_dim, c, theta)
        self.neighbor_list = neighbor_list
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors

        self.mu = mu
        self.c = c # Here, C is the F-norm contraint, not squared
        self.step_size = step_size
        self.PGD_step_size = PGD_step_size
        self.gamma = gamma
        self.scale = scale
        self.PGD_iters = PGD_iters

        # could be removed later
        self.use_block_coordinate = use_block_coordinate

    def scale_adj(self, adj):
        degree = torch.sum(adj, dim=1)  # (N,)
        scale_factor = 1
        if self.scale:
            L_norm_square = (degree ** 2).sum() + (adj ** 2).sum()
            scale_factor = self.c / torch.sqrt(L_norm_square + 1e-8)

        return adj * scale_factor, scale_factor
    
    def apply_L(self, X, adj):
        # adj in (N, k), X in (B, N)
        B = X.size(0)
        N = adj.size(0)
        k = adj.size(1)
        neighbor_X = X[:, self.neighbor_list.view(-1)].reshape(B, N, k)  # (B, N, k)
        Lx = torch.sum(adj * (X.unsqueeze(2) - neighbor_X), dim=2)  # (B, N)
        return Lx
        
    def LHS_E_step(self, x, adj):
        # solve equation (I + mu*L)X = Y
        return self.mu * self.apply_L(x, adj) + x
    
    def solve_E_step(self, y, adj):
        # y in (B, N)
        # solve equation (I + mu*L)X = Y 
        from scipy.sparse.linalg import LinearOperator, cg
        def func_LHS_E_step(x):
            x_tensor = torch.tensor(x, dtype=y.dtype, device=y.device).view(y.size())
            result = self.LHS_E_step(x_tensor, adj)
            return result.cpu().numpy().ravel()
        
        LHS_E_step_op = LinearOperator((y.numel(), y.numel()), matvec=func_LHS_E_step)
        x = torch.zeros_like(y)
        for i in range(y.size(0)):
            b = y[i].cpu().numpy()
            x_i, info = cg(LHS_E_step_op, b, atol=1e-6, maxiter=100)
            x[i] = torch.tensor(x_i, dtype=y.dtype, device=y.device).view(y.size(1))

        return x  # return in shape (B, N)
    
    def adj_to_dense(self, adj):
        N = adj.size(0)
        dense_adj = torch.zeros((N, N), device=adj.device)
        for i in range(N):
            neighbors = self.neighbor_list[i]  # (k,)
            dense_adj[i, neighbors] = adj[i]
        return dense_adj
    
    def L_matrix(self, adj):
        N = adj.size(0)
        degree = torch.sum(adj, dim=1)  # (N,)
        D = torch.diag(degree)  # (N, N)
        L = D - self.adj_to_dense(adj)  # (N, N)
        return L
    
    def direct_E_step(self, y, adj):
        # y in (B, N)
        B = y.size(0)
        N = adj.size(0)
        L = self.L_matrix(adj)
        cov = torch.eye(N, device=adj.device).unsqueeze(0) + self.mu * L  # (N, N)
        return torch.linalg.solve(cov, y.t()).t()
    
    def inv_LJ_matrix(self, adj):
        # return 
    
    def trace_SL(self, S, adj):
        # sparse adj in (N, k), compute in sparse way

    
    def M_step_1(self, x, adj):
        N = self.num_nodes
        J = torch.ones((N, N), device=x.device) / N
        B = x.size(0)
        X_centered = x - x.mean(dim=1, keepdim=True)  # (B, N)
        S = (X_centered.t() @ X_centered) / B  # (N, N)

        L = self.L_matrix(adj)  # (N, N)
        trace_SL = torch.trace(S @ L)
        grad_W = self.mu * (torch.diag(L.sum(1)) - S)  # (N, N)

        # gradient descent step
        dense_adj = self.adj_to_dense(adj)
        W_new = dense_adj - self.step_size * grad_W

        # projection to non-negative and zero-diagonal
        W_new = torch.clamp(W_new, min=0)
        W_new.fill_diagonal_(0)

        # projection to F-norm ball
        W_new, scale_factor = self.scale_adj(W_new)

        # convert back to sparse representation
        adj_new = W_new[torch.arange(N).unsqueeze(1), self.neighbor_list]  # (N, k)

        return adj_new