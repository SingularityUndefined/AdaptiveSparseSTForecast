import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_compact import draw_graph_from_adj, GLR
import math
# Graph Learning Module

class GraphLearningModule(nn.Module):
    # generate weight matrix from node embeddings
    def __init__(self, num_nodes, emb_dim=6, feature_dim=3, c=8):
        super(GraphLearningModule, self).__init__()
        self.num_nodes = num_nodes
        self.emb_dim = emb_dim
        self.feature_dim = feature_dim

        # embedding vectors
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, emb_dim))  
        self.fc = nn.Linear(emb_dim, feature_dim)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.c = c

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
        f = self.leakyrelu(f)

        # 3. pairwise difference
        df = f.unsqueeze(2) - f.unsqueeze(1)  # (B, N, N, feature_dim)

        # 4. adjacency by RBF kernel
        adj = torch.exp(-(df ** 2).sum(-1)).mean(0)  # (N, N)
        adj.fill_diagonal_(0)

        return adj

class GEM(nn.Module):
    def __init__(self, num_nodes, mu, gamma, step_size, emb_dim=6, feature_dim=3, c=8, PGD_iters=100, PGD_step_size=0.01, use_block_coordinate=False, scale=True):
        super(GEM, self).__init__()
        self.glm = GraphLearningModule(num_nodes, emb_dim, feature_dim, c)
        # self.S = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)  # all-one matrix with zero diagonal
        self.mu = mu
        self.c = c
        self.step_size = step_size
        self.PGD_step_size = PGD_step_size
        self.gamma = gamma
        self.scale = scale
        self.PGD_iters = PGD_iters
        self.use_block_coordinate = use_block_coordinate
    
    def scale_W(self, adj, S):
        W = adj * S
        scale_factor = 1
        if self.scale:
            scale_factor = math.sqrt(self.c) / W.norm() if W.norm() != 0 else 1
            W = W * scale_factor
        return W, scale_factor

    def E_step(self, y, W):
        # y in (B, N)
        B = y.size(0)
        D = torch.diag(W.sum(1))
        L = D - W
        cov = torch.eye(self.glm.num_nodes) + self.mu * L
        return torch.linalg.solve(cov, y.t()).t()  # return in shape (B, N)
    
    def M_step_1(self, x, L):
        N = self.glm.num_nodes
        J = torch.ones((N, N), device=x.device) / N
        loss = GLR(x, L) - torch.logdet(L + J)
        # backward to gradient descent
        grads = torch.autograd.grad(loss, self.glm.parameters(), create_graph=True)
        new_params = {}
        for (name, param), grad in zip(self.glm.named_parameters(), grads):
            new_params[name] = param - self.step_size * grad
        return new_params

    def tilde_operation(self, Sigma):
        N = Sigma.size(0)
        diag_Sigma = torch.diag(Sigma)
        tilde_Sigma = Sigma - (diag_Sigma.unsqueeze(0) + diag_Sigma.unsqueeze(1)) / 2
        return tilde_Sigma
    
    def soft_thresholding(self, W, tau):
        return torch.sign(W) * torch.clamp(torch.abs(W) - tau, min=0.0)

    def M_step_2(self, x, W0, S_init=None):
        # for fixed x, Sigma is fixed
        N = self.glm.num_nodes
        if S_init is None:
            S_init = torch.ones((N, N), device=x.device) - torch.eye(N, device=x.device)  # all-one matrix with zero diagonal
        S = S_init
        Sigma = x.T @ x / x.shape[0] # (N, N)
        tilde_Sigma = self.tilde_operation(Sigma)

        J = torch.ones((N, N), device=x.device) / N
        H = torch.ones((N, N), device=x.device) - torch.eye(N, device=x.device)

        # pre-define the first tilde_R
        W = W0 * S
        L = torch.diag(W.sum(1)) - W
        R = torch.inverse(J + L)
        tilde_R = self.tilde_operation(R)

        if self.use_block_coordinate:

            for iter in range(self.PGD_iters):
                S_new = S.clone()
                for i in range(N):
                    for j in range(i+1, N): # iterate over all upper triangular entries
                        # if i == j:
                        #     continue
                        # PGD
                        # tilde_R_old = tilde_R_old.clone()
                        S_old_ij = S_new[i, j]
                        S_ij = S_new[i, j] - self.PGD_step_size * (W0[i, j] * (tilde_R[i, j] - tilde_Sigma[i, j]))
                        S_ij = self.soft_thresholding(S_ij, tau=self.PGD_step_size * self.gamma)
                        S_ij = torch.clamp(S_ij, min=0.0, max=1.0)
                        delta_S_ij = S_ij - S_old_ij
                        # OUT-OF-PLACE index update
                        S_new = S_new.index_put((torch.tensor([i]), torch.tensor([j])), S_ij)
                        S_new = S_new.index_put((torch.tensor([j]), torch.tensor([i])), S_ij)
                        # update tilde_R after each row update
                        if delta_S_ij != 0:
                            delta_r = tilde_R[i] - tilde_R[j]
                            tilde_Q = self.tilde_operation(torch.ger(delta_r, delta_r))  # (N, N)
                            tilde_R = tilde_R - (delta_S_ij / (1 - 2 * delta_S_ij * tilde_R[i, j])) * tilde_Q

                if iter % 20 == 0:
                    print(f'Block Coordinate PGD iter {iter+1}/{self.PGD_iters}, ||S_new - S||_F = {torch.norm(S_new - S):.4f}')
                S = S_new
        else:
            for i in range(self.PGD_iters):
                # update full tilde_R
                # PGD
                S_new = S - self.PGD_step_size * (W0 * (tilde_R - tilde_Sigma)) 
                S_new = self.soft_thresholding(S_new, tau=self.PGD_step_size * self.gamma)
                S_new = torch.clamp(S_new, min=0.0, max=1.0)
                S_new.fill_diagonal_(0)
                if i % 20 == 0:
                    print(f'PGD iter {i+1}/{self.PGD_iters}, ||S_new - S||_F = {torch.norm(S_new - S):.4f}')
                # update tilde_R
                S = S_new
                W = W0 * S  # initialize W
                L = torch.diag(W.sum(1)) - W
                R = torch.inverse(J + L)
                tilde_R = self.tilde_operation(R)

        return S
       
    # needs to be further simplified to be more efficient


    def single_step(self, y, adj, S, params=None):
        # y in (B, N)
        B = y.size(0)
        W0, _ = self.scale_W(adj, S)

        # E-step
        x = self.E_step(y, W0)
        # recompute graph
        adj1 = self.glm(x, params=params) # unregularized adjacency
        W1, _ = self.scale_W(adj1, S)  # element-wise product to enforce sparsity pattern
        L1 = torch.diag(W1.sum(1)) - W1
        print(f'after E-step: delta_W norm {torch.norm(W1 - W0):.4f}, GLR {GLR(x, L1):.4f}, adj norm^2 {adj1.norm()**2:.4f}')

        # M-step-1: minimize GLR
        new_params = self.M_step_1(x, L1) # update parameters in glm
        adj2 = self.glm(x, params=new_params) # unregularized adjacency
        W2, alpha2 = self.scale_W(adj2, S)  # element-wise product to enforce sparsity pattern
        L2 = torch.diag(W2.sum(1)) - W2
        print(f'after M-step-1: delta_W norm {torch.norm(W2 - W1):.4f}, GLR {GLR(x, L2):.4f}, adj norm^2 {adj2.norm()**2:.4f}')

        # M-step-2: optimizing alpha with gradient steps
        S1 = self.M_step_2(x, adj2 * alpha2)
        print(S1)
        return x, adj2 * alpha2, S1, params
    
    def forward(self, y, S_init, num_iters=10, adj_init=None, params=None):
        adj = self.glm(y) if adj_init is None else adj_init
        S = S_init
        _, alpha = self.scale_W(adj, S)
        adj = adj * alpha  # scale initial adjacency
        
        for it in range(num_iters):
            print(f'Iteration {it+1}/{num_iters}')
            x, adj, S, params = self.single_step(y, adj, S, params=params)
            W = adj * S
            print('W norm^2 at Iteration', it+1, W.norm()**2)
            threshold = 5e-3
            W_plot = W.clone()
            W_plot[W_plot < threshold] = 0.0
            draw_graph_from_adj(W_plot.detach().cpu(), title=f'Learned Graph at Iteration {it+1}')
        return x, adj, S