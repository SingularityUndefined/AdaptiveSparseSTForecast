import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
# NOTE: here uses true CG methods
from model.sparse_CG import *
import numpy as np
import math
from sksparse.cholmod import cholesky
import scipy.sparse as sp
from .utils import draw_graph

class GraphLearningModule(L.LightningModule):
    # generate weight matrix from node embeddings
    def __init__(self, num_nodes, num_neighbors, neighbor_list, emb_dim=6, feature_dim=3, c=8, theta=0.5):
        super(GraphLearningModule, self).__init__()
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors
        self.neighbor_list = neighbor_list # in (N, k)
        self.neighbor_mask = (neighbor_list != -1).to('cuda:0')# .to(self.device)  # (N, k), original w mask
        self.emb_dim = emb_dim
        self.feature_dim = feature_dim

        # embedding vectors
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, emb_dim))  
        self.fc = nn.Linear(emb_dim + 1, feature_dim)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.c = c
        self.theta = nn.Parameter(torch.tensor(theta))

    def forward(self, x):
        """
        x: (B, N)
        params: dict[str, tensor] (可选，用于 MAML/unrolled optimization)
        """

        B = x.size(0)

        # 1. embed each node (add)
        e = torch.cat([x.unsqueeze(-1), self.node_embeddings.unsqueeze(0).repeat(B, 1, 1)], dim=-1)  # (B, N, emb_dim + 1)

        # 2. Feature generation: linear + activation
        f = self.fc(e)   # (B, N, feature_dim)
        f = self.leakyrelu(f) # in (B, N, feature_dim)

        # 3. pairwise difference
        print(f.device, self.neighbor_list.device)
        df = f.unsqueeze(2) - f[:, self.neighbor_list.view(-1)].reshape(B, self.num_nodes, self.num_neighbors, self.feature_dim)  # (B, N, k, feature_dim)

        # 4. wacency by RBF kernel
        w = torch.exp(-(df ** 2).sum(-1) / (2 * self.theta)).mean(0)  # (N, k)
        if torch.isnan(w).any():
            print('embedded x + e:', torch.isnan(e).any())
            print('features f:', torch.isnan(f).any())
            raise ValueError('NaN detected in w computation!')

        w = w * self.neighbor_mask.float()  # mask non-edges
        assert torch.allclose(w * self.neighbor_mask.float(), w), "w has non-zero weights on non-edges!"

        return w # (N, k)

class GEM_GNN(L.LightningModule):
    def __init__(self, num_nodes, num_neighbors, neighbor_list, mu, gamma, emb_dim=6, feature_dim=3, c=8, theta=0.5, method='CG', inv_method='L+eI', CG_iters=10, PGD_iters=100, PGD_step_size=0.01, scale=True, GEM_iters=5, lr=0.01):
        super().__init__()
        self.automatic_optimization = False  # 手动优化

        # GNN backbone
        self.glm = GraphLearningModule(num_nodes, num_neighbors, neighbor_list, emb_dim, feature_dim, c, theta)
        # Graph hyperparameters
        self.neighbor_list = neighbor_list
        self.neighbor_mask = (neighbor_list >= 0).to('cuda:0')  # (N, k), original w mask
        # self.A_ori = self.neighbor_mask.float()
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors

        # filter strength and energy constraints
        self.mu = mu
        self.c = c # Here, C is the F-norm contraint, not squared

        # PGD hyperparameters
        self.PGD_step_size = PGD_step_size
        self.gamma = gamma
        self.scale = scale

        # linear system solver
        self.method = method
        assert method in ['CG', 'cholmod'], "Only 'CG' and 'cholmod' are supported currently."
        self.CG_iters = CG_iters
        self.PGD_iters = PGD_iters

        self.inv_method = inv_method
        assert inv_method in ['L+J', 'L+eI'], "Only 'L+J' and 'L+eI' are supported currently."
        # GEM hyperparameters

        # learn rate
        self.lr = lr

        # initialize A
        self.A = self.neighbor_mask.float()# .to('cuda:0')  # (N, k)
        self.graph_list = []
        self.register_buffer('w_0', None)  # initial weight matrix
    
    def scale_w(self, w):
        '''
        scale to fit F-norm constraint of Laplacian (||L||_F = c)
        '''
        # print(w.device, self.neighbor_mask.device)
        assert torch.allclose(w * self.neighbor_mask.float().to(w.device), w), "w has non-zero weights on non-edges!"
        degree = torch.sum(w, dim=1)  # (N,)
        scale_factor = 1
        if self.scale:
            L_norm_square = (degree ** 2).sum() + (w ** 2).sum()
            scale_factor = self.c / torch.sqrt(L_norm_square)

        return w * scale_factor, scale_factor

    def apply_L(self, X, w):
        # w in (N, k), X in (B, N)
        B = X.size(0)
        N = w.size(0)
        k = w.size(1)
        neighbor_X = X[:, self.neighbor_list.view(-1)].reshape(B, N, k)  # (B, N, k)
        neighbor_X = neighbor_X * self.neighbor_mask.unsqueeze(0).float().to(w.device)  # mask non-edges
        assert torch.allclose(w * self.neighbor_mask.float().to(w.device), w), "w has non-zero weights on non-edges!"
        Lx = torch.sum(w * (X.unsqueeze(2) - neighbor_X), dim=2)  # (B, N)
        return Lx
    
    def apply_L_plus_J(self, X, w):
        # w in (N, k), X in (B, N)
        N = w.size(0)
        Jx = X.mean(dim=1, keepdim=True).expand_as(X)  # (B, N)
        Lx = self.apply_L(X, w)  # (B, N)
        return Lx + Jx
    
    def apply_L_plus_epsilon_I(self, X, w, epsilon=4e-3):
        # w in (N, k), X in (B, N)
        Lx = self.apply_L(X, w)  # (B, N)
        return Lx + epsilon * X
        
    def LHS_E_step(self, x, w):
        # solve equation (I + mu*L)X = Y
        return self.mu * self.apply_L(x, w) + x # (B, N)
    
    def E_step(self, y, w):
        # y in (B, N)
        B = y.size(0)

        def A_func(x):
            return self.LHS_E_step(x, w)

        x, res_norms = batch_CG(A_func, y, tol=1e-6, max_iter=self.CG_iters)
        # print(len(res_norms),'E-step CG residual norms:', res_norms)
        return x  # return in shape (B, N)
    
    def _edge_diff_square(self, x):
        # x in (B, N)
        B = x.size(0)
        N = self.glm.num_nodes
        edge_diffs = x.unsqueeze(2) - x[:, self.neighbor_list.view(-1)].reshape(B, N, -1)  # (B, N, k)
        edge_diff_squares = (edge_diffs ** 2).mean(0) * self.neighbor_mask.float().to(x.device)  # (N, k)
        return edge_diff_squares  # (N, k)
    
    def apply_inv_D_plus_J(self, x, adj):
        # x in (B, N), adj in (N, N)
        N = adj.size(0)
        degree = adj.sum(1)  # (N,)
        inv_degree = 1.0 / degree  # (N,)
        c = N + torch.sum(inv_degree)
        inner = x @ inv_degree # (B,)
        return x * inv_degree.unsqueeze(0) - (inner / c).unsqueeze(1) * inv_degree.unsqueeze(0)  # (B, N)
    
    def apply_inv_D(self, x, w):
        # x in (B, N), w in (N, k)
        N = w.size(0)
        degree = torch.sum(w, dim=1)  # (N,)
        inv_degree = 1.0 / degree  # (N,)
        return x * inv_degree.unsqueeze(0)  # (B, N)
    
    def _r_tilde(self, w, CG_method='batch_PCG'):
        """
        solve (e_i - e_j)^T (L + J)^-1 (e_i - e_j) for all edges (i, j)
        """
        # w = w_0 * A
        N = w.size(0)
        k = w.size(1)
        # print(w)
        assert torch.allclose(w * self.neighbor_mask.float().to(w.device), w), "w has non-zero weights on non-edges!"

        e = torch.eye(N, device=w.device)  # (N, N)
        e_input = e[1:]
        e_input[:,0] = e[1:,0] - 1 # reference point 0, e_input in shape (N-1, N)
        # print('e_input shape:', e_input)

        if self.method == 'CG':
            def A_func(x):
                if self.inv_method == 'L+J':
                    return self.apply_L_plus_J(x, w)  # (num_edges, N)
                elif self.inv_method == 'L+eI':
                    return self.apply_L_plus_epsilon_I(x, w, epsilon=5e-3)  # (num_edges, N)
            
            def Minv_func(x):
                if self.inv_method == 'L+J':
                    return self.apply_inv_D_plus_J(x, w)  # (num_edges, N)
                elif self.inv_method == 'L+eI':
                    return self.apply_inv_D(x, w)  # (num_edges, N)

            assert CG_method in ['batch_CG', 'batch_PCG'], "Only 'batch_CG' and 'batch_PCG' are supported currently."
            if CG_method == 'batch_CG':
                inv_ref, res_norms = batch_CG(A_func, e_input, tol=1e-6, max_iter=self.CG_iters)
            elif CG_method == 'batch_PCG':
                inv_ref, res_norms = batch_PCG(A_func, e_input, Minv_func=Minv_func, tol=1e-6, max_iter=self.CG_iters)
            # print(len(res_norms),'r_tilde CG residual norms:', res_norms)
            # inv_ref = self.conjugated_gradients(A_func, e_input, tol=1e-6, max_iter=self.e_step_iters) # (N-1, N)
            

        elif self.method == 'cholmod':
            pass  #TODO To be implemented: use cholmod to solve the system
            detached_w = w.detach().cpu()
            # in a sparse way
            sparse_L = self.sparse_L_from_w(detached_w)  # scipy sparse matrix
            # cholesky factorization
            L_plus_epsilon_I = sparse_L + sp.coo_matrix((np.ones(N) * 5e-3, (np.arange(N), np.arange(N))), shape=(N, N))
            L_plus_epsilon_I = L_plus_epsilon_I.tocsc()
            factor = cholesky(L_plus_epsilon_I) # use cholmod to do Cholesky factorization
            # L_plus_J = sparse_L + sp.coo_matrix((np.ones(N), (np.arange(N), np.arange(N))), shape=(N, N))
            # factor = cholesky(L_plus_J) # use cholmod to do Cholesky factorization    
            inv_ref = torch.Tensor(factor(e_input.T.detach().cpu().numpy()).T)  # (N-1, N)
            # inv_ref = torch.Tensor(factor.solve_A(e.detach().cpu().numpy()))  # (N-1, N)


        inv_ref = torch.concat([torch.zeros((1, N), device=w.device), inv_ref], dim=0)  # (N, N)
        diag_inv_ref = torch.diagonal(inv_ref)  # (N,)

        tilde_inv_ref = diag_inv_ref[:,None] + diag_inv_ref[None,:] - inv_ref - inv_ref.T  # (N, N), all (e_i - e_j)^T (L + J)^-1 (e_i - e_j) including non-edges


        indices = torch.arange(N).unsqueeze(1).repeat(1, k).view(-1)  # (N*k,)
        edge_indices = self.neighbor_list.view(-1)  # (num_edges,)
        r_tilde = tilde_inv_ref[indices, edge_indices].reshape(N, k)  # (N, k)
        r_tilde = r_tilde * self.neighbor_mask.float()  # mask non-edges
        return r_tilde  # (N, k)
    
    def GLR(self, x, w, method='edge_diff'):
        # x in shape (B, N), w in shape (N, k)
        if method == 'Lx':
            B = x.size(0)
            Lx = self.apply_L(x, w)  # (B, N)
            return torch.sum(x * Lx) / B  # scalar, mean of GLR over batch
        elif method == 'edge_diff':
            edge_diff_squares = self._edge_diff_square(x)  # (N, k)
            GLR_value = 0.5 * (w * edge_diff_squares).sum()  # scalar
            return GLR_value  # mean over batch
        
    def soft_thresholding(self, A, tau):
        return torch.sign(A) * torch.clamp(torch.abs(A) - tau, min=0.0)
        
    def M_step_2(self, x, w_0, A_init=None):
        '''
        PGD step to update the wacency mask A
        ''' 
        # N = self.glm.num_nodes
        A = A_init if A_init is not None else self.neighbor_mask.float()
        w, alpha = self.scale_w(w_0 * A)  # initial scaling
        # pre-fill the initial values
        w_0_alpha = w_0 * alpha
        # not detached here, need gradient flow
        r_tilde = self._r_tilde(w)  # (N, k)
        # print('tr(RL)=', (r_tilde * w).sum().item() / 2)  # check validation, = n-1
        # for fixed x, edge_diffs is fixed
        edge_diff_squares = self._edge_diff_square(x)  # (N, k)

        for iter in range(self.PGD_iters):
            # PGD
            A_old = A.clone()
            A_new = A_old - self.PGD_step_size * w_0_alpha * (edge_diff_squares - r_tilde)
            A_new = self.soft_thresholding(A_new, tau=self.PGD_step_size * self.gamma)
            if torch.isnan(A_new).any():
                print('w_0 has nan', torch.isnan(w_0).any())
                print('A_old has nan', torch.isnan(A_old).any())
                print('edge_diff_squares has nan', torch.isnan(edge_diff_squares).any())
                print('r_tilde has nan', torch.isnan(r_tilde).any())
                raise ValueError('NaN detected in A_new computation!')
            assert torch.allclose(A_new * self.neighbor_mask.float(), A_new), "A_new has non-zero weights on non-edges!"
            A_new = torch.clamp(A_new, min=0.0, max=1.0) # project to [0, 1]
            
            A_new = A_new * self.neighbor_mask.float()  # mask non-edges

            A = A_new
            # recompute graphs
            w = w_0_alpha * A  # (N, k)
            r_tilde = self._r_tilde(w)  # (N, k)
            # print(f'  PGD Iter {iter+1}/{self.PGD_iters}, tr(RL)=', (r_tilde * w).sum().item() / 2)  # check validation, = n-1
        
        return A

    def training_step(self, y, batch_idx):
        opt = self.optimizers()

        w_0 = self.w_0 if self.w_0 is not None else self.glm(y)
        device = w_0.device
        A = self.A # self.neighbor_mask.float().to(device)
        w, _ = self.scale_w(w_0 * A)

        # -------- E-step: 用上一轮 M2 输出 --------
        x = self.E_step(y, w)  # (B, N)
        # update weight matrix 
        w_0 = self.glm(x)
        w, _ = self.scale_w(w_0 * A) # rescale

        # -------- M-step Step1: 梯度更新 θ --------
        N = self.num_nodes
        detached_w = w.clone().detach()
        r_tilde = self._r_tilde(detached_w)  # (N, k)
        # print('tr(RL)=', (r_tilde * detached_w).sum().item() / 2) # check validation, = n-1

        proxy_loss = self.GLR(x, w) - (r_tilde * w).sum() / 2 # each edge counted twice
        # backward to gradient descent
        self.manual_backward(proxy_loss)
        opt.step()
        opt.zero_grad()

        # -------- M-step-2--------
        w_0 = self.glm(x)
        assert torch.allclose(w_0 * self.neighbor_mask.float().to(w_0.device), w_0), "w_0 has non-zero weights on non-edges!"
        A_new = self.M_step_2(x, w_0, A)

        # -------- save A and current graph --------
        self.A = A_new.detach()  # detach 避免梯度累积跨轮
        self.w_0 = w_0.detach()

        w, _ = self.scale_w(w_0 * A_new)
        # log graph_list
        self.graph_list.append(w.detach().cpu().numpy())
        print('A statistics: min=', self.A[self.neighbor_mask].min().item(), ', max=', self.A[self.neighbor_mask].max().item(), ', mean=', self.A[self.neighbor_mask].mean().item())
        print('w statistics: min=', w[self.neighbor_mask].min().item(), ', max=', w[self.neighbor_mask].max().item(), ', mean=', w[self.neighbor_mask].mean().item())
        print(f'left edges: {(self.A[self.neighbor_mask] > 0).sum().item() // 2} / {self.neighbor_mask.sum().item() // 2}')

        # logging
        self.log("train_loss", proxy_loss)
        return proxy_loss
    
    def on_train_epoch_end(self):
        if self.current_epoch % 4 == 0:
            print("train_epoch_end called, plotting learned graph")
            if len(self.graph_list) == 0:
                print("No learned graph to plot.")
                return super().on_train_epoch_end()
            
            last_graph = self.graph_list[-1]
            draw_graph(
                self.neighbor_list,
                torch.tensor(last_graph),
                n_row=int(math.sqrt(self.num_nodes)),
                title=f"Learned Graph at Epoch {self.current_epoch}",
                filename=f"learned_graph_epoch_{self.current_epoch}.png"
            )
        return super().on_train_epoch_end()
    
    # def on_validation_epoch_end(self):
    #     # draw last learned graph
    #     print("validation_step called, plotting learned graph")
    #     if len(self.graph_list) == 0:
    #         print("No learned graph to plot.")
    #         return
        
    #     last_graph = self.graph_list[-1]
    #     draw_graph(
    #         self.neighbor_list,
    #         torch.tensor(last_graph),
    #         n_row=int(math.sqrt(self.num_nodes)),
    #         title=f"Learned Graph at Epoch {self.current_epoch}",
    #         filename=f"learned_graph_epoch_{self.current_epoch}.png"
    #     )
    
     # -------- Optimizer --------
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)



