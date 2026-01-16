import torch
import torch.nn as nn
import torch.nn.functional as F
from sksparse.cholmod import cholesky
from copy import deepcopy
import math
import scipy.sparse as sp
import numpy as np
from GEM_sparse.sparse_CG import *
from utils_sparse import draw_graph
from sparsemax import Sparsemax

class GraphLearningModule(nn.Module):
    # generate weight matrix from node embeddings
    def __init__(self, num_nodes, num_neighbors, neighbor_list, emb_dim=6, feature_dim=3, c=8, theta=0.5, method='CG'):
        super(GraphLearningModule, self).__init__()
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors
        self.neighbor_list = neighbor_list # in (N, k)
        self.neighbor_mask = (neighbor_list != -1)  # (N, k), original w mask
        self.emb_dim = emb_dim
        self.feature_dim = feature_dim

        # embedding vectors
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, emb_dim))  
        self.fc = nn.Linear(emb_dim + 1, feature_dim)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.c = c
        self.theta = nn.Parameter(torch.tensor(theta))

    def forward(self, x, params=None):
        """
        x: (B, N)
        params: dict[str, tensor] (可选，用于 MAML/unrolled optimization)
        """
        if params is None:
            node_embeddings = self.node_embeddings
            fc_weight = self.fc.weight
            fc_bias = self.fc.bias
            theta = self.theta
        else:
            node_embeddings = params["node_embeddings"]
            fc_weight = params["fc.weight"]
            fc_bias = params["fc.bias"]
            theta = params["theta"]

        B = x.size(0)

        # 1. embed each node (add)
        e = torch.cat([x.unsqueeze(-1), node_embeddings.unsqueeze(0).repeat(B, 1, 1)], dim=-1)  # (B, N, emb_dim + 1)

        # 2. Feature generation: linear + activation
        f = F.linear(e, fc_weight, fc_bias)   # (B, N, feature_dim)
        f = self.leakyrelu(f) # in (B, N, feature_dim)

        # 3. pairwise difference
        df = f.unsqueeze(2) - f[:, self.neighbor_list.view(-1)].reshape(B, self.num_nodes, self.num_neighbors, self.feature_dim)  # (B, N, k, feature_dim)

        # 4. wacency by RBF kernel
        w = torch.exp(-(df ** 2).sum(-1) / (2 * theta)).mean(0)  # (N, k)
        if torch.isnan(w).any():
            print('embedded x + e:', torch.isnan(e).any())
            print('features f:', torch.isnan(f).any())
            raise ValueError('NaN detected in w computation!')

        w = w * self.neighbor_mask.float()  # mask non-edges
        assert torch.allclose(w * self.neighbor_mask.float(), w), "w has non-zero weights on non-edges!"

        return w # (N, k)
    
class Generalized_EM(nn.Module):
    def __init__(self, num_nodes, num_neighbors, neighbor_list, mu, gamma, step_size, emb_dim=6, feature_dim=3, c=8, theta=0.5, method='CG', inv_method='L+eI', CG_iters=10, PGD_iters=100, PGD_step_size=0.01, scale=True, GEM_iters=5):
        super(Generalized_EM, self).__init__()
        self.glm = GraphLearningModule(num_nodes, num_neighbors, neighbor_list, emb_dim, feature_dim, c, theta)
        self.neighbor_list = neighbor_list
        self.neighbor_mask = (neighbor_list >= 0)  # (N, k), original w mask
        self.A_ori = self.neighbor_mask.float()
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors

        self.mu = mu
        self.c = c # Here, C is the F-norm contraint, not squared
        self.step_size = step_size
        self.PGD_step_size = PGD_step_size
        self.gamma = gamma
        self.scale = scale
        self.method = method
        assert method in ['CG', 'cholmod'], "Only 'CG' and 'cholmod' are supported currently."
        self.CG_iters = CG_iters
        self.PGD_iters = PGD_iters

        self.GEM_iters = GEM_iters  # could be removed later
        self.inv_method = inv_method
        assert inv_method in ['L+J', 'L+eI'], "Only 'L+J' and 'L+eI' are supported currently."

    def scale_w(self, w):
        '''
        scale to fit F-norm constraint of Laplacian (||L||_F = c)
        '''
        assert torch.allclose(w * self.neighbor_mask.float(), w), "w has non-zero weights on non-edges!"
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
        neighbor_X = neighbor_X * self.neighbor_mask.unsqueeze(0).float()  # mask non-edges
        assert torch.allclose(w * self.neighbor_mask.float(), w), "w has non-zero weights on non-edges!"
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
        edge_diff_squares = (edge_diffs ** 2).mean(0) * self.neighbor_mask.float()  # (N, k)
        return edge_diff_squares  # (N, k)
    
    def sparse_L_from_w(self, w):
        # w in (N, k)
        N = w.size(0)
        k = w.size(1)
        degree = torch.sum(w, dim=1)  # (N,)
        row_indices = torch.arange(N).unsqueeze(1).repeat(1, k).view(-1)  # (N*k,)
        col_indices = self.neighbor_list.view(-1)  # (N*k,)
        values = -w.view(-1)  # (N*k,)
        # mask non-edges
        col_mask = (col_indices != -1)
        row_indices = row_indices[col_mask]
        col_indices = col_indices[col_mask]
        values = values[col_mask]

        # Diagonal entries
        diag_row_indices = torch.arange(N)
        diag_col_indices = torch.arange(N)
        diag_values = degree

        all_row_indices = torch.cat([row_indices, diag_row_indices])
        all_col_indices = torch.cat([col_indices, diag_col_indices])
        all_values = torch.cat([values, diag_values])

        detached_all_row_indices = all_row_indices.detach().cpu().numpy()
        detached_all_col_indices = all_col_indices.detach().cpu().numpy()
        detached_all_values = all_values.detach().cpu().numpy()

        sparse_L = sp.coo_matrix((detached_all_values, (detached_all_row_indices, detached_all_col_indices)), shape=(N, N))

        return sparse_L
    
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
        assert torch.allclose(w * self.neighbor_mask.float(), w), "w has non-zero weights on non-edges!"

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

    
    def M_step_1(self, x, w):
        '''
        backward the proxy loss trace(X^T L X)/B - tr(R L) = sum_ij W_ij * ((x_i - x_j)^2/B - r_tilde_ij)
        '''
        N = self.num_nodes
        detached_w = w.clone().detach()
        r_tilde = self._r_tilde(detached_w)  # (N, k)
        print('tr(RL)=', (r_tilde * detached_w).sum().item() / 2) # check validation, = n-1

        proxy_loss = self.GLR(x, w) - (r_tilde * w).sum() / 2 # each edge counted twice
        # backward to gradient descent
        grads = torch.autograd.grad(proxy_loss, self.glm.parameters(), retain_graph=True) #create_graph=True)
        new_params = {}
        for (name, param), grad in zip(self.glm.named_parameters(), grads):
            new_params[name] = param - self.step_size * grad
        return new_params
    
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
        print('tr(RL)=', (r_tilde * w).sum().item() / 2)  # check validation, = n-1
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
            print(f'  PGD Iter {iter+1}/{self.PGD_iters}, tr(RL)=', (r_tilde * w).sum().item() / 2)  # check validation, = n-1
        
        return A
    
    def single_step(self, y, w_0, A, params=None):
        '''
        GEM framework:
            0. rescale weights to fit F-norm constraint

            1a. E step: x = argmin_x 0.5||y - x||^2 + 0.5 * mu * x^T L x
            1b. recompute wacency matrix w_0 from updated x, rescale to fit F-norm constraint

            2a. M step 1: update GLM parameters by minimizing proxy loss
            2b. recompute w_0, rescale to fit F-norm constraint

            3. M step 2: update wacency mask A by PGD
        '''

        w = w_0 * A  # (N, k)

        w, _ = self.scale_w(w)
        # E step
        x = self.E_step(y, w)  # (B, N)
        # update weight matrix 
        w_0 = self.glm(x, params=params)
        w, _ = self.scale_w(w_0 * A) # rescale

        # M step 1
        new_params = self.M_step_1(x, w)
        # update glm parameters
        # for name, param in self.glm.named_parameters():
        #     param.data = new_params[name].data

        # recompute wacency
        w_0 = self.glm(x, params=new_params)  # (N, k)
        assert torch.allclose(w_0 * self.neighbor_mask.float(), w_0), "w_0 has non-zero weights on non-edges!"

        # M step 2
        A_new = self.M_step_2(x, w_0, A)


        return x, w_0, A_new
    
    def forward(self, y, w_init=None, A_init=None, epochs=1):
    
        # init w and A
        w_0 = self.glm(y) if w_init is None else w_init
        A = self.neighbor_mask.float() if A_init is None else A_init
        # rescale
        w, _ = self.scale_w(w_0 * A)
        w_plot = w.clone().detach().numpy()
        # w_plot[w_plot < 5e-3] = 0.0
        # TODO: init plot
        draw_graph(self.neighbor_list, w, int(math.sqrt(self.num_nodes)), title='learned graph Iteration 0')
        
        for it in range(self.GEM_iters):
            print(f'Iteration {it+1}/{self.GEM_iters}')
            x, w_0, A = self.single_step(y, w_0, A)
            size_A = A.size()
            w, _ = self.scale_w(w_0 * A)

            # A = Sparsemax(dim=-1)(A.view(1,-1)).view(size_A)  # project A by sparsemax
            # A = Sparsemax(dim=1)(A)  # project A by sparsemax
            # A_plot = A.clone().detach().numpy()
            A_fill = A[self.neighbor_mask]

            A_fill_sparsemax = Sparsemax(dim=1)(A_fill.view(1, -1))
            
            A_sparse = torch.zeros(size_A) 
            A_sparse[self.neighbor_mask] = A_fill_sparsemax.view(-1)
              # project A by sparsemax
            # A = A_sparse.to(A.device)
            sparse_W, _ = self.scale_w(w_0 * A_sparse)

            w_plot = w.clone().detach().numpy()
            w_sparsemax = sparse_W.clone().detach().numpy()
            if it % epochs == 0:
                print('A statistics: min=', A[self.neighbor_mask].min().item(), ', max=', A[self.neighbor_mask].max().item(), ', mean=', A[self.neighbor_mask].mean().item())
                print('w_0 statistics: min=', w_0[self.neighbor_mask].min().item(), ', max=', w_0[self.neighbor_mask].max().item(), ', mean=', w_0[self.neighbor_mask].mean().item())
                print('w statistics: min=', w[self.neighbor_mask].min().item(), ', max=', w[self.neighbor_mask].max().item(), ', mean=', w[self.neighbor_mask].mean().item())
                print(f'left edges: {(A[self.neighbor_mask] > 0).sum().item() // 2} / {self.neighbor_mask.sum().item() // 2}')
                draw_graph(self.neighbor_list, w_plot, int(math.sqrt(self.num_nodes)), title=f'learned graph Iteration {it+1}')
                draw_graph(self.neighbor_list, w_sparsemax, int(math.sqrt(self.num_nodes)), title=fr'Learned $\mathrm{{Sparsemax}}(A) \circ W$ at Iteration {it+1}')
            # TODO: print

        return x, w_0, A, w
    
#     ################## DEPRECATED ########################
    
#     def solve_E_step_scipy(self, y, w):
#         # y in (B, N)
#         # solve equation (I + mu*L)X = Y 
#         from scipy.sparse.linalg import LinearOperator, cg
#         def func_LHS_E_step(x):
#             x_tensor = torch.tensor(x, dtype=y.dtype, device=y.device).view(y.size())
#             result = self.LHS_E_step(x_tensor, w)
#             return result.cpu().numpy().ravel()
        
#         LHS_E_step_op = LinearOperator((y.numel(), y.numel()), matvec=func_LHS_E_step)
#         x = torch.zeros_like(y)
#         for i in range(y.size(0)):
#             b = y[i].cpu().numpy()
#             x_i, info = cg(LHS_E_step_op, b, atol=1e-6, maxiter=100)
#             x[i] = torch.tensor(x_i, dtype=y.dtype, device=y.device).view(y.size(1))

#         return x  # return in shape (B, N)
    
#     def w_to_dense(self, w):
#         N = w.size(0)
#         dense_w = torch.zeros((N, N), device=w.device)
#         for i in range(N):
#             neighbors = self.neighbor_list[i]  # (k,)
#             dense_w[i, neighbors] = w[i]
#         return dense_w
    
#     def L_matrix(self, w):
#         N = w.size(0)
#         degree = torch.sum(w, dim=1)  # (N,)
#         D = torch.diag(degree)  # (N, N)
#         L = D - self.w_to_dense(w)  # (N, N)
#         return L
    
#     def direct_E_step(self, y, w):
#         # y in (B, N)
#         B = y.size(0)
#         N = w.size(0)
#         L = self.L_matrix(w)
#         cov = torch.eye(N, device=w.device).unsqueeze(0) + self.mu * L  # (N, N)
#         return torch.linalg.solve(cov, y.t()).t()
    
#     def direct_M_step_1(self, x, w):
#         pass

#     def direct_M_step_2(self, w):
#         pass    
#     def inv_LJ_matrix(self, w):
#         # return 
#         pass
    
#     def trace_SL(self, S, w):
#         # sparse w in (N, k), compute in sparse way
#         pass
#   #########################################################