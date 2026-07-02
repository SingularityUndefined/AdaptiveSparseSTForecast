import torch
import torch.nn as nn
from unrolled_GEM_block import UnrolledGEMBlock
from utils_modules import UnrolledCG, GraphLearningModule, SparseGraphOperators

class UnrolledGEM(nn.Module):
    '''
    Unrolled Graph Embedding Module (GEM) for multi-graph learning.
    
    E-step: given signal y and current graph W^o, S, solve the smoothness problem $(I+\mu L)^{-1} x = y$ by unrolled CG.

    M-step: given signal x and current graph W^o, S, update the graph by unrolling the gradient descent steps for the learned graph W, and its connectivity mask S.

    Solve with multi-head graph learning.
    
    Shape of input y: (batch_size, num_nodes)
    Shape of input W^o: (num_graphs, num_nodes, k)
    Shape of input S: (num_graphs, num_nodes, k)
    Shape of neighbor list: (num_nodes, k) (possible neighbors are pre-defined)
    '''
    def __init__(self, num_nodes, neighbor_list, input_neighbor_mask, num_heads, num_blocks, E_iters, M_iters, GD_step_init=0.1, mu_init=0.2, gamma_init=0.4, c=20, scale=True, epsilon=0.2, alpha_init=0.5):
        super(UnrolledGEM, self).__init__()
        self.num_nodes = num_nodes
        self.neighbor_list = neighbor_list
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.E_iters = E_iters
        self.M_iters = M_iters

        if input_neighbor_mask.ndim() == 2:
            input_neighbor_mask = input_neighbor_mask.unsqueeze(0).repeat(num_heads, 1, 1) # (1, num_nodes, k) -> (num_heads, num_nodes, k)

        # input mask in (G, N, k)
        self.input_S = input_neighbor_mask

        # Unrolled GEM Block
        self.GLM_list = nn.ModuleList([
            GraphLearningModule(num_nodes, neighbor_list, input_neighbor_mask, num_heads, E_iters, M_iters, GD_step_init, mu_init, gamma_init, c, scale, epsilon)
            for _ in range(num_blocks)
        ])

        self.GEM_block_list = nn.ModuleList([
            UnrolledGEMBlock(num_nodes, neighbor_list, input_neighbor_mask, num_heads, E_iters, M_iters, GD_step_init, mu_init, gamma_init, c, scale, epsilon)
            for _ in range(num_blocks)
        ])

        # learnable parameters for skip connections
        self.alpha_list = nn.Parameter(torch.ones(num_blocks) * alpha_init, requires_grad=True)  # learnable skip connection weights for each block

    def forward(self, y, W_o):
        '''
        Forward pass of the Unrolled GEM module.
        
        Args:
            y: input signal, shape (batch_size, num_nodes)
            W_o: initial graph weights, shape (num_graphs, num_nodes, k)
        
        Returns:
            x: output signal after E-step and M-step, shape (batch_size, num_nodes)
            W: updated graph weights after M-step, shape (num_graphs, num_nodes, k)
            S: updated connectivity mask after M-step, shape (num_graphs, num_nodes, k)
        '''
        # TODO: other initial recovery of x from y, e.g., x = y, or x = (I + mu L)^-1 y, or parameterized, etc.
        x = y
        S = self.input_S
        W_old = None
        for i in range(self.num_blocks):
            self.GLM_list[i].update_neighbor_mask(S)  # Update the neighbor mask for the current block
            if W_old is not None:
                W = self.GLM_list[i](x) * self.alpha_list[i] + W_old * (1 - self.alpha_list[i])  # Weighted combination of learned graph and previous graph
            else:
                W = self.GLM_list[i](x)
            
            x, W, S = self.GEM_block_list[i](y, W)
            W_old = W

        return x, W, S


