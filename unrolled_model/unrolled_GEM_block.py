import torch
import torch.nn as nn
from effective_resistance import MultiGraphEffectiveResistance
from utils_modules import UnrolledCG, GraphLearningModule, SparseGraphOperators

class UnrolledGEMBlock(nn.Module):
    '''
    Unrolled Graph Embedding Module (GEM) Block for multi-graph learning.
    
    E-step: given signal y and current graph W^o, S, solve the smoothness problem $(I+\mu L)^{-1} x = y$ by unrolled CG.

    M-step: given signal x and current graph W^o, S, update the graph by unrolling the gradient descent steps for the learned graph W, and its connectivity mask S.

    Solve with multi-head graph learning.
    
    Shape of input y: (batch_size, num_nodes)
    Shape of input W^o: (num_graphs, num_nodes, k)
    Shape of input S: (num_graphs, num_nodes, k)
    Shape of neighbor list: (num_nodes, k) (possible neighbors are pre-defined)
    '''
    def __init__(self, num_nodes, neighbor_list, input_neighbor_mask, num_heads, E_iters=5, M_iters=5, GD_step_init=0.1, mu_init=0.2, c=20, scale=True, epsilon=0.2):
        super(UnrolledGEMBlock, self).__init__()
        self.num_nodes = num_nodes
        self.neighbor_list = neighbor_list
        self.num_heads = num_heads
        self.E_iters = E_iters
        self.M_iters = M_iters

        if input_neighbor_mask.ndim() == 2:
            input_neighbor_mask = input_neighbor_mask.unsqueeze(0).repeat(num_heads, 1, 1) # (1, num_nodes, k) -> (num_heads, num_nodes, k)

        # input mask, in (G, N, k)
        self.input_S = input_neighbor_mask

        # E-step block: Unrolled Conjugate Gradient for solving smoothness problem
        self.CG_solver = UnrolledCG(self.E_iters, 0.1, 0, self.num_heads, init_method='uniform', init_scale=0.02)

        # M-step block: Graph Learning with GD for updating graph weights and connectivity mask
        self.c = c
        self.scale = scale
        self.epsilon = epsilon
        # TODO: check whether mu should be in (1,) or in (M_iters,)
        self.mu = nn.Parameter(torch.ones(1,) * mu_init, requires_grad=True)
        self.ER_solver = MultiGraphEffectiveResistance(self.neighbor_list, self.input_S, inv_method="L+J", epsilon=self.epsilon) # reusable block for pure computation without parameters
        self.step_size_list = nn.Parameter(torch.ones(self.M_iters) * GD_step_init, requires_grad=True)

        # Graph Operator block
        self.graph_op = SparseGraphOperators(self.num_nodes, self.neighbor_list, self.input_S, c=self.c, scale=self.scale, epsilon=self.epsilon)
        

    def E_step(self, y, W):
        if y.ndim() == 2:
            y = y.unsqueeze(1).repeat(1, self.num_heads, 1) # (batch_size, num_nodes) -> (batch_size, num_heads, num_nodes)

        def LHS(x):
            """
            Compute the left-hand side of the smoothness problem: (I + mu * L) * x
            where L is the graph Laplacian derived from W.
            """
            # Compute graph Laplacian L from W
            return x + self.mu * self.graph_op.apply_L(x, W)  # (I + mu * L) * x

        x = self.CG_solver(LHS, y)
        return x

    def M_step(self, x, W):
        pass
    
    def forward(self, x, edge_index_list):
        """
        x: Node features of shape (batch_size, num_nodes, in_channels)
        edge_index_list: List of edge indices for each graph (length=num_graphs)
                         Each edge index is of shape (2, num_edges)
        """
        batch_size = x.size(0)
        
        # Apply linear transformation to node features
        x_transformed = self.linear(x)  # Shape: (batch_size, num_nodes, out_channels)
        
        # Compute effective resistance for each graph and combine them
        combined_resistance = torch.zeros(batch_size, self.num_nodes, device=x.device)
        
        for i in range(self.num_graphs):
            edge_index = edge_index_list[i]
            if self.num_graphs == 1:
                resistance = EffectiveResistance.compute(x_transformed, edge_index)  # Shape: (batch_size, num_nodes)
            else:
                resistance = MultiGraphEffectiveResistance.compute(x_transformed, edge_index)  # Shape: (batch_size, num_nodes)
            
            combined_resistance += self.alpha[i] * resistance * self.beta[i]
        
        return combined_resistance
