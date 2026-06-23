import torch
import torch.nn as nn
from effective_resistance import MultiGraphEffectiveResistance
from utils_modules import UnrolledCG, GraphLearningModule

class UnrolledGEMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, num_graphs=1):
        super(UnrolledGEMBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.num_graphs = num_graphs
        
        # Learnable parameters for the GEM block
        self.alpha = nn.Parameter(torch.Tensor(num_graphs))  # Graph weights
        self.beta = nn.Parameter(torch.Tensor(num_graphs))   # Graph scaling factors
        
        # Linear transformation for node features
        self.linear = nn.Linear(in_channels, out_channels)
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.uniform_(self.alpha)
        nn.init.uniform_(self.beta)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def E_step(self, x, edge_index_list):
        pass

    def M_step(self, x, edge_index_list):
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