"""
DAG model implementation with Node-to-Edge and Edge-to-Node transformations.

This module implements the DAG-based neural network architecture with:
- NodeToEdge: Transform node features to edge features
- EdgeToNode: Aggregate edge features back to nodes
- DAGConstraint: Enforce acyclicity in the graph structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np

class NodeToEdge(nn.Module):
    """Transform node features to edge features using attention mechanism."""
    
    def __init__(self, node_dim: int, edge_dim: int):
        """
        Initialize the node to edge transformation.
        
        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        # Transformations for source and target nodes (wider networks)
        self.source_transform = nn.Sequential(
            nn.Linear(1, 64),  # Increased width
            nn.ReLU(),
            nn.Linear(64, edge_dim)
        )
        self.target_transform = nn.Sequential(
            nn.Linear(1, 64),  # Increased width
            nn.ReLU(),
            nn.Linear(64, edge_dim)
        )
        
        # Edge attention with wider network and stronger initialization
        self.edge_attention = nn.Sequential(
            nn.Linear(2 * edge_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Initialize weights with larger values
        for layer in self.source_transform:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.4)
        for layer in self.target_transform:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.4)
        for layer in self.edge_attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.4)
    
    def forward(self, nodes: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Transform node features to edge features.
        
        Args:
            nodes: Node features [batch_size, n_nodes, 1]
            adj_matrix: Adjacency matrix [n_nodes, n_nodes]
            
        Returns:
            Edge features [batch_size, n_nodes, n_nodes, edge_dim]
        """
        B, N, _ = nodes.shape
        
        # Transform source and target nodes
        source_h = self.source_transform(nodes)  # [B, N, edge_dim]
        target_h = self.target_transform(nodes)  # [B, N, edge_dim]
        
        # Compute edge features for all pairs
        source_e = source_h.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, edge_dim]
        target_e = target_h.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, edge_dim]
        
        # Concatenate and compute attention
        edge_input = torch.cat([source_e, target_e], dim=-1)  # [B, N, N, 2*edge_dim]
        edge_attn = self.edge_attention(edge_input)  # [B, N, N, 1]
        
        # Apply adjacency as multiplicative weight with stronger influence
        A = adj_matrix.unsqueeze(0).unsqueeze(-1)  # [1, N, N, 1]
        edge_weights = torch.sigmoid(edge_attn) * A  # Now magnitude matters
        
        # Compute edge features with residual connection
        edge_features = edge_weights * (source_e + target_e)
        
        return edge_features


class EdgeToNode(nn.Module):
    """Aggregate edge features back to nodes."""
    
    def __init__(self, edge_dim: int, node_dim: int):
        """
        Initialize the edge to node transformation.
        
        Args:
            edge_dim: Dimension of edge features
            node_dim: Dimension of node features
        """
        super().__init__()
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        
        # Edge aggregation with wider network
        self.edge_aggregate = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Node update with wider network
        self.node_update = nn.Sequential(
            nn.Linear(2, 64),  # Wider network
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Skip connection scaling factor (learnable)
        self.skip_scale = nn.Parameter(torch.ones(1))
        
        # Initialize weights with larger values
        for layer in self.edge_aggregate:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.4)
        
        for layer in self.node_update:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.4)
    
    def forward(self, edges: torch.Tensor, nodes: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Aggregate edge features to update node features.
        
        Args:
            edges: Edge features [batch_size, n_nodes, n_nodes, edge_dim]
            nodes: Node features [batch_size, n_nodes, 1]
            adj_matrix: Adjacency matrix [n_nodes, n_nodes]
            
        Returns:
            Updated node features [batch_size, n_nodes, 1]
        """
        # Aggregate incoming edges
        edge_aggr = self.edge_aggregate(edges)  # [B, N, N, 1]
        
        # Apply adjacency mask with stronger influence
        mask = adj_matrix.unsqueeze(0).unsqueeze(-1)  # [1, N, N, 1]
        edge_aggr = edge_aggr * mask
        
        # Sum over neighbors
        node_update = edge_aggr.sum(dim=2)  # [B, N, 1]
        
        # Combine with original node features
        combined = torch.cat([nodes, node_update], dim=-1)  # [B, N, 2]
        transformed = self.node_update(combined)  # [B, N, 1]
        
        # Add skip connection with learnable scaling
        skip_scale = torch.sigmoid(self.skip_scale)  # Bound between 0 and 1
        updated_nodes = transformed + skip_scale * nodes
        
        return updated_nodes


class DAGConstraint(nn.Module):
    """Enforce acyclicity in the graph structure using strict triangular constraint."""
    
    def __init__(self, n_nodes: int, sparsity_weight: float = 0.1, temperature: float = 1.0):
        """
        Initialize the DAG constraint module.
        
        Args:
            n_nodes: Number of nodes in the graph
            sparsity_weight: Weight for the sparsity penalty
            temperature: Initial temperature for Gumbel-Softmax
        """
        super().__init__()
        self.n_nodes = n_nodes
        self.sparsity_weight = sparsity_weight
        self.temperature = temperature
        
        # Initialize adjacency logits with strong negative bias for sparsity
        self.adj_logits = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.1 - 3.0)
        
        # Create mask for strictly upper triangular matrix
        mask = torch.triu(torch.ones(n_nodes, n_nodes), diagonal=1)
        self.register_buffer('triangular_mask', mask.bool())
    
    def gumbel_softmax(self, logits: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Gumbel-Softmax sampling with straight-through gradients.
        
        Args:
            logits: Input logits
            tau: Temperature parameter
            
        Returns:
            Sampled probabilities
        """
        if self.training:
            # Sample from Gumbel distribution
            g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
            
            # Gumbel-Softmax with straight-through estimator
            y_soft = torch.sigmoid((logits + g) / tau)
            
            # Straight-through: use hard values in forward pass but soft in backward
            y_hard = (y_soft > 0.5).float()
            y = y_hard.detach() - y_soft.detach() + y_soft
        else:
            # During evaluation, use deterministic thresholding
            y = (torch.sigmoid(logits / tau) > 0.5).float()
        
        return y
    
    def get_adjacency(self) -> torch.Tensor:
        """
        Get the current adjacency matrix using Gumbel-Softmax sampling.
        This enforces unidirectional edges and allows learning discrete structure.
        """
        # Apply Gumbel-Softmax sampling with current temperature
        adj = self.gumbel_softmax(self.adj_logits, self.temperature)
        
        # Apply mask to ensure strictly upper triangular form
        adj = adj * self.triangular_mask
        
        return adj
    
    def update_temperature(self, epoch: int, total_epochs: int, min_temp: float = 0.1):
        """
        Update temperature using exponential decay schedule.
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
            min_temp: Minimum temperature
        """
        # Exponential decay is more aggressive than cosine
        progress = epoch / total_epochs
        self.temperature = max(
            min_temp,
            np.exp(-10 * progress)  # Even faster decay
        )
    
    def dag_loss(self) -> torch.Tensor:
        """
        Compute the DAG constraint loss with sparsity penalty.
        With strictly upper triangular form, we only need sparsity penalty
        as acyclicity is guaranteed by construction.
        
        Returns:
            Loss term combining sparsity and entropy
        """
        adj = self.get_adjacency()
        
        # L1 sparsity with stronger penalty
        sparsity_loss = torch.sum(torch.abs(adj))
        
        # Add entropy penalty to encourage binary decisions
        probs = torch.sigmoid(self.adj_logits)
        entropy_loss = -torch.mean(
            probs * torch.log(probs + 1e-9) + 
            (1 - probs) * torch.log(1 - probs + 1e-9)
        )
        
        # Add edge diversity penalty to encourage different patterns
        edge_diversity = -torch.std(adj[self.triangular_mask])
        
        return self.sparsity_weight * (
            sparsity_loss + 
            0.1 * entropy_loss + 
            0.2 * edge_diversity
        )


class DAGModel(nn.Module):
    """
    Complete DAG-based model combining NodeToEdge and EdgeToNode transformations.
    """
    
    def __init__(
        self,
        n_nodes: int,
        node_dim: int,
        edge_dim: int,
        n_layers: int = 3,
        sparsity_weight: float = 0.1
    ):
        """
        Initialize the DAG model.
        
        Args:
            n_nodes: Number of nodes in the graph
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            n_layers: Number of message passing layers
            sparsity_weight: Weight for the sparsity penalty
        """
        super().__init__()
        self.n_nodes = n_nodes
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_layers = n_layers
        
        # DAG constraint
        self.dag = DAGConstraint(n_nodes, sparsity_weight)
        
        # Node and edge transformations
        self.node_to_edge = NodeToEdge(node_dim, edge_dim)
        self.edge_to_node = EdgeToNode(edge_dim, node_dim)
        
        # Node embedding
        self.node_embedding = nn.Linear(node_dim, node_dim)
        nn.init.xavier_uniform_(self.node_embedding.weight)
        
        # Output projection
        self.output = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        
        for layer in self.output:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the DAG model.
        
        Args:
            nodes: Input node features [batch_size, n_nodes, node_dim]
            
        Returns:
            Tuple of (output node features, adjacency matrix)
        """
        adj = self.dag.get_adjacency()
        
        # Initial node embedding
        h = self.node_embedding(nodes)
        
        # Message passing layers
        for _ in range(self.n_layers):
            # Node to edge
            edge_features = self.node_to_edge(h, adj)
            
            # Edge to node
            h = self.edge_to_node(edge_features, h, adj)
        
        # Output projection
        out = self.output(h)
        
        return out, adj
    
    def get_dag_loss(self) -> torch.Tensor:
        """Get the DAG constraint loss."""
        return self.dag.dag_loss() 