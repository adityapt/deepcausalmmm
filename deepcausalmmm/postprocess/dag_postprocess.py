"""
Post-processing utilities for DAG visualization and analysis.

.. deprecated:: 1.0.0
    This module is deprecated and will be removed in v2.0.0.
    Please use the modern VisualizationManager class instead:
    
    from deepcausalmmm.core.visualization import VisualizationManager
    
    viz_manager = VisualizationManager(config)
    viz_manager.create_dag_network_plot(...)
    viz_manager.create_dag_heatmap_plot(...)
"""

import torch
import numpy as np
import plotly.graph_objects as go
from typing import List, Optional, Dict, Any
import warnings


def plot_dag_structure(
    adjacency_matrix: torch.Tensor,
    channel_names: Optional[List[str]] = None,
    threshold: float = 0.1,
    title: str = "Media Channel DAG Structure"
) -> go.Figure:
    """
    .. deprecated:: 1.0.0
        This function is deprecated. Use VisualizationManager.create_dag_network_plot() instead.
    """
    warnings.warn(
        "plot_dag_structure() is deprecated and will be removed in v2.0.0. "
        "Please use VisualizationManager.create_dag_network_plot() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    """
    Visualize the DAG structure using Plotly.
    
    Args:
        adjacency_matrix: Adjacency matrix from the DAG model [n_nodes, n_nodes]
        channel_names: List of channel names. If None, uses indices
        threshold: Threshold for edge visibility
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Convert to numpy for processing
    adj_matrix = adjacency_matrix.detach().cpu().numpy()
    n_nodes = adj_matrix.shape[0]
    
    if channel_names is None:
        channel_names = [f"Channel {i+1}" for i in range(n_nodes)]
    
    # Create node positions in a circular layout
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    radius = 1
    node_x = radius * np.cos(angles)
    node_y = radius * np.sin(angles)
    
    # Create edges (arrows)
    edge_x = []
    edge_y = []
    edge_text = []
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adj_matrix[i, j] > threshold:
                # Calculate arrow
                start_x, start_y = node_x[i], node_y[i]
                end_x, end_y = node_x[j], node_y[j]
                
                # Add edge with arrow
                edge_x.extend([start_x, end_x, None])
                edge_y.extend([start_y, end_y, None])
                
                # Add edge weight text
                edge_text.append(f"{channel_names[i]} → {channel_names[j]}: {adj_matrix[i,j]:.3f}")
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines+text',
        line=dict(width=1, color='gray'),
        hoverinfo='text',
        text=edge_text,
        name='Edges'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=30,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        ),
        text=channel_names,
        textposition="middle center",
        hoverinfo='text',
        name='Channels'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center'
        ),
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        plot_bgcolor='white',
        width=800,
        height=800
    )
    
    return fig


def analyze_dag_structure(
    adjacency_matrix: torch.Tensor,
    channel_names: Optional[List[str]] = None,
    threshold: float = 0.1
) -> Dict[str, Any]:
    """
    .. deprecated:: 1.0.0
        This function is deprecated. Use VisualizationManager for DAG analysis instead.
    """
    warnings.warn(
        "analyze_dag_structure() is deprecated and will be removed in v2.0.0. "
        "Please use VisualizationManager for DAG analysis instead.",
        DeprecationWarning,
        stacklevel=2
    )
    """
    Analyze the DAG structure and return key metrics.
    
    Args:
        adjacency_matrix: Adjacency matrix from the DAG model
        channel_names: List of channel names
        threshold: Threshold for edge significance
        
    Returns:
        Dictionary containing analysis results
    """
    adj_matrix = adjacency_matrix.detach().cpu().numpy()
    n_nodes = adj_matrix.shape[0]
    
    if channel_names is None:
        channel_names = [f"Channel {i+1}" for i in range(n_nodes)]
    
    # Initialize results
    results = {
        'n_edges': 0,
        'avg_edge_weight': 0.0,
        'max_edge_weight': 0.0,
        'significant_edges': [],
        'influential_channels': [],
        'influenced_channels': []
    }
    
    # Count edges and compute metrics
    significant_edges = adj_matrix > threshold
    results['n_edges'] = significant_edges.sum()
    
    if results['n_edges'] > 0:
        results['avg_edge_weight'] = adj_matrix[significant_edges].mean()
        results['max_edge_weight'] = adj_matrix.max()
    
    # Find significant relationships
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adj_matrix[i,j] > threshold:
                results['significant_edges'].append({
                    'from': channel_names[i],
                    'to': channel_names[j],
                    'weight': float(adj_matrix[i,j])
                })
    
    # Identify influential and influenced channels
    out_degree = adj_matrix.sum(axis=1)
    in_degree = adj_matrix.sum(axis=0)
    
    # Top influential channels (high out-degree)
    influential_idx = np.argsort(-out_degree)
    results['influential_channels'] = [
        {
            'channel': channel_names[i],
            'out_degree': float(out_degree[i])
        }
        for i in influential_idx if out_degree[i] > threshold
    ]
    
    # Top influenced channels (high in-degree)
    influenced_idx = np.argsort(-in_degree)
    results['influenced_channels'] = [
        {
            'channel': channel_names[i],
            'in_degree': float(in_degree[i])
        }
        for i in influenced_idx if in_degree[i] > threshold
    ]
    
    return results 