"""
Reusable VisualizationManager class for creating consistent plots.
Eliminates code duplication and provides config-driven visualization.
"""

import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple
import os

from deepcausalmmm.core.config import get_default_config


class VisualizationManager:
    """
    Visualization manager for creating consistent plots in DeepCausalMMM analysis.
    
    Provides a unified interface for creating training progress, coefficient analysis,
    contribution plots, DAG visualizations, and other MMM-related charts.
    All plot parameters are driven by configuration for consistency.
    
    Parameters
    ----------
    config : Dict[str, Any], optional
        Configuration dictionary. If None, uses default configuration.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the visualization manager.
        
        Args:
            config: Configuration dictionary. If None, uses default config.
        """
        self.config = config or get_default_config()
        self.viz_params = self._get_viz_params()
        
    def _get_viz_params(self) -> Dict[str, Any]:
        """Get visualization parameters from config with defaults"""
        viz_config = self.config.get('visualization', {})
        return {
            'node_opacity': viz_config.get('node_opacity', 0.7),
            'line_opacity': viz_config.get('line_opacity', 0.6),
            'fill_opacity': viz_config.get('fill_opacity', 0.1),
            'marker_size': viz_config.get('marker_size', 8),
            'correlation_threshold': viz_config.get('correlation_threshold', 0.2),
            'edge_width_multiplier': viz_config.get('edge_width_multiplier', 8),
            'subplot_vertical_spacing': viz_config.get('subplot_vertical_spacing', 0.08),
            'subplot_horizontal_spacing': viz_config.get('subplot_horizontal_spacing', 0.06),
        }
        
    def create_training_progress_plot(self, train_losses: List[float], 
                                    train_rmses: List[float], 
                                    train_r2s: List[float],
                                    title: str = "Training Progress") -> go.Figure:
        """
        Create a training progress plot with loss, RMSE, and R².
        
        Args:
            train_losses: Training losses over epochs
            train_rmses: Training RMSEs over epochs
            train_r2s: Training R² scores over epochs
            title: Plot title
            
        Returns:
            Plotly figure
        """
        epochs = list(range(1, len(train_losses) + 1))
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Loss', 'RMSE', 'R²'],
            horizontal_spacing=self.viz_params['subplot_horizontal_spacing']
        )
        
        # Loss plot
        fig.add_trace(
            go.Scatter(x=epochs, y=train_losses, name='Loss', 
                      line=dict(color='blue'), opacity=self.viz_params['line_opacity']),
            row=1, col=1
        )
        
        # RMSE plot
        fig.add_trace(
            go.Scatter(x=epochs, y=train_rmses, name='RMSE', 
                      line=dict(color='red'), opacity=self.viz_params['line_opacity']),
            row=1, col=2
        )
        
        # R² plot
        fig.add_trace(
            go.Scatter(x=epochs, y=train_r2s, name='R²', 
                      line=dict(color='green'), opacity=self.viz_params['line_opacity']),
            row=1, col=3
        )
        
        fig.update_layout(
            title=title,
            showlegend=False,
            height=400
        )
        
        return fig
        
    def create_actual_vs_predicted_plot(self, y_actual: np.ndarray, 
                                      y_predicted: np.ndarray,
                                      title: str = "Actual vs Predicted",
                                      weeks: Optional[List[int]] = None) -> go.Figure:
        """
        Create an actual vs predicted time series plot.
        
        Args:
            y_actual: Actual values
            y_predicted: Predicted values
            title: Plot title
            weeks: Optional week indices for x-axis
            
        Returns:
            Plotly figure
        """
        if weeks is None:
            weeks = list(range(len(y_actual)))
            
        fig = go.Figure()
        
        # Actual values
        fig.add_trace(go.Scatter(
            x=weeks,
            y=y_actual,
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2),
            opacity=self.viz_params['line_opacity']
        ))
        
        # Predicted values
        fig.add_trace(go.Scatter(
            x=weeks,
            y=y_predicted,
            mode='lines',
            name='Predicted',
            line=dict(color='red', width=2, dash='dot'),
            opacity=self.viz_params['line_opacity']
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Week',
            yaxis_title='Value',
            height=500,
            hovermode='x unified'
        )
        
        return fig
        
    def create_scatter_plot(self, x: np.ndarray, y: np.ndarray,
                          title: str = "Scatter Plot",
                          x_label: str = "X", y_label: str = "Y",
                          color: str = 'blue') -> go.Figure:
        """
        Create a scatter plot with perfect correlation line.
        
        Args:
            x: X values
            y: Y values
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            color: Marker color
            
        Returns:
            Plotly figure
        """
        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(x, y) if len(np.unique(x)) > 1 else 0.0
        
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            name=f'Data (R²={r2:.3f})',
            marker=dict(
                size=self.viz_params['marker_size'],
                color=color,
                opacity=self.viz_params['node_opacity']
            )
        ))
        
        # Perfect correlation line
        min_val, max_val = min(x.min(), y.min()), max(x.max(), y.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Correlation',
            line=dict(color='gray', dash='dash', width=1),
            opacity=0.5
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=500
        )
        
        return fig
        
    def create_waterfall_chart(self, categories: List[str], values: List[float],
                             title: str = "Waterfall Chart") -> go.Figure:
        """
        Create a proper waterfall chart using Plotly's go.Waterfall.
        
        Args:
            categories: Category names
            values: Values for each category
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(go.Waterfall(
            name="Contributions",
            orientation="v",
            measure=["relative"] * (len(categories) - 1) + ["total"],
            x=categories,
            textposition="outside",
            text=[f"{v:,.0f}" for v in values],
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title=title,
            showlegend=False,
            height=600
        )
        
        return fig
        
    def create_contribution_stacked_bar(self, media_contributions: np.ndarray,
                                      control_contributions: np.ndarray,
                                      baseline: np.ndarray,
                                      media_names: List[str],
                                      control_names: List[str],
                                      weeks: Optional[List[int]] = None,
                                      title: str = "Contributions Over Time") -> go.Figure:
        """
        Create a stacked bar chart of contributions over time.
        
        Args:
            media_contributions: Media contributions [n_weeks, n_media]
            control_contributions: Control contributions [n_weeks, n_controls]
            baseline: Baseline values [n_weeks]
            media_names: Media channel names
            control_names: Control variable names
            weeks: Optional week indices
            title: Chart title
            
        Returns:
            Plotly figure
        """
        if weeks is None:
            weeks = list(range(len(baseline)))
            
        fig = go.Figure()
        
        # Add baseline
        fig.add_trace(go.Bar(
            x=weeks,
            y=baseline,
            name='Baseline',
            marker_color='lightgray',
            opacity=self.viz_params['node_opacity']
        ))
        
        # Add media contributions
        colors = px.colors.qualitative.Set3
        for i, name in enumerate(media_names):
            fig.add_trace(go.Bar(
                x=weeks,
                y=media_contributions[:, i],
                name=f'Media: {name}',
                marker_color=colors[i % len(colors)],
                opacity=self.viz_params['node_opacity']
            ))
            
        # Add control contributions
        for i, name in enumerate(control_names):
            fig.add_trace(go.Bar(
                x=weeks,
                y=control_contributions[:, i],
                name=f'Control: {name}',
                marker_color=colors[(len(media_names) + i) % len(colors)],
                opacity=self.viz_params['node_opacity']
            ))
            
        fig.update_layout(
            title=title,
            xaxis_title='Week',
            yaxis_title='Contribution',
            barmode='stack',
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
        
    def create_dag_network_plot(self, adjacency_matrix: np.ndarray,
                              node_names: List[str],
                              title: str = "DAG Network") -> go.Figure:
        """
        Create a DAG network visualization.
        
        Args:
            adjacency_matrix: Adjacency matrix [n_nodes, n_nodes]
            node_names: Node names
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Create network graph
        G = nx.DiGraph()
        
        # Add nodes
        for i, name in enumerate(node_names):
            G.add_node(i, name=name, label=name)
            
        # Add edges based on adjacency matrix
        for i in range(len(node_names)):
            for j in range(len(node_names)):
                if adjacency_matrix[i, j] > self.viz_params['correlation_threshold']:
                    G.add_edge(i, j, weight=adjacency_matrix[i, j])
                    
        # Create layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add edges
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = G[edge[0]][edge[1]]['weight']
            
            fig.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(
                    width=weight * self.viz_params['edge_width_multiplier'],
                    color=f'rgba(125,125,125,{self.viz_params["line_opacity"]})'
                ),
                hoverinfo='none',
                showlegend=False
            ))
            
        # Add nodes
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = [node_names[node] for node in G.nodes()]
        
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=self.viz_params['marker_size'] * 5,  # Scale up for nodes
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            text=node_text,
            textposition="middle center",
            textfont=dict(size=10, color='black'),
            hovertemplate='<b>%{text}</b><extra></extra>',
            name='Nodes'
        ))
        
        fig.update_layout(
            title=title,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig
        
    def create_dag_heatmap(self, adjacency_matrix: np.ndarray,
                         node_names: List[str],
                         title: str = "DAG Adjacency Matrix") -> go.Figure:
        """
        Create a DAG adjacency matrix heatmap.
        
        Args:
            adjacency_matrix: Adjacency matrix [n_nodes, n_nodes]
            node_names: Node names
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=adjacency_matrix,
            x=node_names,
            y=node_names,
            colorscale='RdYlBu_r',
            hoverongaps=False,
            hovertemplate='From: %{y}<br>To: %{x}<br>Strength: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Influenced Channel',
            yaxis_title='Influencing Channel',
            height=600
        )
        
        return fig
        
    def save_plot(self, fig: go.Figure, filepath: str, 
                 include_plotlyjs: str = 'cdn') -> bool:
        """
        Save a Plotly figure to HTML file.
        
        Args:
            fig: Plotly figure to save
            filepath: Output file path
            include_plotlyjs: How to include Plotly.js ('cdn', 'inline', etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save figure
            fig.write_html(
                filepath,
                include_plotlyjs=include_plotlyjs,
                config={'displayModeBar': True, 'responsive': True}
            )
            return True
        except Exception as e:
            print(f"   ⚠️ Failed to save plot to {filepath}: {e}")
            return False
            
    def create_comprehensive_dashboard(self, results: Dict[str, Any],
                                     output_dir: str = "dashboard_comprehensive") -> List[Tuple[str, str]]:
        """
        Create a comprehensive dashboard with multiple plots.
        
        Args:
            results: Training results dictionary
            output_dir: Output directory for plots
            
        Returns:
            List of (plot_name, filepath) tuples for created plots
        """
        plots_created = []
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data from results
        model = results['model']
        train_losses = results.get('train_losses', [])
        train_rmses = results.get('train_rmses', [])
        train_r2s = results.get('train_r2s', [])
        
        # 1. Training Progress
        if train_losses:
            fig = self.create_training_progress_plot(
                train_losses, train_rmses, train_r2s,
                "Training Progress"
            )
            filepath = os.path.join(output_dir, "training_progress.html")
            if self.save_plot(fig, filepath):
                plots_created.append(("Training Progress", filepath))
                
        # Add more plots as needed...
        
        return plots_created
