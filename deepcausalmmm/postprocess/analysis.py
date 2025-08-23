"""
Post-processing utilities for analyzing and visualizing DeepCausalMMM results.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
import os

from deepcausalmmm.core.inference import InferenceManager, ModelInference  # ModelInference for legacy compatibility
from deepcausalmmm.core.scaling import SimpleGlobalScaler
from deepcausalmmm.core.config import get_default_config
import torch


class ModelAnalyzer:
    """Analyze and visualize DeepCausalMMM model results with modern class-based architecture."""
    def __init__(
        self,
        inference: Optional[InferenceManager] = None,  # Modern InferenceManager
        legacy_inference: Optional[ModelInference] = None,  # Legacy compatibility
        scaler: Optional[SimpleGlobalScaler] = None,  # Legacy compatibility
        pipeline = None,  # UnifiedDataPipeline instance
        config: Optional[Dict] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the enhanced analyzer with unified pipeline support.
        
        Args:
            inference: Modern InferenceManager instance (preferred)
            legacy_inference: Legacy ModelInference instance (for compatibility)
            scaler: SimpleGlobalScaler for proper inverse transformations (legacy)
            pipeline: UnifiedDataPipeline instance (preferred)
            config: Model configuration dictionary
            output_dir: Directory to save outputs
        """
        # Use modern InferenceManager if available, otherwise fall back to legacy
        self.inference = inference or legacy_inference
        self.scaler = scaler  # Legacy support
        self.pipeline = pipeline  # Unified pipeline support
        self.config = config or get_default_config()
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def analyze_with_unified_pipeline(
        self,
        model,
        X_media: np.ndarray,
        X_control: np.ndarray,
        y_true: np.ndarray,
        channel_names: List[str],
        control_names: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze model results using the unified pipeline.
        
        Args:
            model: Trained model
            X_media: Media data
            X_control: Control data
            y_true: True target values
            channel_names: Media channel names
            control_names: Control variable names
            
        Returns:
            Analysis results dictionary
        """
        if self.pipeline is None:
            raise ValueError("UnifiedDataPipeline is required for this method")
            
        print(f"\n📊 ModelAnalyzer: Unified Pipeline Analysis")
        
        # Get predictions and contributions
        results = self.pipeline.predict_and_postprocess(
            model=model,
            X_media=X_media,
            X_control=X_control,
            channel_names=channel_names,
            control_names=control_names,
            combine_with_holdout=True
        )
        
        # Calculate metrics
        metrics = self.pipeline.calculate_metrics(
            y_true, results['predictions'], prefix='pipeline_'
        )
        
        # Combine results
        analysis_results = {
            **results,
            **metrics,
            'y_true': y_true
        }
        
        print(f"   ✅ Pipeline analysis complete")
        print(f"   📈 R²: {metrics['pipeline_r2']:.3f}")
        print(f"   📊 RMSE: {metrics['pipeline_rmse']:.0f}")
        
        return analysis_results
    
    def analyze_predictions(
        self,
        X_m: np.ndarray,
        X_c: np.ndarray,
        R: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        generate_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze model predictions and generate visualizations.
        
        Args:
            X_m: Media variables [n_regions, n_weeks, n_channels]
            X_c: Control variables [n_regions, n_weeks, n_controls]
            R: Region indices [n_regions]
            y_true: Optional ground truth values
            generate_plots: Whether to generate and save plots
            
        Returns:
            Dictionary containing analysis results and metrics
        """
        # Convert inputs to tensors
        X_m_tensor = torch.tensor(X_m, dtype=torch.float32)
        X_c_tensor = torch.tensor(X_c, dtype=torch.float32)
        R_tensor = torch.tensor(R, dtype=torch.long)
        y_true_tensor = torch.tensor(y_true, dtype=torch.float32) if y_true is not None else None
        
        # Get predictions and contributions
        results = self.inference.predict(X_m_tensor, X_c_tensor, R_tensor, y_true_tensor)
        
        # Generate plots if requested
        if generate_plots:
            self._generate_plots(results)
        
        return results
    
    def _generate_plots(self, results: Dict[str, Any]) -> None:
        """Generate all visualization plots."""
        # 1. Coefficients over time
        coeff_fig = self.plot_coefficients_over_time(
            results['coefficients'],
            self.inference.channel_names
        )
        
        # 2. Actual vs Predicted
        comparison_fig = self.plot_contribution_comparison(
            results['predictions'],
            results.get('actual_revenue'),  # May be None
            self.inference.burn_in_weeks
        )
        
        # 3. Waterfall chart
        waterfall_fig = self.plot_waterfall_chart(
            results['contributions'],
            results.get('control_contributions'),
            results.get('baseline'),
            self.inference.channel_names,
            self.inference.control_names
        )
        
        # 4. Contribution donut
        donut_fig = self.plot_contribution_donut(
            results['contributions'],
            results.get('control_contributions'),
            results.get('baseline')
        )
        
        # Save plots if output directory is specified
        if self.output_dir:
            coeff_fig.write_html(os.path.join(self.output_dir, 'coefficients.html'))
            comparison_fig.write_html(os.path.join(self.output_dir, 'comparison.html'))
            waterfall_fig.write_html(os.path.join(self.output_dir, 'waterfall.html'))
            donut_fig.write_html(os.path.join(self.output_dir, 'donut.html'))
        
        # Display plots - Commented out to prevent browser popups
        # coeff_fig.show()
        # comparison_fig.show()
        # waterfall_fig.show()
        # donut_fig.show()
    
    @staticmethod
    def plot_coefficients_over_time(
        coefficients: np.ndarray,
        channel_names: List[str]
    ) -> go.Figure:
        """Plot mean coefficients over time for each channel."""
        mean_coeffs = coefficients.mean(axis=0)  # Average across regions
        
        fig = go.Figure()
        for i, name in enumerate(channel_names):
            fig.add_trace(
                go.Scatter(
                    y=mean_coeffs[:, i],
                    name=name,
                    mode='lines'
                )
            )
        
        fig.update_layout(
            title="Channel Coefficients Over Time",
            xaxis_title="Week",
            yaxis_title="Coefficient Value",
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def plot_contribution_comparison(
        predictions: np.ndarray,
        actuals: Optional[np.ndarray],
        burn_in_weeks: int
    ) -> go.Figure:
        """Plot actual vs predicted revenue comparison."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Time Series Comparison', 'Scatter Plot'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Sum across regions
        total_pred = predictions.sum(axis=0)[burn_in_weeks:]
        
        if actuals is not None:
            total_actual = actuals.sum(axis=0)[burn_in_weeks:]
            
            # Time series plot
            fig.add_trace(
                go.Scatter(
                    y=total_actual,
                    name='Actual Revenue',
                    mode='lines',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Scatter plot
            fig.add_trace(
                go.Scatter(
                    x=total_actual,
                    y=total_pred,
                    mode='markers',
                    name='Actual vs Predicted',
                    marker=dict(color='green')
                ),
                row=1, col=2
            )
            
            # Add 45-degree line
            min_val = min(total_actual.min(), total_pred.min())
            max_val = max(total_actual.max(), total_pred.max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='45° line',
                    line=dict(dash='dash', color='gray')
                ),
                row=1, col=2
            )
        
        # Add predicted line
        fig.add_trace(
            go.Scatter(
                y=total_pred,
                name='Predicted Revenue',
                mode='lines',
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        fig.update_layout(
            title="Actual vs Predicted Revenue Comparison",
            showlegend=True,
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_waterfall_chart(
        marketing_contribs: np.ndarray,
        control_contribs: Optional[np.ndarray],
        baseline: Optional[np.ndarray],
        channel_names: List[str],
        control_names: List[str]
    ) -> go.Figure:
        """Create waterfall chart of contributions."""
        # Calculate mean contributions
        mean_marketing = marketing_contribs.mean(axis=(0, 1))  # Average across regions and time
        measures = ['relative'] * len(channel_names)
        values = list(mean_marketing)
        labels = channel_names.copy()
        
        if control_contribs is not None:
            mean_control = control_contribs.mean(axis=(0, 1))
            measures.extend(['relative'] * len(control_names))
            values.extend(mean_control)
            labels.extend(control_names)
        
        if baseline is not None:
            mean_baseline = baseline.mean()
            measures.append('total')
            values.append(mean_baseline)
            labels.append('Baseline')
        
        fig = go.Figure(go.Waterfall(
            name="Contribution Breakdown",
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title="Contribution Waterfall Chart",
            showlegend=True,
            xaxis_title="Components",
            yaxis_title="Contribution"
        )
        
        return fig
    
    @staticmethod
    def plot_contribution_donut(
        marketing_contribs: np.ndarray,
        control_contribs: Optional[np.ndarray],
        baseline: Optional[np.ndarray]
    ) -> go.Figure:
        """Create donut chart of contribution percentages."""
        # Calculate total contributions
        total_marketing = marketing_contribs.sum()
        total_control = control_contribs.sum() if control_contribs is not None else 0
        total_baseline = baseline.sum() if baseline is not None else 0
        
        total = total_marketing + total_control + total_baseline
        
        values = [
            (total_marketing / total) * 100,
            (total_control / total) * 100 if control_contribs is not None else 0,
            (total_baseline / total) * 100 if baseline is not None else 0
        ]
        
        labels = ['Marketing', 'Control', 'Baseline']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.4,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title="Contribution Split",
            annotations=[dict(text='Total', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        return fig 