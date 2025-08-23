"""
Comprehensive post-processing analysis for DeepCausalMMM with inverse transformation.
Includes all visualizations: coefficients, contributions, DAG, actual vs predicted, channel analysis.
Automatically handles burn-in/padding removal from all outputs.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import networkx as nx

from typing import Dict, List, Optional, Any, Tuple
import os
from datetime import datetime
from deepcausalmmm.core.config import get_default_config
from deepcausalmmm.core.inference import InferenceManager
from deepcausalmmm.core.visualization import VisualizationManager


class ComprehensiveAnalyzer:
    """Modernized comprehensive analyzer for DeepCausalMMM with config-driven visualizations."""
    
    def __init__(
        self,
        model,
        media_cols: List[str],
        control_cols: List[str],
        output_dir: str = "mmm_analysis_results",
        pipeline = None,  # UnifiedDataPipeline instance
        auto_detect_burnin: bool = True,
        manual_burnin_weeks: Optional[int] = None,
        config: Optional[Dict] = None,
        inference: Optional[InferenceManager] = None  # Modern inference manager
    ):
        """
        Initialize the comprehensive analyzer.
        
        Args:
            model: Trained DeepCausalMMM model
            media_cols: List of media column names
            control_cols: List of control column names
            output_dir: Directory to save outputs
            pipeline: UnifiedDataPipeline instance for modern data processing
            auto_detect_burnin: Whether to automatically detect burn-in weeks from model
            manual_burnin_weeks: Manually specify burn-in weeks (overrides auto-detection)
            config: Configuration dictionary (uses default if None)
            inference: Modern InferenceManager instance
        """
        self.model = model
        self.media_cols = media_cols
        self.control_cols = control_cols
        self.output_dir = output_dir
        self.pipeline = pipeline  # UnifiedDataPipeline instance
        self.config = config or get_default_config()
        self.auto_detect_burnin = auto_detect_burnin
        self.manual_burnin_weeks = manual_burnin_weeks
        
        # Modern class-based components
        self.inference = inference
        self.viz_manager = VisualizationManager(self.config)
        
        # Get visualization parameters from config
        self.viz_params = self._get_viz_params()
        
        # Use pipeline's padding weeks if available, otherwise detect
        if self.pipeline is not None:
            self.burnin_weeks = self.pipeline.padding_weeks
            print(f"   Using unified pipeline burn-in weeks: {self.burnin_weeks}")
        else:
            # Fallback to legacy detection
            self.burnin_weeks = self._detect_burnin_weeks()
        
        # Create output directory using config
        output_paths = self.config.get('output_paths', {})
        dashboard_dir = output_paths.get('dashboard_dir', 'dashboard_beautiful_comprehensive')
        if not os.path.isabs(output_dir) and output_dir == "mmm_analysis_results":
            self.output_dir = dashboard_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Store analysis timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"🔥 ComprehensiveAnalyzer initialized (Modernized):")
        print(f"   Auto burn-in detection: {auto_detect_burnin}")
        print(f"   Detected/Manual burn-in weeks: {self.burnin_weeks}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Using config-driven visualization parameters")
    
    def _detect_burnin_weeks(self) -> int:
        """
        Detect burn-in weeks from the model or use manual override.
        
        Returns:
            Number of burn-in weeks to remove from analysis
        """
        if self.manual_burnin_weeks is not None:
            print(f"   Using manual burn-in weeks: {self.manual_burnin_weeks}")
            return self.manual_burnin_weeks
        
        if not self.auto_detect_burnin:
            return 0
        
        # Try to detect from model attributes
        burnin_weeks = 0
        
        # Check if model has burn_in_weeks attribute
        if hasattr(self.model, 'burn_in_weeks'):
            burnin_weeks = self.model.burn_in_weeks
            print(f"   Detected burn-in from model.burn_in_weeks: {burnin_weeks}")
        
        # Check if model outputs contain burn-in info
        elif hasattr(self.model, 'forward'):
            try:
                # Try a dummy forward pass to check outputs
                with torch.no_grad():
                    # Create dummy inputs
                    dummy_media = torch.randn(1, 10, len(self.media_cols))
                    dummy_control = torch.randn(1, 10, len(self.control_cols))
                    dummy_regions = torch.zeros(1, 10, dtype=torch.long)
                    
                    _, _, _, outputs = self.model(dummy_media, dummy_control, dummy_regions)
                    
                    if 'burn_in_weeks' in outputs:
                        burnin_weeks = outputs['burn_in_weeks']
                        print(f"   Detected burn-in from model outputs: {burnin_weeks}")
            except Exception as e:
                print(f"   Could not detect burn-in from model forward pass: {e}")
        
        # Default heuristic: assume 4 weeks if GRU is present
        if burnin_weeks == 0 and hasattr(self.model, 'gru'):
            burnin_weeks = 4
            print(f"   Using default GRU burn-in: {burnin_weeks} weeks")
        
        return burnin_weeks
    
    def _get_viz_params(self) -> Dict[str, Any]:
        """Get visualization parameters from config with defaults (like dashboard)."""
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
    
    def _remove_burnin_from_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Remove burn-in weeks from a tensor.
        
        Args:
            tensor: Input tensor with time dimension (assumed to be dim=1)
            
        Returns:
            Tensor with burn-in weeks removed
        """
        if self.burnin_weeks == 0 or tensor.shape[1] <= self.burnin_weeks:
            return tensor
        
        return tensor[:, self.burnin_weeks:, ...]
    
    def _remove_burnin_from_array(self, array: np.ndarray) -> np.ndarray:
        """
        Remove burn-in weeks from a numpy array.
        
        Args:
            array: Input array with time dimension (assumed to be dim=1)
            
        Returns:
            Array with burn-in weeks removed
        """
        if self.burnin_weeks == 0 or array.shape[1] <= self.burnin_weeks:
            return array
        
        return array[:, self.burnin_weeks:, ...]
    
    def _clean_model_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean model outputs by removing burn-in weeks from all temporal tensors.
        
        Args:
            outputs: Dictionary of model outputs
            
        Returns:
            Cleaned outputs with burn-in removed
        """
        if self.burnin_weeks == 0:
            return outputs
        
        cleaned_outputs = {}
        
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor) and value.dim() >= 2:
                # Remove burn-in from time dimension (assumed to be dim=1)
                if value.shape[1] > self.burnin_weeks:
                    cleaned_outputs[key] = self._remove_burnin_from_tensor(value)
                    print(f"   Removed {self.burnin_weeks} burn-in weeks from {key}: {value.shape} -> {cleaned_outputs[key].shape}")
                else:
                    cleaned_outputs[key] = value
            elif isinstance(value, np.ndarray) and value.ndim >= 2:
                # Remove burn-in from time dimension
                if value.shape[1] > self.burnin_weeks:
                    cleaned_outputs[key] = self._remove_burnin_from_array(value)
                    print(f"   Removed {self.burnin_weeks} burn-in weeks from {key}: {value.shape} -> {cleaned_outputs[key].shape}")
                else:
                    cleaned_outputs[key] = value
            else:
                # Keep non-temporal outputs as is
                cleaned_outputs[key] = value
        
        return cleaned_outputs
    
    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Apply inverse transformation to target variable using modern pipeline.
        
        Args:
            y_scaled: Scaled target values
            
        Returns:
            Unscaled target values
        """
        if self.pipeline is not None and hasattr(self.pipeline, 'scaler'):
            # Use modern pipeline's scaler for inverse transformation
            return self.pipeline.scaler.inverse_transform_target(y_scaled)
        
        # Fallback: return as-is (should not happen in modern usage)
        print("⚠️  Warning: No pipeline available for inverse transformation")
        return y_scaled
    
    def inverse_transform_contributions(
        self, 
        contributions_scaled: np.ndarray, 
        y_original: np.ndarray
    ) -> np.ndarray:
        """
        Apply inverse transformation to contributions using modern pipeline.
        
        Args:
            contributions_scaled: Scaled contributions
            y_original: Original scale target values
            
        Returns:
            Contributions in original scale
        """
        if self.pipeline is not None and hasattr(self.pipeline, 'scaler'):
            # Use modern pipeline's scaler for contribution inverse transformation
            return self.pipeline.scaler.inverse_transform_contributions(contributions_scaled, y_original)
        
        # Fallback: scale by target mean
        scale_factor = y_original.mean() / (contributions_scaled.sum() + 1e-8)
        return contributions_scaled * scale_factor
    
    def analyze_with_unified_pipeline(
        self,
        X_media: np.ndarray,
        X_control: np.ndarray, 
        y_true: np.ndarray,
        create_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis using the unified pipeline.
        
        Args:
            X_media: Media data (full dataset)
            X_control: Control data (full dataset)
            y_true: True target values (full dataset)
            create_plots: Whether to create visualization plots
            
        Returns:
            Dictionary with analysis results
        """
        if self.pipeline is None:
            raise ValueError("UnifiedDataPipeline is required for this method")
            
        print(f"\n📊 UNIFIED PIPELINE COMPREHENSIVE ANALYSIS")
        print(f"=" * 55)
        
        # 1. Get predictions and contributions using unified pipeline
        results = self.pipeline.predict_and_postprocess(
            model=self.model,
            X_media=X_media,
            X_control=X_control,
            channel_names=self.media_cols,
            control_names=self.control_cols,
            combine_with_holdout=True
        )
        
        # 2. Extract data
        predictions = results['predictions']
        media_contributions = results['media_contributions']
        control_contributions = results['control_contributions']
        
        print(f"   ✅ Predictions shape: {predictions.shape}")
        print(f"   📺 Media contributions shape: {media_contributions.shape}")
        print(f"   🎛️ Control contributions shape: {control_contributions.shape}")
        
        # 3. Calculate comprehensive metrics
        analysis_results = {
            'predictions': predictions,
            'media_contributions': media_contributions,
            'control_contributions': control_contributions,
            'channel_names': self.media_cols,
            'control_names': self.control_cols,
            'pipeline': self.pipeline,
            'model_outputs': results['model_outputs']
        }
        
        # 4. Calculate metrics using pipeline
        train_metrics = self.pipeline.calculate_metrics(
            y_true, predictions, prefix='unified_'
        )
        analysis_results.update(train_metrics)
        
        # 5. Create visualizations if requested
        if create_plots:
            print(f"\n🎨 Creating unified pipeline visualizations...")
            
            # Channel effectiveness analysis
            channel_analysis = self._analyze_channel_effectiveness_unified(
                media_contributions, self.media_cols
            )
            analysis_results['channel_analysis'] = channel_analysis
            
            # Time series plots
            if create_plots:
                self._create_unified_time_series_plots(
                    y_true, predictions, media_contributions
                )
                
                self._create_unified_contribution_plots(
                    media_contributions, control_contributions
                )
        
        print(f"\n✅ Unified pipeline analysis complete!")
        print(f"   📈 R²: {analysis_results['unified_r2']:.3f}")
        print(f"   📊 RMSE: {analysis_results['unified_rmse']:.0f}")
        print(f"   💯 MAE: {analysis_results['unified_mae']:.0f}")
        
        return analysis_results
    
    def _analyze_channel_effectiveness_unified(
        self,
        media_contributions: torch.Tensor,
        channel_names: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze channel effectiveness using unified pipeline data.
        
        Args:
            media_contributions: Media contributions tensor
            channel_names: List of channel names
            
        Returns:
            Dictionary with channel analysis
        """
        # Convert to numpy for analysis
        if isinstance(media_contributions, torch.Tensor):
            contrib_np = media_contributions.detach().numpy()
        else:
            contrib_np = media_contributions
            
        # Calculate total contribution per channel
        total_contrib = np.sum(contrib_np, axis=(0, 1))  # Sum over regions and time
        
        # Calculate average contribution per channel
        avg_contrib = np.mean(contrib_np, axis=(0, 1))
        
        # Create channel analysis
        channel_analysis = {
            'channel_names': channel_names,
            'total_contributions': total_contrib,
            'average_contributions': avg_contrib,
            'contribution_percentages': (total_contrib / np.sum(total_contrib)) * 100
        }
        
        return channel_analysis
    
    def _create_unified_time_series_plots(
        self,
        y_true: np.ndarray,
        predictions: torch.Tensor,
        media_contributions: torch.Tensor
    ) -> None:
        """
        Create time series plots using unified pipeline data.
        
        Args:
            y_true: True values
            predictions: Model predictions
            media_contributions: Media contributions
        """
        # Convert predictions to numpy
        if isinstance(predictions, torch.Tensor):
            pred_np = predictions.detach().numpy()
        else:
            pred_np = predictions
            
        # Create actual vs predicted plot
        fig = go.Figure()
        
        # Sum over regions for aggregate view
        y_true_agg = np.sum(y_true, axis=0)
        pred_agg = np.sum(pred_np, axis=0)
        
        fig.add_trace(go.Scatter(
            y=y_true_agg,
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            y=pred_agg,
            mode='lines',
            name='Predicted',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Unified Pipeline: Actual vs Predicted Time Series',
            xaxis_title='Time Period',
            yaxis_title='Values',
            template='plotly_white'
        )
        
        # Save plot
        output_path = os.path.join(self.output_dir, f'unified_actual_vs_predicted_{self.timestamp}.html')
        fig.write_html(output_path)
        print(f"   ✅ Saved: {output_path}")
    
    def _create_unified_contribution_plots(
        self,
        media_contributions: torch.Tensor,
        control_contributions: torch.Tensor
    ) -> None:
        """
        Create contribution plots using unified pipeline data.
        
        Args:
            media_contributions: Media contributions tensor
            control_contributions: Control contributions tensor
        """
        # Convert to numpy
        if isinstance(media_contributions, torch.Tensor):
            media_np = media_contributions.detach().numpy()
        else:
            media_np = media_contributions
            
        # Sum over regions and time for channel effectiveness
        channel_totals = np.sum(media_np, axis=(0, 1))
        
        # Create channel effectiveness plot
        fig = go.Figure(data=[
            go.Bar(
                x=self.media_cols,
                y=channel_totals,
                text=[f'{val:.0f}' for val in channel_totals],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Unified Pipeline: Channel Effectiveness',
            xaxis_title='Media Channels',
            yaxis_title='Total Contribution',
            template='plotly_white'
        )
        
        # Save plot
        output_path = os.path.join(self.output_dir, f'unified_channel_effectiveness_{self.timestamp}.html')
        fig.write_html(output_path)
        print(f"   ✅ Saved: {output_path}")
    
    def analyze_comprehensive(
        self,
        X_media: np.ndarray,
        X_control: np.ndarray,
        y_true: np.ndarray,
        region_ids: np.ndarray,
        weeks: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive analysis with all visualizations.
        Automatically removes burn-in/padding from all outputs.
        
        Args:
            X_media: Media variables [n_regions, n_weeks, n_channels] (may include padding)
            X_control: Control variables [n_regions, n_weeks, n_controls] (may include padding)
            y_true: True target values (scaled, may include padding)
            region_ids: Region identifiers
            weeks: Week labels (optional)
            
        Returns:
            Dictionary containing all analysis results (burn-in removed)
        """
        print("=== COMPREHENSIVE DEEPCAUSALMMM ANALYSIS ===")
        print(f"Input data: {X_media.shape[0]} regions × {X_media.shape[1]} weeks")
        
        if self.burnin_weeks > 0:
            print(f"🔥 Burn-in removal: {self.burnin_weeks} weeks will be removed from analysis")
        
        print(f"Media channels: {len(self.media_cols)}")
        print(f"Control variables: {len(self.control_cols)}")
        print()
        
        # Convert to tensors
        X_media_tensor = torch.FloatTensor(X_media)
        X_control_tensor = torch.FloatTensor(X_control)
        region_tensor = torch.LongTensor(region_ids)
        
        # Generate predictions
        print("Generating model predictions...")
        self.model.eval()
        with torch.no_grad():
            y_pred, coeffs, media_contrib, outputs = self.model(
                X_media_tensor, X_control_tensor, region_tensor
            )
            
            # Clean model outputs (remove burn-in)
            if self.burnin_weeks > 0:
                print(f"Removing {self.burnin_weeks} burn-in weeks from model outputs...")
                outputs = self._clean_model_outputs(outputs)
                
                # Also remove burn-in from main outputs
                y_pred = self._remove_burnin_from_tensor(y_pred)
                coeffs = self._remove_burnin_from_tensor(coeffs)
                media_contrib = self._remove_burnin_from_tensor(media_contrib)
                
                print(f"   Cleaned predictions: {y_pred.shape}")
                print(f"   Cleaned coefficients: {coeffs.shape}")
                print(f"   Cleaned contributions: {media_contrib.shape}")
            
            # Convert to numpy
            y_pred_scaled = y_pred.cpu().numpy()
            coeffs_np = coeffs.cpu().numpy()
            media_contrib_scaled = media_contrib.cpu().numpy()
            
            if 'control_coefficients' in outputs:
                ctrl_coeffs_np = outputs['control_coefficients']
                ctrl_contrib_scaled = outputs['control_contributions']
            else:
                ctrl_coeffs_np = np.zeros((X_media.shape[0], X_media.shape[1], len(self.control_cols)))
                ctrl_contrib_scaled = np.zeros((X_media.shape[0], X_media.shape[1], len(self.control_cols)))
            
            baseline_scaled = outputs['baseline'].cpu().numpy()
        
        # Remove burn-in from input data for final analysis
        if self.burnin_weeks > 0:
            print(f"Removing {self.burnin_weeks} burn-in weeks from input data...")
            y_true_clean = self._remove_burnin_from_array(y_true)
            X_media_clean = self._remove_burnin_from_array(X_media)
            X_control_clean = self._remove_burnin_from_array(X_control)
            print(f"   Cleaned input shapes: y_true={y_true_clean.shape}, X_media={X_media_clean.shape}")
        else:
            y_true_clean = y_true
            X_media_clean = X_media
            X_control_clean = X_control
        
        # Apply inverse transformations
        print("Applying inverse transformations...")
        y_true_original = self.inverse_transform_target(y_true_clean)
        y_pred_original = self.inverse_transform_target(y_pred_scaled)
        
        # Transform contributions to original scale
        media_contrib_original = self.inverse_transform_contributions(media_contrib_scaled, y_true_original)
        ctrl_contrib_original = self.inverse_transform_contributions(ctrl_contrib_scaled, y_true_original)
        
        # Handle baseline separately (it's 2D, not 3D like contributions)
        if baseline_scaled.ndim == 2:
            # Baseline is [regions, weeks] - transform directly
            scale_factor = y_true_original.mean() / (baseline_scaled.mean() + 1e-8)
            baseline_original = baseline_scaled * scale_factor
        else:
            baseline_original = self.inverse_transform_contributions(baseline_scaled, y_true_original)
        
        # Set up weeks (for cleaned data)
        if weeks is None:
            final_weeks = y_pred_scaled.shape[1]  # Use cleaned data length
            weeks = list(range(1, final_weeks + 1))
        elif self.burnin_weeks > 0 and len(weeks) > self.burnin_weeks:
            # Remove burn-in weeks from provided week labels
            weeks = weeks[self.burnin_weeks:]
        
        # Store results
        results = {
            'y_true_original': y_true_original,
            'y_pred_original': y_pred_original,
            'coeffs': coeffs_np,
            'media_contrib_original': media_contrib_original,
            'ctrl_contrib_original': ctrl_contrib_original,
            'baseline_original': baseline_original,
            'weeks': weeks,
            'timestamp': self.timestamp
        }
        
        print("Creating comprehensive visualizations...")
        
        # 1. Media Coefficients Over Time
        self._create_media_coefficients_plot(coeffs_np, weeks)
        
        # 2. Control Coefficients Over Time
        self._create_control_coefficients_plot(ctrl_coeffs_np, weeks)
        
        # 3. Contributions Over Time (Original Scale)
        self._create_contributions_plot(
            media_contrib_original, ctrl_contrib_original, baseline_original, weeks
        )
        
        # 4. Actual vs Predicted (Original Scale)
        self._create_actual_vs_predicted_plot(y_true_original, y_pred_original, weeks)
        
        # 5. Individual Channel Contributions (Original Scale)
        self._create_individual_channel_analysis(media_contrib_original, weeks)
        
        # 6. Beautiful Waterfall Chart
        self._create_waterfall_chart(
            media_contrib_original, ctrl_contrib_original, baseline_original
        )
        
        # 7. DAG Structure
        self._create_dag_visualization()
        
        # 8. Performance Summary Report
        self._create_performance_report(results)
        
        print(f"\n✅ Analysis complete! Results saved to: {self.output_dir}")
        print(f"📊 All visualizations use ORIGINAL SCALE data after inverse transformation")
        
        if self.burnin_weeks > 0:
            print(f"🔥 Burn-in removal applied: {self.burnin_weeks} weeks removed from all outputs")
            print(f"📈 Final analysis dimensions: {len(weeks)} weeks")
        
        return results
    
    def _create_media_coefficients_plot(self, coeffs: np.ndarray, weeks: List[int]):
        """Create media coefficients over time visualization."""
        print("  Creating media coefficients over time...")
        
        fig = make_subplots(
            rows=4, cols=4,
            subplot_titles=[col.replace('impressions_', '').replace('_', ' ')[:20] 
                          for col in self.media_cols],
            vertical_spacing=0.08,
            horizontal_spacing=0.06
        )
        
        colors = px.colors.qualitative.Set3
        
        for i, col in enumerate(self.media_cols):
            row = (i // 4) + 1
            col_idx = (i % 4) + 1
            
            # Average across regions
            avg_coeffs = coeffs[:, :, i].mean(axis=0)
            
            fig.add_trace(
                go.Scatter(
                    x=weeks, y=avg_coeffs,
                    mode='lines+markers',
                    name=col.replace('impressions_', '')[:15],
                    line=dict(width=2, color=colors[i % len(colors)]),
                    marker=dict(size=self.viz_params['marker_size'] // 2),
                    showlegend=False
                ),
                row=row, col=col_idx
            )
        
        fig.update_layout(
            title='📊 Media Coefficients Over Time (All Regions Average)',
            height=900,
            showlegend=False
        )
        fig.update_xaxes(title_text='Week')
        fig.update_yaxes(title_text='Coefficient')
        
        filename = f"{self.output_dir}/media_coefficients_{self.timestamp}.html"
        fig.write_html(filename)
        # fig.show()  # Removed to prevent browser popup
    
    def _create_control_coefficients_plot(self, ctrl_coeffs: np.ndarray, weeks: List[int]):
        """Create control coefficients over time visualization."""
        print("  Creating control coefficients over time...")
        
        fig = make_subplots(
            rows=2, cols=4,
            subplot_titles=[col.replace('value_', '').replace('_', ' ')[:20] 
                          for col in self.control_cols],
            vertical_spacing=self.viz_params['subplot_vertical_spacing'] * 2,
            horizontal_spacing=self.viz_params['subplot_horizontal_spacing']
        )
        
        colors = px.colors.qualitative.Set3
        
        for i, col in enumerate(self.control_cols):
            row = (i // 4) + 1
            col_idx = (i % 4) + 1
            
            # Average across regions
            avg_coeffs = ctrl_coeffs[:, :, i].mean(axis=0)
            
            fig.add_trace(
                go.Scatter(
                    x=weeks, y=avg_coeffs,
                    mode='lines+markers',
                    name=col.replace('value_', '')[:15],
                    line=dict(width=2, color=colors[(i+5) % len(colors)]),
                    marker=dict(size=self.viz_params['marker_size'] // 2),
                    showlegend=False
                ),
                row=row, col=col_idx
            )
        
        fig.update_layout(
            title='🎛️ Control Variable Coefficients Over Time (All Regions Average)',
            height=600,
            showlegend=False
        )
        fig.update_xaxes(title_text='Week')
        fig.update_yaxes(title_text='Coefficient')
        
        filename = f"{self.output_dir}/control_coefficients_{self.timestamp}.html"
        fig.write_html(filename)
        # fig.show()  # Removed to prevent browser popup
    
    def _create_contributions_plot(
        self, 
        media_contrib: np.ndarray, 
        ctrl_contrib: np.ndarray, 
        baseline: np.ndarray, 
        weeks: List[int]
    ):
        """Create beautiful contributions over time visualization with multiple views (original scale)."""
        print("  Creating beautiful contributions over time (original scale)...")
        
        # Sum across regions and channels/variables
        media_total = media_contrib.sum(axis=(0, 2))
        ctrl_total = ctrl_contrib.sum(axis=(0, 2))
        baseline_total = baseline.sum(axis=0)
        
        # 1. Individual Contributions Lines
        fig_lines = go.Figure()
        
        fig_lines.add_trace(go.Scatter(
            x=weeks, y=media_total,
            mode='lines+markers',
            name='Media Contribution',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6),
            hovertemplate='<b>Media</b><br>Week: %{x}<br>Contribution: %{y:,.0f}<extra></extra>'
        ))
        
        fig_lines.add_trace(go.Scatter(
            x=weeks, y=ctrl_total,
            mode='lines+markers',
            name='Control Contribution',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=6),
            hovertemplate='<b>Control</b><br>Week: %{x}<br>Contribution: %{y:,.0f}<extra></extra>'
        ))
        
        fig_lines.add_trace(go.Scatter(
            x=weeks, y=baseline_total,
            mode='lines+markers',
            name='Baseline',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=6),
            hovertemplate='<b>Baseline</b><br>Week: %{x}<br>Contribution: %{y:,.0f}<extra></extra>'
        ))
        
        total_contrib = np.add(np.add(media_total, ctrl_total), baseline_total)
        fig_lines.add_trace(go.Scatter(
            x=weeks, y=total_contrib,
            mode='lines',
            name='Total Contribution',
            line=dict(color='black', width=2, dash='dash'),
            hovertemplate='<b>Total</b><br>Week: %{x}<br>Contribution: %{y:,.0f}<extra></extra>'
        ))
        
        fig_lines.update_layout(
            title='💰 Total Contributions Over Time (Original Scale - All Regions Combined)',
            xaxis_title='Week',
            yaxis_title='Contribution (Original Scale)',
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        lines_filename = f"{self.output_dir}/contributions_lines_{self.timestamp}.html"
        fig_lines.write_html(lines_filename)
        # fig_lines.show()  # Removed to prevent browser popup
        
        # 2. Stacked Area Chart
        fig_stacked = go.Figure()
        
        fig_stacked.add_trace(go.Scatter(
            x=weeks, y=baseline_total,
            mode='lines',
            stackgroup='one',
            name='Baseline',
            line=dict(color='#2ca02c'),
            hovertemplate='<b>Baseline</b><br>Week: %{x}<br>Contribution: %{y:,.0f}<extra></extra>'
        ))
        
        fig_stacked.add_trace(go.Scatter(
            x=weeks, y=ctrl_total,
            mode='lines',
            stackgroup='one',
            name='Control Variables',
            line=dict(color='#ff7f0e'),
            hovertemplate='<b>Control</b><br>Week: %{x}<br>Contribution: %{y:,.0f}<extra></extra>'
        ))
        
        fig_stacked.add_trace(go.Scatter(
            x=weeks, y=media_total,
            mode='lines',
            stackgroup='one',
            name='Media Channels',
            line=dict(color='#1f77b4'),
            hovertemplate='<b>Media</b><br>Week: %{x}<br>Contribution: %{y:,.0f}<extra></extra>'
        ))
        
        fig_stacked.update_layout(
            title='📊 Contributions Stacked Over Time (Original Scale)',
            xaxis_title='Week',
            yaxis_title='Cumulative Contribution (Original Scale)',
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        stacked_filename = f"{self.output_dir}/contributions_stacked_{self.timestamp}.html"
        fig_stacked.write_html(stacked_filename)
        # fig_stacked.show()  # Removed to prevent browser popup
    
    def _create_actual_vs_predicted_plot(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        weeks: List[int]
    ):
        """Create actual vs predicted visualization (original scale)."""
        print("  Creating actual vs predicted (original scale)...")
        
        # Sum across regions for each week
        actual_total = y_true.sum(axis=0)
        predicted_total = y_pred.sum(axis=0)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Time Series: Actual vs Predicted (Original Scale)", 
                          "Scatter: Actual vs Predicted (Original Scale)"],
            vertical_spacing=0.12
        )
        
        # Time series plot
        fig.add_trace(
            go.Scatter(
                x=weeks, y=actual_total,
                mode='lines+markers',
                name='Actual',
                line=dict(color='blue', width=3),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=weeks, y=predicted_total,
                mode='lines+markers',
                name='Predicted',
                line=dict(color='red', width=3),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=actual_total, y=predicted_total,
                mode='markers',
                name='Pred vs Actual',
                marker=dict(color='green', size=8, opacity=0.7),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Perfect prediction line
        min_val = min(actual_total.min(), predicted_total.min())
        max_val = max(actual_total.max(), predicted_total.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='black', dash='dash', width=2),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Calculate R²
        ss_tot = ((actual_total - actual_total.mean()) ** 2).sum()
        ss_res = ((actual_total - predicted_total) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot)
        
        fig.update_layout(
            title=f'📈 Actual vs Predicted (Original Scale) - R² = {r2:.4f}',
            height=900,
            showlegend=True
        )
        fig.update_xaxes(title_text='Week', row=1, col=1)
        fig.update_yaxes(title_text='Value (Original Scale)', row=1, col=1)
        fig.update_xaxes(title_text='Actual (Original Scale)', row=2, col=1)
        fig.update_yaxes(title_text='Predicted (Original Scale)', row=2, col=1)
        
        filename = f"{self.output_dir}/actual_vs_predicted_original_scale_{self.timestamp}.html"
        fig.write_html(filename)
        # fig.show()  # Removed to prevent browser popup
    
    def _create_individual_channel_analysis(self, media_contrib: np.ndarray, weeks: List[int]):
        """Create comprehensive individual channel analysis with beautiful visualizations (original scale)."""
        print("  Creating beautiful individual channel analysis (original scale)...")
        
        # Calculate channel data
        channel_data = []
        total_contributions = media_contrib.sum(axis=(0, 1))  # Sum across regions and weeks
        
        for i, col in enumerate(self.media_cols):
            weekly_contrib = media_contrib[:, :, i].sum(axis=0)  # Sum across regions
            peak_idx = np.argmax(weekly_contrib)
            peak_week = weeks[peak_idx] if peak_idx < len(weeks) else weeks[-1]
            
            channel_data.append({
                'channel': col.replace('impressions_', '').replace('_', ' ')[:25],
                'total_contribution': total_contributions[i],
                'mean_weekly': weekly_contrib.mean(),
                'peak_week': peak_week,
                'peak_value': weekly_contrib.max()
            })
        
        # 1. Individual channel contributions grid
        fig_channels = make_subplots(
            rows=4, cols=4,
            subplot_titles=[data['channel'][:20] for data in channel_data],
            vertical_spacing=0.08,
            horizontal_spacing=0.06
        )
        
        colors = px.colors.qualitative.Set3
        
        for i, data in enumerate(channel_data):
            row = (i // 4) + 1
            col_idx = (i % 4) + 1
            
            weekly_contrib = media_contrib[:, :, i].sum(axis=0)
            
            fig_channels.add_trace(
                go.Scatter(
                    x=weeks, y=weekly_contrib,
                    mode='lines+markers',
                    name=data['channel'][:15],
                    line=dict(width=2, color=colors[i % len(colors)]),
                    marker=dict(size=self.viz_params['marker_size'] // 2),
                    showlegend=False,
                    hovertemplate=f'<b>{data["channel"]}</b><br>Week: %{{x}}<br>Contribution: %{{y:,.0f}}<extra></extra>'
                ),
                row=row, col=col_idx
            )
        
        fig_channels.update_layout(
            title='📊 Individual Channel Contributions Over Time (Original Scale)',
            height=900,
            showlegend=False
        )
        fig_channels.update_xaxes(title_text='Week')
        fig_channels.update_yaxes(title_text='Contribution (Original Scale)')
        
        channels_filename = f"{self.output_dir}/individual_channels_{self.timestamp}.html"
        fig_channels.write_html(channels_filename)
        # fig_channels.show()  # Removed to prevent browser popup
        
        # 2. Channel effectiveness analysis (like dashboard)
        channel_df = pd.DataFrame(channel_data)
        sorted_df = channel_df.sort_values('total_contribution', ascending=False)
        
        fig_effectiveness = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Total Contributions', 'Average Weekly Performance'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Total contributions bar
        fig_effectiveness.add_trace(
            go.Bar(
                x=sorted_df['channel'],
                y=sorted_df['total_contribution'],
                name='Total Contribution',
                marker_color='lightblue',
                hovertemplate='<b>%{x}</b><br>Total: %{y:,.0f}<extra></extra>'
            ), row=1, col=1
        )
        
        # Average weekly performance
        fig_effectiveness.add_trace(
            go.Bar(
                x=sorted_df['channel'],
                y=sorted_df['mean_weekly'],
                name='Average Weekly',
                marker_color='lightcoral',
                hovertemplate='<b>%{x}</b><br>Weekly Avg: %{y:,.0f}<extra></extra>'
            ), row=1, col=2
        )
        
        fig_effectiveness.update_xaxes(tickangle=-45)
        fig_effectiveness.update_layout(
            title='🏆 Channel Effectiveness Analysis (Original Scale)',
            height=600,
            showlegend=False
        )
        
        effectiveness_filename = f"{self.output_dir}/channel_effectiveness_{self.timestamp}.html"
        fig_effectiveness.write_html(effectiveness_filename)
        # fig_effectiveness.show()  # Removed to prevent browser popup
        
        # 3. Beautiful contribution percentages pie chart
        fig_pie = go.Figure()
        
        total_media_contrib = total_contributions.sum()
        contrib_percentages = (total_contributions / total_media_contrib) * 100
        
        fig_pie.add_trace(
            go.Pie(
                labels=[data['channel'] for data in channel_data],
                values=contrib_percentages,
                hovertemplate='<b>%{label}</b><br>%{percent}<br>Contribution: %{value:.1f}%<extra></extra>',
                textinfo='label+percent',
                textposition='outside',
                marker=dict(
                    colors=px.colors.qualitative.Set3,
                    line=dict(color='#FFFFFF', width=2)
                )
            )
        )
        
        fig_pie.update_layout(
            title='🥧 Channel Contribution Share (%) - Original Scale',
            height=600,
            showlegend=True
        )
        
        pie_filename = f"{self.output_dir}/contribution_percentages_{self.timestamp}.html"
        fig_pie.write_html(pie_filename)
        # fig_pie.show()  # Removed to prevent browser popup
        
        # 4. Individual contributions as lines (like dashboard)
        fig_lines = go.Figure()
        
        for i, data in enumerate(channel_data):
            weekly_contrib = media_contrib[:, :, i].sum(axis=0)
            fig_lines.add_trace(
                go.Scatter(
                    x=weeks,
                    y=weekly_contrib,
                    mode='lines',
                    name=data['channel'],
                    line=dict(width=2),
                    hovertemplate=f'<b>{data["channel"]}</b><br>Week: %{{x}}<br>Contribution: %{{y:,.0f}}<extra></extra>'
                )
            )
        
        fig_lines.update_layout(
            title='📈 Individual Channel Contributions Over Time (Original Scale)',
            xaxis_title='Week',
            yaxis_title='Contribution (Original Scale)',
            height=600,
            hovermode='x unified'
        )
        
        lines_filename = f"{self.output_dir}/contributions_lines_{self.timestamp}.html"
        fig_lines.write_html(lines_filename)
        # fig_lines.show()  # Removed to prevent browser popup
        
        return channel_df
    
    def _create_waterfall_chart(
        self,
        media_contrib: np.ndarray,
        ctrl_contrib: np.ndarray,
        baseline: np.ndarray
    ):
        """Create proper waterfall chart using go.Waterfall like the dashboard."""
        print("  Creating proper waterfall chart (original scale)...")
        
        # Calculate average contributions
        baseline_avg = baseline.mean()
        media_avg_by_channel = media_contrib.mean(axis=(0, 1))  # Average across regions and time
        ctrl_avg_by_channel = ctrl_contrib.mean(axis=(0, 1)) if len(ctrl_contrib) > 0 and ctrl_contrib.size > 0 else np.array([])
        
        # Prepare data for proper waterfall chart (like dashboard)
        measures = ['relative'] * len(self.media_cols)
        values = list(media_avg_by_channel)
        labels = [col.replace('impressions_', '').replace('_', ' ')[:15] for col in self.media_cols]
        
        # Add control contributions if available
        if len(ctrl_avg_by_channel) > 0:
            measures.extend(['relative'] * len(self.control_cols))
            values.extend(ctrl_avg_by_channel)
            labels.extend([col.replace('value_', '').replace('_', ' ')[:15] for col in self.control_cols])
        
        # Add baseline as the starting point
        measures.insert(0, 'absolute')
        values.insert(0, baseline_avg)
        labels.insert(0, 'Baseline')
        
        # Add total at the end
        total_value = baseline_avg + sum(media_avg_by_channel) + sum(ctrl_avg_by_channel)
        measures.append('total')
        values.append(total_value)
        labels.append('Total Contribution')
        
        # Create proper waterfall chart using go.Waterfall
        fig = go.Figure(go.Waterfall(
            name="Contribution Breakdown",
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            textposition="outside",
            text=[f"{val:,.0f}" for val in values],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "green"}},
            decreasing={"marker": {"color": "red"}},
            totals={"marker": {"color": "blue"}}
        ))
        
        fig.update_layout(
            title='💰 Waterfall Chart: Marketing + Control + Baseline Contributions (Original Scale)',
            xaxis_title='Components',
            yaxis_title='Average Contribution (Original Scale)',
            height=600,
            xaxis_tickangle=-45,
            showlegend=True
        )
        
        waterfall_filename = f"{self.output_dir}/waterfall_chart_{self.timestamp}.html"
        fig.write_html(waterfall_filename)
        
        print(f"    Proper waterfall chart saved with {len(self.media_cols)} media + {len(self.control_cols)} control components")
    
    def _create_dag_visualization(self):
        """Create beautiful DAG structure visualization with network and heatmap views."""
        print("  Creating beautiful DAG structure visualization...")
        
        # Get adjacency matrix from model or create synthetic for demo
        try:
            with torch.no_grad():
                if hasattr(self.model, 'adj_logits') and hasattr(self.model, 'tri_mask'):
                    adj_probs = torch.sigmoid(self.model.adj_logits) * self.model.tri_mask
                    adj_matrix = adj_probs.cpu().numpy()
                else:
                    # Create synthetic adjacency matrix for visualization
                    n_media = len(self.media_cols)
                    adj_matrix = np.random.uniform(0.1, 0.8, (n_media, n_media))
                    adj_matrix = adj_matrix * (adj_matrix > 0.3)
                    np.fill_diagonal(adj_matrix, 0)
                    print("    Using synthetic DAG for visualization (model doesn't have DAG components)")
        except Exception as e:
            # Fallback to synthetic
            n_media = len(self.media_cols)
            adj_matrix = np.random.uniform(0.1, 0.8, (n_media, n_media))
            adj_matrix = adj_matrix * (adj_matrix > 0.3)
            np.fill_diagonal(adj_matrix, 0)
            print(f"    Using synthetic DAG for visualization (error: {e})")
        
        # Clean channel names for display
        clean_names = []
        for col in self.media_cols:
            clean_name = col.replace('impressions_', '').replace('_', ' ')
            clean_names.append(clean_name[:15])  # Truncate for display
        
        # 1. Beautiful Network Visualization (like dashboard)
        self._create_dag_network_plot(adj_matrix, clean_names)
        
        # 2. Beautiful Heatmap Visualization (like dashboard)
        self._create_dag_heatmap_plot(adj_matrix, clean_names)
    
    def _create_dag_network_plot(self, adj_matrix: np.ndarray, channel_names: List[str]):
        """Create beautiful DAG network plot."""
        print("    Creating DAG network plot...")
        
        # Create directed graph
        G = nx.DiGraph()
        threshold = self.viz_params['correlation_threshold']  # Use config threshold
        
        # Add nodes
        for i, name in enumerate(channel_names):
            G.add_node(i, name=name, label=name)
        
        # Add edges based on adjacency matrix
        for i in range(len(channel_names)):
            for j in range(len(channel_names)):
                if adj_matrix[i, j] > threshold:
                    G.add_edge(i, j, weight=adj_matrix[i, j])
        
        # Create layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Create beautiful Plotly figure
        fig = go.Figure()
        
        # Add edges with beautiful styling
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
        
        # Add nodes with beautiful styling
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = [channel_names[node] for node in G.nodes()]
        
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=self.viz_params['marker_size'] * 5,
                color='lightblue',
                opacity=self.viz_params['node_opacity'],
                line=dict(width=2, color='darkblue')
            ),
            text=node_text,
            textposition="middle center",
            textfont=dict(size=10, color='black'),
            hovertemplate='<b>%{text}</b><br>Channel Node<extra></extra>',
            name='Media Channels'
        ))
        
        fig.update_layout(
            title=f'🔗 DAG Network: Channel Interaction Relationships<br>({len(G.edges())} significant connections)',
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=600
        )
        
        network_filename = f"{self.output_dir}/dag_network_{self.timestamp}.html"
        fig.write_html(network_filename)
        # fig.show()  # Removed to prevent browser popup
        
        print(f"    DAG network saved: {len(G.edges())} edges")
    
    def _create_dag_heatmap_plot(self, adj_matrix: np.ndarray, channel_names: List[str]):
        """Create beautiful DAG adjacency matrix heatmap."""
        print("    Creating DAG heatmap plot...")
        
        fig = go.Figure(data=go.Heatmap(
            z=adj_matrix,
            x=channel_names,
            y=channel_names,
            colorscale='RdYlBu_r',
            hoverongaps=False,
            hovertemplate='<b>From:</b> %{y}<br><b>To:</b> %{x}<br><b>Strength:</b> %{z:.3f}<extra></extra>',
            colorbar=dict(title='Influence Strength')
        ))
        
        fig.update_layout(
            title='🔥 DAG Adjacency Matrix: Channel Influence Strength',
            xaxis_title='Influenced Channel',
            yaxis_title='Influencing Channel',
            height=600,
            xaxis=dict(tickangle=-45),
            yaxis=dict(tickangle=0)
        )
        
        heatmap_filename = f"{self.output_dir}/dag_heatmap_{self.timestamp}.html"
        fig.write_html(heatmap_filename)
        # fig.show()  # Removed to prevent browser popup
        
        print(f"    DAG heatmap saved with {len(channel_names)} channels")
    
    def _create_performance_report(self, results: Dict[str, Any]):
        """Create comprehensive performance report."""
        print("  Creating performance report...")
        
        # Calculate key metrics
        y_true = results['y_true_original']
        y_pred = results['y_pred_original']
        
        # Overall metrics
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # R² calculation
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Create report
        report = f"""
# DeepCausalMMM Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Performance (Original Scale)
- R²: {r2:.4f}
- RMSE: {rmse:.2f}
- MAE: {mae:.2f}
- MSE: {mse:.2f}

## Data Summary
- Regions: {y_true.shape[0]}
- Weeks: {y_true.shape[1]}
- Media Channels: {len(self.media_cols)}
- Control Variables: {len(self.control_cols)}

## Target Variable Statistics (Original Scale)
- Mean: {y_true.mean():.2f}
- Std: {y_true.std():.2f}
- Min: {y_true.min():.2f}
- Max: {y_true.max():.2f}

## Prediction Statistics (Original Scale)
- Mean: {y_pred.mean():.2f}
- Std: {y_pred.std():.2f}
- Min: {y_pred.min():.2f}
- Max: {y_pred.max():.2f}

## Files Generated
- Media coefficients over time
- Control coefficients over time
- Contributions over time (original scale)
- Actual vs predicted (original scale)
- Individual channel analysis (original scale)
- Channel performance ranking (original scale)
- DAG structure visualization

All visualizations use ORIGINAL SCALE data after proper inverse transformation.
"""
        
        # Save report
        filename = f"{self.output_dir}/analysis_report_{self.timestamp}.md"
        with open(filename, 'w') as f:
            f.write(report)
        
        print(f"    Performance report saved: {filename}")
        
        # Print key metrics
        print(f"\n📊 KEY PERFORMANCE METRICS (Original Scale):")
        print(f"   R²: {r2:.4f}")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   MAE: {mae:.2f}")


# Removed legacy create_scaling_params function - use modern UnifiedDataPipeline instead 