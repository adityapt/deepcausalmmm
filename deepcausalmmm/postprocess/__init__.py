"""
Post-processing utilities for DeepCausalMMM analysis and visualization.
"""

from deepcausalmmm.postprocess.comprehensive_analysis import ComprehensiveAnalyzer
from deepcausalmmm.postprocess.analysis import ModelAnalyzer
from deepcausalmmm.postprocess.response_curves import ResponseCurveFit, ResponseCurveFitter
from deepcausalmmm.postprocess.optimization import (
    BudgetOptimizer,
    OptimizationResult,
    optimize_budget_from_curves
)
from deepcausalmmm.postprocess.optimization_utils import (
    prepare_optimization_data,
    fit_response_curves_batch,
    create_optimizer_from_model_output,
    compare_current_vs_optimal,
    generate_optimization_report
)

# Unified pipeline integration
def create_unified_analyzer(
    model,
    pipeline,
    media_cols: list,
    control_cols: list,
    output_dir: str = "unified_analysis_results"
) -> ComprehensiveAnalyzer:
    """
    Create a ComprehensiveAnalyzer configured for unified pipeline.
    
    Args:
        model: Trained DeepCausalMMM model
        pipeline: UnifiedDataPipeline instance
        media_cols: Media column names
        control_cols: Control column names
        output_dir: Output directory
        
    Returns:
        Configured ComprehensiveAnalyzer
    """
    return ComprehensiveAnalyzer(
        model=model,
        media_cols=media_cols,
        control_cols=control_cols,
        output_dir=output_dir,
        pipeline=pipeline,
        auto_detect_burnin=False  # Use pipeline's burn-in
    )

__all__ = [
    'ComprehensiveAnalyzer',
    'ModelAnalyzer',
    'ResponseCurveFit',
    'ResponseCurveFitter',  # Backward compatibility alias
    'create_unified_analyzer',
    # Budget Optimization
    'BudgetOptimizer',
    'OptimizationResult',
    'optimize_budget_from_curves',
    'prepare_optimization_data',
    'fit_response_curves_batch',
    'create_optimizer_from_model_output',
    'compare_current_vs_optimal',
    'generate_optimization_report'
]
