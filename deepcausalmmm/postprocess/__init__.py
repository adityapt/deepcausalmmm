"""
Post-processing utilities for DeepCausalMMM analysis and visualization.
"""

from deepcausalmmm.postprocess.comprehensive_analysis import ComprehensiveAnalyzer
from deepcausalmmm.postprocess.analysis import ModelAnalyzer

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
    'create_unified_analyzer'
]
