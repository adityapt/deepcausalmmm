"""
Core components of DeepCausalMMM.
"""

from .unified_model import DeepCausalMMM
from .config import get_default_config, update_config
from .trainer import ModelTrainer
from .inference import InferenceManager
from .visualization import VisualizationManager
from .data import UnifiedDataPipeline
from .scaling import SimpleGlobalScaler, GlobalScaler
from .dag_model import NodeToEdge, EdgeToNode, DAGConstraint

# Deprecated imports with warnings
import warnings

def train_mmm(*args, **kwargs):
    """
    .. deprecated:: 1.0.0
        train_mmm() is deprecated. Use ModelTrainer class instead.
    """
    warnings.warn(
        "train_mmm() is deprecated and will be removed in v2.0.0. "
        "Please use ModelTrainer class instead.",
        DeprecationWarning,
        stacklevel=2
    )
    from .train_model import train_mmm as _train_mmm
    return _train_mmm(*args, **kwargs)

__all__ = [
    # Core model
    'DeepCausalMMM',
    
    # Configuration
    'get_default_config',
    'update_config',
    
    # Modern classes (recommended)
    'ModelTrainer',
    'InferenceManager', 
    'VisualizationManager',
    'UnifiedDataPipeline',
    
    # Scaling
    'SimpleGlobalScaler',
    'GlobalScaler',
    
    # DAG components
    'NodeToEdge',
    'EdgeToNode',
    'DAGConstraint',
    
    # Deprecated (backward compatibility)
    'train_mmm',  # Use ModelTrainer instead
]
