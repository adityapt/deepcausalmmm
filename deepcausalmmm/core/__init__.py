"""
Core components of DeepCausalMMM.
"""

from deepcausalmmm.core.unified_model import DeepCausalMMM
from deepcausalmmm.core.config import get_default_config, update_config
from deepcausalmmm.core.trainer import ModelTrainer
from deepcausalmmm.core.inference import InferenceManager
from deepcausalmmm.core.visualization import VisualizationManager
from deepcausalmmm.core.data import UnifiedDataPipeline
from deepcausalmmm.core.scaling import SimpleGlobalScaler, GlobalScaler
from deepcausalmmm.core.dag_model import NodeToEdge, EdgeToNode, DAGConstraint

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
    from deepcausalmmm.core.train_model import train_mmm as _train_mmm
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
