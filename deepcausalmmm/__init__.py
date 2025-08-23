"""
DeepCausalMMM: Deep Learning Marketing Mix Modeling with Causal Structure
========================================================================

A PyTorch-based implementation of Marketing Mix Modeling that incorporates:
- GRU-based time-varying coefficients with advanced stabilization
- DAG (Directed Acyclic Graph) structure for causal relationships
- Channel interaction modeling
- Regional scaling and analysis

Main Components:
- DeepCausalMMM: Core model class
- ComprehensiveAnalyzer: Advanced post-processing and visualization
- Configuration system for reproducible experiments
"""

__version__ = "1.0.0"

# Core model (essential)
from deepcausalmmm.core.unified_model import DeepCausalMMM
from deepcausalmmm.core.config import get_default_config, update_config

# Post-processing (essential)
from deepcausalmmm.postprocess import ComprehensiveAnalyzer
# Scaling (essential)
from deepcausalmmm.core.scaling import SimpleGlobalScaler, GlobalScaler

# Utilities (essential)
from deepcausalmmm.utils.device import get_device

__all__ = [
    # Core model
    'DeepCausalMMM',
    
    # Configuration
    'get_default_config',
    'update_config',
    
    # Analysis and visualization
    'ComprehensiveAnalyzer',
    
    
    # Scaling
    'SimpleGlobalScaler',
    'GlobalScaler',
    
    # Utilities
    'get_device',
]
