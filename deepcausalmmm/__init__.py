"""
DeepCausalMMM: Deep Learning Marketing Mix Modeling with Causal Structure
========================================================================

A PyTorch-based implementation of Marketing Mix Modeling that incorporates:
- GRU-based time-varying coefficients with advanced stabilization
- DAG (Directed Acyclic Graph) structure for causal relationships
  (upper-triangular mask by default; opt-in NOTEARS continuous learning)
- Channel interaction modeling
- Regional scaling and analysis

Main Components:
- DeepCausalMMM: Core model class
- ComprehensiveAnalyzer: Advanced post-processing and visualization
- Configuration system for reproducible experiments
"""

import logging
import sys

# Configure package-level logging
def _setup_logging():
    """Configure default logging for DeepCausalMMM."""
    logger = logging.getLogger('deepcausalmmm')
    logger.setLevel(logging.INFO)
    
    # Only add handler if none exists (avoid duplicates)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False  # Don't propagate to root logger

_setup_logging()

try:
    # Prefer local project metadata when imported directly from a checkout.
    # This keeps the CLI and Sphinx docs aligned with a pending release before
    # an editable install is refreshed. Installed wheels have no adjacent
    # pyproject.toml and therefore use distribution metadata below.
    from pathlib import Path

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    if pyproject_path.exists():
        for line in pyproject_path.read_text().splitlines():
            if line.strip().startswith('version = '):
                __version__ = line.split('=', 1)[1].strip().strip('"\'')
                break
        else:
            raise RuntimeError("project version is missing")
    else:
        from importlib.metadata import version

        __version__ = version("deepcausalmmm")
except Exception:
    __version__ = "unknown"

# Core model (essential)
from deepcausalmmm.core.unified_model import DeepCausalMMM
from deepcausalmmm.core.config import get_default_config, update_config

# Post-processing (essential)
from deepcausalmmm.postprocess import (
    ComprehensiveAnalyzer,
    ResponseCurveFit,
    ResponseCurveFitter,
    BudgetOptimizer,
    OptimizationResult,
    optimize_budget_from_curves
)
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
    'ResponseCurveFit',
    'ResponseCurveFitter',  # Backward compatibility
    
    # Budget Optimization
    'BudgetOptimizer',
    'OptimizationResult',
    'optimize_budget_from_curves',
    
    # Scaling
    'SimpleGlobalScaler',
    'GlobalScaler',
    
    # Utilities
    'get_device',
]
