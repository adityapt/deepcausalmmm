"""
DeepCausalMMM: Deep Learning + Bayesian Networks based causal Marketing Mix Modeling

A novel approach combining GRU-based time-series modeling with Bayesian networks
for causal effect estimation in marketing mix modeling.
"""

from .version import __version__
from .core.model import GRUCausalMMM, CausalEncoder
from .core.data import prepare_data_for_training, create_belief_vectors, create_media_adjacency
from .core.train import train_model
try:
    from .core.train import train_model_with_validation
except ImportError:
    train_model_with_validation = None
try:
    from .core.infer import predict, forecast, get_feature_importance
except ImportError:
    predict = forecast = get_feature_importance = None
try:
    from .utils.metrics import calculate_metrics, plot_results
except ImportError:
    calculate_metrics = plot_results = None
from .config import DEFAULT_CONFIG

__all__ = [
    "__version__",
    "GRUCausalMMM",
    "CausalEncoder", 
    "prepare_data_for_training",
    "create_belief_vectors",
    "create_media_adjacency",
    "train_model",
    "train_model_with_validation",
    "predict",
    "forecast",
    "get_feature_importance",
    "calculate_metrics",
    "plot_results",
    "DEFAULT_CONFIG",
] 
