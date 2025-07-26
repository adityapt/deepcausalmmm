"""
Core functionality for DeepCausalMMM package.

This module contains the main components:
- Model definitions (GRU + Bayesian Networks)
- Data preprocessing and loading
- Training loops and optimization
- Inference and forecasting
"""

from .model import GRUCausalMMM, CausalEncoder
from .data import prepare_data_for_training, create_belief_vectors, create_media_adjacency
from .train import train_model, train_model_with_validation
from .infer import predict, forecast, get_feature_importance, get_contributions

__all__ = [
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
    "get_contributions",
] 