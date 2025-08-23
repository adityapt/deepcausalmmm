"""
Custom exceptions for DeepCausalMMM package.
"""

class DeepCausalMMMError(Exception):
    """Base exception for DeepCausalMMM package."""
    pass


class DataError(DeepCausalMMMError):
    """Raised when there are issues with data loading or preprocessing."""
    pass


class ModelError(DeepCausalMMMError):
    """Raised when there are issues with model initialization or training."""
    pass


class ConfigurationError(DeepCausalMMMError):
    """Raised when there are issues with configuration parameters."""
    pass


class ValidationError(DeepCausalMMMError):
    """Raised when data validation fails."""
    pass


class TrainingError(DeepCausalMMMError):
    """Raised when model training fails."""
    pass


class InferenceError(DeepCausalMMMError):
    """Raised when model inference/prediction fails."""
    pass


class BayesianNetworkError(DeepCausalMMMError):
    """Raised when Bayesian network operations fail."""
    pass


class AdstockError(DeepCausalMMMError):
    """Raised when adstock transformation fails."""
    pass


class SaturationError(DeepCausalMMMError):
    """Raised when saturation transformation fails."""
    pass


class ScalingError(DeepCausalMMMError):
    """Raised when data scaling operations fail."""
    pass


class MissingDependencyError(DeepCausalMMMError):
    """Raised when required dependencies are not available."""
    pass 