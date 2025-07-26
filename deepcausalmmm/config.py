"""
Configuration and default parameters for DeepCausalMMM.
"""

from typing import Dict, Any, List

# Default configuration for the model
DEFAULT_CONFIG = {
    # Model architecture
    "hidden_size": 64,
    "learning_rate": 1e-3,
    "epochs": 10000,
    "batch_size": 32,
    "dropout": 0.1,
    
    # Data preprocessing
    "burn_in_weeks": 4,
    "test_size": 0.2,
    "validation_size": 0.2,
    "random_state": 42,
    
    # Adstock parameters
    "adstock_decay_range": (0.1, 0.9),
    "adstock_init_value": 0.5,
    
    # Hill saturation parameters
    "hill_alpha_range": (0.6, 1.4),
    "hill_gamma_range": (0.1, 0.4),
    
    # Bayesian Network
    "use_bayesian_network": True,
    "bn_score_method": "bic",
    "bn_algorithm": "hill_climbing",
    
    # Training
    "early_stopping_patience": 50,
    "learning_rate_scheduler": "reduce_on_plateau",
    "weight_decay": 1e-5,
    "gradient_clipping": 1.0,
    
    # Feature engineering
    "apply_adstock": True,
    "apply_saturation": True,
    "apply_log_transform": False,
    "apply_lag_features": False,
    "lag_periods": [1, 2, 4],
    
    # Output and logging
    "save_model": True,
    "save_predictions": True,
    "plot_results": True,
    "verbose": True,
    "log_level": "INFO",
}

# Model hyperparameter search space
HYPERPARAMETER_SPACE = {
    "hidden_size": [32, 64, 128, 256],
    "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "adstock_init_value": [0.3, 0.5, 0.7],
    "hill_alpha_range": [(0.5, 1.0), (0.6, 1.4), (0.8, 1.8)],
    "hill_gamma_range": [(0.05, 0.2), (0.1, 0.4), (0.2, 0.6)],
}

# Default column mappings
DEFAULT_COLUMN_MAPPINGS = {
    "date_column": "date",
    "region_column": "region", 
    "target_column": "revenue",
    "media_columns": [
        "tv_spend", "digital_spend", "radio_spend", "print_spend",
        "outdoor_spend", "social_spend", "search_spend", "display_spend",
        "video_spend", "audio_spend"
    ],
    "control_columns": [
        "price", "promotion", "seasonality", "competition",
        "economic_indicator", "weather", "events"
    ]
}

# Validation schemas
REQUIRED_COLUMNS = ["date", "revenue"]
OPTIONAL_COLUMNS = ["region", "price", "promotion"]

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated configuration dictionary
    """
    validated_config = DEFAULT_CONFIG.copy()
    validated_config.update(config)
    
    # Validate ranges
    if not (0 < validated_config["learning_rate"] < 1):
        raise ValueError("learning_rate must be between 0 and 1")
    
    if not (0 <= validated_config["dropout"] < 1):
        raise ValueError("dropout must be between 0 and 1")
    
    if validated_config["hidden_size"] <= 0:
        raise ValueError("hidden_size must be positive")
    
    if validated_config["epochs"] <= 0:
        raise ValueError("epochs must be positive")
    
    return validated_config

def get_model_config(model_type: str = "gru_causal") -> Dict[str, Any]:
    """
    Get configuration for specific model type.
    
    Args:
        model_type: Type of model ("gru_causal", "simple_nn", etc.)
        
    Returns:
        Model-specific configuration
    """
    if model_type == "gru_causal":
        return DEFAULT_CONFIG
    elif model_type == "simple_nn":
        simple_config = DEFAULT_CONFIG.copy()
        simple_config.update({
            "hidden_size": 32,
            "epochs": 1000,
            "use_bayesian_network": False,
        })
        return simple_config
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    import json
    import yaml
    import os
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            config = json.load(f)
        elif config_path.endswith(('.yaml', '.yml')):
            config = yaml.safe_load(f)
        else:
            raise ValueError("Configuration file must be JSON or YAML")
    
    return validate_config(config) 