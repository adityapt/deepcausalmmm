"""
Training utilities for DeepCausalMMM.

This module handles:
- Model training loops
- Optimization and learning rate scheduling
- Early stopping and validation
- Training history tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import json
import os
from datetime import datetime

from ..exceptions import TrainingError, ModelError
from .model import GRUCausalMMM


def train_model(
    model: GRUCausalMMM,
    train_data: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    val_data: Optional[Dict[str, torch.Tensor]] = None
) -> Dict[str, Any]:
    """
    Train the GRU Causal MMM model.
    
    Args:
        model: The model to train
        train_data: Dictionary containing training data (X_m, X_c, y, R)
        config: Training configuration
        val_data: Optional validation data
        
    Returns:
        Dictionary containing training results and history
    """
    # Extract training data
    X_m_train = train_data['X_m']
    X_c_train = train_data['X_c']
    y_train = train_data['y']
    R_train = train_data['R']
    
    # Training parameters
    learning_rate = config.get('learning_rate', 1e-3)
    epochs = config.get('epochs', 10000)
    weight_decay = config.get('weight_decay', 1e-5)
    gradient_clipping = config.get('gradient_clipping', 1.0)
    early_stopping_patience = config.get('early_stopping_patience', 50)
    verbose = config.get('verbose', True)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=verbose)
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    if verbose:
        print(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        
        # Forward pass
        y_pred, _, _ = model(X_m_train, X_c_train, R_train)
        loss = criterion(y_pred, y_train)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Validation
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                X_m_val = val_data['X_m']
                X_c_val = val_data['X_c']
                y_val = val_data['y']
                R_val = val_data['R']
                
                y_val_pred, _, _ = model(X_m_val, X_c_val, R_val)
                val_loss = criterion(y_val_pred, y_val)
                val_losses.append(val_loss.item())
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    # Restore best model
                    model.load_state_dict(best_model_state)
                    break
        
        # Progress reporting
        if verbose and epoch % 100 == 0:
            if val_data is not None:
                print(f"Epoch {epoch:4d}: Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
            else:
                print(f"Epoch {epoch:4d}: Train Loss: {loss.item():.6f}")
    
    # Prepare results
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses if val_data is not None else None,
        'best_val_loss': best_val_loss if val_data is not None else None,
        'final_train_loss': train_losses[-1],
        'epochs_trained': len(train_losses),
        'early_stopped': patience_counter >= early_stopping_patience if val_data is not None else False,
    }
    
    return results


def train_model_with_validation(
    model: GRUCausalMMM,
    data_dict: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Train model with automatic train/validation split.
    
    Args:
        model: The model to train
        data_dict: Dictionary containing all data and scalers
        config: Training configuration
        
    Returns:
        Dictionary containing training results
    """
    # Extract data
    X_m = data_dict['X_m']
    X_c = data_dict['X_c']
    y = data_dict['y']
    R = data_dict['R']
    
    # Split data
    test_size = config.get('test_size', 0.2)
    val_size = config.get('validation_size', 0.2)
    
    # Calculate split indices
    n_regions, n_time_steps, _ = X_m.shape
    train_cut = int(n_time_steps * (1 - test_size - val_size))
    val_cut = int(n_time_steps * (1 - test_size))
    
    # Split data
    train_data = {
        'X_m': X_m[:, :train_cut, :],
        'X_c': X_c[:, :train_cut, :],
        'y': y[:, :train_cut],
        'R': R
    }
    
    val_data = {
        'X_m': X_m[:, train_cut:val_cut, :],
        'X_c': X_c[:, train_cut:val_cut, :],
        'y': y[:, train_cut:val_cut],
        'R': R
    }
    
    test_data = {
        'X_m': X_m[:, val_cut:, :],
        'X_c': X_c[:, val_cut:, :],
        'y': y[:, val_cut:],
        'R': R
    }
    
    # Train model
    training_results = train_model(model, train_data, config, val_data)
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        y_test_pred, w_test, contrib_test = model(
            test_data['X_m'], 
            test_data['X_c'], 
            test_data['R']
        )
        
        # Calculate test metrics
        test_loss = nn.MSELoss()(y_test_pred, test_data['y'])
        
        # Inverse transform predictions for metrics
        y_scaler = data_dict['y_scaler']
        y_test_pred_original = y_scaler.inverse_transform(
            y_test_pred.cpu().numpy().reshape(-1, 1)
        ).reshape(test_data['y'].shape)
        y_test_original = y_scaler.inverse_transform(
            test_data['y'].cpu().numpy().reshape(-1, 1)
        ).reshape(test_data['y'].shape)
        
        # Calculate additional metrics
        from ..utils.metrics import calculate_metrics
        test_metrics = calculate_metrics(y_test_original.flatten(), y_test_pred_original.flatten())
    
    # Combine results
    results = {
        **training_results,
        'test_loss': test_loss.item(),
        'test_metrics': test_metrics,
        'test_predictions': y_test_pred_original,
        'test_actual': y_test_original,
        'test_coefficients': w_test,
        'test_contributions': contrib_test,
        'data_splits': {
            'train_cut': train_cut,
            'val_cut': val_cut,
            'n_time_steps': n_time_steps
        }
    }
    
    return results


def save_training_results(
    results: Dict[str, Any],
    model: GRUCausalMMM,
    config: Dict[str, Any],
    output_dir: str = "ml_output"
) -> str:
    """
    Save training results, model, and visualizations.
    
    Args:
        results: Training results dictionary
        model: Trained model
        config: Configuration used for training
        output_dir: Directory to save results
        
    Returns:
        Path to output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    if config.get('save_model', True):
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'results': results
        }, os.path.join(output_dir, 'model.pth'))
    
    # Save results as JSON
    if config.get('save_predictions', True):
        # Convert tensors to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                json_results[key] = value.cpu().numpy().tolist()
            elif isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        json_results[key][k] = v.cpu().numpy().tolist()
                    else:
                        json_results[key][k] = v
            else:
                json_results[key] = value
        
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(json_results, f, indent=2)
    
    # Generate plots
    if config.get('plot_results', True):
        from ..utils.metrics import plot_results
        plot_results(results, output_dir)
    
    print(f"Results saved to {output_dir}/")
    return output_dir


def load_trained_model(model_path: str) -> Tuple[GRUCausalMMM, Dict[str, Any], Dict[str, Any]]:
    """
    Load a trained model from file.
    
    Args:
        model_path: Path to saved model file
        
    Returns:
        Tuple of (model, config, results)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Reconstruct model
    config = checkpoint['config']
    A_media = torch.tensor(config.get('media_adjacency', np.zeros((10, 10))))
    n_media = config.get('n_media', 10)
    ctrl_dim = config.get('ctrl_dim', 15)
    hidden_size = config.get('hidden_size', 64)
    n_regions = config.get('n_regions', 2)
    
    model = GRUCausalMMM(
        A_media, 
        n_media=n_media, 
        ctrl_dim=ctrl_dim, 
        hidden=hidden_size, 
        n_regions=n_regions
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config, checkpoint['results'] 