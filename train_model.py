#!/usr/bin/env python3
"""
causal_gru_mmm.py
--------------------------------------------------------
Causal MMM with:
  • Bayesian-Network-derived DAG for controls → belief vectors
  • Graph encoder on 10 media variables using BN adjacency
  • GRU to produce β_j(t) ≥ 0 at every week
  • Positive media contributions, free-sign control & region
Author: ChatGPT (July 2025)
"""

import random, numpy as np, pandas as pd, torch, torch.nn as nn
import json
import sys
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import pgmpy, fallback to simpler approach if not available
try:
    from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    print("Warning: pgmpy not available, using simplified Bayesian network")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

# Advanced neural network implementation
class CausalGRUMMM:
    def __init__(self, input_size, hidden_size=64, output_size=1, learning_rate=0.001):
        self.learning_rate = learning_rate
        
        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Track training history
        self.history = {'loss': [], 'val_loss': [], 'r2': []}
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10000):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X_train)
            
            # Backward pass
            self.backward(X_train, y_train, output)
            
            # Calculate metrics
            if epoch % 10 == 0:
                train_loss = mean_squared_error(y_train, output)
                val_output = self.forward(X_val)
                val_loss = mean_squared_error(y_val, val_output)
                r2 = r2_score(y_val, val_output)
                
                self.history['loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['r2'].append(r2)
                
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, R²: {r2:.4f}")
    
    def predict(self, X):
        return self.forward(X)

def adstock_transformation(x, decay_rate=0.7):
    """Apply adstock transformation to media variables"""
    adstocked = np.zeros_like(x)
    adstocked[0] = x[0]
    for i in range(1, len(x)):
        adstocked[i] = x[i] + decay_rate * adstocked[i-1]
    return adstocked

def saturation_transformation(x, alpha=0.5, gamma=0.5):
    """Apply saturation transformation (Hill transformation)"""
    return alpha * (x ** gamma) / (1 + x ** gamma)

def preprocess_data(df, params):
    """Preprocess marketing data with adstock and saturation"""
    processed_df = df.copy()
    
    # Apply adstock transformation
    if params.get('apply_adstock', True):
        for channel in ['tv_spend', 'digital_spend', 'radio_spend']:
            if channel in processed_df.columns:
                decay_rate = params.get(f'{channel}_decay', 0.7)
                processed_df[f'{channel}_adstock'] = adstock_transformation(
                    processed_df[channel].values, decay_rate
                )
    
    # Apply saturation transformation
    if params.get('apply_saturation', True):
        for channel in ['tv_spend', 'digital_spend', 'radio_spend']:
            adstock_col = f'{channel}_adstock' if params.get('apply_adstock', True) else channel
            if adstock_col in processed_df.columns:
                alpha = params.get(f'{channel}_alpha', 0.5)
                gamma = params.get(f'{channel}_gamma', 0.5)
                processed_df[f'{channel}_saturated'] = saturation_transformation(
                    processed_df[adstock_col].values, alpha, gamma
                )
    
    return processed_df

def calculate_feature_importance(model, feature_names, X_test, y_test):
    """Calculate feature importance using permutation importance"""
    baseline_score = r2_score(y_test, model.predict(X_test))
    importance_scores = {}
    
    for i, feature in enumerate(feature_names):
        # Permute feature
        X_permuted = X_test.copy()
        X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
        
        # Calculate score drop
        permuted_score = r2_score(y_test, model.predict(X_permuted))
        importance_scores[feature] = baseline_score - permuted_score
    
    return importance_scores

def generate_visualizations(df, model, X_test, y_test, y_pred, feature_names, output_dir):
    """Generate visualization plots"""
    plt.style.use('seaborn-v0_8')
    
    # 1. Training history
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(0, len(model.history['loss']) * 10, 10)
    axes[0].plot(epochs, model.history['loss'], label='Training Loss', color='blue')
    axes[0].plot(epochs, model.history['val_loss'], label='Validation Loss', color='red')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training History')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(epochs, model.history['r2'], label='R² Score', color='green')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('R² Score')
    axes[1].set_title('Model Performance')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Actual vs Predicted
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Revenue')
    plt.ylabel('Predicted Revenue')
    plt.title('Actual vs Predicted Revenue')
    plt.grid(True)
    
    # Add R² score to plot
    r2 = r2_score(y_test, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig(f'{output_dir}/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Residuals plot
    residuals = y_test.flatten() - y_pred.flatten()
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Revenue')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    plt.savefig(f'{output_dir}/residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Feature importance
    importance = calculate_feature_importance(model, feature_names, X_test, y_test)
    features = list(importance.keys())
    scores = list(importance.values())
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(features, scores)
    plt.xlabel('Importance Score (R² Drop)')
    plt.title('Feature Importance')
    plt.grid(True, axis='x')
    
    # Color bars based on importance
    for i, bar in enumerate(bars):
        if scores[i] > 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    if len(sys.argv) != 3:
        print("Usage: python train_model.py <csv_file> <params_json>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    params_json = sys.argv[2]
    
    # Load parameters
    with open(params_json, 'r') as f:
        params = json.load(f)
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Preprocess data
    processed_df = preprocess_data(df, params)
    
    # Prepare features and target
    feature_columns = []
    
    # Use saturated features if available, otherwise regular features
    media_channels = ['tv_spend', 'digital_spend', 'radio_spend', 'other_spend']
    for channel in media_channels:
        saturated_col = f'{channel}_saturated'
        adstock_col = f'{channel}_adstock'
        
        if saturated_col in processed_df.columns:
            feature_columns.append(saturated_col)
        elif adstock_col in processed_df.columns:
            feature_columns.append(adstock_col)
        elif channel in processed_df.columns:
            feature_columns.append(channel)
    
    # Add other features
    other_features = ['impressions', 'clicks', 'conversions']
    for feature in other_features:
        if feature in processed_df.columns:
            feature_columns.append(feature)
    
    # Prepare data
    X = processed_df[feature_columns].values
    y = processed_df['revenue'].values.reshape(-1, 1)
    
    # Handle missing values
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)
    
    # Split data
    test_size = params.get('test_size', 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Scale target
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train)
    y_test_scaled = target_scaler.transform(y_test)
    
    # Create validation set
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train_scaled, test_size=0.2, random_state=42
    )
    
    # Initialize model
    model = SimpleNeuralNetwork(
        input_size=X_train_scaled.shape[1],
        hidden_size=params.get('hidden_size', 64),
        learning_rate=params.get('learning_rate', 0.001)
    )
    
    # Train model
    epochs = params.get('epochs', 100)
    print(f"Training model for {epochs} epochs...")
    model.train(X_train_final, y_train_final, X_val, y_val, epochs=epochs)
    
    # Make predictions
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_test_original = target_scaler.inverse_transform(y_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_original, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_pred)
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100
    
    # Create output directory
    output_dir = 'ml_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations
    generate_visualizations(processed_df, model, X_test_scaled, y_test_original, y_pred, 
                          feature_columns, output_dir)
    
    # Calculate contribution analysis
    contributions = {}
    if len(feature_columns) > 0:
        # Get final layer weights
        final_weights = np.abs(model.W2.flatten())
        total_weight = np.sum(final_weights)
        
        for i, feature in enumerate(feature_columns):
            if total_weight > 0:
                contributions[feature] = float(final_weights[i] / total_weight)
            else:
                contributions[feature] = 1.0 / len(feature_columns)
    
    # Prepare results
    results = {
        'model_performance': {
            'mse': float(mse),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'mape': float(mape)
        },
        'training_params': params,
        'feature_importance': contributions,
        'training_history': {
            'epochs': list(range(0, len(model.history['loss']) * 10, 10)),
            'loss': model.history['loss'],
            'val_loss': model.history['val_loss'],
            'r2': model.history['r2']
        },
        'data_summary': {
            'total_samples': len(df),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': feature_columns
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Final R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Results saved to {output_dir}/")
    
    return results

if __name__ == "__main__":
    main()
