"""
Reusable ModelTrainer class for DeepCausalMMM training.
Eliminates code duplication and provides consistent training interface.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, LambdaLR
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score

from deepcausalmmm.core.unified_model import DeepCausalMMM
from deepcausalmmm.core.config import get_default_config
from deepcausalmmm.utils.device import get_device


class ModelTrainer:
    """
    Reusable trainer class for DeepCausalMMM models.
    
    This class provides a complete training pipeline for DeepCausalMMM models with
    advanced features including early stopping, learning rate scheduling, gradient
    clipping, and comprehensive logging. It supports both MSE and Huber loss functions
    with automatic device detection and mixed precision training.
    
    Features:
    - Config-driven model initialization (zero hardcoding)
    - Automatic device detection (CPU/CUDA)
    - Multiple loss functions (MSE, Huber, optional Focal)
    - Early stopping with patience
    - Learning rate scheduling (StepLR, Cosine Annealing)
    - Gradient clipping (global and parameter-specific)
    - Comprehensive metrics tracking (RMSE, R², MAE)
    - Progress bars with detailed statistics
    - Holdout evaluation during training
    
    Parameters
    ----------
    config : Dict[str, Any], optional
        Configuration dictionary containing all training parameters.
        If None, uses default configuration from get_default_config().
        
    Attributes
    ----------
    model : DeepCausalMMM
        The initialized model instance
    optimizer : torch.optim.Optimizer
        The optimizer (Adam by default)
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler if enabled
    device : torch.device
        Training device (CPU or CUDA)
    best_rmse : float
        Best holdout RMSE achieved during training
    train_losses : List[float]
        Training loss history
    train_rmses : List[float]
        Training RMSE history
    train_r2s : List[float]
        Training R² history
        
    Examples
    --------
    >>> from deepcausalmmm.core.trainer import ModelTrainer
    >>> from deepcausalmmm.core.config import get_default_config
    >>> 
    >>> # Initialize trainer with custom config
    >>> config = get_default_config()
    >>> config['n_epochs'] = 1000
    >>> config['learning_rate'] = 0.01
    >>> trainer = ModelTrainer(config)
    >>> 
    >>> # Train model (assumes processed_data is available)
    >>> model, results = trainer.train(processed_data)
    >>> 
    >>> # Access training history
    >>> print(f"Final RMSE: {results['holdout_rmse']:.0f}")
    >>> print(f"Final R²: {results['holdout_r2']:.3f}")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the trainer with configuration.
        
        Args:
            config: Configuration dictionary. If None, uses default config.
        """
        self.config = config or get_default_config()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Revert to original device handling
        
        # Note: Seeds should be set by main script for global reproducibility
        # Model creation will use current RNG state
        
        # Training state
        self.best_rmse = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.train_rmses = []
        self.train_r2s = []
        
    def create_model(self, n_media: int, n_control: int, n_regions: int) -> DeepCausalMMM:
        """
        Create and initialize model from config with reproducible initialization.
        
        Args:
            n_media: Number of media channels
            n_control: Number of control variables
            n_regions: Number of regions
            
        Returns:
            Initialized DeepCausalMMM model
        """
        self.model = DeepCausalMMM(
            n_media=n_media,
            ctrl_dim=n_control,
            n_regions=n_regions,
            hidden=self.config.get('hidden_dim', 64),
            dropout=self.config.get('dropout', 0.1),
            l1_weight=self.config.get('l1_weight', 0.001),
            l2_weight=self.config.get('l2_weight', 0.001),
            burn_in_weeks=self.config.get('burn_in_weeks', 4),
            momentum_decay=self.config.get('momentum_decay', 0.9),
            warm_start_epochs=self.config.get('warm_start_epochs', 50),
            enable_dag=self.config.get('enable_dag', True),
            enable_interactions=self.config.get('enable_interactions', True),
            # COEFFICIENT REGULARIZATION: Pass parameters to prevent explosion
            coeff_l2_weight=self.config.get('coeff_l2_weight', 0.1),
            coeff_gen_l2_weight=self.config.get('coeff_gen_l2_weight', 0.05)
        ).to(self.device)
        
        return self.model
        
    def create_optimizer_and_scheduler(self):
        """Create optimizer and learning rate scheduler from config."""
        if self.model is None:
            raise ValueError("Model must be created before optimizer")
            
        # Get optimizer config
        opt_config = self.config.get('optimizer', {})
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            betas=opt_config.get('betas', (0.9, 0.999)),
            eps=opt_config.get('eps', 1e-8),
            weight_decay=opt_config.get('weight_decay', 1e-5)
        )
        
        # Create advanced scheduler based on config
        if self.config.get('use_cosine_annealing', False):
            # Cosine Annealing with Warm Restarts for better convergence
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.get('cosine_t_initial', 500),
                T_mult=int(self.config.get('cosine_t_mult', 1.2)),
                eta_min=self.config.get('cosine_eta_min', 1e-6)
            )
            
            # Add warmup scheduler if specified
            warmup_epochs = self.config.get('warmup_epochs', 0)
            if warmup_epochs > 0:
                def lr_lambda(epoch):
                    if epoch < warmup_epochs:
                        return epoch / warmup_epochs
                    return 1.0
                
                self.warmup_scheduler = LambdaLR(self.optimizer, lr_lambda)
            else:
                self.warmup_scheduler = None
        else:
            # Default ReduceLROnPlateau scheduler
            scheduler_config = self.config.get('scheduler', {})
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=scheduler_config.get('patience', 300),
                factor=scheduler_config.get('factor', 0.8),
                min_lr=scheduler_config.get('min_lr', 1e-8)
            )
            self.warmup_scheduler = None
        
    def warm_start_training(self, X_media: torch.Tensor, X_control: torch.Tensor, 
                           R: torch.Tensor, y: torch.Tensor, verbose: bool = True) -> None:
        """
        Perform warm-start training to stabilize GRU coefficients.
        
        Args:
            X_media: Media data tensor
            X_control: Control data tensor
            R: Region tensor
            y: Target tensor
            verbose: Whether to show progress
        """
        if self.model is None or self.optimizer is None:
            raise ValueError("Model and optimizer must be created before training")
            
        warm_epochs = self.config.get('warm_start_epochs', 50)
        if warm_epochs <= 0:
            return
            
        if verbose:
            print(f"\n🔥 Warm-start Training ({warm_epochs} epochs)...")
            
        # Create separate optimizer for warm-start with lower learning rate
        warm_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'] * 0.01,  # Reduced LR for warm-start
            weight_decay=self.config.get('optimizer', {}).get('weight_decay', 1e-7)
        )
        
        self.model.train()
        pbar = tqdm(range(warm_epochs), desc="Warm-start", leave=False) if verbose else range(warm_epochs)
        
        for epoch in pbar:
            warm_optimizer.zero_grad()
            
            # Forward pass
            predictions, media_coeffs, media_contributions, outputs = self.model(X_media, X_control, R)
            
            # Compute loss
            mse_loss = nn.MSELoss()(predictions, y)
            dag_loss = self.model.get_dag_loss() if hasattr(self.model, 'get_dag_loss') else 0
            sparsity_loss = self.model.get_sparsity_loss() if hasattr(self.model, 'get_sparsity_loss') else 0
            
            # Add L1 and L2 regularization
            l1_reg = sum(torch.sum(torch.abs(param)) for param in self.model.parameters())
            l2_reg = sum(torch.sum(param ** 2) for param in self.model.parameters())
            
            total_loss = (mse_loss + 
                         self.config.get('dag_weight', 1.0) * dag_loss + 
                         self.config.get('sparsity_weight', 0.1) * sparsity_loss +
                         self.config.get('l1_weight', 0.0) * l1_reg +
                         self.config.get('l2_weight', 0.0) * l2_reg)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                         self.config.get('max_grad_norm', 1.0))
            
            warm_optimizer.step()
            
            if verbose and isinstance(pbar, tqdm):
                pbar.set_postfix({'Loss': f'{total_loss.item():.4f}'})
                
    def train_epoch(self, X_media: torch.Tensor, X_control: torch.Tensor, 
                   R: torch.Tensor, y: torch.Tensor) -> Tuple[float, float, float]:
        """
        Train for one epoch.
        
        Args:
            X_media: Media data tensor
            X_control: Control data tensor  
            R: Region tensor
            y: Target tensor
            
        Returns:
            Tuple of (loss, rmse, r2)
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions, media_coeffs, media_contributions, outputs = self.model(X_media, X_control, R)
        
        # Use same loss function as today's working version (Huber Loss)
        if self.config.get('use_huber_loss', True):  # Default to Huber loss (today's working version)
            huber_delta = self.config.get('huber_delta', 0.3)
            base_loss = nn.HuberLoss(delta=huber_delta)(predictions, y)
        else:
            base_loss = nn.MSELoss()(predictions, y)
        
        # Add focal loss component for hard examples
        if self.config.get('use_focal_loss', False):
            alpha = self.config.get('focal_alpha', 0.25)
            gamma = self.config.get('focal_gamma', 1.5)
            
            # Calculate focal weight based on prediction error
            abs_error = torch.abs(predictions - y)
            normalized_error = abs_error / (abs_error.mean() + 1e-8)
            focal_weight = alpha * torch.pow(normalized_error, gamma)
            
            focal_loss = focal_weight * base_loss
            focal_weight_config = self.config.get('focal_loss_weight', 0.1)
            mse_loss = base_loss + focal_weight_config * focal_loss.mean()  # CONFIGURABLE focal loss contribution
        else:
            mse_loss = base_loss
        dag_loss = self.model.get_dag_loss() if hasattr(self.model, 'get_dag_loss') else 0
        sparsity_loss = self.model.get_sparsity_loss() if hasattr(self.model, 'get_sparsity_loss') else 0
        
        # HYBRID APPROACH: Fixed regularization weights for stable training
        # Only core model parameters (coefficients, ranges) are learnable - not loss balancing
        l1_reg = sum(torch.sum(torch.abs(param)) for param in self.model.parameters())
        l2_reg = sum(torch.sum(param ** 2) for param in self.model.parameters())
        
        # Use minimal fixed regularization for maximum learning capability (from config)
        total_loss = (mse_loss + 
                     self.config.get('dag_weight', 0.005) * dag_loss +      # Minimal DAG regularization
                     self.config.get('sparsity_weight', 0.001) * sparsity_loss + # Minimal sparsity regularization  
                     self.config.get('l1_weight', 1e-5) * l1_reg +         # Ultra-light L1
                     self.config.get('l2_weight', 5e-5) * l2_reg)          # Ultra-light L2
        
        # Backward pass
        total_loss.backward()
        
        # COEFFICIENT-SPECIFIC GRADIENT CLIPPING: Prevent coefficient explosion
        # Stronger clipping for coefficient-related parameters
        coeff_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'coeff' in name.lower() or 'range_raw' in name:
                    coeff_params.append(param)
                else:
                    other_params.append(param)
        
        # Advanced gradient clipping for stability
        if coeff_params:
            coeff_grad_clip = self.config.get('coeff_grad_clip', 1.0)  # Updated from config
            torch.nn.utils.clip_grad_norm_(coeff_params, max_norm=coeff_grad_clip)
        
        # Global gradient clipping for all parameters
        gradient_clip_norm = self.config.get('gradient_clip_norm', 2.0)
        if other_params:
            torch.nn.utils.clip_grad_norm_(other_params, max_norm=gradient_clip_norm)
        
        self.optimizer.step()
        
        # Calculate metrics IN LOG SPACE for training stability
        # Original-space conversion should ONLY be done for final reporting
        with torch.no_grad():
            y_np = y.detach().cpu().numpy().flatten()
            pred_np = predictions.detach().cpu().numpy().flatten()
            
            rmse = np.sqrt(mean_squared_error(y_np, pred_np))
            r2 = r2_score(y_np, pred_np) if len(np.unique(y_np)) > 1 else 0.0
            
        return total_loss.item(), rmse, r2
        
    def evaluate_holdout(self, X_media: torch.Tensor, X_control: torch.Tensor,
                        R: torch.Tensor, y: torch.Tensor) -> Tuple[float, float, float]:
        """
        Evaluate model on holdout data.
        
        Args:
            X_media: Holdout media data tensor
            X_control: Holdout control data tensor
            R: Holdout region tensor
            y: Holdout target tensor
            
        Returns:
            Tuple of (loss, rmse, r2)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            predictions, _, _, _ = self.model(X_media, X_control, R)
            
            # Enhanced validation loss matching training loss
            if self.config.get('use_huber_loss', True):  # Default to Huber loss
                huber_delta = self.config.get('huber_delta', 0.25)  # Tighter delta for precision
                base_loss = nn.HuberLoss(delta=huber_delta)(predictions, y)
            else:
                base_loss = nn.MSELoss()(predictions, y)
            
            # Add focal loss component for validation too
            if self.config.get('use_focal_loss', False):
                alpha = self.config.get('focal_alpha', 0.25)
                gamma = self.config.get('focal_gamma', 1.5)
                
                # Calculate focal weight based on prediction error
                abs_error = torch.abs(predictions - y)
                normalized_error = abs_error / (abs_error.mean() + 1e-8)
                focal_weight = alpha * torch.pow(normalized_error, gamma)
                
                focal_loss = focal_weight * base_loss
                focal_weight_config = self.config.get('focal_loss_weight', 0.1)
                mse_loss = base_loss + focal_weight_config * focal_loss.mean()  # CONFIGURABLE focal loss contribution
            else:
                mse_loss = base_loss
                
            dag_loss = self.model.get_dag_loss() if hasattr(self.model, 'get_dag_loss') else 0
            sparsity_loss = self.model.get_sparsity_loss() if hasattr(self.model, 'get_sparsity_loss') else 0
            
            # HYBRID APPROACH: Fixed regularization weights for consistent evaluation (from config)
            total_loss = (mse_loss + 
                         self.config.get('dag_weight', 0.005) * dag_loss +      # Minimal DAG regularization
                         self.config.get('sparsity_weight', 0.001) * sparsity_loss)  # Minimal sparsity regularization
            
            # Calculate metrics IN LOG SPACE for training stability
            # Original-space conversion should ONLY be done for final reporting
            y_np = y.detach().cpu().numpy().flatten()
            pred_np = predictions.detach().cpu().numpy().flatten()
            
            rmse = np.sqrt(mean_squared_error(y_np, pred_np))
            r2 = r2_score(y_np, pred_np) if len(np.unique(y_np)) > 1 else 0.0
            
        return total_loss.item(), rmse, r2
        
    def should_stop_early(self, current_rmse: float) -> bool:
        """
        Check if training should stop early based on RMSE improvement.
        
        Args:
            current_rmse: Current epoch's RMSE
            
        Returns:
            True if training should stop
        """
        if not self.config.get('early_stopping', False):
            return False
            
        if current_rmse < self.best_rmse:
            self.best_rmse = current_rmse
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            patience = self.config.get('patience', 600)
            return self.patience_counter >= patience
            
    def train(self, X_media_train: torch.Tensor, X_control_train: torch.Tensor,
              R_train: torch.Tensor, y_train: torch.Tensor,
              X_media_holdout: Optional[torch.Tensor] = None,
              X_control_holdout: Optional[torch.Tensor] = None,
              R_holdout: Optional[torch.Tensor] = None,
              y_holdout: Optional[torch.Tensor] = None,
              y_full_for_baseline: Optional[torch.Tensor] = None,
              verbose: bool = True) -> Dict[str, Any]:
        """
        Full training loop with warm-start, main training, and holdout evaluation.
        
        Args:
            X_media_train: Training media data
            X_control_train: Training control data
            R_train: Training region data
            y_train: Training target data
            X_media_holdout: Optional holdout media data
            X_control_holdout: Optional holdout control data
            R_holdout: Optional holdout region data
            y_holdout: Optional holdout target data
            verbose: Whether to show progress
            
        Returns:
            Dictionary with training results
        """
        if self.model is None or self.optimizer is None:
            raise ValueError("Model and optimizer must be created before training")
            
        # Move data to device
        X_media_train = X_media_train.to(self.device)
        X_control_train = X_control_train.to(self.device)
        R_train = R_train.to(self.device)
        y_train = y_train.to(self.device)
        
        if X_media_holdout is not None:
            X_media_holdout = X_media_holdout.to(self.device)
            X_control_holdout = X_control_holdout.to(self.device)
            R_holdout = R_holdout.to(self.device)
            y_holdout = y_holdout.to(self.device)
            
        # Initialize model with data - use full dataset for baseline if provided
        if hasattr(self.model, 'initialize_baseline'):
            baseline_data = y_full_for_baseline if y_full_for_baseline is not None else y_train
            self.model.initialize_baseline(baseline_data)
        if hasattr(self.model, 'initialize_stable_coefficients_from_data'):
            self.model.initialize_stable_coefficients_from_data(X_media_train, X_control_train, y_train)
            
        # Warm-start training
        self.warm_start_training(X_media_train, X_control_train, R_train, y_train, verbose)
        
        # Main training
        n_epochs = self.config.get('n_epochs', 1000)
        
        if verbose:
            print(f"\n🚀 Main Training ({n_epochs} epochs)...")
            
        # Storage for holdout metrics
        last_holdout_loss = None
        last_holdout_rmse = None
        last_holdout_r2 = None
        
        pbar = tqdm(range(n_epochs), desc="Training") if verbose else range(n_epochs)
        
        for epoch in pbar:
            # Training step
            train_loss, train_rmse, train_r2 = self.train_epoch(
                X_media_train, X_control_train, R_train, y_train
            )
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_rmses.append(train_rmse)
            self.train_r2s.append(train_r2)
            
            # Holdout evaluation (every 10 epochs to save time)
            if X_media_holdout is not None and epoch % 10 == 0:
                holdout_loss, holdout_rmse, holdout_r2 = self.evaluate_holdout(
                    X_media_holdout, X_control_holdout, R_holdout, y_holdout
                )
                last_holdout_loss = holdout_loss
                last_holdout_rmse = holdout_rmse
                last_holdout_r2 = holdout_r2
                
                # Advanced learning rate scheduling
                if self.config.get('use_cosine_annealing', False):
                    # For Cosine Annealing, step every epoch (not based on metrics)
                    if self.warmup_scheduler and epoch < self.config.get('warmup_epochs', 0):
                        self.warmup_scheduler.step()
                    else:
                        self.scheduler.step()
                else:
                    # Traditional ReduceLROnPlateau scheduling
                    self.scheduler.step(holdout_rmse)
                
                # Update best RMSE for monitoring (regardless of early stopping)
                if holdout_rmse < self.best_rmse:
                    self.best_rmse = holdout_rmse
                
                # Early stopping check
                if self.should_stop_early(holdout_rmse):
                    if verbose:
                        print(f"\n⏹️ Early stopping at epoch {epoch}")
                    break
            else:
                # Advanced learning rate scheduling (no holdout case)
                if self.config.get('use_cosine_annealing', False):
                    # For Cosine Annealing, step every epoch
                    if self.warmup_scheduler and epoch < self.config.get('warmup_epochs', 0):
                        self.warmup_scheduler.step()
                    else:
                        self.scheduler.step()
                else:
                    # Use training RMSE for scheduling if no holdout
                    self.scheduler.step(train_rmse)
                
                # Update best RMSE using training RMSE when no holdout available
                if train_rmse < self.best_rmse:
                    self.best_rmse = train_rmse
                
            # Update progress bar
            if verbose and isinstance(pbar, tqdm):
                progress_dict = {
                    'TrL': f'{train_loss:.2f}',
                    'TrR': f'{train_rmse:.4f}',
                    'TrR²': f'{train_r2:.3f}',
                    'Best': f'{self.best_rmse:.4f}'
                }
                
                # Add holdout metrics if available
                if last_holdout_rmse is not None:
                    r2_min = self.config.get('training_display', {}).get('r2_display_min', -10.0)
                    r2_display = last_holdout_r2 if last_holdout_r2 > r2_min else r2_min
                    progress_dict.update({
                        'HoL': f'{last_holdout_loss:.1f}',
                        'HoR': f'{last_holdout_rmse:.4f}',
                        'HoR²': f'{r2_display:.3f}'
                    })
                    
                pbar.set_postfix(progress_dict)
                
        # Final evaluation: Convert ONLY final results to original scale for reporting
        # Keep all training metrics in log space for stability
        self.model.eval()
        with torch.no_grad():
            # Final training evaluation in original scale
            train_pred_log, _, _, _ = self.model(X_media_train, X_control_train, R_train)
            train_pred_orig = torch.expm1(torch.clamp(train_pred_log, max=20.0))
            train_true_orig = torch.expm1(torch.clamp(y_train, max=20.0))
            
            final_train_rmse_orig = np.sqrt(mean_squared_error(
                train_true_orig.detach().cpu().numpy().flatten(),
                train_pred_orig.detach().cpu().numpy().flatten()
            ))
            final_train_r2_orig = r2_score(
                train_true_orig.detach().cpu().numpy().flatten(),
                train_pred_orig.detach().cpu().numpy().flatten()
            )
            
            # Final holdout evaluation in original scale (if available)
            if X_media_holdout is not None and y_holdout is not None:
                holdout_pred_log, _, _, _ = self.model(X_media_holdout, X_control_holdout, R_holdout)
                holdout_pred_orig = torch.expm1(torch.clamp(holdout_pred_log, max=20.0))
                holdout_true_orig = torch.expm1(torch.clamp(y_holdout, max=20.0))
                
                # Convert to numpy for robust metrics
                y_true_np = holdout_true_orig.detach().cpu().numpy().flatten()
                y_pred_np = holdout_pred_orig.detach().cpu().numpy().flatten()
                y_true_log_np = y_holdout.detach().cpu().numpy().flatten()
                y_pred_log_np = holdout_pred_log.detach().cpu().numpy().flatten()
                
                # Standard metrics (original scale)
                final_holdout_rmse_orig = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
                final_holdout_r2_orig = r2_score(y_true_np, y_pred_np)
                
                # ROBUST METRICS - Option 1
                from sklearn.metrics import mean_absolute_error
                
                # 1. Median Absolute Error (original scale)
                holdout_mae_orig = mean_absolute_error(y_true_np, y_pred_np)
                holdout_median_ae = np.median(np.abs(y_true_np - y_pred_np))
                
                # 2. Trimmed RMSE (remove top 5% outliers)
                abs_errors = np.abs(y_true_np - y_pred_np)
                trimmed_threshold = np.percentile(abs_errors, 95)
                trimmed_mask = abs_errors <= trimmed_threshold
                if np.sum(trimmed_mask) > 10:  # Need enough data points
                    holdout_trimmed_rmse = np.sqrt(mean_squared_error(
                        y_true_np[trimmed_mask], y_pred_np[trimmed_mask]
                    ))
                else:
                    holdout_trimmed_rmse = final_holdout_rmse_orig
                
                # 3. Log-space R² (should be excellent!)
                holdout_r2_log = r2_score(y_true_log_np, y_pred_log_np)
                
                # 4. Log-space RMSE 
                holdout_rmse_log = np.sqrt(mean_squared_error(y_true_log_np, y_pred_log_np))
                
                final_holdout_loss_orig = last_holdout_loss  # Keep log-space loss
            else:
                final_holdout_rmse_orig = None
                final_holdout_r2_orig = None
                final_holdout_loss_orig = None
                # Initialize robust metrics as None
                holdout_mae_orig = None
                holdout_median_ae = None
                holdout_trimmed_rmse = None
                holdout_r2_log = None
                holdout_rmse_log = None
        
        final_results = {
            'model': self.model,
            'train_losses': self.train_losses,
            'train_rmses': self.train_rmses,
            'train_r2s': self.train_r2s,
            'best_rmse': self.best_rmse,
            'final_train_loss': self.train_losses[-1] if self.train_losses else 0.0,
            'final_train_rmse': final_train_rmse_orig,  # ORIGINAL SCALE (for final reporting)
            'final_train_r2': final_train_r2_orig,      # ORIGINAL SCALE (for final reporting)
        }
        
        if final_holdout_rmse_orig is not None:
            final_results.update({
                'final_holdout_loss': final_holdout_loss_orig,
                'final_holdout_rmse': final_holdout_rmse_orig,  # ORIGINAL SCALE (for final reporting)
                'final_holdout_r2': final_holdout_r2_orig,      # ORIGINAL SCALE (for final reporting)
                # ROBUST METRICS - More reliable evaluation
                'holdout_mae_orig': holdout_mae_orig,           # Mean Absolute Error (original scale)
                'holdout_median_ae': holdout_median_ae,         # Median Absolute Error (original scale)  
                'holdout_trimmed_rmse': holdout_trimmed_rmse,   # Trimmed RMSE (95% of data, removes outliers)
                'holdout_r2_log': holdout_r2_log,               # R² in log space (should be excellent!)
                'holdout_rmse_log': holdout_rmse_log,           # RMSE in log space (training metric)
            })
            
        return final_results
    
    # REMOVED: _set_random_seeds method
    # Seeds should be set once by the main script and not interfered with during training.
    # Multiple seed resets can disrupt the intended random number sequence.
