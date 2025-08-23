"""
Data preprocessing and loading utilities for DeepCausalMMM.

This module handles:
- Data loading and validation
- Bayesian Network creation
- Feature engineering (adstock, saturation)
- Data scaling and preparation
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from deepcausalmmm.exceptions import DataError, BayesianNetworkError, ValidationError
from deepcausalmmm.core.scaling import SimpleGlobalScaler

# Try to import pgmpy, fallback to simpler approach if not available
try:
    from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Validate that the dataframe contains required columns.
    
    Args:
        df: Input dataframe
        required_columns: List of required column names
        
    Raises:
        ValidationError: If required columns are missing
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValidationError(f"Missing required columns: {missing_columns}")


def create_belief_vectors(
    df: pd.DataFrame, 
    control_vars: List[str]
) -> Tuple[pd.DataFrame, Any]:
    """
    Create belief vectors from control variables using Bayesian Network.
    
    Args:
        df: Input dataframe
        control_vars: List of control variable names
        
    Returns:
        Tuple of (belief_vectors_df, bayesian_network_structure)
    """
    if PGMPY_AVAILABLE and len(control_vars) > 0:
        try:
            # Use pgmpy for Bayesian Network
            disc_ctrl = df[control_vars].astype(str)
            bn_struct = HillClimbSearch(disc_ctrl).estimate(BicScore(disc_ctrl))
            bn = DiscreteBayesianNetwork(bn_struct.edges())
            bn.fit(disc_ctrl, MaximumLikelihoodEstimator)
            bn_inf = VariableElimination(bn)
            
            def belief(row):
                beliefs = []
                for v in control_vars:
                    evidence = {var: disc_ctrl.loc[row.name, var] for var in control_vars if var != v}
                    q = bn_inf.query(variables=[v], evidence=evidence, show_progress=False)
                    beliefs.append(np.argmax(q.values))
                return beliefs
            
            Z_ctrl = np.vstack(df.apply(belief, axis=1))
            result_df = pd.DataFrame(Z_ctrl, columns=[f'belief_{i}' for i in range(Z_ctrl.shape[1])])
            return result_df, bn_struct
            
        except Exception as e:
            print(f"Bayesian Network failed: {e}, using fallback")
    
    # Fallback: use control variables directly
    if len(control_vars) > 0:
        Z_ctrl = df[control_vars].values
        result_df = pd.DataFrame(Z_ctrl, columns=[f'belief_{i}' for i in range(Z_ctrl.shape[1])])
    else:
        # Create dummy control variables
        Z_ctrl = np.random.randint(0, 5, (len(df), 15))
        result_df = pd.DataFrame(Z_ctrl, columns=[f'belief_{i}' for i in range(15)])
    
    # Create simple adjacency matrix for controls
    n_ctrl = len(control_vars) if len(control_vars) > 0 else 15
    bn_struct = type('MockStruct', (), {
        'edges': lambda: [(f'control_{i}', f'control_{i+1}') for i in range(n_ctrl-1)]
    })()
    
    return result_df, bn_struct


def create_media_adjacency(
    media_vars: List[str], 
    bn_struct: Optional[Any] = None
) -> torch.Tensor:
    """
    Create adjacency matrix for media variables.
    
    Args:
        media_vars: List of media variable names
        bn_struct: Bayesian network structure (optional)
        
    Returns:
        Adjacency matrix as torch tensor
    """
    n_media = len(media_vars)
    A_media = np.zeros((n_media, n_media))
    
    if bn_struct and hasattr(bn_struct, 'edges'):
        try:
            for u, v in bn_struct.edges():
                if u in media_vars and v in media_vars:
                    u_idx = media_vars.index(u)
                    v_idx = media_vars.index(v)
                    if u_idx < n_media and v_idx < n_media:
                        A_media[u_idx, v_idx] = 1.0
        except Exception:
            pass
    
    # If no edges found, create simple chain structure
    if A_media.sum() == 0:
        for i in range(min(n_media - 1, n_media - 1)):
            A_media[i, i + 1] = 1.0
    
    return torch.tensor(A_media, dtype=torch.float32)


def prepare_data_for_training(
    df: pd.DataFrame, 
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prepare data for training with proper scaling and structure.
    
    Args:
        df: Input dataframe
        params: Configuration parameters
        
    Returns:
        Dictionary containing prepared data and scalers
    """
    burn_in = params.get("burn_in_weeks", 4)
    df_work = df.copy()
    
    # Get variable names from config
    marketing_vars = params.get('marketing_vars', [])
    control_vars = params.get('control_vars', [])
    dependent_var = params.get('dependent_var', 'revenue')
    region_var = params.get('region_var', None)
    date_var = params.get('date_var', None)
    
    # Auto-detect variables if not provided
    if not marketing_vars:
        marketing_vars = [col for col in df_work.columns 
                         if any(keyword in col.lower() for keyword in 
                               ['spend', 'media', 'tv', 'digital', 'radio', 'social'])]
    
    if not control_vars:
        control_vars = [col for col in df_work.columns 
                       if col not in marketing_vars + [dependent_var, region_var, date_var] 
                       and col not in ['date', 'week', 'region']
                       and df_work[col].dtype in ['int64', 'float64']]
    
    # Handle region creation/validation
    if region_var and region_var in df_work.columns:
        regions = df_work[region_var].unique()
        print(f"Using existing region column: {len(regions)} regions found")
    else:
        df_work['region'] = 'All_Data'
        regions = df_work['region'].unique()
        region_var = 'region'
        print(f"No region column specified, creating single region with {len(df_work)} rows")
    
    # Add week if not exists
    if 'week' not in df_work.columns:
        if date_var and date_var in df_work.columns:
            df_work['week'] = pd.to_datetime(df_work[date_var]).dt.isocalendar().week
        else:
            df_work['week'] = np.arange(len(df_work))
    
    # Ensure target variable exists
    if dependent_var not in df_work.columns:
        if 'revenue' in df_work.columns:
            dependent_var = 'revenue'
        elif 'sales' in df_work.columns:
            dependent_var = 'sales'
        else:
            raise DataError(f"Target variable '{dependent_var}' not found in data")
    
    # Group data by region and ensure equal sequence lengths
    region_data = {}
    min_length = float('inf')
    
    for region in regions:
        region_df = df_work[df_work[region_var] == region].copy()
        
        # Sort by time
        if date_var and date_var in region_df.columns:
            region_df = region_df.sort_values(date_var)
        elif 'week' in region_df.columns:
            region_df = region_df.sort_values('week')
        
        region_data[region] = region_df
        min_length = min(min_length, len(region_df))
    
    # Truncate all regions to same length
    for region in regions:
        region_df_trimmed = region_data[region].iloc[:min_length]
        
        # Add burn-in padding
        if burn_in > 0:
            pad_block = region_df_trimmed.iloc[:1].copy()
            pad_block = pd.concat([pad_block] * burn_in, ignore_index=True)
            
            # Add small jitter to media columns
            for col in marketing_vars:
                if col in pad_block.columns:
                    pad_block[col] += np.random.normal(0, 0.01, size=burn_in)
            
            region_df_trimmed = pd.concat([pad_block, region_df_trimmed], ignore_index=True)
        
        region_data[region] = region_df_trimmed
    
    # Create standardized feature matrices
    n_regions = len(regions)
    n_time_steps = min_length + burn_in
    n_media_target = len(marketing_vars)
    n_control_target = max(len(control_vars), 1)
    
    # Media variables matrix
    media_matrix = np.zeros((n_regions, n_time_steps, n_media_target))
    for i, region in enumerate(regions):
        region_df = region_data[region]
        for j, var in enumerate(marketing_vars):
            if var in region_df.columns:
                media_matrix[i, :, j] = region_df[var].values
    
    # Control variables matrix
    control_matrix = np.zeros((n_regions, n_time_steps, n_control_target))
    for i, region in enumerate(regions):
        region_df = region_data[region]
        for j, var in enumerate(control_vars):
            if var in region_df.columns:
                control_matrix[i, :, j] = region_df[var].values
        
        # If no control variables, create intercept
        if len(control_vars) == 0:
            control_matrix[i, :, 0] = 1.0
    
    # Target variable matrix
    y_matrix = np.zeros((n_regions, n_time_steps))
    for i, region in enumerate(regions):
        region_df = region_data[region]
        y_matrix[i, :] = region_df[dependent_var].values
    
    # Create region IDs
    region_ids = np.arange(n_regions)
    
    # Scale the data
    media_scaler = MinMaxScaler()
    media_matrix_flat = media_matrix.reshape(-1, n_media_target)
    media_matrix_scaled = media_scaler.fit_transform(media_matrix_flat)
    media_matrix_scaled = media_matrix_scaled.reshape(n_regions, n_time_steps, n_media_target)
    
    control_scaler = MinMaxScaler()
    control_matrix_flat = control_matrix.reshape(-1, n_control_target)
    control_matrix_scaled = control_scaler.fit_transform(control_matrix_flat)
    control_matrix_scaled = control_matrix_scaled.reshape(n_regions, n_time_steps, n_control_target)
    
    y_scaler = MinMaxScaler()
    y_matrix_flat = y_matrix.reshape(-1, 1)
    y_matrix_scaled = y_scaler.fit_transform(y_matrix_flat)
    y_matrix_scaled = y_matrix_scaled.reshape(n_regions, n_time_steps)
    
    # Convert to tensors
    X_m = torch.tensor(media_matrix_scaled, dtype=torch.float32)
    X_c = torch.tensor(control_matrix_scaled, dtype=torch.float32)
    y = torch.tensor(y_matrix_scaled, dtype=torch.float32)
    R = torch.tensor(region_ids, dtype=torch.long)
    
    print(f"Data preparation complete:")
    print(f"  - Regions: {n_regions}")
    print(f"  - Time steps: {n_time_steps}")
    print(f"  - Media variables: {len(marketing_vars)} → {n_media_target}")
    print(f"  - Control variables: {len(control_vars)} → {n_control_target}")
    print(f"  - Target variable: {dependent_var}")
    print(f"  - Shapes: X_m={X_m.shape}, X_c={X_c.shape}, y={y.shape}, R={R.shape}")
    
    return {
        'X_m': X_m,
        'X_c': X_c,
        'R': R,
        'y': y,
        'burn_in': burn_in,
        'media_scaler': media_scaler,
        'control_scaler': control_scaler,
        'y_scaler': y_scaler,
        'marketing_vars': marketing_vars,
        'control_vars': control_vars,
        'dependent_var': dependent_var,
        'regions': regions,
    }


def load_and_preprocess_data(
    file_path: str, 
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Load data from file and preprocess for training.
    
    Args:
        file_path: Path to data file (CSV, Excel, etc.)
        params: Configuration parameters
        
    Returns:
        Dictionary containing prepared data and metadata
    """
    # Load data
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise DataError(f"Unsupported file format: {file_path}")
    
    # Validate data
    required_columns = params.get('required_columns', ['revenue'])
    validate_dataframe(df, required_columns)
    
    # Prepare data
    data_dict = prepare_data_for_training(df, params)
    
    # Create Bayesian Network structure
    Z_ctrl, bn_struct = create_belief_vectors(df, data_dict['control_vars'])
    A_media = create_media_adjacency(data_dict['marketing_vars'], bn_struct)
    
    data_dict.update({
        'belief_vectors': Z_ctrl,
        'bayesian_network': bn_struct,
        'media_adjacency': A_media,
        'original_dataframe': df,
    })
    
    return data_dict


class UnifiedDataPipeline:
    """
    Unified data processing pipeline for DeepCausalMMM models.
    
    This pipeline ensures consistent data transformations between training and holdout
    datasets, implementing the complete preprocessing workflow required for MMM analysis.
    It handles temporal splitting, multi-scale normalization, seasonal decomposition,
    and tensor preparation for PyTorch models.
    
    Key Features:
    - Temporal train/holdout splitting (respects time series nature)
    - SOV (Share of Voice) scaling for media channels
    - Z-score normalization for control variables  
    - Min-Max scaling for seasonal components (per region)
    - Burn-in padding for GRU stabilization
    - Automatic tensor conversion and device handling
    - Inverse transformation utilities for interpretation
    - Region encoding and validation
    
    The pipeline maintains data integrity by:
    - Using the same scaler fit on training data for holdout
    - Preserving temporal order in all transformations
    - Handling missing values and outliers appropriately
    - Ensuring consistent tensor shapes across regions
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing:
        - 'holdout_ratio': Fraction of data for holdout (default 0.08)
        - 'burn_in_weeks': Number of weeks for padding (default 6)
        - 'random_seed': Seed for reproducible operations (default 42)
        - Media channel names, control variable names, etc.
        
    Attributes
    ----------
    scaler : SimpleGlobalScaler
        Fitted scaler for consistent transformations
    seasonal_detector : DetectSeasonality
        Seasonal decomposition utility
    media_columns : List[str]
        Names of media channel columns
    control_columns : List[str]
        Names of control variable columns
    region_column : str
        Name of region identifier column
    target_column : str
        Name of target variable column
        
    Examples
    --------
    >>> import pandas as pd
    >>> from deepcausalmmm.core.data import UnifiedDataPipeline
    >>> from deepcausalmmm.core.config import get_default_config
    >>> 
    >>> # Load your MMM dataset
    >>> df = pd.read_csv('mmm_data.csv')
    >>> config = get_default_config()
    >>> 
    >>> # Initialize and fit pipeline
    >>> pipeline = UnifiedDataPipeline(config)
    >>> processed_data = pipeline.fit_transform(df)
    >>> 
    >>> # Access processed tensors
    >>> X_media_train = processed_data['X_media_train']
    >>> y_train = processed_data['y_train']
    >>> 
    >>> # Get holdout data
    >>> X_media_holdout = processed_data['X_media_holdout']
    >>> y_holdout = processed_data['y_holdout']
    >>> 
    >>> print(f"Training shape: {X_media_train.shape}")
    >>> print(f"Holdout shape: {X_media_holdout.shape}")
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the unified data pipeline.
        
        Args:
            config: Configuration dictionary with all parameters
        """
        self.config = config
        self.scaler = None
        self.padding_weeks = config.get('burn_in_weeks', 20)
        self.fitted = False
        
    def temporal_split(self, 
                      X_media: np.ndarray, 
                      X_control: np.ndarray, 
                      y: np.ndarray,
                      holdout_ratio: Optional[float] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Perform time series split of data using ratio-based approach.
        This ensures adequate holdout data regardless of burn-in weeks.
        
        Args:
            X_media: Media data [regions, weeks, channels]
            X_control: Control data [regions, weeks, controls]
            y: Target data [regions, weeks]
            holdout_ratio: Fraction of data for holdout (uses config if None)
            
        Returns:
            Tuple of (train_data_dict, holdout_data_dict)
        """
        n_weeks = X_media.shape[1]
        
        # Use config holdout_ratio if not provided
        if holdout_ratio is None:
            holdout_ratio = self.config.get('holdout_ratio', 0.27)
        
        # Calculate split point using ratio
        holdout_weeks = int(n_weeks * holdout_ratio)
        train_weeks = n_weeks - holdout_weeks
        self.train_weeks = train_weeks  # Store for holdout seasonality alignment
        
        # Ensure we have reasonable amounts of data for both splits
        min_train_weeks = self.config.get('min_train_weeks', 60)
        burn_in_weeks = self.config.get('burn_in_weeks', 20)
        
        if train_weeks < min_train_weeks:
            raise ValueError(f"Not enough training data: need at least {min_train_weeks} training weeks, "
                           f"but only have {train_weeks} with {holdout_ratio:.1%} holdout ratio")
                           
        if holdout_weeks < burn_in_weeks + 5:  # Need at least 5 weeks after burn-in for meaningful evaluation
            print(f"   ⚠️ WARNING: Holdout weeks ({holdout_weeks}) is close to burn-in weeks ({burn_in_weeks})")
            print(f"   ⚠️ After burn-in removal, only {holdout_weeks - burn_in_weeks} weeks will remain for evaluation")
        
        # Time series split: train on first weeks, test on last weeks
        # Training data (chronologically first)
        train_data = {
            'X_media': X_media[:, :train_weeks, :].astype(np.float32),
            'X_control': X_control[:, :train_weeks, :].astype(np.float32),
            'y': y[:, :train_weeks].astype(np.float32)
        }
        
        # Holdout data (chronologically last - most recent weeks)
        holdout_data = {
            'X_media': X_media[:, train_weeks:, :].astype(np.float32),
            'X_control': X_control[:, train_weeks:, :].astype(np.float32),
            'y': y[:, train_weeks:].astype(np.float32)
        }
        
        print(f"\n📊 Unified Data Pipeline - Time Series Split (Ratio-Based):")
        print(f"   🚂 Training: {train_weeks} actual weeks (weeks 1-{train_weeks}) - {train_weeks/n_weeks*100:.1f}%")
        print(f"   🧪 Holdout: {holdout_weeks} actual weeks (weeks {train_weeks+1}-{n_weeks}) - {holdout_weeks/n_weeks*100:.1f}%")
        print(f"   🔥 Burn-in Padding: {burn_in_weeks} weeks will be added to BOTH train and holdout")
        print(f"   📊 Model sees: Train {train_weeks + burn_in_weeks} weeks, Holdout {holdout_weeks + burn_in_weeks} weeks")
        print(f"   🎯 Evaluation: Remove {burn_in_weeks} padding weeks, evaluate on ALL actual data")
        print(f"   ⏰ Time series approach: Training on historical data, testing on most recent data")
        
        return train_data, holdout_data
    
    def fit_and_transform_training(self, train_data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Fit scaler on training data and transform it.
        
        Args:
            train_data: Dictionary with training data arrays
            
        Returns:
            Dictionary with transformed and padded tensors
        """
        print(f"\n🔧 Unified Data Pipeline - Processing Training Data:")
        
        # 1. Add seasonality features to training data BEFORE scaling (CRITICAL FIX)
        print(f"   🌊 Adding seasonality features to training data...")
        X_control_with_seasonality_raw = self._add_seasonality_features(
            torch.tensor(train_data['X_control']), start_week=0
        )
        
        # 2. Initialize and fit scaler on training data WITH seasonality
        self.scaler = SimpleGlobalScaler(config=self.config)
        X_media_scaled, X_control_with_seasonality, y_scaled = self.scaler.fit_transform(
            train_data['X_media'], X_control_with_seasonality_raw.numpy(), train_data['y']
        )
        
        # 3. Add padding for burn-in
        X_media_padded, X_control_padded, y_padded = self._add_padding(
            X_media_scaled, X_control_with_seasonality, y_scaled
        )
        
        # 4. Create region tensor
        n_regions = X_media_padded.shape[0]
        R = torch.arange(n_regions, dtype=torch.long)
        
        self.fitted = True
        
        # Store train tensors for get_processed_full_data
        self.train_tensors = {
            'X_media': X_media_padded,
            'X_control': X_control_padded,
            'y': y_padded,
            'R': R
        }
        
        print(f"   ✅ Training data processed:")
        print(f"   📊 Shape: {X_media_padded.shape[0]} regions × {X_media_padded.shape[1]} weeks")
        print(f"   📺 Media channels: {X_media_padded.shape[2]}")
        print(f"   🎛️ Control variables: {X_control_padded.shape[2]} (seasonality now handled by model baseline)")
        print(f"   🔧 Scaler fitted on training data only")
        print(f"   📏 Added {self.padding_weeks} weeks padding")
        
        return self.train_tensors
    
    def transform_holdout(self, holdout_data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Transform holdout data using the fitted scaler (same transformations as training).
        
        Args:
            holdout_data: Dictionary with holdout data arrays
            
        Returns:
            Dictionary with transformed and padded tensors
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted on training data first")
            
        print(f"\n🧪 Unified Data Pipeline - Processing Holdout Data:")
        
        # 1. Add seasonality features to holdout data BEFORE scaling (CRITICAL FIX)
        print(f"   🌊 Adding seasonality features to holdout data BEFORE scaling...")
        # CRITICAL FIX: Add seasonality BEFORE scaling, then scale everything together
        # This ensures seasonality features are properly scaled with the same distribution
        X_control_with_seasonality_raw = self._add_seasonality_features(
            torch.tensor(holdout_data['X_control']), start_week=self.train_weeks
        )
        
        # Now scale ALL data WITH seasonality using the training scaler
        X_media_scaled, X_control_with_seasonality, y_scaled = self.scaler.transform(
            holdout_data['X_media'], X_control_with_seasonality_raw.numpy(), holdout_data['y']
        )
        
        # 2. Add SAME padding as training data
        X_media_padded, X_control_padded, y_padded = self._add_padding(
            X_media_scaled, X_control_with_seasonality, y_scaled
        )
        
        # 3. Create region tensor (same as training)
        n_regions = X_media_padded.shape[0]
        R = torch.arange(n_regions, dtype=torch.long)
        
        # Store holdout tensors for get_processed_full_data
        self.holdout_tensors = {
            'X_media': X_media_padded,
            'X_control': X_control_padded,
            'y': y_padded,
            'R': R
        }
        
        print(f"   ✅ Holdout data processed with IDENTICAL transformations:")
        print(f"   📊 Shape: {X_media_padded.shape[0]} regions × {X_media_padded.shape[1]} weeks")
        print(f"   📺 Media channels: {X_media_padded.shape[2]}")
        print(f"   🎛️ Control variables: {X_control_padded.shape[2]} (seasonality now handled by model baseline)")
        print(f"   🔧 Used SAME scaler as training (no data leakage)")
        print(f"   📏 Added SAME {self.padding_weeks} weeks padding")
        
        return self.holdout_tensors
    
    def _add_seasonality_features(self, X_control: torch.Tensor, start_week: int = 0) -> torch.Tensor:
        """
        UPDATED: Seasonality is now handled by the model using actual data decomposition.
        This method now returns control variables unchanged.
        """
        print("   🌊 Seasonality now handled by model's data-driven seasonal decomposition (not as control variable)")
        return X_control  # Return unchanged - no artificial seasonality added
    
    def _add_padding(self, 
                    X_media: torch.Tensor, 
                    X_control: torch.Tensor, 
                    y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Add consistent padding to all data splits.
        
        Args:
            X_media: Media tensor [regions, weeks, channels]
            X_control: Control tensor [regions, weeks, controls] 
            y: Target tensor [regions, weeks]
            
        Returns:
            Tuple of padded tensors
        """
        # Create padding tensors
        media_padding = torch.zeros(X_media.shape[0], self.padding_weeks, X_media.shape[2])
        control_padding = torch.zeros(X_control.shape[0], self.padding_weeks, X_control.shape[2])
        y_padding = torch.zeros(y.shape[0], self.padding_weeks)
        
        # Add padding to the beginning
        X_media_padded = torch.cat([media_padding, X_media], dim=1)
        X_control_padded = torch.cat([control_padding, X_control], dim=1)
        y_padded = torch.cat([y_padding, y], dim=1)
        
        return X_media_padded, X_control_padded, y_padded
    
    def inverse_transform_predictions(self, 
                                    y_pred_scaled: torch.Tensor,
                                    remove_padding: bool = True) -> torch.Tensor:
        """
        Inverse transform predictions to original scale.
        
        Args:
            y_pred_scaled: Predictions in scaled space
            remove_padding: Whether to remove padding weeks
            
        Returns:
            Predictions in original scale
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted first")
            
        # Remove padding if requested
        if remove_padding:
            y_pred_eval = y_pred_scaled[:, self.padding_weeks:]
        else:
            y_pred_eval = y_pred_scaled
            
        # Inverse transform
        y_pred_orig = self.scaler.inverse_transform_target(y_pred_eval)
        
        return y_pred_orig
    
    def get_evaluation_data(self, 
                           y_true_padded: torch.Tensor, 
                           y_pred_padded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract evaluation data (removing burn-in padding).
        
        Args:
            y_true_padded: True values with padding
            y_pred_padded: Predicted values with padding
            
        Returns:
            Tuple of (y_true_eval, y_pred_eval) without padding
        """
        y_true_eval = y_true_padded[:, self.padding_weeks:].contiguous()
        y_pred_eval = y_pred_padded[:, self.padding_weeks:].contiguous()
        
        return y_true_eval, y_pred_eval
    
    def inverse_transform_contributions(self, 
                                       media_contributions: torch.Tensor, 
                                       y_true: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform media contributions to original scale.
        
        Args:
            media_contributions: Media contributions in scaled space
            y_true: True values in original scale (for scaling reference)
            
        Returns:
            Media contributions in original scale
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted first")
        
        return self.scaler.inverse_transform_contributions(media_contributions, y_true)
    
    def get_scaler(self) -> SimpleGlobalScaler:
        """
        Get the fitted scaler for external use.
        
        Returns:
            Fitted SimpleGlobalScaler instance
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted first")
        return self.scaler
    
    def predict_and_postprocess(self, 
                               model,
                               X_media: np.ndarray,
                               X_control: np.ndarray,
                               channel_names: List[str],
                               control_names: List[str],
                               combine_with_holdout: bool = True) -> Dict[str, Any]:
        """
        Generate predictions and contributions using the unified pipeline.
        
        Args:
            model: Trained model
            X_media: Media data (full dataset for contributions)
            X_control: Control data (full dataset for contributions) 
            channel_names: Media channel names
            control_names: Control variable names
            combine_with_holdout: Whether to combine train+holdout for contributions
            
        Returns:
            Dictionary with predictions, contributions, and metadata
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted first")
            
        print(f"\n📊 Unified Post-Processing:")
        
        # 1. Process full dataset for contributions (if requested)
        if combine_with_holdout:
            print(f"   🔗 Processing full dataset (train + holdout) for contributions...")
            
            # CRITICAL FIX: Add seasonality features with PROPER temporal alignment
            # Must match how training/holdout data was processed during training
            print(f"   🌊 Adding seasonality features to full dataset with proper temporal alignment...")
            X_control_with_seasonality_raw = self._add_seasonality_features(
                torch.tensor(X_control), start_week=0  # This is correct for full dataset
            )
            
            # Transform full data WITH seasonality using fitted scaler
            X_media_scaled, X_control_with_seasonality, _ = self.scaler.transform(
                X_media, X_control_with_seasonality_raw.numpy(), np.zeros_like(X_media[:, :, 0])  # Dummy y for consistency
            )
            
            # Add padding
            X_media_padded, X_control_padded, _ = self._add_padding(
                X_media_scaled, X_control_with_seasonality, torch.zeros(X_media_scaled.shape[0], X_media_scaled.shape[1])
            )
            
            # Create region tensor
            n_regions = X_media_padded.shape[0]
            R = torch.arange(n_regions, dtype=torch.long)
            
            # Get model predictions and contributions
            model.eval()
            with torch.no_grad():
                y_pred_scaled, media_contributions, control_contributions, outputs = model(
                    X_media_padded, X_control_padded, R
                )
            
            # Remove padding from predictions and contributions
            y_pred_eval = y_pred_scaled[:, self.padding_weeks:]
            media_contrib_eval = media_contributions[:, self.padding_weeks:, :]
            control_contrib_eval = control_contributions[:, self.padding_weeks:, :]
            
            # Inverse transform predictions to original scale
            y_pred_orig = self.scaler.inverse_transform_target(y_pred_eval)
            
            print(f"   ✅ Full dataset processed: {X_media.shape[0]} regions × {X_media.shape[1]} weeks")
            print(f"   📈 Predictions shape: {y_pred_orig.shape}")
            print(f"   📺 Media contributions shape: {media_contrib_eval.shape}")
            print(f"   🎛️ Control contributions shape: {control_contrib_eval.shape}")
            
        else:
            # Use only training data
            y_pred_orig = None
            media_contrib_eval = None
            control_contrib_eval = None
            outputs = {}
        
        # 2. Prepare results dictionary
        results = {
            'predictions': y_pred_orig,
            'media_contributions': media_contrib_eval,
            'control_contributions': control_contrib_eval,
            'channel_names': channel_names,
            'control_names': control_names,
            'model_outputs': outputs,
            'scaler': self.scaler,
            'config': self.config,
            'padding_weeks': self.padding_weeks
        }
        
        return results
    
    def calculate_metrics(self, 
                         y_true: torch.Tensor, 
                         y_pred: torch.Tensor,
                         prefix: str = "") -> Dict[str, float]:
        """
        Calculate comprehensive metrics for model evaluation.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            prefix: Prefix for metric names (e.g., 'train_', 'holdout_')
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        import numpy as np
        
        # Convert to numpy if needed
        if isinstance(y_true, torch.Tensor):
            y_true_np = y_true.numpy().flatten()
        else:
            y_true_np = np.array(y_true).flatten()
            
        if isinstance(y_pred, torch.Tensor):
            y_pred_np = y_pred.numpy().flatten()
        else:
            y_pred_np = np.array(y_pred).flatten()
        
        # Calculate metrics
        r2 = r2_score(y_true_np, y_pred_np)
        rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
        mae = mean_absolute_error(y_true_np, y_pred_np)
        
        # Relative metrics
        y_mean = np.mean(y_true_np)
        relative_rmse = (rmse / y_mean) * 100 if y_mean != 0 else 0
        relative_mae = (mae / y_mean) * 100 if y_mean != 0 else 0
        
        return {
            f'{prefix}r2': r2,
            f'{prefix}rmse': rmse,
            f'{prefix}mae': mae,
            f'{prefix}relative_rmse': relative_rmse,
            f'{prefix}relative_mae': relative_mae,
            f'{prefix}mean': y_mean
        }
    
    def get_processed_full_data(self):
        """
        Get the processed full dataset (train + holdout) with all transformations applied.
        This includes seasonality features, scaling, and padding - exactly as the model expects.
        
        Returns:
            Dictionary containing processed X_media and X_control tensors
        """
        if not hasattr(self, 'train_tensors') or not hasattr(self, 'holdout_tensors'):
            raise ValueError("Pipeline must be fitted and holdout data processed before getting full data")
        
        # Combine train and holdout data (both already include padding and seasonality)
        X_media_full = torch.cat([
            self.train_tensors['X_media'],  # [n_regions, train_weeks + padding, n_media]
            self.holdout_tensors['X_media']  # [n_regions, holdout_weeks + padding, n_media]
        ], dim=1)
        
        X_control_full = torch.cat([
            self.train_tensors['X_control'],  # [n_regions, train_weeks + padding, n_control + seasonality]
            self.holdout_tensors['X_control']  # [n_regions, holdout_weeks + padding, n_control + seasonality]
        ], dim=1)
        
        print(f"   🔗 Combined full dataset:")
        print(f"      📺 Media: {X_media_full.shape}")
        print(f"      🎛️ Control: {X_control_full.shape} (includes seasonality)")
        
        return {
            'X_media': X_media_full.numpy(),
            'X_control': X_control_full.numpy()
        } 
