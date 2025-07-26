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

from ..exceptions import DataError, BayesianNetworkError, ValidationError

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