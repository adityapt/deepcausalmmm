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
import torch.nn.functional as F
import json
import sys
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
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

def gs(logits, tau=0.5):
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
    return torch.sigmoid((logits + g) / tau)

class CausalEncoder(nn.Module):
    def __init__(self, A_prior, d_in=10, d_hid=10):
        super().__init__()
        self.register_buffer("A", torch.tensor(A_prior, dtype=torch.float32))           # fixed DAG
        self.edge = nn.Sequential(nn.Linear(d_in*2, d_hid), nn.ReLU(),
                                  nn.Linear(d_hid, d_hid), nn.ReLU())
        self.node = nn.Sequential(nn.Linear(d_in+d_hid, d_hid), nn.ReLU())
    
    def forward(self, X):                           # X [B,d_in]
        send, recv = X @ self.A.t(), X @ self.A      # [B,d_in] each
        He = self.edge(torch.cat([send, recv], -1))  # [B,d_in]
        Z  = self.node(torch.cat([X, He], -1))       # [B,d_in]
        return Z

class GRUCausalMMM(nn.Module):
    def __init__(self, A_prior, n_media=10, ctrl_dim=15, hidden=64, n_regions=2):
        super().__init__()
        self.enc   = CausalEncoder(A_prior, d_in=n_media, d_hid=n_media)
        self.gru   = nn.GRU(input_size=n_media+ctrl_dim, hidden_size=hidden, batch_first=True)
        # small-gain init keeps β₀ from exploding
        self.w_raw = nn.Linear(hidden, n_media)
        nn.init.xavier_uniform_(self.w_raw.weight, gain=0.10)
        nn.init.zeros_(self.w_raw.bias)
        # REMOVED w_base - let GRU learn channel-specific weights from scratch
        self.alpha = nn.Parameter(torch.full((n_media,), 0.5))
        # Hill saturation parameters - learnable with diverse initialization
        # Initialize gamma values to be more appropriate for scaled data (0-1 range after MinMaxScaler)
        torch.manual_seed(42)  # For reproducible initialization
        self.hill_a = nn.Parameter(torch.rand(n_media) * 0.8 + 0.6)  # Shape: 0.6-1.4
        self.hill_g = nn.Parameter(torch.rand(n_media) * 0.3 + 0.1)  # Half-saturation: 0.1-0.4 (for 0-1 scaled data)
        self.ctrl_mlp = nn.Linear(ctrl_dim, hidden)
        self.reg_emb  = nn.Embedding(n_regions, hidden)
        self.bias     = nn.Parameter(torch.zeros(1))
        self.hidden_size = hidden
        
        # Learnable warm-start: trainable initial hidden state to fix week-0 spike
        self.h0 = nn.Parameter(torch.zeros(1, 1, hidden))

    def adstock(self, x):                               # x [B,T,10]
        B, T, C = x.shape
        alpha = torch.clamp(self.alpha, 0, 1).view(1, 1, -1)
        
        # Use list to collect results and avoid in-place operations
        out_list = [x[:, 0:1]]  # Start with first timestep [B, 1, C]
        
        for t in range(1, T):
            prev_adstock = out_list[-1]  # [B, 1, C]
            current = x[:, t:t+1] + alpha * prev_adstock  # [B, 1, C]
            out_list.append(current)
        
        # Concatenate all timesteps
        out = torch.cat(out_list, dim=1)  # [B, T, C]
        return out
    
    def hill(self, x):
        a = torch.clamp(self.hill_a, 0.1, 3.0).view(1,1,-1)  # Reasonable alpha range
        g = torch.clamp(self.hill_g, 0.1, 2.0).view(1,1,-1)  # Reasonable gamma range for MMM
        num = x.clamp(min=0).pow(a)
        return num / (num + g.pow(a) + 1e-8)

    def forward(self, Xm, Xc, R):                      # all seq [B,T,*]
        B, T, _ = Xm.shape
        # causal node embeddings each week
        Z_seq = torch.stack([self.enc(x) for x in Xm.unbind(1)], 1)  # [B,T,10]
        
        # Apply transformations in correct order: Raw → Adstock → Hill
        # Step 1: Apply adstock transformation first (on original raw data)
        adstock_out = self.adstock(Xm)  # [B,T,n_media] - raw spending with carryover
        
        # Step 2 – channel-wise 0-1 scaling **per region**
        channel_max = adstock_out.amax(dim=1, keepdim=True).clamp_min(1e-6)
        adstock_norm = adstock_out / channel_max
        hill_out = self.hill(adstock_norm)              # [B,T,n_media]
        media_in = hill_out + Z_seq  # [B,T,n_media] - causal-adjusted
        
        gru_in = torch.cat([media_in, Xc], -1)  # [B,T,n_media+ctrl_dim]
        
        # Initialize GRU with learnable warm-start to fix week-0 spike
        h0 = self.h0.repeat(1, B, 1)  # Learnable initialization [1, B, hidden]
        h_seq, _ = self.gru(gru_in, h0)  # [B, T, hidden]
        
        # Generate time-varying coefficients directly from GRU hidden states
        # Let GRU learn channel-specific weights from scratch (no w_base, no shared time_factor)
        w_raw = self.w_raw(h_seq)  # [B,T,n_media] - raw coefficients from GRU
        
        # Drop shared time_factor to let GRU learn distinct patterns per channel
        w_pos = F.softplus(w_raw)  # [B,T,n_media] - always positive, channel-specific
        
        if T > 1:                                       # 50% blend
            w_pos[:,0,:] = 0.5 * w_pos[:,1,:].detach() + 0.5 * w_pos[:,0,:]
        
        # 3️⃣  Contributions must mirror the signal fed into the GRU
        media_contrib_scaled = hill_out * w_pos         # [B,T,n_media] in *scaled* space
        media_term = media_contrib_scaled.sum(-1)       # [B,T]  — drives ŷ
        ctrl_term = torch.relu(self.ctrl_mlp(Xc)).sum(-1) * 0.3
        reg_term = self.reg_emb(torch.zeros(B, dtype=torch.long)).sum(-1).unsqueeze(1).expand(-1, T) * 0.3
        y_scaled = media_term + ctrl_term + reg_term + self.bias
        return y_scaled, w_pos, media_contrib_scaled

def create_belief_vectors(df, control_vars):
    """Create belief vectors from control variables using Bayesian Network or fallback"""
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
            # Convert to DataFrame to preserve length and structure
            result_df = pd.DataFrame(Z_ctrl, columns=[f'belief_{i}' for i in range(Z_ctrl.shape[1])])
            return result_df, bn_struct
        except Exception as e:
            print(f"Bayesian Network failed: {e}, using fallback")
    
    # Fallback: use control variables directly
    if len(control_vars) > 0:
        Z_ctrl = df[control_vars].values
        # Convert to DataFrame to preserve length and structure
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

def create_media_adjacency(media_vars, bn_struct=None):
    """Create adjacency matrix for media variables"""
    n_media = len(media_vars)
    A_media = np.zeros((n_media, n_media))  # Use actual number of media variables
    
    if bn_struct and hasattr(bn_struct, 'edges'):
        # Use BN structure if available
        try:
            for u, v in bn_struct.edges():
                if u in media_vars and v in media_vars:
                    u_idx = media_vars.index(u)
                    v_idx = media_vars.index(v)
                    if u_idx < n_media and v_idx < n_media:
                        A_media[u_idx, v_idx] = 1.0
        except:
            pass
    
    # If no edges found, create simple chain structure
    if A_media.sum() == 0:
        for i in range(min(n_media - 1, n_media - 1)):
            A_media[i, i + 1] = 1.0
    
    return torch.tensor(A_media, dtype=torch.float32)

def prepare_data_for_training(df, params):
    """Prepare data for training with proper padding masks and scaling order"""
    
    burn_in = params.get("burn_in_weeks", 4)          # << NEW (default 4)
    df_work = df.copy()
    
    # Get actual marketing variables from user config (no padding/fake data)
    marketing_vars = params.get('marketing_vars', [])
    control_vars = params.get('control_vars', [])
    dependent_var = params.get('dependent_var', 'revenue')
    region_var = params.get('region_var', None)
    date_var = params.get('date_var', None)
    
    # Ensure we have the minimum required variables
    if not marketing_vars:
        # Auto-detect marketing variables
        marketing_vars = [col for col in df_work.columns 
                         if any(keyword in col.lower() for keyword in ['spend', 'media', 'tv', 'digital', 'radio', 'social'])]
    
    if not control_vars:
        # Auto-detect control variables
        control_vars = [col for col in df_work.columns 
                       if col not in marketing_vars + [dependent_var, region_var, date_var] 
                       and col not in ['date', 'week', 'region']
                       and df_work[col].dtype in ['int64', 'float64']]
    
    # Handle region creation/validation
    if region_var and region_var in df_work.columns:
        regions = df_work[region_var].unique()
        print(f"Using existing region column: {len(regions)} regions found")
    else:
        # Create single dummy region for modeling
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
            raise ValueError(f"Target variable '{dependent_var}' not found in data")
    
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
    
    # Truncate all regions to same length to avoid padding issues
    for region in regions:
        region_df_trimmed = region_data[region].iloc[:min_length]
        # ----------  BURN-IN PADDING  ----------
        if burn_in > 0:
            pad_block = region_df_trimmed.iloc[:1].copy()            # first week
            pad_block = pd.concat([pad_block]*burn_in, ignore_index=True)
            # jitter each media column slightly so gradients flow
            for col in marketing_vars:
                if col in pad_block.columns:
                    pad_block[col] += np.random.normal(0, 0.01, size=burn_in)
            region_df_trimmed = pd.concat([pad_block, region_df_trimmed],
                                          ignore_index=True)
        region_data[region] = region_df_trimmed
    
    # Create standardized feature matrices
    n_regions = len(regions)
    n_time_steps = min_length + burn_in  # Account for burn-in weeks
    
    # Use actual number of variables (no artificial padding)
    n_media_target = len(marketing_vars)
    n_control_target = max(len(control_vars), 1)  # At least 1 control variable
    
    # Media variables - use actual data only (no artificial padding)
    media_matrix = np.zeros((n_regions, n_time_steps, n_media_target))
    for i, region in enumerate(regions):
        region_df = region_data[region]
        for j, var in enumerate(marketing_vars):
            if var in region_df.columns:
                media_matrix[i, :, j] = region_df[var].values
    
    # Control variables - use actual data only (no artificial padding)
    control_matrix = np.zeros((n_regions, n_time_steps, n_control_target))
    for i, region in enumerate(regions):
        region_df = region_data[region]
        for j, var in enumerate(control_vars):
            if var in region_df.columns:
                control_matrix[i, :, j] = region_df[var].values
        
        # If no control variables, create a simple constant (intercept)
        if len(control_vars) == 0:
            control_matrix[i, :, 0] = 1.0  # Intercept term
    
    # Target variable
    y_matrix = np.zeros((n_regions, n_time_steps))
    for i, region in enumerate(regions):
        region_df = region_data[region]
        y_matrix[i, :] = region_df[dependent_var].values
    
    # Create region IDs
    region_ids = np.arange(n_regions)
    
    # Scale the data to proper ranges for model training
    from sklearn.preprocessing import MinMaxScaler
    
    # Scale media variables for Hill transformation
    media_scaler = MinMaxScaler()
    media_matrix_flat = media_matrix.reshape(-1, n_media_target)
    media_matrix_scaled = media_scaler.fit_transform(media_matrix_flat)
    media_matrix_scaled = media_matrix_scaled.reshape(n_regions, n_time_steps, n_media_target)
    
    # Scale control variables
    control_scaler = MinMaxScaler()
    control_matrix_flat = control_matrix.reshape(-1, n_control_target)
    control_matrix_scaled = control_scaler.fit_transform(control_matrix_flat)
    control_matrix_scaled = control_matrix_scaled.reshape(n_regions, n_time_steps, n_control_target)
    
    # Scale target variable
    y_scaler = MinMaxScaler()
    y_matrix_flat = y_matrix.reshape(-1, 1)
    y_matrix_scaled = y_scaler.fit_transform(y_matrix_flat)
    y_matrix_scaled = y_matrix_scaled.reshape(n_regions, n_time_steps)
    
    # Convert to tensors - all data properly scaled
    X_m = torch.tensor(media_matrix_scaled, dtype=torch.float32)  # Scaled for Hill transformation
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
    
    # Return scalers as well for proper inverse transformations
    result = {
        'X_m': X_m,
        'X_c': X_c,
        'R': R,
        'y': y,
        'burn_in': burn_in,
        'media_scaler': media_scaler,
        'control_scaler': control_scaler,
        'y_scaler': y_scaler
    }
    
    return result

def main():
    if len(sys.argv) != 3:
        print("Usage: python causal_gru_mmm.py <csv_file> <params_json>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    params_json = sys.argv[2]
    
    # Load parameters
    with open(params_json, 'r') as f:
        params = json.load(f)
    
    # Set seed
    SEED = params.get('seed', 42)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    
    # Load data
    df = pd.read_csv(csv_file)
    print(f"Loaded data with shape: {df.shape}")
    
    # Prepare data following the exact structure
    data_dict = prepare_data_for_training(df, params)
    MEDIA_VARS = params.get('marketing_vars', [])
    CONTROL_VARS = params.get('control_vars', [])
    
    # Encode regions exactly as in original code
    le_region = LabelEncoder()
    df["region_id"] = le_region.fit_transform(df["region"])
    N_REGIONS = len(le_region.classes_)
    
    # Create belief vectors exactly as in original code
    Z_ctrl, bn_struct = create_belief_vectors(df, CONTROL_VARS)
    
    # Create media adjacency matrix exactly as in original code  
    A_media = create_media_adjacency(MEDIA_VARS, bn_struct)
    
    # Train/test split exactly as in original code (last 20 weeks = test)
    train_cut = df["week"].max() - 20
    train_mask = df["week"] <= train_cut
    test_mask = ~train_mask
    
    # Scale data exactly as in original code
    m_scaler = StandardScaler().fit(df.loc[train_mask, MEDIA_VARS])
    c_scaler = StandardScaler().fit(Z_ctrl[train_mask])
    y_scaler = StandardScaler().fit(df.loc[train_mask, ["sales"]])
    
    Xm = m_scaler.transform(df[MEDIA_VARS])
    Xc = c_scaler.transform(Z_ctrl)
    Y = y_scaler.transform(df[["sales"]]).ravel()
    R = df["region_id"].values
    
    # Convert to sequences exactly as in original code
    def to_seq(arr):
        """stack rows into [B=regions, T=weeks, ...]"""
        return np.stack([arr[df["region_id"]==r] for r in range(N_REGIONS)], 0)
    
    Xm_seq = torch.tensor(to_seq(Xm), dtype=torch.float32)
    Xc_seq = torch.tensor(to_seq(Xc), dtype=torch.float32)
    Y_seq = torch.tensor(to_seq(Y.reshape(-1,1))[:,:,0], dtype=torch.float32)
    R_ids = torch.arange(N_REGIONS)
    
    T_train = train_cut + 1
    Xm_tr, Xm_te = Xm_seq[:,:T_train], Xm_seq[:,T_train:]
    Xc_tr, Xc_te = Xc_seq[:,:T_train], Xc_seq[:,T_train:]
    Y_tr, Y_te = Y_seq[:,:T_train], Y_seq[:,T_train:]
    
    # Initialize model with dynamic dimensions
    n_media = len(MEDIA_VARS)
    n_control = len(CONTROL_VARS) if len(CONTROL_VARS) > 0 else 1
    hidden_size = params.get('hidden_size', 64)
    model = GRUCausalMMM(A_media, n_media=n_media, ctrl_dim=n_control, hidden=hidden_size, n_regions=N_REGIONS)
    
    # Training setup exactly as in original code
    learning_rate = params.get('learning_rate', 1e-3)
    epochs = params.get('epochs', 10000)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lossf = nn.MSELoss()
    
    # Training loop exactly as in original code
    train_losses = []
    print(f"Training for {epochs} epochs...")
    
    for ep in range(epochs):
        model.train()
        y_hat, _, _ = model(Xm_tr, Xc_tr, R_ids)
        loss = lossf(y_hat, Y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        train_losses.append(loss.item())
        
        if ep % 100 == 0:
            print(f"ep{ep:3d}  loss {loss.item():.4f}")
    
    # Inference exactly as in original code
    model.eval()
    with torch.no_grad():
        y_scaled_te, w_te, contrib_te = model(Xm_te, Xc_te, R_ids)
        y_pred = y_scaler.inverse_transform(y_scaled_te.cpu().numpy().reshape(-1,1)).reshape(N_REGIONS, -1)
        y_test_actual = y_scaler.inverse_transform(Y_te.cpu().numpy().reshape(-1,1)).reshape(N_REGIONS, -1)
        
        print("Sample forecast (Region-0 last 5 weeks):", np.round(y_pred[0,-5:], 3))
        print("β_media_1(t) Region-0:", np.round(w_te[0,:,0].cpu().numpy()[:10], 3), "...")
    
    # Calculate metrics from actual model outputs
    mse = mean_squared_error(y_test_actual.flatten(), y_pred.flatten())
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_actual.flatten(), y_pred.flatten())
    mape = np.mean(np.abs((y_test_actual.flatten() - y_pred.flatten()) / (y_test_actual.flatten() + 1e-8))) * 100
    
    # Create output directory
    output_dir = 'ml_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations
    # Training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig(f'{output_dir}/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Coefficient trajectory plot exactly as in original code
    plt.figure(figsize=(10, 6))
    plt.plot(w_te[0,:,0].cpu())
    plt.title("β_media_1 over time (Region-0)")
    plt.xlabel("Week")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.savefig(f'{output_dir}/coefficient_trajectories.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate feature importance from model coefficients
    media_importance = {}
    w_mean = w_te.mean(dim=1).mean(dim=0).cpu().numpy()  # Average across time and regions
    total_weight = w_mean.sum()
    for i in range(len(MEDIA_VARS)):
        media_importance[MEDIA_VARS[i]] = float(w_mean[i] / total_weight) if total_weight > 0 else 0.0
    
    # Extract learned parameters from model
    model_coefficients = {
        'adstock_params': model.alpha.detach().cpu().numpy().tolist(),
        'saturation_params': {
            'hill_a': model.hill_a.detach().cpu().numpy().tolist(),
            'hill_g': model.hill_g.detach().cpu().numpy().tolist()
        },
        'media_weights_sample': w_te[0,:5,0].detach().cpu().numpy().tolist()  # First 5 weeks, first media
    }
    
    # Prepare results with actual model outputs
    results = {
        'model_performance': {
            'mse': float(mse),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'mape': float(mape)
        },
        'training_params': params,
        'feature_importance': media_importance,
        'training_history': {
            'epochs': list(range(len(train_losses))),
            'train_loss': train_losses
        },
        'model_coefficients': model_coefficients,
        'data_summary': {
            'total_samples': len(df),
            'training_samples': int(len(df[train_mask])),
            'test_samples': int(len(df[test_mask])),
            'n_regions': N_REGIONS,
            'media_channels': MEDIA_VARS,
            'control_variables': len(CONTROL_VARS)
        },
        'forecast_sample': {
            'region_0_last_5_weeks': y_pred[0, -5:].tolist() if y_pred.shape[1] >= 5 else y_pred[0].tolist()
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