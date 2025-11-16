"""
Budget Optimization for Marketing Mix Modeling.

This module provides budget optimization capabilities using response curves
from DeepCausalMMM. It uses constrained optimization to find the optimal
allocation of marketing budget across channels to maximize predicted response.

The optimizer uses Hill equation saturation curves fitted by ResponseCurveFit
to predict channel responses at different spend levels and finds the allocation
that maximizes total response subject to budget and business constraints.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
import scipy.optimize as op

logger = logging.getLogger('deepcausalmmm')


@dataclass
class OptimizationResult:
    """
    Result from budget optimization.
    
    Attributes
    ----------
    success : bool
        Whether optimization converged successfully
    allocation : Dict[str, float]
        Optimal spend allocation by channel
    predicted_response : float
        Total predicted response at optimal allocation
    by_channel : pd.DataFrame
        Detailed results by channel with spend, response, and ROI
    message : str
        Optimization status message
    method : str
        Optimization method used
    
    Examples
    --------
    >>> result = optimizer.optimize()
    >>> if result.success:
    ...     print(f"Optimal allocation: {result.allocation}")
    ...     print(f"Expected response: {result.predicted_response:,.0f}")
    ...     print(result.by_channel)
    """
    success: bool
    allocation: Dict[str, float]
    predicted_response: float
    by_channel: pd.DataFrame
    message: str = ""
    method: str = "trust-constr"


class BudgetOptimizer:
    """
    Optimize marketing budget allocation using response curves.
    
    Uses constrained optimization (trust-constr, SLSQP, or differential evolution)
    with Hill transformation curves from ResponseCurveFit to find optimal spend
    allocation that maximizes total response subject to business constraints.
    
    Parameters
    ----------
    budget : float
        Total budget to allocate across all channels
    channels : List[str]
        List of channel names to include in optimization
    response_curves : Dict[str, Dict]
        Response curve parameters by channel from ResponseCurveFit.
        Each channel dict should contain: 'top', 'bottom', 'saturation', 'slope'
    num_weeks : int, default=52
        Number of weeks for planning horizon (annual by default)
    method : str, default='trust-constr'
        Optimization method: 'trust-constr', 'SLSQP', 'differential_evolution', 'hybrid'
        
    Attributes
    ----------
    constraints_df : pd.DataFrame or None
        DataFrame with channel-level constraints (lower, upper bounds)
        
    Examples
    --------
    >>> # After fitting response curves with ResponseCurveFit
    >>> curves = {
    ...     'TV': {'top': 1000000, 'bottom': 0, 'saturation': 50000, 'slope': 1.5},
    ...     'Search': {'top': 800000, 'bottom': 0, 'saturation': 30000, 'slope': 2.0},
    ...     'Social': {'top': 600000, 'bottom': 0, 'saturation': 20000, 'slope': 1.8}
    ... }
    >>> 
    >>> optimizer = BudgetOptimizer(
    ...     budget=1000000,
    ...     channels=['TV', 'Search', 'Social'],
    ...     response_curves=curves,
    ...     num_weeks=52
    ... )
    >>> 
    >>> # Optional: Set channel-specific constraints
    >>> optimizer.set_constraints({
    ...     'TV': {'lower': 50000, 'upper': 500000},
    ...     'Search': {'lower': 100000, 'upper': 400000}
    ... })
    >>> 
    >>> # Run optimization
    >>> result = optimizer.optimize()
    >>> 
    >>> # View results
    >>> if result.success:
    ...     print("Optimal Allocation:")
    ...     for channel, spend in result.allocation.items():
    ...         print(f"  {channel}: ${spend:,.0f}")
    ...     print(f"\\nTotal Response: {result.predicted_response:,.0f}")
    ...     print(f"\\nDetailed Results:\\n{result.by_channel}")
    
    Notes
    -----
    The optimizer maximizes total response using the Hill equation:
    
    .. math::
        response = bottom + (top - bottom) * \\frac{spend^{slope}}{saturation^{slope} + spend^{slope}}
    
    Where:
    - `top`: Maximum response (saturation level)
    - `bottom`: Minimum response (typically 0)
    - `saturation`: Spend level at half-maximum response
    - `slope`: Steepness of the response curve
    
    The optimization problem is:
    
    .. math::
        \\max_{x_1, ..., x_n} \\sum_{i=1}^{n} response_i(x_i)
        
        s.t. \\sum_{i=1}^{n} x_i = budget
        
        lower_i \\leq x_i \\leq upper_i \\quad \\forall i
    """
    
    def __init__(
        self,
        budget: float,
        channels: List[str],
        response_curves: Dict[str, Dict],
        *,
        num_weeks: int = 52,
        method: str = 'trust-constr'
    ):
        """Initialize BudgetOptimizer with budget, channels, and response curves."""
        self.budget = budget
        self.channels = channels
        self.response_curves = response_curves
        self.num_weeks = num_weeks
        self.method = method
        
        # Validate method
        valid_methods = ['trust-constr', 'SLSQP', 'differential_evolution', 'hybrid']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got '{method}'")
        
        # Validate channels have curves
        missing_curves = [ch for ch in channels if ch not in response_curves]
        if missing_curves:
            raise ValueError(f"Missing response curves for channels: {missing_curves}")
        
        # Validate response curve parameters
        required_params = ['top', 'bottom', 'saturation', 'slope']
        for channel in channels:
            missing_params = [p for p in required_params if p not in response_curves[channel]]
            if missing_params:
                raise ValueError(
                    f"Channel '{channel}' missing required parameters: {missing_params}"
                )
        
        # Initialize constraints
        self.constraints_df: Optional[pd.DataFrame] = None
        
        logger.info(
            f"Initialized BudgetOptimizer: "
            f"Budget=${budget:,.0f}, Channels={len(channels)}, Weeks={num_weeks}, Method={method}"
        )
    
    def set_constraints(self, constraints: Dict[str, Dict[str, float]]) -> None:
        """
        Set spend constraints for channels.
        
        Parameters
        ----------
        constraints : Dict[str, Dict[str, float]]
            Channel constraints: {'channel': {'lower': min_spend, 'upper': max_spend}}
            
        Examples
        --------
        >>> optimizer.set_constraints({
        ...     'TV': {'lower': 50000, 'upper': 500000},
        ...     'Search': {'lower': 100000, 'upper': 400000},
        ...     'Social': {'lower': 25000, 'upper': 300000}
        ... })
        
        Notes
        -----
        - Channels not specified in constraints get default bounds: [0, budget]
        - Upper bounds are automatically capped at total budget
        - If lower > upper, lower is reset to 0
        - Upper bounds cannot be 0 (would make channel unusable)
        """
        constraints_list = []
        for channel in self.channels:
            if channel in constraints:
                constraints_list.append({
                    'channel': channel,
                    'lower': constraints[channel].get('lower', 0),
                    'upper': constraints[channel].get('upper', self.budget)
                })
            else:
                # Default: no minimum, max is total budget
                constraints_list.append({
                    'channel': channel,
                    'lower': 0,
                    'upper': self.budget
                })
        
        self.constraints_df = pd.DataFrame(constraints_list)
        
        # Validate constraints
        invalid = self.constraints_df[self.constraints_df['upper'] == 0]
        if len(invalid) > 0:
            raise ValueError(
                f"Upper constraints cannot be 0 for channels: {invalid['channel'].tolist()}"
            )
        
        # Ensure upper doesn't exceed budget
        self.constraints_df['upper'] = np.minimum(self.constraints_df['upper'], self.budget)
        
        # Ensure lower doesn't exceed upper
        self.constraints_df['lower'] = np.where(
            self.constraints_df['lower'] > self.constraints_df['upper'],
            0,
            self.constraints_df['lower']
        )
        
        logger.info(f"Set constraints for {len(self.channels)} channels")
    
    def _predict_response(self, spend: float, channel: str) -> float:
        """
        Predict response using Hill equation.
        
        Parameters
        ----------
        spend : float
            Weekly spend amount
        channel : str
            Channel name
            
        Returns
        -------
        float
            Predicted response (weekly)
        """
        params = self.response_curves[channel]
        bottom = params['bottom']
        top = params['top']
        saturation = params['saturation']
        slope = params['slope']
        
        # Hill equation with safe exponentiation
        try:
            spend_pow = spend ** slope
            sat_pow = saturation ** slope
        except (OverflowError, RuntimeWarning):
            # Handle overflow with clipping
            spend_pow = np.clip(spend ** slope, 0, 1e100)
            sat_pow = np.clip(saturation ** slope, 0, 1e100)
        
        response = bottom + (top - bottom) * spend_pow / (sat_pow + spend_pow)
        return float(response)
    
    def _objective(self, x: np.ndarray) -> float:
        """
        Objective function to minimize (negative of total response).
        
        Parameters
        ----------
        x : np.ndarray
            Array of total spend allocations (one per channel)
            
        Returns
        -------
        float
            Negative total response (for minimization)
        """
        # x contains total spend per channel, convert to weekly
        weekly_spend = x / self.num_weeks
        
        # Calculate total response across all channels
        total_response = 0.0
        for i, channel in enumerate(self.channels):
            weekly_response = self._predict_response(weekly_spend[i], channel)
            total_response += weekly_response * self.num_weeks
        
        # Return negative (we minimize, but want to maximize response)
        return -total_response
    
    def _constraint_budget(self, x: np.ndarray) -> float:
        """
        Equality constraint: sum of spend equals budget.
        
        Parameters
        ----------
        x : np.ndarray
            Array of spend allocations
            
        Returns
        -------
        float
            Difference from budget (should be 0 at optimum)
        """
        return float(np.sum(x) - self.budget)
    
    def _get_bounds(self) -> List[Tuple[float, float]]:
        """
        Get bounds for optimization.
        
        Returns
        -------
        List[Tuple[float, float]]
            List of (lower, upper) bounds for each channel
        """
        if self.constraints_df is None:
            # Default: no minimum, max is total budget
            return [(0.0, float(self.budget)) for _ in self.channels]
        
        bounds = []
        for channel in self.channels:
            row = self.constraints_df[self.constraints_df['channel'] == channel]
            if len(row) > 0:
                bounds.append((float(row['lower'].iloc[0]), float(row['upper'].iloc[0])))
            else:
                bounds.append((0.0, float(self.budget)))
        
        return bounds
    
    def _get_initial_guess(self) -> np.ndarray:
        """
        Get initial guess for optimization.
        
        Uses equal allocation as starting point, adjusted to respect bounds.
        
        Returns
        -------
        np.ndarray
            Initial spend allocation
        """
        # Start with equal allocation
        x0 = np.full(len(self.channels), self.budget / len(self.channels))
        
        # Adjust to respect bounds
        bounds = self._get_bounds()
        for i, (lower, upper) in enumerate(bounds):
            x0[i] = np.clip(x0[i], lower, upper)
        
        # Normalize to match budget exactly
        if np.sum(x0) > 0:
            x0 = x0 * (self.budget / np.sum(x0))
        
        return x0
    
    def optimize(self) -> OptimizationResult:
        """
        Run optimization to find optimal budget allocation.
        
        Returns
        -------
        OptimizationResult
            Optimization results including allocation, predicted response, and details
            
        Examples
        --------
        >>> result = optimizer.optimize()
        >>> if result.success:
        ...     print("Optimization successful!")
        ...     print(f"Predicted response: {result.predicted_response:,.0f}")
        ...     for channel, spend in result.allocation.items():
        ...         roi = result.by_channel[result.by_channel['channel']==channel]['roi'].iloc[0]
        ...         print(f"{channel}: ${spend:,.0f} (ROI: {roi:.2f})")
        ... else:
        ...     print(f"Optimization failed: {result.message}")
        """
        logger.info(f"Starting budget optimization with method={self.method}...")
        
        # Get initial guess, bounds, and constraints
        x0 = self._get_initial_guess()
        bounds = self._get_bounds()
        constraints = {'type': 'eq', 'fun': self._constraint_budget}
        
        # Run optimization based on method
        if self.method == 'differential_evolution':
            result = self._optimize_global(bounds, constraints)
        elif self.method == 'hybrid':
            result = self._optimize_hybrid(bounds, constraints, x0)
        else:
            result = self._optimize_gradient(x0, bounds, constraints, self.method)
        
        if result.success:
            logger.info(f"Optimization converged: {result.message}")
            
            # Extract results
            allocation = {ch: float(spend) for ch, spend in zip(self.channels, result.x)}
            predicted_response = -result.fun  # Negate back to positive
            
            # Build detailed results by channel
            by_channel = self._calculate_by_channel(result.x)
            
            return OptimizationResult(
                success=True,
                allocation=allocation,
                predicted_response=float(predicted_response),
                by_channel=by_channel,
                message=result.message,
                method=self.method
            )
        else:
            logger.error(f"Optimization failed: {result.message}")
            
            return OptimizationResult(
                success=False,
                allocation={ch: 0.0 for ch in self.channels},
                predicted_response=0.0,
                by_channel=pd.DataFrame(),
                message=result.message,
                method=self.method
            )
    
    def _optimize_gradient(
        self,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
        constraints: Dict,
        method: str
    ):
        """Gradient-based optimization (trust-constr or SLSQP)."""
        if method == 'trust-constr':
            # trust-constr: More robust than SLSQP, handles constraints better
            return op.minimize(
                fun=self._objective,
                x0=x0,
                method='trust-constr',
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': 500,
                    'verbose': 1,
                    'gtol': 1e-8,
                    'xtol': 1e-10
                }
            )
        else:  # SLSQP
            return op.minimize(
                fun=self._objective,
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': 400,
                    'disp': True,
                    'ftol': 1e-9
                },
                jac='3-point'
            )
    
    def _optimize_global(self, bounds: List[Tuple[float, float]], constraints: Dict):
        """Global optimization using differential evolution."""
        from scipy.optimize import differential_evolution
        
        logger.info("Running global optimization (may take longer)...")
        return differential_evolution(
            func=self._objective,
            bounds=bounds,
            constraints=constraints,
            strategy='best1bin',
            maxiter=1000,
            popsize=15,
            tol=0.01,
            atol=0,
            seed=42,
            polish=True,  # Local refinement at end
            workers=1
        )
    
    def _optimize_hybrid(
        self,
        bounds: List[Tuple[float, float]],
        constraints: Dict,
        x0: np.ndarray
    ):
        """Hybrid: global search followed by local refinement."""
        from scipy.optimize import differential_evolution
        
        logger.info("Running hybrid optimization (global + local)...")
        
        # Stage 1: Quick global search
        logger.info("Stage 1: Global search...")
        global_result = differential_evolution(
            func=self._objective,
            bounds=bounds,
            constraints=constraints,
            maxiter=100,  # Quick scan
            popsize=10,
            polish=False,
            workers=1
        )
        
        # Stage 2: Local refinement from global optimum
        logger.info("Stage 2: Local refinement...")
        local_result = op.minimize(
            fun=self._objective,
            x0=global_result.x,
            method='trust-constr',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 300, 'verbose': 1}
        )
        
        return local_result
    
    def _calculate_by_channel(self, optimal_spend: np.ndarray) -> pd.DataFrame:
        """
        Calculate detailed metrics by channel.
        
        Parameters
        ----------
        optimal_spend : np.ndarray
            Optimal spend allocation (total over planning horizon)
            
        Returns
        -------
        pd.DataFrame
            Results by channel with spend, response, ROI, and saturation metrics
        """
        results = []
        
        weekly_spend = optimal_spend / self.num_weeks
        
        for i, channel in enumerate(self.channels):
            weekly_response = self._predict_response(weekly_spend[i], channel)
            total_response = weekly_response * self.num_weeks
            
            params = self.response_curves[channel]
            
            # Calculate saturation percentage
            saturation_pct = (weekly_spend[i] / params['saturation']) * 100 if params['saturation'] > 0 else 0
            
            results.append({
                'channel': channel,
                'total_spend': optimal_spend[i],
                'weekly_spend': weekly_spend[i],
                'weekly_response': weekly_response,
                'total_response': total_response,
                'roi': total_response / optimal_spend[i] if optimal_spend[i] > 0 else 0,
                'spend_pct': (optimal_spend[i] / self.budget) * 100,
                'response_pct': 0,  # Will calculate after summing
                'saturation_point': params['saturation'],
                'saturation_pct': saturation_pct,
                'slope': params['slope']
            })
        
        df = pd.DataFrame(results)
        
        # Calculate response percentage
        total_resp = df['total_response'].sum()
        if total_resp > 0:
            df['response_pct'] = (df['total_response'] / total_resp) * 100
        
        # Sort by total spend descending
        df = df.sort_values('total_spend', ascending=False).reset_index(drop=True)
        
        return df
    
    def compare_scenarios(
        self,
        scenarios: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Compare different budget allocation scenarios.
        
        Parameters
        ----------
        scenarios : Dict[str, Dict[str, float]]
            Dictionary of scenarios: {'scenario_name': {'channel': spend, ...}}
            
        Returns
        -------
        pd.DataFrame
            Comparison of scenarios with predicted responses and ROIs
            
        Examples
        --------
        >>> scenarios = {
        ...     'Current': {'TV': 400000, 'Search': 350000, 'Social': 250000},
        ...     'Optimized': result.allocation,
        ...     'Heavy TV': {'TV': 600000, 'Search': 250000, 'Social': 150000}
        ... }
        >>> comparison = optimizer.compare_scenarios(scenarios)
        >>> print(comparison)
        """
        comparison = []
        
        for scenario_name, allocation in scenarios.items():
            # Validate allocation
            if set(allocation.keys()) != set(self.channels):
                logger.warning(
                    f"Scenario '{scenario_name}' has different channels, skipping"
                )
                continue
            
            total_spend = sum(allocation.values())
            total_response = 0.0
            
            # Calculate response for this allocation
            for channel, spend in allocation.items():
                weekly_spend = spend / self.num_weeks
                weekly_response = self._predict_response(weekly_spend, channel)
                total_response += weekly_response * self.num_weeks
            
            comparison.append({
                'scenario': scenario_name,
                'total_spend': total_spend,
                'total_response': total_response,
                'roi': total_response / total_spend if total_spend > 0 else 0,
                **allocation
            })
        
        df = pd.DataFrame(comparison)
        return df


def optimize_budget_from_curves(
    budget: float,
    curve_params: pd.DataFrame,
    *,
    channel_col: str = 'channel',
    num_weeks: int = 52,
    constraints: Optional[Dict[str, Dict[str, float]]] = None,
    method: str = 'trust-constr'
) -> OptimizationResult:
    """
    Convenience function to optimize budget directly from curve parameters DataFrame.
    
    This function is useful when you have response curve parameters in a DataFrame
    (e.g., from ResponseCurveFit fitted on multiple channels) and want to quickly
    run optimization without manually setting up the BudgetOptimizer.
    
    Parameters
    ----------
    budget : float
        Total budget to allocate
    curve_params : pd.DataFrame
        DataFrame with response curve parameters.
        Required columns: channel, top, bottom, saturation, slope
    channel_col : str, default='channel'
        Name of the channel column in curve_params
    num_weeks : int, default=52
        Number of weeks for planning horizon
    constraints : Dict[str, Dict[str, float]], optional
        Channel-specific constraints: {'channel': {'lower': min, 'upper': max}}
    method : str, default='trust-constr'
        Optimization method
        
    Returns
    -------
    OptimizationResult
        Optimization results
        
    Examples
    --------
    >>> # After fitting curves for multiple channels
    >>> curves_df = pd.DataFrame({
    ...     'channel': ['TV', 'Search', 'Social'],
    ...     'top': [1000000, 800000, 600000],
    ...     'bottom': [0, 0, 0],
    ...     'saturation': [50000, 30000, 20000],
    ...     'slope': [1.5, 2.0, 1.8]
    ... })
    >>> 
    >>> result = optimize_budget_from_curves(
    ...     budget=1000000,
    ...     curve_params=curves_df,
    ...     constraints={'TV': {'lower': 100000, 'upper': 600000}}
    ... )
    >>> print(result.allocation)
    """
    # Validate required columns
    required_cols = [channel_col, 'top', 'bottom', 'saturation', 'slope']
    missing_cols = [col for col in required_cols if col not in curve_params.columns]
    if missing_cols:
        raise ValueError(f"curve_params missing required columns: {missing_cols}")
    
    # Convert DataFrame to dictionary format
    channels = curve_params[channel_col].tolist()
    response_curves = {}
    
    for _, row in curve_params.iterrows():
        channel = row[channel_col]
        response_curves[channel] = {
            'top': row['top'],
            'bottom': row['bottom'],
            'saturation': row['saturation'],
            'slope': row['slope']
        }
    
    # Create optimizer
    optimizer = BudgetOptimizer(
        budget=budget,
        channels=channels,
        response_curves=response_curves,
        num_weeks=num_weeks,
        method=method
    )
    
    # Set constraints if provided
    if constraints:
        optimizer.set_constraints(constraints)
    
    # Run optimization
    return optimizer.optimize()

