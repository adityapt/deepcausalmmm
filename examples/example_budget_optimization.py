#!/usr/bin/env python3
"""
Budget Optimization Example
============================

This example demonstrates how to optimize marketing budget allocation
using response curves fitted from a trained DeepCausalMMM model.

Steps:
1. Fit response curves from your trained model's output
2. Use optimize_budget_from_curves() to find optimal allocation
3. Apply constraints based on business requirements
"""

import pandas as pd
from deepcausalmmm import optimize_budget_from_curves

# ============================================================================
# Step 1: Fit Response Curves from Your Model
# ============================================================================
# After training your DeepCausalMMM model, use ResponseCurveFitter to fit
# Hill equation curves for each channel. This gives you a DataFrame with:
# - channel: Channel name
# - top: Maximum response (saturation level)
# - bottom: Minimum response (typically 0)
# - saturation: Spend level at half-maximum response
# - slope: Curve steepness (elasticity)

# Example: Response curves from a trained model
fitted_curves = pd.DataFrame([
    {'channel': 'TV', 'top': 500000, 'bottom': 0, 'saturation': 300000, 'slope': 1.5},
    {'channel': 'Search', 'top': 400000, 'bottom': 0, 'saturation': 250000, 'slope': 1.3},
    {'channel': 'Social', 'top': 300000, 'bottom': 0, 'saturation': 200000, 'slope': 1.2},
    {'channel': 'Display', 'top': 250000, 'bottom': 0, 'saturation': 180000, 'slope': 1.4},
    {'channel': 'Email', 'top': 150000, 'bottom': 0, 'saturation': 100000, 'slope': 1.1}
])

print("=" * 80)
print("BUDGET OPTIMIZATION EXAMPLE")
print("=" * 80)
print("\nFitted Response Curves:")
print(fitted_curves)

# ============================================================================
# Step 2: Set Budget and Constraints
# ============================================================================

# Total budget to allocate
total_budget = 1_000_000

# Planning horizon (weeks)
num_weeks = 52

# Channel-specific constraints (optional but recommended)
# These should be based on business requirements, historical bounds, etc.
constraints = {
    'TV': {'lower': 100000, 'upper': 600000},        # TV must be between 100K-600K
    'Search': {'lower': 150000, 'upper': 500000},    # Search must be between 150K-500K
    'Social': {'lower': 50000, 'upper': 300000},     # Social must be between 50K-300K
    'Display': {'lower': 30000, 'upper': 250000},    # Display must be between 30K-250K
    'Email': {'lower': 20000, 'upper': 150000}       # Email must be between 20K-150K
}

print(f"\nTotal Budget: ${total_budget:,.0f}")
print(f"Planning Horizon: {num_weeks} weeks")
print("\nConstraints:")
for channel, bounds in constraints.items():
    print(f"  {channel}: ${bounds['lower']:>10,} - ${bounds['upper']:>10,}")

# ============================================================================
# Step 3: Run Optimization
# ============================================================================
print("\n" + "=" * 80)
print("Running optimization...")
print("=" * 80)

result = optimize_budget_from_curves(
    budget=total_budget,
    curve_params=fitted_curves,
    num_weeks=num_weeks,
    constraints=constraints,
    method='SLSQP'  # Options: 'SLSQP', 'trust-constr', 'differential_evolution', 'hybrid'
)

# ============================================================================
# Step 4: View Results
# ============================================================================

if result.success:
    print("\nOptimization Successful!")
    print(f"\nOptimization Method: {result.method}")
    print(f"Predicted Total Response: {result.predicted_response:,.0f}")
    
    print("\n" + "=" * 80)
    print("OPTIMAL BUDGET ALLOCATION")
    print("=" * 80)
    
    # Show allocation
    print("\nChannel Allocation:")
    for channel, spend in sorted(result.allocation.items(), key=lambda x: x[1], reverse=True):
        pct = (spend / total_budget) * 100
        print(f"  {channel:.<20s} ${spend:>12,.0f} ({pct:>5.1f}%)")
    
    # Show detailed metrics
    print("\n" + "=" * 80)
    print("DETAILED CHANNEL METRICS")
    print("=" * 80)
    print(result.by_channel.to_string(index=False))
    
    # Save results
    output_file = 'optimization_results.csv'
    result.by_channel.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
else:
    print(f"\nOptimization Failed")
    print(f"Message: {result.message}")

# ============================================================================
# Advanced: Compare with Current Allocation
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON: CURRENT VS OPTIMAL")
print("=" * 80)

# Example current allocation
current_allocation = {
    'TV': 400000,
    'Search': 350000,
    'Social': 150000,
    'Display': 70000,
    'Email': 30000
}

print("\nCurrent Allocation:")
for channel, spend in current_allocation.items():
    print(f"  {channel}: ${spend:,.0f}")

print("\nOptimal Allocation:")
for channel, spend in result.allocation.items():
    current = current_allocation.get(channel, 0)
    change = spend - current
    change_pct = (change / current * 100) if current > 0 else 0
    print(f"  {channel}: ${spend:,.0f} (Change: ${change:+,.0f}, {change_pct:+.1f}%)")

print("\n" + "=" * 80)

# ============================================================================
# Tips for Your Own Data
# ============================================================================
print("\nTIPS FOR YOUR OWN DATA:")
print("-" * 80)
print("""
1. Fit Response Curves:
   After training your DeepCausalMMM model, use ResponseCurveFitter to
   fit Hill equation curves for each channel.

2. Set Realistic Constraints:
   Base your constraints on:
   - Historical spending patterns (e.g., 0.5x to 2x historical range)
   - Business requirements (minimum brand spend, contract commitments)
   - Operational limits (team capacity, media availability)

3. Validate Results:
   - Check saturation_pct: values >100% indicate diminishing returns
   - Review ROI: ensure it aligns with expectations
   - Compare with current allocation to understand changes

4. Iterate:
   - Try different optimization methods
   - Adjust constraints based on business feedback
   - Run sensitivity analysis with different budget levels
""")
print("=" * 80)

