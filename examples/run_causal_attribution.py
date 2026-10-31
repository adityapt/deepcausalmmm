"""Fit the post-fit conditional-NOTEARS causal attribution model on MMM Data.

This script does not train or alter the GRU MMM.  It emits a machine-readable
report of model-based direct, mediated, and total intervention effects.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dashboard_rmse_optimized import load_real_mmm_data
from deepcausalmmm.core.causal_attribution import NonlinearStructuralCausalAttributor
from deepcausalmmm.core.conditional_notears import ConditionalNotears
from deepcausalmmm.core.causal_validation import CausalValidationSuite
from deepcausalmmm.core.causal_visualization import render_constrained_causal_dag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=0,
        help="Blocked-time NOTEARS refits for edge stability. Use 100+ for a release diagnostic; 0 skips it.",
    )
    parser.add_argument(
        "--bootstrap-block-size",
        type=int,
        default=None,
        help="Weeks per resampled contiguous block (default: sqrt(number of weeks)).",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=42, help="Random seed for blocked bootstrap refits.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_repetitions < 0:
        raise ValueError("--bootstrap-repetitions must be non-negative")
    media, controls, outcome, names, control_names, dma_ids, week_ids = load_real_mmm_data(return_panel_ids=True)
    dag = ConditionalNotears(
        lambda1=0.01, max_outer=15, max_iter=300, h_tolerance=1e-6
    ).fit(media, controls, names)
    estimate = NonlinearStructuralCausalAttributor().fit(media, controls, outcome, dag, threshold=0.15)
    validation = CausalValidationSuite().run(
        media,
        controls,
        outcome,
        dag,
        threshold=0.15,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_block_size=args.bootstrap_block_size,
        bootstrap_seed=args.bootstrap_seed,
    )
    attribution = estimate.attribution
    intervention_weights, _ = dag._thresholded_order(0.15, positive_only=True)
    rows = [
        {
            "channel": name,
            "direct_visits": float(attribution.direct[index]),
            "mediated_visits": float(attribution.mediated[index]),
            "total_visits": float(attribution.total[index]),
            "positive_total_share_pct": float(attribution.positive_total_share[index] * 100),
        }
        for index, name in enumerate(attribution.channel_names)
    ]
    rows.sort(key=lambda row: row["total_visits"], reverse=True)
    report = {
        "method": "nonlinear structural outcome model + conditional NOTEARS intervention propagation",
        "assumptions": [
            "controls, DMA fixed effects, and Fourier time terms block measured backdoor paths",
            "the thresholded contemporaneous media graph is acyclic",
            "media-to-media intervention paths are positive-only",
            "the outcome is non-decreasing in each media innovation",
            "the nonlinear outcome equation is correctly specified",
        ],
        "dag_acyclicity": estimate.dag_acyclicity,
        "outcome_r2_in_sample_log_visits": estimate.outcome_r2_in_sample,
        "outcome_r2_temporal_holdout_log_visits": estimate.outcome_r2_temporal_holdout,
        "positive_total_effect_visits": float(np.maximum(attribution.total, 0).sum()),
        "net_total_effect_visits": float(attribution.total.sum()),
        "channels": rows,
        "raw_dag_edges_diagnostic_only": [
            {"parent": parent, "child": child, "weight": weight}
            for parent, child, weight in dag.edges(0.15)
        ],
        "intervention_dag_edges": [
            {"parent": dag.node_names[parent], "child": dag.node_names[child], "weight": float(intervention_weights[parent, child])}
            for parent, child in zip(*np.nonzero(intervention_weights))
        ],
        "validation": validation.to_dict(),
    }
    output = Path("dashboard_outputs_notears_causal") / "causal_attribution_report.json"
    output.parent.mkdir(exist_ok=True)
    granular_output = output.parent / "causal_attribution_dma_week_channel.csv"
    estimate.dma_week_channel_frame(media, dma_ids, week_ids).to_csv(granular_output, index=False)
    dag_output = render_constrained_causal_dag(dag, control_names, output.parent / "causal_dag_positive")
    report["constrained_dag_svg"] = str(dag_output)
    report["dma_week_channel_output"] = str(granular_output)
    report["granular_output_definition"] = (
        "One row per DMA-week-channel. Effects are time-varying model scenario estimates, "
        "not separately estimated weekly coefficients."
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved causal attribution report: {output}")


if __name__ == "__main__":
    main()
