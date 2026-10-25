"""Fit the post-fit conditional-NOTEARS causal attribution model on MMM Data.

This script does not train or alter the GRU MMM.  It emits a machine-readable
report of model-based direct, mediated, and total intervention effects.
"""

import json
from pathlib import Path

import numpy as np

from dashboard_rmse_optimized import load_real_mmm_data
from deepcausalmmm.core.causal_attribution import NonlinearStructuralCausalAttributor
from deepcausalmmm.core.conditional_notears import ConditionalNotears
from deepcausalmmm.core.causal_visualization import render_constrained_causal_dag


def main() -> None:
    media, controls, outcome, names, control_names = load_real_mmm_data()
    dag = ConditionalNotears(
        lambda1=0.01, max_outer=15, max_iter=300, h_tolerance=1e-6
    ).fit(media, controls, names)
    estimate = NonlinearStructuralCausalAttributor().fit(media, controls, outcome, dag, threshold=0.15)
    attribution = estimate.attribution
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
        "dag_edges": [
            {"parent": parent, "child": child, "weight": weight}
            for parent, child, weight in dag.edges(0.15)
        ],
    }
    output = Path("dashboard_outputs_notears_causal") / "causal_attribution_report.json"
    output.parent.mkdir(exist_ok=True)
    dag_output = render_constrained_causal_dag(dag, control_names, output.parent / "causal_dag_positive")
    report["constrained_dag_svg"] = str(dag_output)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved causal attribution report: {output}")


if __name__ == "__main__":
    main()
