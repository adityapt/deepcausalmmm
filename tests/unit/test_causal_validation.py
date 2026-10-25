import numpy as np

from deepcausalmmm.core.causal_validation import CausalValidationSuite
from deepcausalmmm.core.conditional_notears import ConditionalNotears


def _synthetic_data(seed=21):
    rng = np.random.default_rng(seed)
    n_dma, weeks = 8, 18
    controls = rng.normal(size=(n_dma, weeks, 1))
    parent = 0.4 * controls[..., 0] + rng.normal(scale=0.3, size=(n_dma, weeks))
    child = 0.8 * parent + rng.normal(scale=0.2, size=(n_dma, weeks))
    media = np.exp(np.stack([parent, child], axis=-1))
    media[0, 0, 0] = 0.0
    outcome = np.exp(2.0 + 0.3 * parent + 0.5 * child + 0.1 * controls[..., 0])
    dag = ConditionalNotears(lambda1=0.02, max_outer=5, max_iter=80).fit(media, controls, ["parent", "child"])
    dag.adjacency = np.array([[0.0, 0.7], [0.0, 0.0]])
    return media, controls, outcome, dag


def test_validation_reports_support_placebo_and_explicit_scope():
    media, controls, outcome, dag = _synthetic_data()
    report = CausalValidationSuite(lead_periods=2).run(media, controls, outcome, dag, threshold=0.1)

    assert len(report.support) == 2
    assert report.support[0].zero_intervention_observed
    assert len(report.lead_placebo) == 2
    assert report.bootstrap_repetitions == 0
    assert "do not identify causal effects" in report.to_dict()["scope"]


def test_blocked_bootstrap_returns_selection_frequency():
    media, controls, outcome, dag = _synthetic_data()
    report = CausalValidationSuite(lead_periods=2).run(
        media, controls, outcome, dag, threshold=0.1, bootstrap_repetitions=2, bootstrap_block_size=4
    )

    assert report.bootstrap_repetitions == 2
    assert "warm-started bounded NOTEARS" in report.edge_stability_method
    assert all(0.0 < edge.selection_frequency <= 1.0 for edge in report.edge_stability)
