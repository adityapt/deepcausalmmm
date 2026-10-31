import numpy as np

from deepcausalmmm.core.causal_attribution import NonlinearStructuralCausalAttributor, StructuralCausalAttributor
from deepcausalmmm.core.conditional_notears import ConditionalNotears


def test_structural_attributor_separates_mediated_effect_from_direct_effect():
    rng = np.random.default_rng(123)
    n_dma, weeks = 12, 20
    controls = rng.normal(size=(n_dma, weeks, 1))
    parent = 0.3 * controls[..., 0] + rng.normal(scale=0.35, size=(n_dma, weeks))
    child = 0.8 * parent + 0.2 * controls[..., 0] + rng.normal(scale=0.2, size=(n_dma, weeks))
    media = np.exp(np.stack([parent, child], axis=-1))
    outcome = 4.0 * parent + 7.0 * child + 1.5 * controls[..., 0] + rng.normal(scale=0.2, size=(n_dma, weeks))

    dag = ConditionalNotears(lambda1=0.01, max_outer=12).fit(media, controls, ["parent", "child"])
    # Isolate attribution arithmetic from graph-discovery sampling variation.
    dag.adjacency = np.array([[0.0, 0.8], [0.0, 0.0]])
    estimate = StructuralCausalAttributor().fit(media, controls, outcome, dag, threshold=0.05)

    assert estimate.dag_acyclicity < 1e-4
    assert estimate.attribution.mediated[0] > 0
    assert estimate.attribution.total[0] > estimate.attribution.direct[0]
    assert np.isclose(estimate.attribution.total, estimate.attribution.direct + estimate.attribution.mediated).all()


def test_nonlinear_attributor_exposes_dma_week_channel_effects():
    rng = np.random.default_rng(45)
    n_dma, weeks = 5, 14
    controls = rng.normal(size=(n_dma, weeks, 1))
    parent = 0.4 * controls[..., 0] + rng.normal(scale=0.25, size=(n_dma, weeks))
    child = 0.6 * parent + rng.normal(scale=0.2, size=(n_dma, weeks))
    media = np.exp(np.stack([parent, child], axis=-1))
    outcome = np.exp(3.0 + 0.4 * parent + 0.7 * child + 0.2 * controls[..., 0])
    dag = ConditionalNotears(lambda1=0.02, max_outer=5, max_iter=80).fit(media, controls, ["parent", "child"])
    dag.adjacency = np.array([[0.0, 0.7], [0.0, 0.0]])

    estimate = NonlinearStructuralCausalAttributor(max_iter=30).fit(media, controls, outcome, dag, threshold=0.1)
    frame = estimate.dma_week_channel_frame(media, dma_ids=["A", "B", "C", "D", "E"], week_ids=np.arange(weeks))

    assert len(frame) == n_dma * weeks * 2
    assert set(frame.columns) >= {"dma", "week", "channel", "time_varying_total_effect_visits"}
    assert np.isclose(frame["time_varying_total_effect_visits"].sum(), estimate.attribution.total.sum())
