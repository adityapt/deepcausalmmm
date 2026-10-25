import numpy as np

from deepcausalmmm.core.conditional_notears import ConditionalNotears


def test_recovers_sparse_acyclic_channel_edge_after_adjustment():
    rng = np.random.default_rng(4)
    n_dma, weeks = 24, 24
    controls = rng.normal(size=(n_dma, weeks, 1))
    parent = 0.7 * controls[..., 0] + rng.normal(scale=0.5, size=(n_dma, weeks))
    child = 0.9 * parent + 0.4 * controls[..., 0] + rng.normal(scale=0.35, size=(n_dma, weeks))
    independent = rng.normal(scale=0.6, size=(n_dma, weeks))
    media = np.exp(np.stack([parent, child, independent], axis=-1))

    result = ConditionalNotears(lambda1=0.02, max_outer=15).fit(
        media, controls, media_names=["parent", "child", "independent"]
    )

    assert result.acyclicity < 1e-4
    assert abs(result.adjacency[0, 1]) > abs(result.adjacency[1, 0])


def test_intervention_propagates_to_descendant_without_changing_parent_target():
    rng = np.random.default_rng(9)
    controls = rng.normal(size=(10, 12, 1))
    parent = rng.normal(size=(10, 12))
    child = 0.8 * parent + rng.normal(scale=0.1, size=(10, 12))
    media = np.exp(np.stack([parent, child], axis=-1))
    result = ConditionalNotears(lambda1=0.01).fit(media, controls, ["parent", "child"])
    # Isolate counterfactual propagation from graph-estimation variability.
    result.adjacency = np.array([[0.0, 0.8], [0.0, 0.0]])

    counterfactual = result.intervene(media, controls, channel=0, threshold=0.05)
    assert np.allclose(counterfactual[..., 0], 0.0)
    assert counterfactual[..., 1].mean() < media[..., 1].mean()


def test_intervention_projects_nearly_acyclic_weights_to_an_exact_dag():
    rng = np.random.default_rng(13)
    controls = rng.normal(size=(3, 6, 1))
    media = np.exp(rng.normal(size=(3, 6, 3)))
    result = ConditionalNotears(max_outer=1).fit(media, controls, ["a", "b", "c"])
    # A thresholded 3-cycle must not be used directly for an intervention.
    result.adjacency = np.array([[0.0, 0.8, 0.0], [0.0, 0.0, 0.7], [0.6, 0.0, 0.0]])
    projected, order = result._thresholded_order(0.1)
    assert len(order) == 3
    assert np.count_nonzero(projected) == 2
