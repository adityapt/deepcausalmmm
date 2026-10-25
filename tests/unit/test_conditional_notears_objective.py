import numpy as np

from deepcausalmmm.core.conditional_notears import ConditionalNotears


def test_covariance_loss_and_gradient_match_panel_least_squares():
    """The fast objective must be mathematically identical to the panel form."""
    rng = np.random.default_rng(55)
    x = rng.normal(size=(17, 4))
    w = rng.normal(scale=0.1, size=(4, 4))
    np.fill_diagonal(w, 0.0)
    covariance = x.T @ x / len(x)
    identity = np.eye(x.shape[1])
    residual = x - x @ w

    panel_loss = 0.5 * np.sum(residual * residual) / len(x)
    covariance_loss = 0.5 * np.trace((identity - w).T @ covariance @ (identity - w))
    assert np.isclose(panel_loss, covariance_loss)

    expected_gradient = -(x.T @ residual) / len(x)
    covariance_gradient = covariance @ (w - identity)
    assert np.allclose(expected_gradient, covariance_gradient)


def test_covariance_implementation_recovers_expected_direction():
    rng = np.random.default_rng(56)
    controls = rng.normal(size=(12, 18, 1))
    parent = 0.5 * controls[..., 0] + rng.normal(scale=0.4, size=(12, 18))
    child = 0.9 * parent + rng.normal(scale=0.25, size=(12, 18))
    media = np.exp(np.stack([parent, child], axis=-1))

    result = ConditionalNotears(lambda1=0.02, max_outer=10, max_iter=100).fit(media, controls, ["parent", "child"])
    assert result.acyclicity < 1e-4
    assert abs(result.adjacency[0, 1]) > abs(result.adjacency[1, 0])
