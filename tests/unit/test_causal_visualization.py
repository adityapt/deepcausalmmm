import numpy as np

from deepcausalmmm.core.causal_visualization import build_constrained_causal_dag
from deepcausalmmm.core.conditional_notears import ConditionalNotears


def test_visualization_uses_positive_acyclic_intervention_edges():
    rng = np.random.default_rng(8)
    controls = rng.normal(size=(3, 8, 1))
    media = np.exp(rng.normal(size=(3, 8, 3)))
    dag = ConditionalNotears(max_outer=1, max_iter=20).fit(media, controls, ["A", "B", "C"])
    dag.adjacency = np.array([[0.0, 0.7, 0.0], [0.0, 0.0, 0.6], [0.5, -0.4, 0.0]])

    source = build_constrained_causal_dag(dag, ["Control 01"]).source

    assert 'label=0.7' in source
    assert 'label="-0.400"' not in source
    assert 'direct outcome' in source
