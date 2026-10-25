"""Graphviz rendering for the constrained causal-attribution DAG."""

from pathlib import Path
from typing import Sequence, Union

import numpy as np

from deepcausalmmm.core.conditional_notears import ConditionalNotearsResult


def build_constrained_causal_dag(
    dag: ConditionalNotearsResult,
    control_names: Sequence[str],
    threshold: float = 0.15,
    positive_only: bool = True,
):
    """Build the exact intervention graph, including direct outcome edges.

    This deliberately calls the same acyclic projection used by
    ``ConditionalNotearsResult.intervene``.  It therefore cannot visualize a
    raw edge that attribution excluded because it was negative or cycle-closing.
    """
    try:
        from graphviz import Digraph
    except ImportError as exc:  # pragma: no cover - depends on optional renderer
        raise ImportError("Install the 'graphviz' Python package and Graphviz's 'dot' renderer") from exc

    weights, _ = dag._thresholded_order(threshold, positive_only=positive_only)
    graph = Digraph("constrained_causal_dag", format="svg")
    graph.attr(rankdir="LR", bgcolor="transparent", pad="0.3", nodesep="0.25", ranksep="0.8", splines="spline")
    graph.attr("node", shape="box", style="rounded,filled", fontname="Arial", fontsize="11", color="#334155", fillcolor="#eff6ff")
    graph.attr("edge", fontname="Arial", fontsize="9", color="#15803d", fontcolor="#166534", arrowsize="0.7")

    with graph.subgraph(name="cluster_adjustment") as adjustment:
        adjustment.attr(label="Backdoor adjustment set — conditioned, never attributed", color="#94a3b8", style="rounded,dashed")
        adjustment.attr("node", shape="ellipse", fillcolor="#f8fafc", color="#94a3b8", fontsize="9")
        for index, name in enumerate(control_names):
            adjustment.node(f"control_{index}", name)
        adjustment.node("dma", "DMA fixed effects")
        adjustment.node("time", "Fourier time terms")

    graph.node("adjustment", "Conditioning /\nresidualization", fillcolor="#f1f5f9", color="#64748b")
    graph.node("outcome", "Visits", shape="doubleoctagon", fillcolor="#fef3c7", color="#b45309")
    for index, name in enumerate(dag.node_names):
        graph.node(f"channel_{index}", name)
    for index in range(len(control_names)):
        graph.edge(f"control_{index}", "adjustment", style="dashed", color="#94a3b8")
    graph.edge("dma", "adjustment", style="dashed", color="#94a3b8")
    graph.edge("time", "adjustment", style="dashed", color="#94a3b8")
    graph.edge("adjustment", "outcome", label="adjusts outcome", style="dashed", color="#64748b")

    for parent, child in zip(*np.nonzero(weights)):
        graph.edge(f"channel_{parent}", f"channel_{child}", label=f"{weights[parent, child]:.3f}")
    for index in range(len(dag.node_names)):
        graph.edge(f"channel_{index}", "outcome", label="direct outcome", style="dashed", color="#b45309", fontcolor="#92400e")
    return graph


def render_constrained_causal_dag(
    dag: ConditionalNotearsResult,
    control_names: Sequence[str],
    output_path: Union[str, Path],
    threshold: float = 0.15,
    positive_only: bool = True,
) -> Path:
    """Render the exact attribution DAG to an SVG path."""
    destination = Path(output_path)
    graph = build_constrained_causal_dag(dag, control_names, threshold, positive_only)
    rendered = graph.render(filename=destination.stem, directory=str(destination.parent), cleanup=True)
    return Path(rendered)
