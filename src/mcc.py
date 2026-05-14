from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Hashable, Literal, TypeAlias

import numpy as np
from scipy.linalg import eigh

Node: TypeAlias = Hashable
PairwiseEdge: TypeAlias = tuple[Node, Node]
Hyperedge: TypeAlias = set[Node]
ScoreMap: TypeAlias = dict[Node, float]
WeightMode: TypeAlias = Literal["uniform", "size"]
NormalizeMode: TypeAlias = Literal["none", "minmax", "zscore", "l1"]


def load_hypergraph(filepath: str | Path) -> tuple[list[int], list[set[int]]]:
    """Load a text hypergraph with one integer hyperedge per line.
    We ignore empty lines and lines starting with ``#``, and we skip singleton hyperedges, but we still include their nodes in th returned node list.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath}")

    hyperedges: list[set[int]] = []
    node_set: set[int] = set()
    singleton_count = 0

    with open(filepath, "r") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            try:
                tokens = [int(tok) for tok in line.split()]
            except ValueError as e:
                raise ValueError(f"No integer token at {lineno} of '{filepath}': {e}")

            if len(tokens) == 0:
                continue

            if len(tokens) == 1:
                singleton_count += 1
                node_set.add(tokens[0])
                continue

            hyperedge = set(tokens)
            hyperedges.append(hyperedge)
            node_set.update(hyperedge)

    if singleton_count > 0:
        print(f"[load_hypergraph] w {singleton_count} singleton hyperedge(s) "
            f"found and excluded (nodes still registered).")

    nodes = sorted(node_set)
    print(f"[load_hypergraph] Loaded {len(nodes)} nodes and {len(hyperedges)} hyperedges from '{filepath}'.")
    return nodes, hyperedges


def fiedler_value(nodes: list[Node], edges: list[PairwiseEdge]) -> float:
    """Return the second-smallest Laplacian eigenvalue for a pairwise graph."""
    n = len(nodes)
    if n < 2:
        return 0.0

    idx = {node: i for i, node in enumerate(nodes)}

    L = np.zeros((n, n), dtype=float)
    for u, v in edges:
        i, j = idx[u], idx[v]
        L[i, i] += 1.0
        L[j, j] += 1.0
        L[i, j] -= 1.0
        L[j, i] -= 1.0

    if n == 2:
        eigenvalues = eigh(L, eigvals_only=True)
    else:
        eigenvalues = eigh(L, eigvals_only=True, subset_by_index=[0, 1])

    return max(0.0, eigenvalues[1])


def induced_subgraph(
    hyperedge: Hyperedge,
    hyperedges: list[Hyperedge],
) -> tuple[list[Node], list[PairwiseEdge]]:
    """Build the pairwise graph induced by shared memberships outside ``hyperedge``."""
    nodes = sorted(hyperedge)
    node_set = set(nodes)
    edge_set: set[PairwiseEdge] = set()

    for other_edge in hyperedges:
        if other_edge == hyperedge:
            continue
        shared = node_set & other_edge
        if len(shared) < 2:
            continue
        shared_sorted = sorted(shared)
        for i in range(len(shared_sorted)):
            for j in range(i + 1, len(shared_sorted)):
                edge_set.add((shared_sorted[i], shared_sorted[j]))

    return nodes, list(edge_set)


def _normalize_scores(scores: ScoreMap, method: NormalizeMode) -> ScoreMap:
    """Normalize a node-score mapping."""
    if method == "none":
        return scores

    node_list = list(scores.keys())
    values = np.array([scores[n] for n in node_list], dtype=float)

    if method == "minmax":
        lo, hi = values.min(), values.max()
        denom = hi - lo
        normalized = np.zeros_like(values) if denom == 0.0 else (values - lo) / denom

    elif method == "zscore":
        mean, std = values.mean(), values.std()
        normalized = np.zeros_like(values) if std == 0.0 else (values - mean) / std

    elif method == "l1":
        total = np.abs(values).sum()
        normalized = np.zeros_like(values) if total == 0.0 else values / total

    else:
        raise ValueError(f"Unknown normalization method: '{method}'.")

    return {node: float(normalized[i]) for i, node in enumerate(node_list)}


def compute_mcc(
    nodes: list[Node],
    hyperedges: list[Hyperedge],
    weight: WeightMode = "uniform",
    rectified: bool = False,
    normalize: NormalizeMode = "none",
) -> ScoreMap:
    """Compute Marginal Cohesion Centrality scores for all nodes.

    ``rectified=True`` computes MCC+ by dropping negative marginal
    contributions before aggregation.
    """
    valid_weights: tuple[WeightMode, ...] = ("uniform", "size")
    valid_norms: tuple[NormalizeMode, ...] = ("none", "minmax", "zscore", "l1")

    if weight not in valid_weights:
        raise ValueError(f"weight must be one of {valid_weights}.")
    if normalize not in valid_norms:
        raise ValueError(f"normalize must be one of {valid_norms}.")

    hyperedges_sets = [set(e) for e in hyperedges]
    mcc_scores: ScoreMap = {v: 0.0 for v in nodes}
    kappa_cache: dict[frozenset[Node], float] = {}

    def get_kappa(edge_set: Hyperedge) -> float:
        key = frozenset(edge_set)
        if key not in kappa_cache:
            n_e, edges_e = induced_subgraph(edge_set, hyperedges_sets)
            kappa_cache[key] = fiedler_value(n_e, edges_e)
        return kappa_cache[key]

    for edge_set in hyperedges_sets:
        if len(edge_set) < 2:
            continue

        w = 1.0 if weight == "uniform" else float(len(edge_set))
        kappa_full = get_kappa(edge_set)

        for v in edge_set:
            reduced = edge_set - {v}
            kappa_reduced = 0.0 if len(reduced) < 2 else get_kappa(reduced)

            delta = kappa_full - kappa_reduced
            if rectified:
                delta = max(0.0, delta)

            mcc_scores[v] += w * delta

    return _normalize_scores(mcc_scores, normalize)


def compute_tmcc(
    nodes: list[Node],
    snapshots: list[list[Hyperedge]],
    alpha: float = 0.9,
    weight: WeightMode = "uniform",
    rectified: bool = False,
    normalize: NormalizeMode = "none",
) -> ScoreMap:
    """Compute temporally discounted MCC scores over ordered snapshots."""
    if not (0 < alpha <= 1):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}.")

    T = len(snapshots)
    tmcc_scores: ScoreMap = {v: 0.0 for v in nodes}

    for t, hyperedges_t in enumerate(snapshots):
        nodes_t = list({v for e in hyperedges_t for v in e})
        if not nodes_t:
            continue

        mcc_t = compute_mcc(
            nodes_t,
            hyperedges_t,
            weight=weight,
            rectified=rectified,
            normalize="none",
        )

        decay = alpha ** (T - 1 - t)
        for v, score in mcc_t.items():
            if v in tmcc_scores:
                tmcc_scores[v] += decay * score

    return _normalize_scores(tmcc_scores, normalize)


def rank_nodes(scores: ScoreMap, top_k: int | None = None) -> list[tuple[Node, float]]:
    """Return nodes sorted by decreasing score."""
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k] if top_k is not None else ranked


if __name__ == "__main__":
    separator = "=" * 60

    print(separator)
    print("MCC: Co-authorship hypergraph (dalla survey)")
    print(separator)

    name_to_id = {
        "V": 0, "B": 1, "K": 2, "Y": 3,
        "G": 4, "L": 5, "F": 6, "P": 7, "S": 8,
    }
    id_to_name = {v: k for k, v in name_to_id.items()}

    nodes = list(name_to_id.values())
    hyperedges = [
        {0, 1, 2},      # e1: V B K
        {1, 2},         # e2: B K
        {3, 1, 4, 5},   # e3: Y B G L
        {5, 6},         # e4: L F
        {6, 7, 8},      # e5: F P S
        {6, 7},         # e6: F P
        {8, 7},         # e7: S P
    ]

    print("\nHyperedges:")
    for i, e in enumerate(hyperedges, 1):
        print(f"  e{i}: {sorted(id_to_name[v] for v in e)}")

    print(f"\n{'-'*40}")
    print("1. File I/O round-trip")
    print(f"{'-'*40}")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as tmp:
        tmp.write("# Co-authorship hypergraph\n")
        tmp.write("# One hyperedge per line, space-separatorarated integer node IDs\n")
        for e in hyperedges:
            tmp.write(" ".join(str(v) for v in sorted(e)) + "\n")
        tmp_path = tmp.name

    loaded_nodes, loaded_hyperedges = load_hypergraph(tmp_path)
    os.unlink(tmp_path)

    assert loaded_nodes == sorted(nodes)
    assert len(loaded_hyperedges) == len(hyperedges)

    print(f"\n{'-'*40}")
    print("2. Static MCC — normalization comparison")
    print(f"{'-'*40}")

    for norm in ("none", "minmax", "zscore", "l1"):
        print(f"\n  normalize='{norm}'")
        mcc = compute_mcc(nodes, hyperedges, weight="uniform", normalize=norm)
        for nid, score in rank_nodes(mcc):
            print(f"    {id_to_name[nid]}: {score:+.4f}")

    print(f"\n{'-'*40}")
    print("3. Rectified MCC+ (negative contributions zeroed out)")
    print(f"{'-'*40}")

    for norm in ("none", "minmax"):
        print(f"\n  rectified=True, normalize='{norm}'")
        mcc_r = compute_mcc(
            nodes,
            hyperedges,
            weight="uniform",
            rectified=True,
            normalize=norm,
        )
        for nid, score in rank_nodes(mcc_r):
            print(f"    {id_to_name[nid]}: {score:+.4f}")

    print(f"\n{'-'*40}")
    print("4. Temporal MCC (TMCC)")
    print(f"{'-'*40}")

    snapshot_0 = [{0, 1, 2}, {5, 6}]
    snapshot_1 = [{0, 1, 2}, {1, 2}, {3, 1, 4, 5}, {5, 6}]
    snapshot_2 = hyperedges
    snapshots = [snapshot_0, snapshot_1, snapshot_2]

    for alpha in (0.5, 0.9, 0.99):
        print(f"\n  alpha={alpha}, normalize='minmax'")
        tmcc = compute_tmcc(
            nodes,
            snapshots,
            alpha=alpha,
            weight="uniform",
            normalize="minmax",
        )
        for nid, score in rank_nodes(tmcc):
            print(f"    {id_to_name[nid]}: {score:+.4f}")

    print(f"\n{separator}")
