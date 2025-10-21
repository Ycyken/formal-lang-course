from collections import defaultdict
from typing import Set

from scipy.sparse import csr_matrix
from project.task6 import cfg_to_weak_normal_form

import networkx as nx
import pyformlang


def matrix_based_cfpq(
    cfg: pyformlang.cfg.CFG,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> set[tuple[int, int]]:
    start_nodes = start_nodes or graph.nodes
    final_nodes = final_nodes or graph.nodes
    nd_to_idx = {nd: idx for idx, nd in enumerate(graph.nodes)}
    cfg = cfg_to_weak_normal_form(cfg)
    n = graph.number_of_nodes()

    decomp = {nt: csr_matrix((n, n), dtype=bool) for nt in cfg.variables}
    ts_to_nts = defaultdict(list)  # N -> x
    for prod in cfg.productions:
        if len(prod.body) == 1 and isinstance(prod.body[0], pyformlang.cfg.Terminal):
            ts_to_nts[prod.body[0].value].append(prod.head.value)

    for src, dst, label in graph.edges(data="label"):
        src, dst = nd_to_idx[src], nd_to_idx[dst]
        for nt in ts_to_nts.get(label, []):
            decomp[nt][src, dst] = True
    for nt in cfg.get_nullable_symbols():  # N -> ε
        for i in range(0, n):
            decomp[nt.value][i, i] = True

    two_nt_prods = {  # N_i -> N_j N_k
        (prod.head.value, prod.body[0].value, prod.body[1].value)
        for prod in cfg.productions
        if len(prod.body) == 2
    }

    changed = True
    while changed:
        changed = False
        for n_i, n_j, n_k in two_nt_prods:
            prev_nnz = decomp[n_i].nnz
            decomp[n_i] += decomp[n_j] @ decomp[n_k]
            changed = changed or (decomp[n_i].nnz > prev_nnz)

    return {
        (i, j)
        for i in start_nodes
        for j in final_nodes
        if decomp[cfg.start_symbol][nd_to_idx[i], nd_to_idx[j]]
    }
