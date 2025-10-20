from collections import defaultdict

import networkx as nx
import pyformlang
from pyformlang.cfg import Production, Epsilon, CFG, Variable, Terminal


def cfg_to_weak_normal_form(cfg: pyformlang.cfg.CFG) -> pyformlang.cfg.CFG:
    ncfg = cfg.to_normal_form()
    eps_prods = {Production(nt, [Epsilon()]) for nt in cfg.get_nullable_symbols()}
    return CFG(
        ncfg.variables, ncfg.terminals, ncfg.start_symbol, ncfg.productions | eps_prods
    )


def hellings_init(
    cfg: pyformlang.cfg.CFG, graph: nx.DiGraph
) -> set[tuple[Variable, int, int]]:
    r = set()
    for ns in cfg.get_nullable_symbols():
        for v in graph.nodes:
            r.add((ns, v, v))

    ts_to_nts = defaultdict(list)
    for prod in cfg.productions:
        if len(prod.body) == 1 and isinstance(prod.body[0], Terminal):
            ts_to_nts[prod.body[0]].append(prod.head)
    for src, dst, label in graph.edges(data="label"):
        for nt in ts_to_nts.get(Terminal(label), []):
            r.add((nt, src, dst))

    return r


def hellings_based_cfpq(
    cfg: pyformlang.cfg.CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    if start_nodes is None:
        start_nodes = set(graph.nodes)
    if final_nodes is None:
        final_nodes = set(graph.nodes)

    cfg = cfg_to_weak_normal_form(cfg)
    r = hellings_init(cfg, graph)
    m = r.copy()
    two_nt_prods = {
        (prod.head, prod.body[0], prod.body[1])
        for prod in cfg.productions
        if len(prod.body) == 2
        and isinstance(prod.body[0], Variable)
        and isinstance(prod.body[1], Variable)
    }

    while m:
        (n_i, v, u) = m.pop()
        edges_to_add = set()
        for n_j, s, d in r:
            if d != v and s != u:
                continue
            for n_k, n_a, n_b in two_nt_prods:
                if d == v and n_a == n_j and n_b == n_i and (n_k, s, u) not in r:
                    m.add((n_k, s, u))
                    edges_to_add.add((n_k, s, u))
                if s == u and n_a == n_i and n_b == n_j and (n_k, v, d) not in r:
                    m.add((n_k, v, d))
                    edges_to_add.add((n_k, v, d))
        r.update(edges_to_add)

    return {
        (v1, v2)
        for (nt, v1, v2) in r
        if nt == cfg.start_symbol and v1 in start_nodes and v2 in final_nodes
    }
