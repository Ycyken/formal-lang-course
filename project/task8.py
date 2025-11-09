from collections import defaultdict, deque

import networkx as nx
import pyformlang
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State
from pyformlang.rsa import RecursiveAutomaton
from scipy.sparse import csr_matrix

from project.task2 import graph_to_nfa
from project.task3 import AdjacencyMatrixFA, intersect_automata


def rsm_to_nfa(rsm: RecursiveAutomaton) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton()
    for nt, box in rsm.boxes.items():
        for src, dst, label in box.dfa.to_networkx().edges(data="label"):
            nfa.add_transition(State((nt, src)), label, State((nt, dst)))

        for s in box.dfa.start_states:
            nfa.add_start_state(State((nt, s)))
        for s in box.dfa.final_states:
            nfa.add_final_state(State((nt, s)))
    return nfa


def msbfs(
    intersect_adj: AdjacencyMatrixFA,
    graph_adj: AdjacencyMatrixFA,
    rsm_adj: AdjacencyMatrixFA,
) -> set[tuple[str, int, int]]:
    n = intersect_adj.states_count

    succ = [set() for _ in range(n)]
    for m in intersect_adj.matrices.values():
        rows, cols = m.nonzero()
        for u, v in zip(rows, cols):
            succ[u].add(v)

    starts: dict[str, list[tuple[int, int]]] = defaultdict(list)
    finals: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for (g_st, r_st), idx in intersect_adj.states_to_idxs.items():
        nt, _ = r_st.value
        r_idx = rsm_adj.states_to_idxs[r_st]
        g_idx = graph_adj.states_to_idxs[g_st]
        if r_idx in rsm_adj.start_idxs:
            starts[nt].append((idx, g_idx))
        if r_idx in rsm_adj.final_idxs:
            finals[nt].append((idx, g_idx))

    res: set[tuple[str, int, int]] = set()

    for nt, s_list in starts.items():
        f_list = finals.get(nt)
        if not f_list:
            continue

        origin: dict[int, set[int]] = {}
        q = deque()

        for prod_idx, u_idx in s_list:
            origin[prod_idx] = {u_idx}
            q.append(prod_idx)

        while q:
            u = q.popleft()
            srcs = origin[u]
            for v in succ[u]:
                if v not in origin:
                    origin[v] = set(srcs)
                    q.append(v)
                else:
                    new = srcs - origin[v]
                    if new:
                        origin[v] |= new
                        q.append(v)

        for prod_idx, v_idx in f_list:
            if prod_idx not in origin:
                continue
            for u_idx in origin[prod_idx]:
                res.add((nt, u_idx, v_idx))

    return res


def tensor_based_cfpq(
    rsm: pyformlang.rsa.RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    rsm_adj = AdjacencyMatrixFA(rsm_to_nfa(rsm))
    graph_nfa = graph_to_nfa(nx.MultiDiGraph(graph), start_nodes, final_nodes)
    graph_adj = AdjacencyMatrixFA(graph_nfa)

    for nt in rsm.labels:
        graph_adj.matrices.setdefault(
            nt, csr_matrix((graph_adj.states_count, graph_adj.states_count), dtype=bool)
        )
    graph_adj.alphabet |= rsm.labels

    changed = True
    while changed:
        changed = False
        inter = intersect_automata(graph_adj, rsm_adj)
        for nt, u, v in msbfs(inter, graph_adj, rsm_adj):
            mat = graph_adj.matrices[nt]
            if not mat[u, v]:
                mat[u, v] = True
                changed = True

    res = set()
    idx_to_state = {i: s for s, i in graph_adj.states_to_idxs.items()}
    for u, v in zip(*graph_adj.matrices[rsm.initial_label].nonzero()):
        if u in graph_adj.start_idxs and v in graph_adj.final_idxs:
            res.add((idx_to_state[u].value, idx_to_state[v].value))
    return res


def cfg_to_rsm(cfg: pyformlang.cfg.CFG) -> pyformlang.rsa.RecursiveAutomaton:
    return RecursiveAutomaton.from_text(cfg.to_text())


def ebnf_to_rsm(ebnf: str) -> pyformlang.rsa.RecursiveAutomaton:
    return RecursiveAutomaton.from_text(ebnf)
