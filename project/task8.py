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


def ms_bfs(
    graph_adj: AdjacencyMatrixFA,
    rsm_adj: AdjacencyMatrixFA,
    rsm: RecursiveAutomaton,
) -> set[tuple[object, int, int]]:
    inter = intersect_automata(graph_adj, rsm_adj)

    n_g = graph_adj.states_count
    n_r = rsm_adj.states_count

    alphabet = inter.alphabet

    nt_starts = {}
    nt_finals = {}
    for nt, box in rsm.boxes.items():
        starts = {rsm_adj.states_to_idxs[State((nt, s))] for s in box.dfa.start_states}
        finals = {rsm_adj.states_to_idxs[State((nt, f))] for f in box.dfa.final_states}
        nt_starts[nt] = starts
        nt_finals[nt] = finals

    result = set()

    for nt, starts in nt_starts.items():
        finals = nt_finals[nt]
        if not starts or not finals:
            continue

        front = csr_matrix((n_g, inter.states_count), dtype=bool)
        for u in range(n_g):
            base = u * n_r
            for rs in starts:
                front[u, base + rs] = True

        visited = front.copy()

        while True:
            new_front = csr_matrix(front.shape, dtype=bool)
            for sym in alphabet:
                new_front += front @ inter.matrices[sym]

            delta = new_front - new_front.multiply(visited)
            if delta.nnz == 0:
                break

            visited += delta
            front = delta

        for u in range(n_g):
            row = visited.getrow(u)
            if row.nnz == 0:
                continue
            for v in range(n_g):
                base = v * n_r
                for rf in finals:
                    if row[0, base + rf]:
                        result.add((nt, u, v))
                        break

    return result


def tensor_based_cfpq(
    rsm: pyformlang.rsa.RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    rsm_adj = AdjacencyMatrixFA(rsm_to_nfa(rsm))
    graph_nfa = graph_to_nfa(nx.MultiDiGraph(graph), start_nodes, final_nodes)
    graph_adj = AdjacencyMatrixFA(graph_nfa)

    for nt in rsm.boxes.keys():
        graph_adj.matrices.setdefault(
            nt,
            csr_matrix((graph_adj.states_count, graph_adj.states_count), dtype=bool),
        )
        graph_adj.alphabet.add(nt)

    changed = True
    while changed:
        changed = False
        for nt, src, dst in ms_bfs(graph_adj, rsm_adj, rsm):
            m = graph_adj.matrices[nt]
            if not m[src, dst]:
                m[src, dst] = True
                changed = True

    res = set()
    init_nt = rsm.initial_label

    idx_to_state = {i: s for s, i in graph_adj.states_to_idxs.items()}
    m0 = graph_adj.matrices[init_nt]
    for src, dst in zip(*m0.nonzero()):
        if src in graph_adj.start_idxs and dst in graph_adj.final_idxs:
            res.add((idx_to_state[src].value, idx_to_state[dst].value))
    return res


def cfg_to_rsm(cfg: pyformlang.cfg.CFG) -> pyformlang.rsa.RecursiveAutomaton:
    return RecursiveAutomaton.from_text(cfg.to_text())


def ebnf_to_rsm(ebnf: str) -> pyformlang.rsa.RecursiveAutomaton:
    return RecursiveAutomaton.from_text(ebnf)
