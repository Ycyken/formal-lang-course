from functools import reduce
from itertools import product
from typing import Any, Iterable

from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    Symbol,
)
from scipy.sparse import csr_matrix, eye, kron

from project.task2 import regex_to_dfa, graph_to_nfa


class AdjacencyMatrixFA:
    def __init__(self, automaton: NondeterministicFiniteAutomaton):
        self.states_to_idxs: dict[Any, int] = {
            state: idx for idx, state in enumerate(automaton.states)
        }
        self.states_count: int = len(automaton.states)
        self.matrices: dict[Any, csr_matrix] = {}
        self.alphabet: set[Any] = automaton.symbols
        self.start_idx: set[int] = {
            self.states_to_idxs[s] for s in automaton.start_states
        }
        self.final_idxs: set[int] = {
            self.states_to_idxs[s] for s in automaton.final_states
        }

        for symbol in automaton.symbols:
            self.matrices[symbol] = csr_matrix(
                (self.states_count, self.states_count), dtype=bool
            )

        graph = automaton.to_networkx()
        for src, dst, label in graph.edges(data="label"):
            if label is None:
                continue
            src_idx = self.states_to_idxs[src]
            dst_idx = self.states_to_idxs[dst]
            self.matrices[label][src_idx, dst_idx] = True

    def accepts(self, word: Iterable[Symbol]) -> bool:
        front = csr_matrix((1, self.states_count), dtype=bool)

        for s in self.start_idx:
            front[0, s] = True

        for symbol in word:
            if symbol not in self.alphabet:
                return False
            front = front @ self.matrices[symbol]
            if front.nnz == 0:
                return False

        if set(front.nonzero()[1]) & self.final_idxs:
            return True
        return False

    def transitive_closure(self) -> csr_matrix:
        tc = reduce(
            lambda a, b: a + b,
            self.matrices.values(),
            eye(self.states_count, dtype=bool, format="csr"),
        )

        prev_nnz = tc.nnz
        for _ in range(self.states_count - 1):
            tc = tc @ tc
            if tc.nnz == prev_nnz:
                return tc

        return tc

    def is_empty(self) -> bool:
        tc = self.transitive_closure()

        for s in self.start_idx:
            for f in self.final_idxs:
                if tc[s, f]:
                    return False

        return True


def intersect_automata(
        automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    intersected = AdjacencyMatrixFA(NondeterministicFiniteAutomaton())
    intersected.alphabet = automaton1.alphabet & automaton2.alphabet
    intersected.states_count = automaton1.states_count * automaton2.states_count
    intersected.states_to_idxs = {}

    for (s1, idx1), (s2, idx2) in product(
            automaton1.states_to_idxs.items(), automaton2.states_to_idxs.items()
    ):
        new_idx = idx1 * automaton2.states_count + idx2
        intersected.states_to_idxs[(s1, s2)] = new_idx

        if (idx1 in automaton1.start_idx) and (idx2 in automaton2.start_idx):
            intersected.start_idx.add(new_idx)
        if (idx1 in automaton1.final_idxs) and (idx2 in automaton2.final_idxs):
            intersected.final_idxs.add(new_idx)

    for symbol in intersected.alphabet:
        m1 = automaton1.matrices[symbol]
        m2 = automaton2.matrices[symbol]
        intersected.matrices[symbol] = kron(m1, m2, format="csr")

    return intersected


def tensor_based_rpq(
        regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    automaton_regex = regex_to_dfa(regex)
    automaton_graph = graph_to_nfa(graph, start_nodes, final_nodes)

    automaton_regex_m = AdjacencyMatrixFA(automaton_regex)
    automaton_graph_m = AdjacencyMatrixFA(automaton_graph)

    intersected = intersect_automata(automaton_regex_m, automaton_graph_m)
    tc = intersected.transitive_closure()

    result = set()
    for start_regex, final_regex, start_graph, final_graph in product(
            automaton_regex.start_states,
            automaton_regex.final_states,
            automaton_graph.start_states,
            automaton_graph.final_states,
    ):
        start_idx = intersected.states_to_idxs[(start_regex, start_graph)]
        final_idx = intersected.states_to_idxs[(final_regex, final_graph)]
        if tc[start_idx, final_idx]:
            result.add((start_graph.value, final_graph.value))
    return result
