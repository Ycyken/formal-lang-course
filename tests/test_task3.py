import pytest
from networkx import MultiDiGraph
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton

from project.task3 import AdjacencyMatrixFA, tensor_based_rpq

nfa_cases = {
    "empty_graph": ({}, {}, [], [("a", False), ("", False)]),
    "2_node_graph": (
        {0},
        {1},
        [(0, "a", 1), (1, "b", 0)],
        [("ababa", True), ("ab", False)],
    ),
    "3_node_graph": (
        {1},
        {3, 2},
        [(1, "a", 2), (2, "c", 3), (3, "b", 1), (3, "r", 3)],
        [
            ("acba", True),
            ("acrrrrba", True),
            ("ac", True),
            ("a", True),
            ("abc", False),
            ("bacb", False),
        ],
    ),
}

rpq_cases = {
    "empty_graph": (MultiDiGraph(), "a", {}, {}, set()),
    "single_edge": (
        MultiDiGraph([(0, 1, {"label": "b"})]),
        "b",
        {0},
        {1},
        {(0, 1)},
    ),
    "star_regex": (
        MultiDiGraph(
            [(1, 2, {"label": "b"}), (1, 3, {"label": "a"}), (1, 1, {"label": "a"})]
        ),
        "(a)b*",
        {1, 2, 3},
        {1, 2, 3},
        {(1, 1), (1, 2), (1, 3)},
    ),
}


def build_nfa(
    nfa_data,
) -> tuple[NondeterministicFiniteAutomaton, list[tuple[str, bool]]]:
    start_states, final_states, transitions, test_cases = nfa_data
    nfa = NondeterministicFiniteAutomaton()
    for s in start_states:
        nfa.add_start_state(s)
    for s in final_states:
        nfa.add_final_state(s)
    nfa.add_transitions(transitions)
    return nfa, test_cases


@pytest.fixture(params=list(nfa_cases.items()))
def nfa_case(request):
    name, data = request.param
    nfa, test_cases = build_nfa(data)
    return name, nfa, test_cases


@pytest.fixture(params=list(rpq_cases.items()))
def rpq_case(request):
    name, data = request.param
    return name, data


def test_fa_accepts(nfa_case):
    name, nfa, test_cases = nfa_case
    am = AdjacencyMatrixFA(nfa)

    for word, expected in test_cases:
        result = am.accepts(word)
        assert result == expected, (
            f"Failed for case {name}, word '{word}': expected {expected}, got {result}"
        )


def test_rpq(rpq_case):
    name, (graph, regex, start_nodes, final_nodes, expected) = rpq_case
    result = tensor_based_rpq(regex, graph, start_nodes, final_nodes)
    assert result == expected, (
        f"Failed for case {name}: expected {expected}, got {result}"
    )
