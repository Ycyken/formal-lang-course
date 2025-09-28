from networkx import MultiDiGraph
from scipy.sparse import csr_matrix, vstack

from project.task2 import regex_to_dfa, graph_to_nfa
from project.task3 import AdjacencyMatrixFA


def create_front(adj1: AdjacencyMatrixFA, adj2: AdjacencyMatrixFA) -> csr_matrix:
    start_states1 = sorted(list(adj1.start_idx))
    blocks = []
    for s1 in start_states1:
        block = csr_matrix((adj1.states_count, adj2.states_count), dtype=bool)
        for s2 in adj2.start_idx:
            block[s1, s2] = True
        blocks.append(block)
    return vstack(blocks, format="csr", dtype=bool)


def ms_bfs_based_rpq(regex: str, graph: MultiDiGraph, start_nodes: set[int],
                     final_nodes: set[int]) -> set[tuple[int, int]]:
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    regex_dfa = regex_to_dfa(regex)

    graph_adj = AdjacencyMatrixFA(graph_nfa)
    regex_adj = AdjacencyMatrixFA(regex_dfa)

    transposed_graph_decomp = {label: m.transpose() for label, m in graph_adj.matrices.items()}
    intersected_alphabet = graph_adj.alphabet & regex_adj.alphabet

    front = create_front(graph_adj, regex_adj)
    visited = front
    visited_updated = True
    while visited_updated:
        new_front = csr_matrix(front.shape, dtype=bool)
        for symbol in intersected_alphabet:
            blocks = []
            for block_number in range(len(graph_adj.start_idx)):
                block = front[block_number * graph_adj.states_count: (block_number + 1) * graph_adj.states_count, :]
                block = transposed_graph_decomp[symbol] @ block
                blocks.append(block)
            joint_blocks = vstack(blocks, format="csr", dtype=bool)
            symbol_front = joint_blocks @ regex_adj.matrices[symbol]
            new_front += symbol_front

        visited_updated = (new_front - (new_front.multiply(visited))).nnz != 0
        front = new_front
        visited += new_front

    result = set()
    start_nodes = sorted(list(start_nodes))
    for block_number, graph_start_node in enumerate(start_nodes):
        block = visited[block_number * graph_adj.states_count: (block_number + 1) * graph_adj.states_count, :]
        for graph_final_node in final_nodes:
            graph_final_idx = graph_adj.states_to_idxs[graph_final_node]
            for regex_final_idx in regex_adj.final_idxs:
                if block[graph_final_idx, regex_final_idx]:
                    result.add((graph_start_node, graph_final_node))
    return result
