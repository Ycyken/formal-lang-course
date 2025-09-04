from typing import Tuple

import cfpq_data
from dataclasses import dataclass
from networkx.drawing.nx_pydot import write_dot


@dataclass
class GraphInfo:
    vertices_number: int
    edges_number: int
    labels: set[str]


def get_graph_info(name: str) -> GraphInfo:
    graph_path = cfpq_data.download(name)
    graph = cfpq_data.graph_from_csv(graph_path)

    edg_number = graph.number_of_edges()
    vert_number = graph.number_of_nodes()
    labels = {data['label'] for _, _, data in graph.edges.data()}

    return GraphInfo(vert_number, edg_number, labels)


def save_labeled_two_cycles_graph(n: int, m: int, labels: Tuple[str, str], file_path: str) -> None:
    graph = cfpq_data.labeled_two_cycles_graph(n, m, labels=labels)
    write_dot(graph, file_path)
