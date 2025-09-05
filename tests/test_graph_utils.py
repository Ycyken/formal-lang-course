import cfpq_data
import networkx as nx
from networkx.algorithms.isomorphism import MultiDiGraphMatcher

from project.graph_utils import get_graph_info, save_labeled_two_cycles_graph, GraphInfo


def test_get_graph_info():
    actual = get_graph_info("bzip")

    bzip_path = cfpq_data.download("bzip")
    bzip = cfpq_data.graph_from_csv(bzip_path)

    labels = {data['label'] for _, _, data in bzip.edges.data()}
    expected = GraphInfo(len(bzip.nodes), len(bzip.edges), labels)
    assert actual == expected


def test_save_labeled_two_cycles_graph(tmp_path):
    graph_path = tmp_path / "two_cycles.dot"
    save_labeled_two_cycles_graph(3, 4, ("a", "b"), graph_path)

    actual = nx.nx_pydot.read_dot(graph_path)
    expected = cfpq_data.labeled_two_cycles_graph(3, 4, labels=("a", "b"))

    matcher = MultiDiGraphMatcher(
        expected,
        actual,
        node_match=lambda n1, n2: True,
        edge_match=lambda e1, e2: e1.get("label") == e2.get("label")
    )
    assert matcher.is_isomorphic()
