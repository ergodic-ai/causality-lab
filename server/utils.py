from typing import Optional
import pandas
import time

import numpy
from causal_discovery_algs.fci import LearnStructFCI
from causal_discovery_utils.cond_indep_tests import CondIndepParCorr
from causal_discovery_utils.constraint_based import CDLogger, LearnStructBase
from plot_utils.draw_graph import draw_graph
from graphical_models.partial_ancestral_graph import PAG

from graphical_models import arrow_head_types as Mark
import networkx as nx


MARKS = ["---", "<--", "o--"]  # X--*Y  (for PAGs)


def is_DAG(edges):
    G = nx.DiGraph()
    for edge in edges:
        if edge["data"]["edge_type"] == "undirected":
            return False
        G.add_edge(edge["source"], edge["target"])
    return nx.is_directed_acyclic_graph(G)


import numpy as np


def edges_to_adjacency_matrix(edges, mapping):
    n = len(mapping)
    adj_mat = np.zeros((n, n), dtype=int)

    reverse_mapping = {v: k for k, v in enumerate(mapping)}
    relationship_map = {"partially_known": 1, "arrow": 2, "directed": 3}

    inverse_relationship_map = {v: k for k, v in relationship_map.items()}

    for edge in edges:
        source = reverse_mapping[edge["source"]]
        target = reverse_mapping[edge["target"]]
        edge_type = edge["data"]["edge_type"]
        relationship = edge["data"].get("relationship", 1)

        if edge_type == "undirected":
            adj_mat[target, source] = relationship
            adj_mat[source, target] = relationship

        if edge_type == "partially_known":
            adj_mat[target, source] = 1
            adj_mat[source, target] = 2

        if edge_type == "directed":
            adj_mat[target, source] = 3
            adj_mat[source, target] = 2

    return adj_mat


def edges_to_graph_object(edges):
    nodes = set()
    for edge in edges:
        nodes.add(edge["source"])
        nodes.add(edge["target"])

    mapping = {node: i for i, node in enumerate(sorted(list(nodes)))}
    adj_mat = edges_to_adjacency_matrix(edges, mapping)
    pag = PAG(nodes_set=nodes)
    pag.init_from_adj_mat(adj_mat, sorted(list(nodes)))
    return pag


def graph_object_to_edges(graph: dict):
    arrow_type_map = dict()
    arrow_type_map[Mark.Circle] = 1
    arrow_type_map[Mark.Directed] = 2
    arrow_type_map[Mark.Tail] = 3

    num_vars = len(list(graph.keys()))
    adj_mat = numpy.zeros((num_vars, num_vars), dtype=int)
    node_index_map = {node: i for i, node in enumerate(sorted(list(graph.keys())))}

    for node in graph:
        for edge_mark in MARKS:
            for node_p in graph[node][edge_mark]:
                adj_mat[node_index_map[node_p]][node_index_map[node]] = arrow_type_map[
                    edge_mark
                ]

    index_node_map = {i: node for node, i in node_index_map.items()}

    return adjacency_matrix_to_edges(adj_mat, index_node_map)


def adjacency_matrix_to_edges(adj_mat, mapping):
    edges = []
    n = adj_mat.shape[0]

    relationship_map = {1: "partially_known", 2: "arrow", 3: "directed"}

    def edge_dict(i, j, edge_type, relationship):
        return {
            "id": f"{mapping[i]}-{mapping[j]}",
            "source": mapping[i],
            "target": mapping[j],
            "data": {
                "edge_type": edge_type,
                "relationship": relationship,
            },
        }

    for i in range(n):
        for j in range(i + 1, n):
            if adj_mat[i, j] != 0:
                relationship_forward = adj_mat[i, j]
                relationship_backward = adj_mat[j, i]

                if relationship_forward == relationship_backward:
                    edges.append(edge_dict(i, j, "undirected", relationship_forward))
                else:
                    if relationship_forward == 2:
                        edge_type = relationship_map[relationship_backward]
                        edges.append(edge_dict(i, j, edge_type, relationship_backward))
                    else:
                        edge_type = relationship_map[relationship_forward]
                        edges.append(edge_dict(j, i, edge_type, relationship_forward))

    return edges


def initialize_fci(data: pandas.DataFrame, logger: Optional[CDLogger] = None):
    if logger is None:
        logger = CDLogger()
    par_corr_test = CondIndepParCorr(dataset=data, threshold=0.03)
    model = LearnStructFCI(
        set(data.columns), par_corr_test, logger=logger
    )  # instantiate an ICD learner
    return model


def run_cd(model: LearnStructBase):
    # draw_graph(model.graph)
    while model.learn_structure_iterative():
        # draw_graph(model.graph)
        print("Iteration")
        pass
    # model.learn_structure()  # learn the causal graph
    mat, mapping = model.graph.get_adj_mat_with_cols()
    edges = adjacency_matrix_to_edges(mat, mapping)
    print(edges)
    return edges


def generate_random_data():
    n = 10000
    C1 = numpy.random.normal(size=(n,))
    C2 = numpy.random.normal(size=(n,))
    T1 = C1 + C2 + numpy.random.normal(size=(n,))
    T2 = C1 + C2 + numpy.random.normal(size=(n,))
    T3 = T2 + T1 + numpy.random.normal(size=(n,))

    data = pandas.DataFrame({"C1": C1, "C2": C2, "T1": T1, "T2": T2, "T3": T3})

    return data


def generate_retention_data():
    n = 10000
    traffic = numpy.random.normal(size=(n,))
    distance = numpy.random.normal(size=(n,))

    service_problems = numpy.random.normal(size=(n,)) + traffic + 2 * distance
    discount = numpy.random.normal(size=(n,)) + service_problems

    new_items = numpy.random.normal(size=(n,))
    stale_items = numpy.random.normal(size=(n,))

    engagement = numpy.random.normal(size=(n,)) + new_items - stale_items
    retention = (
        numpy.random.normal(size=(n,)) + discount - 1.5 * service_problems + engagement
    )

    retention = (retention > 0).astype(int)

    data = pandas.DataFrame(
        {
            "traffic": traffic,
            "distance": distance,
            "service_problems": service_problems,
            "discount": discount,
            "new_items": new_items,
            "stale_items": stale_items,
            "engagement": engagement,
            "retention": retention,
        }
    )

    return data
