import networkx
import pandas
import numpy
from pandas.io.formats.style_render import Optional
from causal_discovery_algs.icd import LearnStructICD
from causal_discovery_algs.fci import LearnStructFCI
from causal_discovery_utils.cond_indep_tests import CondIndepParCorr
from causal_discovery_utils.constraint_based import (
    CDLogger,
    DomainKnowledge,
    Relationship,
)
from effekx.DataManager import SCM
from plot_utils.draw_graph import draw_graph
from causal_server.utils import (
    adjacency_matrix_to_edges,
    generate_retention_data,
    graph_object_to_edges,
    initialize_fci,
    run_cd,
)


class MyLogger(CDLogger):
    def __init__(self):
        pass

    def log(self, msg: str, metadata: Optional[dict] = None) -> None:
        # raise NotImplementedError
        print(msg + str(metadata))

    def graph(self, graph: dict):
        edges = graph_object_to_edges(graph)
        print(edges)
        # pass

    def info(self, msg: str, metadata: Optional[dict] = None) -> None:
        metadata = metadata or {}
        metadata = {**metadata, "level": "info"}
        self.log(msg, metadata)


def main():
    df = generate_retention_data()
    logger = MyLogger()
    model = initialize_fci(df, logger=logger)
    edges = run_cd(model)
    edge_list = [(x["source"], x["target"]) for x in edges]
    causal_graph = networkx.DiGraph(edge_list)
    nodes = list(networkx.ancestors(causal_graph, "retention"))

    print(df.corr())
    df.to_csv("RetentionData.csv", index=False)

    # print("endnodes: ")
    # print([x for x in causal_graph.nodes() if causal_graph.out_degree(x) == 0])

    # scm = SCM(data=df, graph=causal_graph, logger=logger)
    # scm.fit_all()

    # results = []
    # for node in nodes:
    #     res = scm.get_total_strength("service_problems", "retention")
    #     results.append(res)

    # results = sorted(results, key=lambda x: abs(x["total_strength"]), reverse=True)
    # for res in results:
    #     print(" * * * ")
    #     print(res)


if __name__ == "__main__":
    main()
