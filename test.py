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
from plot_utils.draw_graph import draw_graph
from server.utils import (
    adjacency_matrix_to_edges,
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
    n = 10000
    C1 = numpy.random.normal(size=(n,))
    C2 = numpy.random.normal(size=(n,))
    T1 = C1 + C2 + numpy.random.normal(size=(n,))
    T2 = C1 + C2 + numpy.random.normal(size=(n,))
    T3 = T2 + T1 + numpy.random.normal(size=(n,))

    data = pandas.DataFrame({"C1": C1, "C2": C2, "T1": T1, "T2": T2, "T3": T3})

    data.to_csv("test_data.csv", index=False)

    # return
    # domain_knowledge = [
    #     {"node_from": "C1", "node_to": "T1", "relationship": Relationship.CANNOT_CAUSE}
    # ]
    # domain_knowledge = [DomainKnowledge(**dk) for dk in domain_knowledge]
    model = initialize_fci(data, logger=MyLogger())
    edges = run_cd(model)
    draw_graph(model.graph)


if __name__ == "__main__":
    main()
