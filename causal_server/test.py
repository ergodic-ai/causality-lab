import pandas
import numpy
from causal_discovery_algs.icd import LearnStructICD
from causal_discovery_algs.fci import LearnStructFCI
from causal_discovery_utils.cond_indep_tests import CondIndepParCorr
from plot_utils.draw_graph import draw_graph
from causal_server.utils import adjacency_matrix_to_edges


def main():
    n = 10000
    C1 = numpy.random.normal(size=(n,))
    C2 = numpy.random.normal(size=(n,))
    T1 = -C1 + numpy.random.normal(size=(n,))
    T2 = C1 + C2 + numpy.random.normal(size=(n,))
    T3 = T2 + T1 + numpy.random.normal(size=(n,))

    data = pandas.DataFrame({"C1": C1, "C2": C2, "T1": T1, "T2": T2, "T3": T3})

    par_corr_test = CondIndepParCorr(dataset=data, threshold=0.01)
    icd = LearnStructFCI(set(data.columns), par_corr_test)  # instantiate an ICD learner
    print(icd.graph._graph)
    icd.learn_structure()  # learn the causal graph

    mat, mapping = icd.graph.get_adj_mat_with_cols()
    edges = adjacency_matrix_to_edges(mat, mapping)
    for edge in edges:
        print(edge)

    # draw_graph(icd.graph)


if __name__ == "__main__":
    main()
