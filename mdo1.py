import pandas as pd
import numpy as np
import numpy
import pandas
import torch
from torch import nn
from torch.autograd import Variable
from typing import Optional
from effekx.DataManager import DataTypeManager, prep_data
from scipy.stats import chi2, f

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from causal_server.utils import generate_retention_data

from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2

from causal_discovery_utils.cond_indep_tests import StatCondIndepDF


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.linear(x)


def log_likelihood(model, X, y):
    outputs = model(X)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss = criterion(outputs, y)
    return -loss.item()


def get_rss(X, y):
    X_t = torch.Tensor(X).to(device)
    y_t = torch.Tensor(y).to(device)
    beta, rss, rank, sing = torch.linalg.lstsq(y_t.unsqueeze(1), X_t)
    print(rss)
    residuals = y_t.unsqueeze(1) - X_t @ beta
    print(residuals)
    rss = torch.sum(residuals**2).item()
    return rss


def get_f_and_p_val(rss_full, rss_reduced, p_full, p_reduced, n):
    F_stat = ((rss_reduced - rss_full) / (p_full - p_reduced)) / (
        rss_full / (n - p_full)
    )
    p_value = 1 - f.cdf(F_stat, p_full - p_reduced, n - p_full)
    return F_stat, p_value


def hashFeatureList(feature_list, target):
    return str(sorted(feature_list)) + target


class MixedDataOracle:
    def __init__(self, data: pd.DataFrame, max_n_classes: int = 5):
        self.data = data
        self.data_types = DataTypeManager(data).data_types
        self.prepped_data, self.feature_groups, self.normalization = prep_data(
            data, data_types=self.data_types, n_cat_max=max_n_classes
        )
        self.RSSCache = {}
        self.LLCache = {}

    def logistic_regression_lr_test(self, x: str, y: str, Z: Optional[list[str]] = []):
        x_features = self.feature_groups[x]
        Z_features = sum([self.feature_groups[z] for z in Z], [])
        y = self.data[y]
        n = y.shape[0]

        model_type = "binary"
        if len(np.unique(y)) > 2:
            model_type = "multinomial"

        def get_log_likelihood(X, y, cache_key):
            if cache_key in self.LLCache:
                return self.LLCache[cache_key]
            X_t = torch.Tensor(X).to(device)
            y_t = torch.LongTensor(y.values).to(device)
            input_dim = X_t.shape[1]
            output_dim = len(np.unique(y))
            model = LogisticRegressionModel(input_dim, output_dim).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()

            for _ in range(10000):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_t)
                loss = criterion(outputs, y_t)
                loss.backward()
                optimizer.step()

            LL = log_likelihood(model, X_t, y_t)
            d = input_dim
            self.LLCache[cache_key] = (LL, d)
            return LL, d

        full_features = Z_features + x_features
        X_full = np.hstack(
            [np.ones((n, 1)), self.prepped_data[full_features].astype("float")]
        )
        LLF1, d_full = get_log_likelihood(
            X_full, y, hashFeatureList(full_features, y.name)
        )

        X_reduced = np.hstack(
            [np.ones((n, 1)), self.prepped_data[Z_features].astype("float")]
        )
        LLF0, d_reduced = get_log_likelihood(
            X_reduced, y, hashFeatureList(Z_features, y.name)
        )

        LR_stat = 2 * (LLF1 - LLF0)
        p_value = chi2.sf(LR_stat, d_full - d_reduced)

        return LR_stat, p_value

    def f_test(self, x: str, y: str, Z: Optional[list[str]] = []):
        x_features = self.feature_groups[x]
        Z_features = sum([self.feature_groups[z] for z in Z], [])
        y = self.data[y].astype("float")
        n = self.data.shape[0]

        def get_rss_cached(X, y, cache_key):
            if cache_key in self.RSSCache:
                return self.RSSCache[cache_key]
            rss = get_rss(X, y)
            self.RSSCache[cache_key] = rss
            return rss

        p_full = len(x_features) + len(Z_features) + 1
        p_reduced = len(Z_features) + 1

        X_full = np.hstack(
            [
                np.ones((n, 1)),
                self.prepped_data[Z_features + x_features].astype("float"),
            ]
        )
        RSS_full = get_rss_cached(
            X_full, y, hashFeatureList(x_features + Z_features, y.name)
        )

        X_reduced = np.hstack(
            [np.ones((n, 1)), self.prepped_data[Z_features].astype("float")]
        )
        RSS_reduced = get_rss_cached(X_reduced, y, hashFeatureList(Z_features, y.name))

        return get_f_and_p_val(RSS_full, RSS_reduced, p_full, p_reduced, n)

    def _run_cont_cont(self, x: str, y: str, Z: Optional[list[str]] = []):
        _, p = self.f_test(x, y, Z)
        return p

    def _run_cont_cat(self, x: str, y: str, Z: Optional[list[str]] = []):
        _, p = self.logistic_regression_lr_test(x, y, Z)
        return p

    def _run_cat_cont(self, x: str, y: str, Z: Optional[list[str]] = []):
        _, p = self.f_test(x, y, Z)
        return p

    def _agg_p_values(self, p1, p2):
        return min(2 * min(p1, p2), max(p1, p2))

    def _run(self, x: str, y: str, Z: Optional[list[str]] = []):
        regression_types = ["integer", "numeric"]
        if (
            self.data_types[x] in regression_types
            and self.data_types[y] in regression_types
        ):
            return self._run_cont_cont(x, y, Z)
        if self.data_types[x] in regression_types:
            p1 = self._run_cont_cat(x, y, Z)
            p2 = self._run_cat_cont(y, x, Z)
            return self._agg_p_values(p1, p2)
        if self.data_types[y] in regression_types:
            p1 = self._run_cont_cat(y, x, Z)
            p2 = self._run_cat_cont(x, y, Z)
            return self._agg_p_values(p1, p2)
        p1 = self._run_cont_cat(x, y, Z)
        p2 = self._run_cont_cat(y, x, Z)
        return self._agg_p_values(p1, p2)

    def __call__(self, x: str, y: str, Z: Optional[list[str]] = []):
        print(f"Running test for {x} -> {y} | {Z}")
        return self._run(x, y, Z)


def main():
    df = generate_retention_data()

    zz = numpy.random.normal(size=(500,))
    zz = numpy.round(zz)
    xx = zz + numpy.random.normal(size=(500,))
    yy = zz + numpy.random.normal(size=(500,))
    xx = numpy.round(xx).astype(str)
    zz = numpy.round(zz).astype(str)
    df = pandas.DataFrame({"x": xx, "y": yy, "z": zz})
    # df["y"] = [str(int(z)) for z in df["y"]]
    # df["y"] = df["y"].astype(int).astype("category")

    # print(len(df["y"].unique()))

    y = "y"
    x = "x"
    Z = ["z"]
    oracle = MixedDataOracle(df, max_n_classes=10)
    # print(oracle.logistic_regression_lr_test(x, y, Z))
    print(oracle.f_test(x, y, Z))


class MixedData(StatCondIndepDF):
    def __init__(
        self,
        threshold,
        dataset,
        weights=None,
        retained_edges=None,
        count_tests=False,
        use_cache=False,
        num_records=None,
        num_vars=None,
        max_n_classes=5,
    ):
        if weights is not None:
            raise Exception(
                "weighted Partial-correlation is not supported. Please avoid using weights."
            )
        super().__init__(
            dataset,
            threshold,
            database_type=float,
            weights=weights,
            retained_edges=retained_edges,
            count_tests=count_tests,
            use_cache=use_cache,
            num_records=num_records,
            num_vars=num_vars,
        )

        self.data = None  # no need to store the data as we will just use the oracle to calculate the p-value
        self.oracle = MixedDataOracle(dataset, max_n_classes=max_n_classes)

    def calc_statistic(self, x, y, zz):
        return self.oracle(x, y, zz)


if __name__ == "__main__":
    main()
