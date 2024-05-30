import pandas
import numpy
from typing import Optional
from effekx.DataManager import DataTypeManager, prep_data
from scipy.stats import f, pearsonr

from server.utils import generate_retention_data

from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2

from causal_discovery_utils.cond_indep_tests import StatCondIndepDF


def log_likelihood(model, X, y):
    """
    Calculate the log-likelihood of a logistic regression model.

    Parameters:
    - model: a fitted sklearn LogisticRegression model
    - X: the input features, a NumPy array or Pandas DataFrame
    - y: the true labels, a NumPy array or Pandas Series

    Returns:
    - log_likelihood: the log-likelihood of the model
    """
    # Get the predicted probabilities for each class
    probabilities = model.predict_proba(X)

    classes = model.classes_

    # Convert y to a one-hot encoded matrix
    n_classes = probabilities.shape[1]

    y_one_hot = numpy.zeros((len(y), n_classes))
    for i, label in enumerate(y):
        y_one_hot[i, numpy.where(classes == label)] = 1
    # Calculate the log-likelihood
    log_likelihood = numpy.sum(y_one_hot * numpy.log(probabilities))

    return log_likelihood


def get_rss(X, y):
    """Return the residual sum of squares for the data X and y."""
    params, [rss], rank, s = numpy.linalg.lstsq(X, y, rcond=None)
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
    def __init__(self, data: pandas.DataFrame, max_n_classes: int = 5):
        self.data = data
        self.data_types = DataTypeManager(data).data_types
        prepped_data, feature_groups, normalization = prep_data(
            data, data_types=self.data_types, n_cat_max=max_n_classes
        )
        self.prepped_data = prepped_data
        self.feature_groups = feature_groups
        self.normalization = normalization
        self.RSSCache = {}
        self.LLCache = {}

    def logistic_regression_lr_test(self, x: str, y: str, Z: Optional[list[str]] = []):
        """Perform a likelihood-ratio test to compare the full model with the reduced model."""

        x_features = self.feature_groups[x]
        Z_features = [self.feature_groups[z] for z in Z]
        Z_features = [item for sublist in Z_features for item in sublist]

        n_x_features = len(x_features)

        full_features = Z_features + x_features

        y_feat = y
        y = self.data[y]
        n = y.shape[0]

        # print(y.unique())

        # reduced_features = Z_features
        # print(X.shape)

        model_type = "binary"
        if len(numpy.unique(y)) > 2:
            model_type = "multinomial"

        if model_type == "binary":
            kwargs = {"max_iter": 10000, "fit_intercept": False}
        elif model_type == "multinomial":
            kwargs = {
                "multi_class": "multinomial",
                "solver": "lbfgs",
                "max_iter": 10000,
                "fit_intercept": False,
            }
        else:
            raise ValueError(
                "Unsupported model type. Choose 'binary' or 'multinomial'."
            )

        # Full model

        thisHash = hashFeatureList(Z_features + x_features, y_feat)
        if thisHash in self.LLCache:
            LLF1, d_full = self.LLCache[thisHash]

        else:
            X = self.prepped_data[Z_features + x_features].astype("float")
            ones = numpy.ones(shape=(n, 1))
            X = numpy.hstack([ones, X])

            model_full = LogisticRegression(**kwargs).fit(X, y)
            LLF1 = log_likelihood(model_full, X, y)
            d_full = len(model_full.coef_[0])
            self.LLCache[thisHash] = LLF1, d_full

        # Reduced model (without the last predictor)
        thisHash = hashFeatureList(Z_features, y_feat)
        if thisHash in self.LLCache:
            LLF0, d_reduced = self.LLCache[thisHash]
        else:
            X_reduced = self.prepped_data[Z_features].astype("float")
            ones = numpy.ones(shape=(n, 1))
            X_reduced = numpy.hstack([ones, X_reduced])

            model_reduced = LogisticRegression(**kwargs).fit(X_reduced, y)
            LLF0 = log_likelihood(model_reduced, X_reduced, y)
            d_reduced = len(model_reduced.coef_[0])
            self.LLCache[thisHash] = LLF0, d_reduced

        # Degrees of freedom
        df_diff = d_full - d_reduced

        # print(LLF1, LLF0, df_diff)
        # Likelihood-ratio statistic
        LR_stat = 2 * (LLF1 - LLF0)
        p_value = chi2.sf(LR_stat, df_diff)

        return LR_stat, p_value

    def f_test(self, x: str, y: str, Z: Optional[list[str]] = []):
        """Perform an F-test to compare the full model with the reduced model."""

        x_features = self.feature_groups[x]
        Z_features = [self.feature_groups[z] for z in Z]
        Z_features = [item for sublist in Z_features for item in sublist]

        n_x_features = len(x_features)

        p_full = len(x_features) + len(Z_features) + 1
        p_reduced = len(Z_features) + 1

        y_feat = y
        y = self.data[y].astype("float")

        n = self.data.shape[0]

        thisHash = hashFeatureList(x_features + Z_features, y_feat)
        if thisHash in self.RSSCache:
            RSS_full = self.RSSCache[thisHash]
        else:
            X = self.prepped_data[Z_features + x_features].astype("float")
            ones = numpy.ones(shape=(n, 1))
            X = numpy.hstack([ones, X])
            RSS_full = get_rss(X, y)

            self.RSSCache[thisHash] = RSS_full

        # p_full = X.shape[1]

        # Reduced model (without the last predictor)
        thisHash = hashFeatureList(Z_features, y_feat)
        if thisHash in self.RSSCache:
            RSS_reduced = self.RSSCache[thisHash]
        else:
            X_reduced = self.prepped_data[Z_features].astype("float")
            ones = numpy.ones(shape=(n, 1))
            X_reduced = numpy.hstack([ones, X_reduced])
            RSS_reduced = get_rss(X_reduced, y)

            self.RSSCache[thisHash] = RSS_reduced

        # p_reduced = X_reduced.shape[1]

        return get_f_and_p_val(RSS_full, RSS_reduced, p_full, p_reduced, n)

    def _run_cont_cont(self, x: str, y: str, Z: Optional[list[str]] = []):
        F, p = self.f_test(x, y, Z)
        return p

    def _run_cont_cat(self, x: str, y: str, Z: Optional[list[str]] = []):
        F, p = self.logistic_regression_lr_test(x, y, Z)
        return p

    def _run_cat_cont(self, x: str, y: str, Z: Optional[list[str]] = []):
        F, p = self.f_test(x, y, Z)
        return p

    def _agg_p_values(self, p1, p2):
        """𝑝𝑚𝑚=min{2min(𝑝1,𝑝2),max(𝑝1,𝑝2)}."""
        # print(p1, p2)
        return min(2 * min(p1, p2), max(p1, p2))

    def _run(self, x: str, y: str, Z: Optional[list[str]] = []):
        assert y in self.data.columns, f"Target {y} not in data columns"
        assert x in self.data.columns, f"Source {x} not in data columns"
        for z in Z:
            assert (
                z in self.data.columns
            ), f"Conditioning variable {z} not in data columns"

        regression_types = ["integer", "numeric"]

        if (
            self.data_types[x] in regression_types
            and self.data_types[y] in regression_types
        ):
            return self._run_cont_cont(x, y, Z)

        if self.data_types[x] in regression_types:
            # print("Running cont -> cat")
            p_val_1 = self._run_cont_cat(x, y, Z)
            # print("Running cat -> cont")
            p_val_2 = self._run_cat_cont(y, x, Z)
            return self._agg_p_values(p_val_1, p_val_2)

        if self.data_types[y] in regression_types:
            # print("Running cont -> cat")
            p_val_1 = self._run_cont_cat(y, x, Z)
            # print("Running cat -> cont")
            p_val_2 = self._run_cat_cont(x, y, Z)

            return self._agg_p_values(p_val_1, p_val_2)

        p_val_1 = self._run_cont_cat(x, y, Z)
        p_val_2 = self._run_cont_cat(y, x, Z)

        return self._agg_p_values(p_val_1, p_val_2)

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
