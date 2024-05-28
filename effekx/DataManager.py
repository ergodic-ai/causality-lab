from typing import Optional
import pandas
import numpy
import networkx
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression, LinearRegression
from causal_discovery_utils.constraint_based import CDLogger


def is_integer(value):
    if pandas.isnull(value):  # Handle null values
        return True
    if isinstance(value, (int, numpy.integer)):  # Check if value is an integer type
        return True
    if isinstance(
        value, float
    ):  # Check if value is a float and if it can be safely converted to an int
        return value.is_integer()
    if isinstance(
        value, str
    ):  # Check if value is a string representation of an integer
        try:
            int(value)
            return True
        except ValueError:
            return False
    return False


def truncate_and_one_hot_encode(col: pandas.Series, n_max: int = 5) -> pandas.DataFrame:
    """
    This function truncates a pandas Series to retain only the top n_max most common values,
    replacing the remaining values with "Others", and then performs one-hot encoding.

    Parameters:
    col (pandas.Series): The column to truncate and one-hot encode.
    n_max (int): The number of most common values to retain. Default is 5.

    Returns:
    pandas.DataFrame: The one-hot encoded DataFrame.
    """
    # Identify the top n_max most common values
    top_n_values = col.value_counts().nlargest(n_max).index

    # Replace values not in the top n_max most common values with "Others"
    truncated_col = col.where(col.isin(top_n_values), other="Others")

    # Perform one-hot encoding
    hasNans = truncated_col.isnull().any()

    one_hot_encoded_df = pandas.get_dummies(
        truncated_col, prefix=col.name, dummy_na=hasNans, drop_first=True
    )

    return one_hot_encoded_df


def truncate_column(col: pandas.Series, n_max: int = 5) -> pandas.Series:
    """
    This function identifies the top n_max most common values in the column and replaces the
    remaining values with "Others".

    Parameters:
    col (pandas.Series): The column to truncate.
    n_max (int): The number of most common values to retain. Default is 5.

    Returns:
    pandas.Series: The truncated column with less common values replaced by "Others".
    """
    # Identify the top n_max most common values
    top_n_values = col.value_counts().nlargest(n_max).index

    # Replace values not in the top n_max most common values with "Others"
    truncated_col = col.where(col.isin(top_n_values), other="Others")

    return truncated_col


class DataTypeManager:
    def __init__(self, data: pandas.DataFrame):
        self.data = data
        self.data_types = {col: self._get_data_type(col) for col in data.columns}

    def _validate_col(self, column: str):
        assert column in self.data.columns, f"Column {column} not found in data"

    def _is_object(self, column: str):
        return self.data[column].dtype == numpy.object_

    def _is_integer(self, column):
        return self.data[column].apply(is_integer).all() == True

    def _is_binary(self, column):
        return len(self.data[column].unique()) == 2

    def _is_numeric(self, column):
        return pandas.api.types.is_numeric_dtype(self.data[column])

    def _is_categorical(self, column):
        return pandas.api.types.is_categorical_dtype(self.data[column])

    def _is_uid(self, column):
        return len(self.data[column].unique()) == len(self.data)

    def _is_date(self, column):
        return pandas.api.types.is_datetime64_any_dtype(self.data[column])

    def _can_coerce_to_date(self, column):
        try:
            pandas.to_datetime(self.data[column])
            return True
        except (ValueError, TypeError):
            return False

    def _get_data_type(self, column: str):
        self._validate_col(column)
        if self._is_integer(column):
            return "integer"
        if self._is_binary(column):
            return "binary"
        if self._is_numeric(column):
            return "numeric"
        if self._is_categorical(column):
            return "categorical"
        if self._is_date(column) or self._can_coerce_to_date(column):
            return "date"
        if self._is_object(column):
            return "object"
        return "unknown"


def prep_data(df: pandas.DataFrame, data_types: dict, n_cat_max: int):
    """
    This function prepares the data for modelling by performing the following steps:
    1. Truncate and one-hot encode categorical columns.
    2. Replace binary columns with 0s and 1s.
    3. Replace date columns with the number of days since the earliest date.
    4. Replace object columns with the number of times each unique value appears.

    Parameters:
    df (pandas.DataFrame): The DataFrame to prepare.
    data_types (dict): A dictionary mapping column names to their data types.

    Returns:
    pandas.DataFrame: The prepared DataFrame.
    """
    prepared_df = pandas.DataFrame()
    feature_groups = {}
    normalization = {}

    for column in df.columns:
        assert (
            column in data_types
        ), f"Data type for column {column} not provided in data_types dictionary"

        data_type = data_types[column]
        assert df[column] is not None, f"Column {column} is empty"
        assert isinstance(
            df[column], pandas.Series
        ), f"Column {column} is not a pandas Series"

        col_data = pandas.Series(df[column])

        if data_type == "categorical" or data_type == "object":
            ohe = truncate_and_one_hot_encode(col_data, n_cat_max)
            feature_groups[column] = ohe.columns
            prepared_df = pandas.concat([prepared_df, ohe], axis=1)

        elif data_type == "binary":
            all_values = col_data.unique()
            assert len(all_values) == 2, f"Column {column} is not binary"
            zero_value = all_values[0]
            prepared_df[column] = col_data.apply(lambda x: 1 if x == zero_value else 0)
            feature_groups[column] = [column]

        elif data_type == "date":
            prepared_df[column] = (col_data - col_data.min()).dt.days
            feature_groups[column] = [column]
        elif data_type == "integer" or data_type == "numeric":
            col_data_mean = col_data.mean()
            col_data_std = col_data.std()
            col_data = (col_data - col_data_mean) / col_data_std
            prepared_df[column] = col_data
            feature_groups[column] = [column]
            normalization[column] = {"mean": col_data_mean, "std": col_data_std}
        else:
            prepared_df[column] = col_data
            feature_groups[column] = [column]

    return prepared_df, feature_groups, normalization


def marginal_contribution(x):
    return numpy.exp(x) / (1 + numpy.exp(x)) - 1 / 2


def quantile_abs(values, percentile: int = 90):
    # Reshape values as 1d array
    values = numpy.array(values).flatten()

    # sort based on absolute value
    values = sorted(values, key=abs, reverse=True)

    n_items = len(values)
    n_quantile = int(n_items * percentile / 100)

    return values[n_quantile]


class SCMConfigParams(BaseModel):
    n_cat_max: int = 5


class SCM:
    def __init__(
        self,
        data: pandas.DataFrame,
        graph: networkx.DiGraph,
        params=SCMConfigParams(),
        logger=CDLogger(),
    ):
        self.data = data
        self.graph = graph
        self.data_types = DataTypeManager(data).data_types
        self.params = params
        self.modelling_data = {}
        self.explanations = {}
        self.logger = logger

    def get_parents(self, node: str):
        return list(self.graph.predecessors(node))

    def prepare_modelling_data(self, node: str):
        parents = self.get_parents(node)
        subdf = pandas.DataFrame(self.data[parents])

        X, feature_groups, normalization = prep_data(
            subdf, self.data_types, self.params.n_cat_max
        )
        y = self.data[node]

        regression_types = ["integer", "numeric"]

        isClassification = True
        if self.data_types[node] in regression_types:
            isClassification = False
            y_mean = y.mean()
            y_std = y.std()
            normalization[node] = {"mean": y_mean, "std": y_std}
            y = (y - y_mean) / y_std

        return X, y, isClassification, feature_groups, normalization

    def explain(self, node: str):
        assert node in self.modelling_data, f"Model for node {node} not found"
        model_data = self.modelling_data[node]

        model = model_data["model"]
        feature_groups = model_data["feature_groups"]
        normalization = model_data["normalization"]
        model_type = model_data["model_type"]
        features = model_data["features"]

        explanations = []
        coefs = model.coef_

        if model_type == "regression":
            coefs_df = pandas.DataFrame([coefs], columns=features, index=[node])
            for feature_group in feature_groups:
                d = {}
                this_df = coefs_df[feature_groups[feature_group]]

                d["feature"] = feature_group
                d["target"] = node
                d["target_magnitude"] = normalization[node]["std"]

                explanation = {}

                if feature_group in normalization:
                    d["type"] = "number-number"
                    explanation["magnitude"] = normalization[feature_group]["std"]
                else:
                    d["type"] = "category-number"

                # explanation['strength'] = numpy.quantile(this_df.apply(lambda x: numpy.abs(x)).values, 0.9)
                explanation["strength"] = quantile_abs(this_df.values, 99)
                explanation["details"] = this_df.reset_index().to_dict(
                    orient="records", index=True
                )
                d["explanation"] = explanation
                explanations.append(d)

        else:
            coefs_df = pandas.DataFrame(coefs, columns=features, index=model.classes_)
            for feature_group in feature_groups:
                d = {}
                this_df = coefs_df[feature_groups[feature_group]].apply(
                    lambda x: marginal_contribution(x)
                )

                d["feature"] = feature_group
                d["target"] = node

                explanation = {}

                if feature_group in normalization:
                    d["type"] = "number-category"
                    explanation["magnitude"] = normalization[feature_group]["std"]
                else:
                    d["type"] = "category-category"

                explanation["strength"] = quantile_abs(this_df.values, 90)
                # explanation['strength'] = numpy.max(this_df.apply(lambda x: numpy.abs(x)).values)
                explanation["details"] = this_df.reset_index().to_dict(
                    orient="records", index=True
                )
                d["explanation"] = explanation
                explanations.append(d)

        self.explanations[node] = explanations
        return explanations

    def get_total_strength(self, source: str, target: str):
        assert source in self.graph.nodes, f"Node {source} not found in graph"
        assert target in self.graph.nodes, f"Node {target} not found in graph"

        results = {}
        paths = list(networkx.all_simple_paths(self.graph, source, target))

        results["paths"] = []

        if len(paths) == 0:
            return 0

        total_strength = 0
        for path in paths:
            d = {}
            d["path"] = path

            strength = 1
            d["breakdown"] = []
            for i in range(1, len(path)):
                parent = path[i - 1]
                child = path[i]
                s = self.get_strength(parent, child)
                strength *= s
                d["breakdown"].append(
                    {"source": parent, "target": child, "strength": s}
                )

            d["strength"] = strength
            results["paths"].append(d)
            total_strength += strength

        results["total_strength"] = total_strength
        results["source"] = source
        results["target"] = target
        return results

    def get_strength(self, source: str, target: str):
        assert target in self.explanations, f"Explanations for node {target} not found"
        explanations = self.explanations[target]
        for explanation in explanations:
            if explanation["feature"] == source:
                return explanation["explanation"]["strength"]
        return 0

    def fit(self, node: str):
        (
            X,
            y,
            isClassification,
            feature_groups,
            normalization,
        ) = self.prepare_modelling_data(node)
        features = list(X.columns)
        if isClassification:
            model = LogisticRegression(multi_class="multinomial")
        else:
            model = LinearRegression()

        model.fit(X, y)

        self.modelling_data[node] = {
            "model": model,
            "features": features,
            "feature_groups": feature_groups,
            "normalization": normalization,
            "model_type": "classification" if isClassification else "regression",
        }
        return self.modelling_data[node]

    def fit_all(self):
        for node in self.graph.nodes:
            parents = list(self.graph.predecessors(node))
            if len(parents) == 0:
                continue
            _ = self.fit(node)
            self.logger.log(f"Model for {node} fitted.")
            _ = self.explain(node)
