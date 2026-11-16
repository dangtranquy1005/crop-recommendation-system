import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

# ta tính thêm 4 đặc trưng nữa 
class CropFeatureEngineer(TransformerMixin, BaseEstimator):
   

    def __init__(self, include_total_npk=True):
        self.include_total_npk = include_total_npk

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.to_numpy(copy=True)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        check_is_fitted(self)
        engineered = X.loc[:, self.feature_names_in_].copy()

        if self.include_total_npk:
            engineered["total_npk"] = X["N"] + X["P"] + X["K"]

        # Undefined ratios stay missing for the downstream train-fitted imputer.
        engineered["np_ratio"] = X["N"] / X["P"].replace(0, np.nan)
        engineered["nk_ratio"] = X["N"] / X["K"].replace(0, np.nan)
        engineered["ph_distance_7"] = (X["ph"] - 7).abs()
        return engineered

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        if input_features is not None and not np.array_equal(
            input_features, self.feature_names_in_
        ):
            raise ValueError("input_features must match feature_names_in_")

        added_features = ["np_ratio", "nk_ratio", "ph_distance_7"]
        if self.include_total_npk:
            added_features.insert(0, "total_npk")
        return np.concatenate([self.feature_names_in_, added_features])


def build_crop_pipeline(classifier, use_engineered_features, *, include_total_npk=True):
    features = (
        CropFeatureEngineer(include_total_npk=include_total_npk)
        if use_engineered_features else "passthrough"
    )
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocess = ColumnTransformer(
        [("numeric", numeric_pipeline, make_column_selector(dtype_include=np.number))],
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("features", features),
        ("preprocess", preprocess),
        ("classifier", classifier),
    ])
