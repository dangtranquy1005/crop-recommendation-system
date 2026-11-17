"""Measure feature engineering on validation data with fixed model settings."""

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score

from config import FEATURES, RANDOM_STATE
from train_logistic_regression import build_logistic_model
from train_qda import build_qda_model, load_training_data
from train_random_forest import build_random_forest_model
from trainlda import build_model as build_lda_model


def fit_feature_variants(X_train, y_train):
    builders = {
        "Logistic Regression": build_logistic_model,
        "LDA": build_lda_model,
        "QDA": build_qda_model,
        "Random Forest": build_random_forest_model,
    }
    for name, build in builders.items():
        baseline = build(use_engineered_features=False).fit(X_train, y_train)
        engineered = build(use_engineered_features=True).fit(X_train, y_train)
        yield name, baseline, engineered


def main():
    X_train, X_val, y_train, y_val = load_training_data()
    results = []
    importance_rows = []

    for name, baseline, engineered in fit_feature_variants(X_train, y_train):
        for variant, model in (("Baseline", baseline), ("Engineered", engineered)):
            predictions = model.predict(X_val)
            results.append({
                "Model": name,
                "Features": variant,
                "Count": len(model[:-1].get_feature_names_out()),
                "Accuracy": accuracy_score(y_val, predictions),
                "Macro F1": f1_score(y_val, predictions, average="macro"),
            })

        importance = permutation_importance(
            engineered, X_val, y_val, scoring="f1_macro",
            n_repeats=10, random_state=RANDOM_STATE,
        )
        importance_rows.append(pd.DataFrame({
            "Model": name,
            "Input": FEATURES,
            "Mean F1 drop": importance.importances_mean,
            "Std": importance.importances_std,
        }).sort_values("Mean F1 drop", ascending=False))

    print("\nValidation metrics")
    print(pd.DataFrame(results).to_string(index=False, float_format="{:.4f}".format))
    print("\nValidation permutation importance (engineered pipelines, raw inputs)")
    print(pd.concat(importance_rows).to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
