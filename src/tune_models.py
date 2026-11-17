"""Tune the four crop classifiers using shared stratified CV folds."""

import json

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from config import BASE_DIR, RANDOM_STATE
from train_logistic_regression import build_logistic_model
from train_qda import build_qda_model, load_training_data
from train_random_forest import build_random_forest_model
from trainlda import build_model as build_lda_model


def main():
    X_train, X_val, y_train, y_val = load_training_data()
    X_dev = pd.concat([X_train, X_val], ignore_index=True)
    y_dev = pd.concat([y_train, y_val], ignore_index=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(X_dev, y_dev))

    searches = {
        "Logistic Regression": (
            build_logistic_model(),
            {"classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        "LDA": (
            build_lda_model(),
            [
                {"classifier__solver": ["svd"]},
                {
                    "classifier__solver": ["lsqr"],
                    "classifier__shrinkage": [None, "auto", 0.01, 0.1, 0.5],
                },
            ],
        ),
        "QDA": (
            build_qda_model(),
            # A CV training fold has a singular covariance with no regularization.
            {"classifier__reg_param": [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]},
        ),
        "Random Forest": (
            build_random_forest_model(),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 12, 20],
                "classifier__min_samples_leaf": [1, 2],
            },
        ),
    }

    print(f"{len(y_dev)} development samples | {cv.n_splits} folds | Macro F1")
    rows = []
    for name, (model, param_grid) in searches.items():
        print(f"Searching {name}...", flush=True)
        search = GridSearchCV(
            model,
            param_grid,
            scoring="f1_macro",
            cv=folds,
            refit=False,
            error_score="raise",
        ).fit(X_dev, y_dev)
        index = search.best_index_
        metrics = search.cv_results_
        rows.append({
            "model": name,
            "candidates": len(metrics["params"]),
            "mean_cv_f1_macro": search.best_score_,
            "std_cv_f1_macro": metrics["std_test_score"][index],
            **{
                f"fold_{fold + 1}_f1_macro": metrics[f"split{fold}_test_score"][index]
                for fold in range(cv.n_splits)
            },
            "best_params": json.dumps(search.best_params_, sort_keys=True),
        })
        print(f"Best mean F1: {search.best_score_:.4f}", flush=True)

    summary = pd.DataFrame(rows).sort_values(
        "mean_cv_f1_macro", ascending=False, kind="stable"
    )
    output = BASE_DIR / "results" / "cv_tuning_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)

    columns = ["model", "candidates", "mean_cv_f1_macro", "std_cv_f1_macro", "best_params"]
    print("\n" + summary[columns].to_string(index=False, float_format="{:.4f}".format))
    print(f"\nSaved CV summary: {output}")


if __name__ == "__main__":
    main()
