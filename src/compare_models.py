"""Refit CV-selected configurations and evaluate them on the held-out test split."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import BASE_DIR, FEATURES, PROCESSED_DATA_DIR, TARGET
from evaluate_features import MODEL_BUILDERS, feature_importance_table
from train_qda import load_training_data
from model_artifact import save_best_model

RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
METRICS = {
    "accuracy": "Accuracy",
    "precision_macro": "Precision macro",
    "recall_macro": "Recall macro",
    "f1_macro": "F1 macro",
    "roc_auc_ovr_macro": "ROC-AUC OVR macro",
}


def compute_metrics(y, predictions, probabilities, classes):
    return {
        "accuracy": accuracy_score(y, predictions),
        "precision_macro": precision_score(y, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(y, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(y, predictions, average="macro", zero_division=0),
        "roc_auc_ovr_macro": roc_auc_score(
            y, probabilities, labels=classes, multi_class="ovr", average="macro"
        ),
    }


def plot_confusion_matrices(matrices, classes, *, normalize=False):
    n_rows = (len(matrices) + 1) // 2
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(18, 7.5 * n_rows), layout="constrained", squeeze=False,
    )
    description = "Row-normalized (%)" if normalize else "Counts"
    fig.suptitle(f"Test confusion matrices | {description}", fontsize=17)

    for ax, (name, matrix) in zip(axes.flat, matrices.items()):
        values = matrix
        if normalize:
            support = matrix.sum(axis=1, keepdims=True)
            values = 100 * matrix / np.maximum(support, 1)
        annotations = np.where(matrix > 0, values.round().astype(int).astype(str), "")
        sns.heatmap(
            values, annot=annotations, fmt="", cmap="Blues", square=True, vmin=0,
            vmax=100 if normalize else None, xticklabels=classes, yticklabels=classes,
            annot_kws={"fontsize": 7}, ax=ax,
        )
        ax.set(title=name, xlabel="Predicted crop", ylabel="Actual crop")
        ax.tick_params(axis="x", labelrotation=60, labelsize=8)
        ax.tick_params(axis="y", labelrotation=0, labelsize=8)

    for ax in axes.flat[len(matrices):]:
        ax.set_visible(False)

    suffix = "_normalized" if normalize else ""
    fig.savefig(FIGURES_DIR / f"final_confusion_matrices{suffix}.png", dpi=150)
    plt.close(fig)


def plot_metrics(summary):
    scores = summary.set_index("model")[list(METRICS)].rename(columns=METRICS)
    fig, ax = plt.subplots(
        figsize=(11, max(4, len(summary) * 0.8)), layout="constrained",
    )
    sns.heatmap(scores, annot=True, fmt=".4f", cmap="Blues", vmax=1, ax=ax)
    ax.set(title="Test metrics by model", xlabel="", ylabel="")
    ax.tick_params(axis="y", labelrotation=0)
    fig.savefig(FIGURES_DIR / "final_test_metrics.png", dpi=150)
    plt.close(fig)


def fit_tuned_models():
    """Refit the configurations in the CV summary on train plus validation."""
    cv_summary = pd.read_csv(RESULTS_DIR / "cv_tuning_summary.csv")
    # Accept CSVs written before the search method had its own column.
    cv_summary["model"] = cv_summary["model"].replace({
        "Random Forest (RandomizedSearch)": "Random Forest",
    })
    if len(cv_summary) != len(MODEL_BUILDERS) or set(cv_summary["model"]) != set(MODEL_BUILDERS):
        expected = ", ".join(MODEL_BUILDERS)
        raise ValueError(
            f"The CV summary must contain each model exactly once: {expected}. "
            "Run src/tune_models.py."
        )
    cv_summary = cv_summary.sort_values("mean_cv_f1_macro", ascending=False, kind="stable")
    X_train, X_val, y_train, y_val = load_training_data()
    X_dev = pd.concat([X_train, X_val], ignore_index=True)
    y_dev = pd.concat([y_train, y_val], ignore_index=True)
    models = {}
    for row in cv_summary.itertuples(index=False):
        model = MODEL_BUILDERS[row.model]().set_params(**json.loads(row.best_params))
        models[row.model] = model.fit(X_dev, y_dev)
    winner = cv_summary.iloc[0]["model"]
    models[winner] = save_best_model(models[winner], winner)
    return cv_summary, models


def main():
    cv_summary, models = fit_tuned_models()
    winner = cv_summary.iloc[0]["model"]
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")[FEATURES]
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv")[TARGET]
    classes = models[winner].classes_

    rows = []
    matrices = {}
    errors = []
    for row in cv_summary.itertuples(index=False):
        model = models[row.model]
        predictions = model.predict(X_test)
        scores = compute_metrics(y_test, predictions, model.predict_proba(X_test), model.classes_)
        rows.append({
            "model": row.model,
            "selected_by_cv": row.model == winner,
            "mean_cv_f1_macro": row.mean_cv_f1_macro,
            "best_params": row.best_params,
            **scores,
        })
        matrix = confusion_matrix(y_test, predictions, labels=classes)
        matrices[row.model] = matrix
        errors.extend({
            "model": row.model,
            "actual_crop": classes[actual],
            "predicted_crop": classes[predicted],
            "count": matrix[actual, predicted],
        } for actual, predicted in zip(*np.nonzero(matrix)) if actual != predicted)

        if row.model == winner:
            final_importance = feature_importance_table(model, X_test, y_test)

    X_train, X_val, y_train, y_val = load_training_data()
    reference = MODEL_BUILDERS[winner]().fit(X_train, y_train)
    validation_importance = feature_importance_table(reference, X_val, y_val)
    importance = pd.concat([
        validation_importance.assign(source="validation_reference"),
        final_importance.assign(source="test_tuned"),
    ], ignore_index=True).assign(model=winner)

    summary = pd.DataFrame(rows)
    confusion_pairs = pd.DataFrame(
        errors, columns=["model", "actual_crop", "predicted_crop", "count"]
    ).sort_values(
        ["count", "model", "actual_crop", "predicted_crop"],
        ascending=[False, True, True, True],
    )
    summary.to_csv(RESULTS_DIR / "final_test_metrics.csv", index=False)
    confusion_pairs.to_csv(RESULTS_DIR / "final_confusion_pairs.csv", index=False)
    importance.to_csv(RESULTS_DIR / "final_permutation_importance.csv", index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrices(matrices, classes)
    plot_confusion_matrices(matrices, classes, normalize=True)
    plot_metrics(summary)

    print(f"Test split: {len(y_test)} samples | {len(classes)} crops")
    print(f"Selected by cross-validation: {winner}\n")
    print(summary.to_string(index=False, float_format="{:.6f}".format))
    print("\nConfusion pairs for the selected model")
    print(confusion_pairs[confusion_pairs["model"] == winner].to_string(index=False))
    print("\nPermutation importance for the selected model")
    print(importance.to_string(index=False, float_format="{:.4f}".format))
    return models, X_test, y_test


if __name__ == "__main__":
    main()
