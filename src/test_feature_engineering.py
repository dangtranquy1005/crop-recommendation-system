"""Compare paired validation predictions before and after feature engineering."""

import pandas as pd
from sklearn.metrics import f1_score

from evaluate_features import fit_feature_variants
from train_qda import load_training_data


def main():
    X_train, X_val, y_train, y_val = load_training_data()
    truth = y_val.to_numpy()
    rows = []

    for name, baseline, engineered in fit_feature_variants(X_train, y_train):
        before = baseline.predict(X_val)
        after = engineered.predict(X_val)
        correct_before = before == truth
        correct_after = after == truth
        accuracy_before = correct_before.mean()
        accuracy_after = correct_after.mean()
        f1_before = f1_score(truth, before, average="macro")
        f1_after = f1_score(truth, after, average="macro")
        feature_count = len(engineered[:-1].get_feature_names_out())

        rows.append({
            "Model": name,
            "Features": f"{X_val.shape[1]} -> {feature_count}",
            "Accuracy before": f"{accuracy_before:.2%}",
            "Accuracy after": f"{accuracy_after:.2%}",
            "Accuracy delta (pp)": f"{100 * (accuracy_after - accuracy_before):+.2f}",
            "Macro F1 before": f"{f1_before:.2%}",
            "Macro F1 after": f"{f1_after:.2%}",
            "Macro F1 delta (pp)": f"{100 * (f1_after - f1_before):+.2f}",
            "Wrong -> right": (~correct_before & correct_after).sum(),
            "Right -> wrong": (correct_before & ~correct_after).sum(),
        })

    print(f"Feature engineering comparison on {len(truth)} validation samples")
    print("Both variants use the same training data and model settings.")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
