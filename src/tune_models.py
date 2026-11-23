"""
==============================================================
  tune_models.py – Tinh chỉnh siêu tham số (Hyperparameter Tuning)
  kết hợp 5-Fold Stratified Cross Validation
==============================================================
Phương pháp:
  - GridSearchCV cho các không gian tham số nhỏ (Logistic, LDA, QDA, k-NN)
  - RandomizedSearchCV cho Random Forest (không gian tham số rộng)
  - Đánh giá chéo 5-Fold (StratifiedKFold)
  - Tự động refit và lưu mô hình tốt nhất đạt F1-macro cao nhất
==============================================================
"""

import json
import warnings
import pandas as pd
from scipy.stats import randint

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)

from config import BASE_DIR, BEST_MODEL_PATH, RANDOM_STATE
from model_artifact import save_best_model
from train_logistic_regression import build_logistic_model
from trainlda import build_model as build_lda_model
from train_qda import build_qda_model, load_training_data
from train_random_forest import build_random_forest_model
from train_knn import build_knn_model

warnings.filterwarnings("ignore")


def main():
    print("=" * 65)
    print("  TỐI ƯU SIÊU THAM SỐ VỚI CROSS VALIDATION (5-FOLD)")
    print("=" * 65)

    X_train, X_val, y_train, y_val = load_training_data()
    X_dev = pd.concat([X_train, X_val], ignore_index=True)
    y_dev = pd.concat([y_train, y_val], ignore_index=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(X_dev, y_dev))

    # Cấu hình tuning: (model_pipeline, search_type, param_distribution_or_grid, n_iter)
    tuning_configs = {
        "Logistic Regression": {
            "model": build_logistic_model(),
            "type": "grid",
            "params": {"classifier__C": [0.01, 0.1, 1.0, 10.0, 50.0]},
        },
        "LDA": {
            "model": build_lda_model(),
            "type": "grid",
            "params": [
                {"classifier__solver": ["svd"]},
                {"classifier__solver": ["lsqr"], "classifier__shrinkage": [None, "auto", 0.01, 0.1, 0.5]},
            ],
        },
        "QDA": {
            "model": build_qda_model(),
            "type": "grid",
            "params": {"classifier__reg_param": [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]},
        },
        "k-NN": {
            "model": build_knn_model(),
            "type": "grid",
            "params": {
                "classifier__n_neighbors": [3, 5, 7, 9, 11],
                "classifier__weights": ["uniform", "distance"],
                "classifier__metric": ["euclidean", "manhattan", "minkowski"],
            },
        },
        "Random Forest": {
            "model": build_random_forest_model(),
            "type": "random",
            "n_iter": 12,
            "params": {
                "classifier__n_estimators": randint(100, 350),
                "classifier__max_depth": [None, 10, 15, 20, 25],
                "classifier__min_samples_split": [2, 4, 6],
                "classifier__min_samples_leaf": [1, 2, 3],
                "classifier__max_features": ["sqrt", "log2", None],
            },
        },
    }

    print(f"Tổng mẫu tập phát triển: {len(y_dev)} | K-Fold: {cv.n_splits} | Chỉ số tối ưu: Macro F1\n")
    rows = []
    best_overall_score = -1.0
    best_overall_model = None
    best_overall_name = ""

    for name, cfg in tuning_configs.items():
        print(f"🔍 Đang tối ưu {name} ({cfg['type'].upper()} SEARCH)...", flush=True)

        if cfg["type"] == "grid":
            search = GridSearchCV(
                cfg["model"],
                cfg["params"],
                scoring="f1_macro",
                cv=folds,
                refit=True,
                n_jobs=-1,
                error_score="raise",
            )
        else:
            search = RandomizedSearchCV(
                cfg["model"],
                cfg["params"],
                n_iter=cfg.get("n_iter", 10),
                scoring="f1_macro",
                cv=folds,
                refit=True,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                error_score="raise",
            )

        search.fit(X_dev, y_dev)
        index = search.best_index_
        metrics = search.cv_results_

        score = search.best_score_
        std = metrics["std_test_score"][index]

        rows.append({
            "model": name,
            "search_method": cfg["type"],
            "candidates_tested": len(metrics["params"]),
            "mean_cv_f1_macro": score,
            "std_cv_f1_macro": std,
            "best_params": json.dumps(search.best_params_, sort_keys=True, default=str),
        })

        print(f"  ↳ Điểm Macro F1 tốt nhất: {score:.4f} (±{std:.4f})")

        if score > best_overall_score:
            best_overall_score = score
            best_overall_model = search.best_estimator_
            best_overall_name = name

    summary = pd.DataFrame(rows).sort_values("mean_cv_f1_macro", ascending=False, kind="stable")
    output = BASE_DIR / "results" / "cv_tuning_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)

    print("\n" + "=" * 65)
    print("  BẢNG TỔNG HỢP KẾT QUẢ TUNING (CV F1-MACRO)")
    print("=" * 65)
    cols = ["model", "search_method", "candidates_tested", "mean_cv_f1_macro", "std_cv_f1_macro"]
    print(summary[cols].to_string(index=False, float_format="{:.4f}".format))
    print(f"\n[Kết quả] Đã lưu bảng kết quả vào: {output}")

    # Lưu best model tuned
    save_best_model(best_overall_model, best_overall_name)
    print(f"\n🏆 Mô hình tối ưu toàn diện nhất: {best_overall_name} (F1 = {best_overall_score:.4f})")
    print(f"💾 Đã lưu mô hình tối ưu đã tune tại: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
