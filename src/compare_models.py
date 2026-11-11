"""
==============================================================
  compare_models.py – So sánh 3 mô hình bằng Confusion Matrix
  - Logistic Regression
  - LDA (Linear Discriminant Analysis)
  - QDA (Quadratic Discriminant Analysis)
==============================================================
"""

import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import BASE_DIR, PROCESSED_DATA_DIR, FEATURES

# ==============================================================
# Cấu hình các mô hình
# ==============================================================
MODELS = {
    "Logistic Regression": BASE_DIR / "models" / "logistic_model.joblib",
    "LDA":                 BASE_DIR / "models" / "lda_model.joblib",
    "QDA":                 BASE_DIR / "models" / "qda_model.pkl",
}

FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================
# Load dữ liệu Test
# ==============================================================
def load_test_data():
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")[FEATURES]
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv")["label"]
    print(f"Tập Test: {X_test.shape[0]} mẫu, {y_test.nunique()} lớp\n")
    return X_test, y_test


# ==============================================================
# Tính toán chỉ số đánh giá
# ==============================================================
def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    return {
        "y_pred":             y_pred,
        "accuracy":           accuracy_score(y, y_pred),
        "precision_macro":    precision_score(y, y_pred, average="macro",    zero_division=0),
        "precision_weighted": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall_macro":       recall_score(y, y_pred, average="macro",    zero_division=0),
        "recall_weighted":    recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1_macro":           f1_score(y, y_pred, average="macro",    zero_division=0),
        "f1_weighted":        f1_score(y, y_pred, average="weighted", zero_division=0),
    }


# ==============================================================
# Biểu đồ 1: 3 Confusion Matrix xếp cạnh nhau
# ==============================================================
def plot_confusion_matrices(results, classes, y_test):
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(22, 8))
    fig.suptitle(
        "So Sánh Confusion Matrix – 3 Mô Hình (Tập Test)",
        fontsize=16, fontweight="bold", y=1.01
    )

    cmap_list = ["Blues", "Greens", "Oranges"]

    for ax, (model_name, res), cmap in zip(axes, results.items(), cmap_list):
        cm = confusion_matrix(y_test, res["y_pred"], labels=classes)
        acc = res["accuracy"]

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            xticklabels=classes,
            yticklabels=classes,
            linewidths=0.4,
            linecolor="white",
            cbar=True,
            ax=ax,
        )
        ax.set_title(
            f"{model_name}\nAccuracy: {acc:.4f} ({acc*100:.2f}%)",
            fontsize=12, fontweight="bold", pad=10
        )
        ax.set_xlabel("Nhãn Dự Đoán", fontsize=10)
        ax.set_ylabel("Nhãn Thực Tế", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=7.5)
        ax.tick_params(axis="y", rotation=0,  labelsize=7.5)

    plt.tight_layout()
    out = FIGURES_DIR / "compare_confusion_matrix_3models.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Biểu đồ] Confusion Matrix (3 mô hình) → {out}")


# ==============================================================
# Biểu đồ 2: Confusion Matrix chuẩn hoá (% theo hàng)
# ==============================================================
def plot_normalized_confusion_matrices(results, classes, y_test):
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(22, 8))
    fig.suptitle(
        "Confusion Matrix Chuẩn Hoá (%) – 3 Mô Hình (Tập Test)",
        fontsize=16, fontweight="bold", y=1.01
    )

    cmap_list = ["Blues", "Greens", "Oranges"]

    for ax, (model_name, res), cmap in zip(axes, results.items(), cmap_list):
        cm = confusion_matrix(y_test, res["y_pred"], labels=classes).astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm / row_sums

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            xticklabels=classes,
            yticklabels=classes,
            linewidths=0.4,
            linecolor="white",
            vmin=0, vmax=1,
            cbar=True,
            ax=ax,
        )
        ax.set_title(
            f"{model_name}\nAccuracy: {res['accuracy']*100:.2f}%",
            fontsize=12, fontweight="bold", pad=10
        )
        ax.set_xlabel("Nhãn Dự Đoán", fontsize=10)
        ax.set_ylabel("Nhãn Thực Tế", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=7.5)
        ax.tick_params(axis="y", rotation=0,  labelsize=7.5)

    plt.tight_layout()
    out = FIGURES_DIR / "compare_confusion_matrix_normalized.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Biểu đồ] Confusion Matrix chuẩn hoá     → {out}")


# ==============================================================
# Biểu đồ 3: So sánh tất cả chỉ số (grouped bar chart)
# ==============================================================
def plot_metrics_comparison(results):
    model_names = list(results.keys())
    metric_keys = [
        ("accuracy",           "Accuracy"),
        ("precision_macro",    "Precision\n(macro)"),
        ("precision_weighted", "Precision\n(weighted)"),
        ("recall_macro",       "Recall\n(macro)"),
        ("recall_weighted",    "Recall\n(weighted)"),
        ("f1_macro",           "F1\n(macro)"),
        ("f1_weighted",        "F1\n(weighted)"),
    ]

    keys   = [k for k, _ in metric_keys]
    labels = [l for _, l in metric_keys]

    n_metrics = len(keys)
    n_models  = len(model_names)
    x = np.arange(n_metrics)
    width = 0.22
    colors = ["#4C72B0", "#55A868", "#DD8452"]

    fig, ax = plt.subplots(figsize=(15, 6))

    for i, (name, color) in enumerate(zip(model_names, colors)):
        vals = [results[name][k] for k in keys]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=color, alpha=0.88)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=7.5, rotation=0
            )

    ax.set_title("So Sánh Các Chỉ Số Đánh Giá – 3 Mô Hình (Tập Test)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0.80, 1.07)
    ax.set_ylabel("Điểm số", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out = FIGURES_DIR / "compare_metrics_3models.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Biểu đồ] So sánh chỉ số (bar chart)      → {out}")


# ==============================================================
# In bảng tổng kết
# ==============================================================
def print_summary_table(results):
    print("\n" + "=" * 72)
    print("  BẢNG SO SÁNH 3 MÔ HÌNH – TẬP TEST")
    print("=" * 72)
    header = f"{'Mô hình':<24} {'Accuracy':>9} {'Prec(M)':>9} {'Rec(M)':>8} {'F1(M)':>8} {'F1(W)':>8}"
    print(header)
    print("-" * 72)
    for name, res in results.items():
        print(
            f"{name:<24} "
            f"{res['accuracy']:>9.4f} "
            f"{res['precision_macro']:>9.4f} "
            f"{res['recall_macro']:>8.4f} "
            f"{res['f1_macro']:>8.4f} "
            f"{res['f1_weighted']:>8.4f}"
        )
    print("=" * 72)

    # Mô hình tốt nhất theo Accuracy
    best = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\n🏆 Mô hình tốt nhất (Accuracy): {best} "
          f"({results[best]['accuracy']*100:.2f}%)")


# ==============================================================
# MAIN
# ==============================================================
def main():
    print("=" * 60)
    print("  SO SÁNH 3 MÔ HÌNH – CONFUSION MATRIX")
    print("=" * 60)

    # Load tập Test
    X_test, y_test = load_test_data()
    classes = sorted(y_test.unique())

    # Load và đánh giá từng mô hình
    results = {}
    for model_name, model_path in MODELS.items():
        if not model_path.exists():
            print(f"⚠️  Không tìm thấy: {model_path} – bỏ qua {model_name}")
            continue
        model = joblib.load(model_path)
        results[model_name] = compute_metrics(model, X_test, y_test)
        print(f"✅ Đã đánh giá: {model_name}  "
              f"(Accuracy = {results[model_name]['accuracy']:.4f})")

    if not results:
        print("❌ Không có mô hình nào. Hãy train trước.")
        return

    # Vẽ biểu đồ
    print("\n--- TẠO BIỂU ĐỒ ---")
    plot_confusion_matrices(results, classes, y_test)
    plot_normalized_confusion_matrices(results, classes, y_test)
    plot_metrics_comparison(results)

    # Bảng tổng kết
    print_summary_table(results)

    print(f"\n📁 Tất cả biểu đồ đã lưu tại: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

