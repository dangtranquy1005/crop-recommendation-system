"""
==============================================================
  train_knn.py – Huấn luyện mô hình k-Nearest Neighbors (k-NN)
  cho bài toán Crop Recommendation
==============================================================
Cấu trúc:
  1. Import thư viện
  2. Load dữ liệu đã chia sẵn
  3. Xây dựng Pipeline (Feature Engineering + k-NN)
  4. Huấn luyện & Đánh giá đầy đủ chỉ số
  5. Trực quan hoá (Confusion Matrix, Heatmap, Bar-chart)
  6. Lưu mô hình (knn_model.joblib)

Lưu ý về thuật toán:
  k-NN là thuật toán học có giám sát dựa trên khoảng cách
  (Instance-Based Learning / Lazy Learning), hoàn toàn khác biệt với:
    - LDA/QDA (Bayesian – mô hình xác suất)
    - Logistic Regression (tuyến tính – biên quyết định tuyến tính)
    - Random Forest (Ensemble – kết hợp nhiều cây quyết định)
  k-NN không xây dựng mô hình tường minh; phân loại bằng cách
  tìm k láng giềng gần nhất và bỏ phiếu (vote).
"""

# ==============================================================
# 1. Import thư viện
# ==============================================================
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import BASE_DIR, RANDOM_STATE
from feature_engineering import build_crop_pipeline
from trainlda import load_training_data

warnings.filterwarnings("ignore")

# Tạo thư mục figures nếu chưa có
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================
# 2. Kiểm tra chất lượng dữ liệu
# ==============================================================
def inspect_data(X_train, X_val, y_train, y_val):
    """In thông tin chất lượng dữ liệu đầu vào."""
    print("\n--- KIỂM TRA DỮ LIỆU ---")
    print(f"Train : X={X_train.shape}, y={y_train.shape}")
    print(f"Val   : X={X_val.shape},   y={y_val.shape}")

    train_missing = X_train.isnull().sum().sum()
    val_missing   = X_val.isnull().sum().sum()
    print(f"Giá trị thiếu – Train: {train_missing}, Val: {val_missing}")

    if y_train.isnull().any() or y_val.isnull().any():
        raise ValueError("Phát hiện nhãn bị thiếu (NaN). Kiểm tra lại dữ liệu.")

    print(f"Số lớp: {y_train.nunique()} | "
          f"Phân bố: {dict(y_train.value_counts().head(3))} ...")


# ==============================================================
# 3. Xây dựng Pipeline k-NN
# ==============================================================
def build_knn_model(use_engineered_features=True):
    """
    Pipeline:
      CropFeatureEngineer → Imputer + Scaler → KNeighborsClassifier

    Tham số k-NN ban đầu (sẽ được tune bằng GridSearchCV sau nếu cần):
      - n_neighbors=7       : số láng giềng; lẻ để tránh hoà phiếu
      - weights='distance'  : láng giềng gần hơn được ưu tiên hơn
      - metric='minkowski'  : khoảng cách Minkowski (p=2 → Euclidean mặc định)

    Lưu ý: k-NN rất nhạy với tỉ lệ đặc trưng (feature scale),
    nên bước Scaler trong pipeline là bắt buộc.
    """
    classifier = KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        metric="minkowski",
        n_jobs=-1,
    )
    return build_crop_pipeline(classifier, use_engineered_features)


# ==============================================================
# 4. Đánh giá mô hình
# ==============================================================
def evaluate_model(model, X, y, name=""):
    """
    Đánh giá đầy đủ:
      Accuracy, Precision/Recall/F1 (macro & weighted)
    Trả về dict chứa các chỉ số và y_pred.
    """
    y_pred = model.predict(X)

    metrics = {
        "y_pred":             y_pred,
        "accuracy":           accuracy_score(y, y_pred),
        "precision_macro":    precision_score(y, y_pred, average="macro",    zero_division=0),
        "precision_weighted": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall_macro":       recall_score(y, y_pred, average="macro",    zero_division=0),
        "recall_weighted":    recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1_macro":           f1_score(y, y_pred, average="macro",    zero_division=0),
        "f1_weighted":        f1_score(y, y_pred, average="weighted", zero_division=0),
    }

    print(f"\n===== KẾT QUẢ k-NN – {name} =====")
    print(f"  Accuracy            : {metrics['accuracy']:.4f}")
    print(f"  Precision (macro)   : {metrics['precision_macro']:.4f}")
    print(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"  Recall    (macro)   : {metrics['recall_macro']:.4f}")
    print(f"  Recall    (weighted): {metrics['recall_weighted']:.4f}")
    print(f"  F1-score  (macro)   : {metrics['f1_macro']:.4f}")
    print(f"  F1-score  (weighted): {metrics['f1_weighted']:.4f}")

    return metrics


# ==============================================================
# 5. Trực quan hoá
# ==============================================================
def plot_confusion_matrix(y_true, y_pred, classes):
    """Confusion Matrix (số lượng) bằng Seaborn."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                linewidths=0.4, linecolor="white", ax=ax)
    ax.set_title("Confusion Matrix – k-NN (Validation Set)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Nhãn Dự Đoán (Predicted)", fontsize=11)
    ax.set_ylabel("Nhãn Thực Tế (Actual)",    fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8.5)
    plt.yticks(rotation=0,  fontsize=8.5)
    plt.tight_layout()
    out = FIGURES_DIR / "confusion_matrix_knn.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Biểu đồ] Confusion Matrix          → {out}")


def plot_heatmap_normalized(y_true, y_pred, classes):
    """Heatmap chuẩn hoá % theo hàng (nhãn thực tế vs dự đoán)."""
    cm = confusion_matrix(y_true, y_pred, labels=classes).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1          # tránh chia cho 0
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=classes, yticklabels=classes,
                linewidths=0.4, vmin=0, vmax=1, ax=ax)
    ax.set_title(
        "Heatmap: Thực Tế vs Dự Đoán – k-NN\n(Chuẩn hoá theo hàng)",
        fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Nhãn Dự Đoán", fontsize=11)
    ax.set_ylabel("Nhãn Thực Tế", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8.5)
    plt.yticks(rotation=0,  fontsize=8.5)
    plt.tight_layout()
    out = FIGURES_DIR / "heatmap_actual_vs_predicted_knn.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Biểu đồ] Heatmap chuẩn hoá         → {out}")


def plot_metrics_bar(train_metrics, val_metrics):
    """Bar chart so sánh chỉ số Train vs Validation."""
    metric_labels = [
        "Accuracy",
        "Precision\n(macro)", "Precision\n(weighted)",
        "Recall\n(macro)",    "Recall\n(weighted)",
        "F1\n(macro)",        "F1\n(weighted)",
    ]
    keys = [
        "accuracy",
        "precision_macro", "precision_weighted",
        "recall_macro",    "recall_weighted",
        "f1_macro",        "f1_weighted",
    ]
    train_vals = [train_metrics[k] for k in keys]
    val_vals   = [val_metrics[k]   for k in keys]

    x = np.arange(len(metric_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))
    # Dùng màu xanh dương – phong cách đặc trưng cho k-NN
    b1 = ax.bar(x - width/2, train_vals, width, label="Train",      color="#1565C0", alpha=0.85)
    b2 = ax.bar(x + width/2, val_vals,   width, label="Validation", color="#64B5F6", alpha=0.85)

    # Ghi giá trị lên mỗi cột
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_title("Các Chỉ Số Đánh Giá – k-NN (k=7, weights='distance')",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Điểm số", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out = FIGURES_DIR / "metrics_knn.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Biểu đồ] Metrics bar-chart          → {out}")


# ==============================================================
# 6. Lưu mô hình
# ==============================================================
def save_model(model):
    """Lưu mô hình bằng joblib. Tên file: knn_model.joblib"""
    model_dir  = BASE_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "knn_model.joblib"
    joblib.dump(model, model_path)
    print(f"\n[Lưu mô hình] Đã lưu tại: {model_path}")

    # Kiểm tra load lại để đảm bảo file hợp lệ
    loaded = joblib.load(model_path)
    print("[Kiểm tra]   Load lại thành công – sẵn sàng dự đoán. ✅")
    return model_path


# ==============================================================
# MAIN
# ==============================================================
def main():
    print("=" * 60)
    print("  HUẤN LUYỆN MÔ HÌNH k-NN – CROP RECOMMENDATION")
    print("=" * 60)
    print("  Thuật toán: k-Nearest Neighbors (Instance-Based Learning)")
    print("  Phân loại dựa trên khoảng cách đến k láng giềng gần nhất")
    print("=" * 60)

    # Bước 1: Load & kiểm tra dữ liệu
    X_train, X_val, y_train, y_val = load_training_data()
    inspect_data(X_train, X_val, y_train, y_val)

    # Bước 2: Xây dựng & huấn luyện
    print("\n--- XÂY DỰNG VÀ HUẤN LUYỆN k-NN ---")
    print("  n_neighbors=7, weights='distance', metric='minkowski'")
    model = build_knn_model(use_engineered_features=True)
    model.fit(X_train, y_train)
    print("Huấn luyện k-NN thành công! ✅")

    # Bước 3: Đánh giá trên tập Train và Validation
    train_metrics = evaluate_model(model, X_train, y_train, "TRAIN")
    val_metrics   = evaluate_model(model, X_val,   y_val,   "VALIDATION")

    print(f"\nChênh lệch Accuracy (Train – Val): "
          f"{round(train_metrics['accuracy'] - val_metrics['accuracy'], 4)}")

    # Classification Report chi tiết trên tập Validation
    y_pred = val_metrics["y_pred"]
    print("\n--- CLASSIFICATION REPORT (Validation) ---")
    print(classification_report(y_val, y_pred, zero_division=0))

    # Bước 4: Trực quan hoá
    classes = sorted(y_val.unique())
    print("--- TẠO BIỂU ĐỒ ---")
    plot_confusion_matrix(y_val, y_pred, classes)
    plot_heatmap_normalized(y_val, y_pred, classes)
    plot_metrics_bar(train_metrics, val_metrics)

    # Bước 5: Lưu mô hình
    model_path = save_model(model)

    # Tóm tắt kết quả
    print("\n" + "=" * 60)
    print("  TÓM TẮT KẾT QUẢ – k-NN (Validation)")
    print("=" * 60)
    print(f"  Accuracy            : {val_metrics['accuracy']:.4f}")
    print(f"  Precision (macro)   : {val_metrics['precision_macro']:.4f}")
    print(f"  Precision (weighted): {val_metrics['precision_weighted']:.4f}")
    print(f"  Recall    (macro)   : {val_metrics['recall_macro']:.4f}")
    print(f"  Recall    (weighted): {val_metrics['recall_weighted']:.4f}")
    print(f"  F1-score  (macro)   : {val_metrics['f1_macro']:.4f}")
    print(f"  F1-score  (weighted): {val_metrics['f1_weighted']:.4f}")
    print(f"\n  Mô hình đã lưu tại : {model_path}")
    print(f"  Biểu đồ đã lưu tại : {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
