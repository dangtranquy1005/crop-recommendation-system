"""
==============================================================
  train_qda.py – Huấn luyện mô hình QDA (Quadratic Discriminant Analysis)
  cho bài toán Crop Recommendation
==============================================================
Cấu trúc:
  1. Import thư viện
  2. Load dữ liệu đã chia sẵn
  3. Xây dựng Pipeline QDA
  4. Huấn luyện & Đánh giá
  5. Trực quan hoá (Confusion Matrix, Heatmap, Bar-chart chỉ số)
  6. Lưu mô hình (qda_model.pkl)
"""

# ==============================================================
# 1. Import thư viện
# ==============================================================
import sys
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Import cấu hình từ config.py của dự án
from config import (
    BASE_DIR,
    FEATURES,
    PROCESSED_DATA_DIR,
)

# Bỏ qua cảnh báo không ảnh hưởng kết quả
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ==============================================================
# 2. Load dữ liệu đã chia sẵn
# ==============================================================
def load_training_data():
    """
    Đọc dữ liệu đã được chia bởi preprocessing.py.
    Trả về: X_train, X_val, y_train, y_val
    """
    required_files = [
        PROCESSED_DATA_DIR / "X_train.csv",
        PROCESSED_DATA_DIR / "X_val.csv",
        PROCESSED_DATA_DIR / "y_train.csv",
        PROCESSED_DATA_DIR / "y_val.csv",
    ]

    # Kiểm tra tất cả file tồn tại trước khi đọc
    missing = [str(f) for f in required_files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            "Không tìm thấy các file dữ liệu sau. "
            "Hãy chạy preprocessing.py trước:\n" + "\n".join(missing)
        )

    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    X_val   = pd.read_csv(PROCESSED_DATA_DIR / "X_val.csv")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv")["label"]
    y_val   = pd.read_csv(PROCESSED_DATA_DIR / "y_val.csv")["label"]

    # Đảm bảo thứ tự đặc trưng khớp với config
    X_train = X_train[FEATURES]
    X_val   = X_val[FEATURES]

    return X_train, X_val, y_train, y_val


# ==============================================================
# Kiểm tra chất lượng dữ liệu
# ==============================================================
def inspect_data(X_train, X_val, y_train, y_val):
    """In thông tin chất lượng dữ liệu đầu vào."""
    print("\n--- KIỂM TRA DỮ LIỆU ---")
    print(f"Train : X={X_train.shape}, y={y_train.shape}")
    print(f"Val   : X={X_val.shape},   y={y_val.shape}")

    # Giá trị thiếu
    train_missing = X_train.isnull().sum().sum()
    val_missing   = X_val.isnull().sum().sum()
    print(f"\nGiá trị thiếu – Train: {train_missing}, Val: {val_missing}")
    if train_missing > 0:
        print("  Chi tiết Train:\n", X_train.isnull().sum())

    # Kiểm tra nhãn hợp lệ
    if y_train.isnull().any() or y_val.isnull().any():
        raise ValueError("Phát hiện nhãn bị thiếu (NaN). Kiểm tra lại dữ liệu.")

    print(f"\nSố lớp: {y_train.nunique()}")
    print("Phân bố nhãn (Train):\n", y_train.value_counts().to_string())


# ==============================================================
# 3. Xây dựng Pipeline QDA
# ==============================================================
def build_qda_model():
    """
    Tạo Pipeline gồm:
      - SimpleImputer  : xử lý giá trị thiếu bằng trung vị
      - StandardScaler : chuẩn hoá đặc trưng (giúp QDA ổn định hơn)
      - QDA            : mô hình phân loại

    Lưu ý về QDA:
      - reg_param=0.1 : regularisation để tránh lỗi ma trận hiệp phương sai
        suy biến (singular covariance matrix) khi các đặc trưng có
        đa cộng tuyến cao.
      - store_covariance=True : lưu ma trận hiệp phương sai để kiểm tra sau.
    """
    model = Pipeline(
        [
            ("imputer",    SimpleImputer(strategy="median")),
            ("scaler",     StandardScaler()),
            (
                "classifier",
                QuadraticDiscriminantAnalysis(
                    reg_param=0.1,         # regularisation chống suy biến
                    store_covariance=True, # lưu Σ_k để debug nếu cần
                    tol=1e-4,              # ngưỡng dừng
                ),
            ),
        ]
    )
    return model


# ==============================================================
# 4. Đánh giá mô hình
# ==============================================================
def evaluate_model(model, X, y, name=""):
    """
    Đánh giá mô hình với đầy đủ chỉ số:
      - Accuracy
      - Precision (macro & weighted)
      - Recall    (macro & weighted)
      - F1-score  (macro & weighted)

    Trả về: dict chứa tất cả chỉ số và y_pred
    """
    y_pred = model.predict(X)

    metrics = {
        "accuracy":           accuracy_score(y, y_pred),
        "precision_macro":    precision_score(y, y_pred, average="macro",    zero_division=0),
        "precision_weighted": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall_macro":       recall_score(y, y_pred, average="macro",    zero_division=0),
        "recall_weighted":    recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1_macro":           f1_score(y, y_pred, average="macro",    zero_division=0),
        "f1_weighted":        f1_score(y, y_pred, average="weighted", zero_division=0),
    }

    header = f"===== KẾT QUẢ QDA – {name} ====="
    print(f"\n{header}")
    print(f"  Accuracy           : {metrics['accuracy']:.4f}")
    print(f"  Precision (macro)  : {metrics['precision_macro']:.4f}")
    print(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"  Recall    (macro)  : {metrics['recall_macro']:.4f}")
    print(f"  Recall    (weighted): {metrics['recall_weighted']:.4f}")
    print(f"  F1-score  (macro)  : {metrics['f1_macro']:.4f}")
    print(f"  F1-score  (weighted): {metrics['f1_weighted']:.4f}")

    return metrics, y_pred


# ==============================================================
# 5. Trực quan hoá
# ==============================================================
def plot_confusion_matrix(y_true, y_pred, classes, save_dir):
    """
    Vẽ Confusion Matrix bằng Seaborn Heatmap.
    Lưu file: confusion_matrix_qda.png
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Confusion Matrix – QDA (Validation Set)", fontsize=15, pad=15)
    ax.set_xlabel("Nhãn Dự Đoán (Predicted)", fontsize=12)
    ax.set_ylabel("Nhãn Thực Tế (Actual)",    fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    plt.tight_layout()

    out_path = save_dir / "confusion_matrix_qda.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Biểu đồ] Confusion Matrix đã lưu: {out_path}")
    return out_path


def plot_metrics_bar(train_metrics, val_metrics, save_dir):
    """
    Vẽ Bar-chart so sánh Accuracy, Precision, Recall, F1 (macro & weighted)
    giữa Train và Validation.
    Lưu file: metrics_qda.png
    """
    metric_labels = [
        "Accuracy",
        "Precision\n(macro)",
        "Precision\n(weighted)",
        "Recall\n(macro)",
        "Recall\n(weighted)",
        "F1\n(macro)",
        "F1\n(weighted)",
    ]
    keys = [
        "accuracy",
        "precision_macro",
        "precision_weighted",
        "recall_macro",
        "recall_weighted",
        "f1_macro",
        "f1_weighted",
    ]

    train_vals = [train_metrics[k] for k in keys]
    val_vals   = [val_metrics[k]   for k in keys]

    x = np.arange(len(metric_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    bars_train = ax.bar(x - width / 2, train_vals, width, label="Train",      color="#4C72B0", alpha=0.85)
    bars_val   = ax.bar(x + width / 2, val_vals,   width, label="Validation", color="#DD8452", alpha=0.85)

    # Ghi giá trị lên mỗi cột
    for bar in bars_train:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=8,
        )
    for bar in bars_val:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_title("Các Chỉ Số Đánh Giá – QDA", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Điểm số", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = save_dir / "metrics_qda.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Biểu đồ] Metrics bar-chart đã lưu: {out_path}")
    return out_path


def plot_prediction_heatmap(y_true, y_pred, classes, save_dir):
    """
    Vẽ Heatmap hiển thị Nhãn Thực Tế vs Nhãn Dự Đoán (dạng ma trận %).
    Giống Confusion Matrix nhưng chuẩn hoá theo hàng (row-normalised).
    Lưu file: heatmap_actual_vs_predicted_qda.png
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    # Chuẩn hoá theo hàng để thấy tỷ lệ % từng lớp
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    # Tránh chia cho 0
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.5,
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title(
        "Heatmap: Nhãn Thực Tế vs Nhãn Dự Đoán – QDA\n(Chuẩn hoá theo hàng)",
        fontsize=14, pad=15,
    )
    ax.set_xlabel("Nhãn Dự Đoán (Predicted)", fontsize=12)
    ax.set_ylabel("Nhãn Thực Tế (Actual)",    fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    plt.tight_layout()

    out_path = save_dir / "heatmap_actual_vs_predicted_qda.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Biểu đồ] Heatmap thực tế vs dự đoán đã lưu: {out_path}")
    return out_path


# ==============================================================
# 6. Lưu mô hình
# ==============================================================
def save_model(model, model_dir):
    """
    Lưu mô hình QDA bằng joblib.
    Tên file: qda_model.pkl
    Trả về đường dẫn tuyệt đối.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "qda_model.pkl"
    joblib.dump(model, model_path)
    print(f"\n[Lưu mô hình] Đã lưu QDA tại: {model_path}")
    return model_path


def verify_model_load(model_path, X_sample):
    """
    Kiểm tra có thể load lại mô hình và dự đoán không.
    """
    loaded = joblib.load(model_path)
    _ = loaded.predict(X_sample)
    print("[Kiểm tra] Load lại mô hình thành công – sẵn sàng dự đoán.")
    return loaded


# ==============================================================
# MAIN
# ==============================================================
def main():
    print("=" * 60)
    print("  HUẤN LUYỆN MÔ HÌNH QDA – CROP RECOMMENDATION SYSTEM")
    print("=" * 60)

    # -----------------------------------------------------------
    # Bước 1: Load & kiểm tra dữ liệu
    # -----------------------------------------------------------
    X_train, X_val, y_train, y_val = load_training_data()
    inspect_data(X_train, X_val, y_train, y_val)

    # -----------------------------------------------------------
    # Bước 2: Xây dựng & huấn luyện mô hình
    # -----------------------------------------------------------
    print("\n--- XÂY DỰNG VÀ HUẤN LUYỆN QDA ---")
    model = build_qda_model()

    try:
        model.fit(X_train, y_train)
        print("Huấn luyện QDA thành công!")
    except Exception as exc:
        print(f"[LỖI] Huấn luyện thất bại: {exc}")
        print("Thử tăng reg_param để xử lý ma trận hiệp phương sai suy biến...")
        # Thử lại với reg_param cao hơn
        model.set_params(classifier__reg_param=0.5)
        model.fit(X_train, y_train)
        print("Huấn luyện QDA thành công với reg_param=0.5!")

    # -----------------------------------------------------------
    # Bước 3: Đánh giá mô hình
    # -----------------------------------------------------------
    train_metrics, _       = evaluate_model(model, X_train, y_train, "TRAIN")
    val_metrics,   y_pred  = evaluate_model(model, X_val,   y_val,   "VALIDATION")

    # Chênh lệch train – validation
    print(f"\nChênh lệch Accuracy (Train – Val): "
          f"{round(train_metrics['accuracy'] - val_metrics['accuracy'], 4)}")

    # Classification Report chi tiết
    print("\n--- CLASSIFICATION REPORT (Validation) ---")
    print(classification_report(y_val, y_pred, zero_division=0))

    # -----------------------------------------------------------
    # Bước 4: Trực quan hoá
    # -----------------------------------------------------------
    # Thư mục lưu ảnh
    figures_dir = BASE_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    classes = sorted(y_val.unique())

    print("\n--- TẠO BIỂU ĐỒ ---")
    plot_confusion_matrix(y_val, y_pred, classes, figures_dir)
    plot_prediction_heatmap(y_val, y_pred, classes, figures_dir)
    plot_metrics_bar(train_metrics, val_metrics, figures_dir)

    # -----------------------------------------------------------
    # Bước 5: Lưu mô hình
    # -----------------------------------------------------------
    model_dir  = BASE_DIR / "models"
    model_path = save_model(model, model_dir)
    verify_model_load(model_path, X_val.iloc[:5])

    # -----------------------------------------------------------
    # Tóm tắt kết quả cuối
    # -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  TÓM TẮT KẾT QUẢ CUỐI – QDA (Validation)")
    print("=" * 60)
    print(f"  Accuracy           : {val_metrics['accuracy']:.4f}")
    print(f"  Precision (macro)  : {val_metrics['precision_macro']:.4f}")
    print(f"  Precision (weighted): {val_metrics['precision_weighted']:.4f}")
    print(f"  Recall    (macro)  : {val_metrics['recall_macro']:.4f}")
    print(f"  Recall    (weighted): {val_metrics['recall_weighted']:.4f}")
    print(f"  F1-score  (macro)  : {val_metrics['f1_macro']:.4f}")
    print(f"  F1-score  (weighted): {val_metrics['f1_weighted']:.4f}")
    print(f"\n  Mô hình đã lưu tại : {model_path}")
    print(f"  Biểu đồ đã lưu tại : {figures_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

