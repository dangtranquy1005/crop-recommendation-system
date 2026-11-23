"""
==============================================================
  error_analysis.py – Đánh giá chuyên sâu & Phân tích lỗi
  cho dự án Crop Recommendation
==============================================================
Cấu trúc:
  1. Import thư viện & cấu hình đường dẫn
  2. Load tất cả models và tập Test
  3. Bảng so sánh đa chỉ số (Accuracy, Precision, Recall, F1,
     Cohen's Kappa, MCC)
  4. ROC Curve (macro OvR) cho tất cả models
  5. Top-N cặp nhầm lẫn nhiều nhất (RF)
  6. Giải thích tại sao RF vượt trội
"""

# ==============================================================
# 1. Import thư viện
# ==============================================================
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

# Thêm thư mục src vào sys.path để import config
sys.path.insert(0, os.path.dirname(__file__))
from config import BASE_DIR, FEATURES, PROCESSED_DATA_DIR
from compare_models import fit_tuned_models, main as evaluate_final_models

# ==============================================================
# Cấu hình đường dẫn
# ==============================================================
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"

# Tạo thư mục nếu chưa tồn tại
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Tên các model và đường dẫn file tương ứng
# ==============================================================
# 2. Load models & tập Test
# ==============================================================
def load_models():
    """Refit the five tuned configurations using the shared evaluation workflow."""
    _, models = fit_tuned_models()
    return models


def load_test_data():
    """
    Load tập Test từ PROCESSED_DATA_DIR:
      - X_test.csv  : features
      - y_test.csv  : nhãn (cột 'label')
    Trả về (X_test: DataFrame, y_test: Series).
    """
    x_path = PROCESSED_DATA_DIR / "X_test.csv"
    y_path = PROCESSED_DATA_DIR / "y_test.csv"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu test tại {PROCESSED_DATA_DIR}. "
            "Hãy chạy preprocessing.py trước."
        )

    X_test = pd.read_csv(x_path)
    y_test = pd.read_csv(y_path).squeeze()   # chuyển DataFrame 1 cột → Series

    # Nếu cột tên là 'label', lấy Series đó
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test["label"]

    print(f"\n[Dữ liệu Test] X={X_test.shape}, y={y_test.shape}")
    print(f"[Dữ liệu Test] Số lớp: {y_test.nunique()} | "
          f"Phân bố (5 lớp đầu): {dict(y_test.value_counts().head(5))}")
    return X_test, y_test


# ==============================================================
# 3. Bảng so sánh đa chỉ số
# ==============================================================
def compute_metrics(model, X, y, model_name):
    """
    Tính đầy đủ các chỉ số đánh giá cho một model:
      Accuracy, Precision/Recall/F1 (macro & weighted),
      Cohen's Kappa, Matthews Correlation Coefficient (MCC).
    Trả về dict chỉ số.
    """
    y_pred = model.predict(X)

    metrics = {
        "Model":               model_name,
        "Accuracy":            accuracy_score(y, y_pred),
        "Precision (macro)":   precision_score(y, y_pred, average="macro",    zero_division=0),
        "Precision (weighted)":precision_score(y, y_pred, average="weighted", zero_division=0),
        "Recall (macro)":      recall_score(y, y_pred, average="macro",    zero_division=0),
        "Recall (weighted)":   recall_score(y, y_pred, average="weighted", zero_division=0),
        "F1 (macro)":          f1_score(y, y_pred, average="macro",    zero_division=0),
        "F1 (weighted)":       f1_score(y, y_pred, average="weighted", zero_division=0),
        "Cohen's Kappa":       cohen_kappa_score(y, y_pred),
        "MCC":                 matthews_corrcoef(y, y_pred),
        "ROC-AUC (macro OvR)": roc_auc_score(
            y, model.predict_proba(X), labels=model.classes_,
            multi_class="ovr", average="macro",
        ),
        # Lưu y_pred để dùng lại (không lưu vào CSV)
        "_y_pred":             y_pred,
    }
    return metrics


def compare_all_models(models, X_test, y_test):
    """
    Tính chỉ số cho tất cả models, in bảng đẹp ra màn hình,
    lưu CSV vào results/model_comparison_full.csv.
    Trả về list[dict] metrics của từng model.
    """
    print("\n" + "=" * 70)
    print("  SO SÁNH ĐA CHỈ SỐ – TẤT CẢ MODELS (Test Set)")
    print("=" * 70)

    all_metrics = []
    for name, model in models.items():
        m = compute_metrics(model, X_test, y_test, name)
        all_metrics.append(m)

    # Tạo DataFrame để in & lưu (loại bỏ cột ẩn _y_pred)
    display_cols = [
        "Model", "Accuracy",
        "Precision (macro)", "Precision (weighted)",
        "Recall (macro)",    "Recall (weighted)",
        "F1 (macro)",        "F1 (weighted)",
        "Cohen's Kappa",     "MCC", "ROC-AUC (macro OvR)",
    ]
    df = pd.DataFrame(all_metrics)[display_cols]

    # In bảng đẹp
    print(df.to_string(index=False, float_format="{:.6f}".format))

    # Lưu CSV
    csv_path = RESULTS_DIR / "model_comparison_full.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[Kết quả] Đã lưu bảng so sánh → {csv_path}")

    return all_metrics


# ==============================================================
# 4. ROC Curve (macro OvR) cho tất cả models
# ==============================================================
def plot_roc_all_models(models, X_test, y_test):
    """
    Vẽ ROC Curve macro-average (One-vs-Rest) cho tất cả models
    trên cùng 1 figure. Dùng predict_proba; nếu model không có
    predict_proba thì dùng decision_function; nếu cũng không có
    thì bỏ qua và in cảnh báo.
    Lưu: figures/roc_curve_all_models.png
    """
    # Lấy danh sách lớp theo thứ tự
    classes = sorted(y_test.unique())
    n_classes = len(classes)

    # Binarise nhãn thực tế theo OvR
    y_bin = label_binarize(y_test, classes=classes)   # shape (n_samples, n_classes)

    # Màu sắc cho các model
    colors = ["#E63946", "#2196F3", "#4CAF50", "#FF9800", "#8E44AD"]

    fig, ax = plt.subplots(figsize=(10, 7))

    for idx, (name, model) in enumerate(models.items()):
        color = colors[idx % len(colors)]
        if not np.array_equal(model.classes_, classes):
            raise ValueError(f"Class order mismatch for {name}")

        # Thử lấy xác suất dự đoán
        try:
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)      # (n_samples, n_classes)
            elif hasattr(model, "decision_function"):
                # Dùng decision_function thay thế (LDA/QDA đôi khi dùng cách này)
                y_score = model.decision_function(X_test)
                # Nếu chỉ trả về (n_samples,) → không dùng được OvR
                if y_score.ndim == 1:
                    print(f"[CẢNH BÁO] {name}: decision_function 1D, bỏ qua ROC.")
                    continue
            else:
                print(f"[CẢNH BÁO] {name}: không có predict_proba/decision_function. Bỏ qua ROC.")
                continue

            # Tính macro-average ROC AUC
            fpr_dict, tpr_dict = {}, {}
            for i in range(n_classes):
                fpr_dict[i], tpr_dict[i], _ = roc_curve(y_bin[:, i], y_score[:, i])

            # Nội suy để tính macro
            all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
            mean_tpr /= n_classes

            macro_auc = roc_auc_score(
                y_test, y_score, labels=model.classes_,
                multi_class="ovr", average="macro",
            )

            ax.plot(all_fpr, mean_tpr, color=color, lw=2,
                    label=f"{name} (AUC = {macro_auc:.4f})")

        except Exception as e:
            print(f"[CẢNH BÁO] Không tính được ROC cho {name}: {e}")
            continue

    # Đường diagonal tham chiếu
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random (AUC = 0.50)")

    ax.set_title("ROC Curve – Macro-Average (One-vs-Rest)\nTất cả models – Test Set",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()
    out = FIGURES_DIR / "roc_curve_all_models.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Biểu đồ] ROC Curve (tất cả models) → {out}")


# ==============================================================
# 5. Top-N cặp nhầm lẫn nhiều nhất (RF)
# ==============================================================
def analyze_top_confusion_pairs(rf_model, X_test, y_test, top_n=10):
    """
    Tìm top_n cặp (actual, predicted) bị nhầm lẫn nhiều nhất
    của Random Forest trên tập Test.
    - In bảng ra màn hình
    - Vẽ bar chart: figures/top_confusion_pairs_rf.png
    Trả về DataFrame các cặp nhầm lẫn.
    """
    y_pred = rf_model.predict(X_test)
    classes = sorted(y_test.unique())

    # Ma trận nhầm lẫn → DataFrame
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)

    # Tìm các cặp nhầm lẫn (actual ≠ predicted)
    pairs = []
    for actual in classes:
        for predicted in classes:
            if actual != predicted:
                count = cm_df.loc[actual, predicted]
                if count > 0:
                    pairs.append({
                        "Actual":    actual,
                        "Predicted": predicted,
                        "Count":     int(count),
                    })

    pairs_df = (pd.DataFrame(pairs, columns=["Actual", "Predicted", "Count"])
                  .sort_values("Count", ascending=False)
                  .head(top_n)
                  .reset_index(drop=True))
    pairs_df.index += 1   # đánh số từ 1
    pairs_df.to_csv(RESULTS_DIR / "top_confusion_pairs_rf.csv", index=False)
    predictions = X_test.reset_index(drop=True).copy()
    predictions["actual"] = np.asarray(y_test)
    predictions["predicted"] = y_pred
    predictions["correct"] = predictions["actual"] == predictions["predicted"]
    predictions.to_csv(RESULTS_DIR / "test_predictions_rf.csv", index=False)

    # In bảng
    print("\n" + "=" * 55)
    print(f"  TOP {top_n} CẶP NHẦM LẪN NHIỀU NHẤT – RANDOM FOREST")
    print("=" * 55)
    print(pairs_df.to_string())

    # Vẽ bar chart
    labels = [f"{r['Actual']} →\n{r['Predicted']}"
              for _, r in pairs_df.iterrows()]
    counts = pairs_df["Count"].tolist()

    fig, ax = plt.subplots(figsize=(10, max(3, 0.6 * len(counts) + 2)))
    if pairs_df.empty:
        ax.text(0.5, 0.5, "No misclassified samples", ha="center", transform=ax.transAxes)
    bars = ax.barh(labels[::-1], counts[::-1],
                   color=plt.cm.Reds(np.linspace(0.4, 0.85, len(counts))))
    for bar, cnt in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(cnt), va="center", fontsize=10)

    ax.set_title(f"Top {top_n} Cặp Nhầm Lẫn Nhiều Nhất – Random Forest (Test Set)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Số lần nhầm lẫn", fontsize=11)
    ax.set_xlim(0, max(counts, default=1) * 1.25)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out = FIGURES_DIR / "top_confusion_pairs_rf.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Biểu đồ] Top confusion pairs (RF) → {out}")

    return pairs_df


# ==============================================================
# 6. Giải thích tại sao RF vượt trội
# ==============================================================
def explain_rf_superiority(all_metrics, models, X_test, y_test):
    """
    Phân tích tự động tại sao Random Forest vượt trội:
      (a) So sánh Accuracy giữa RF và các model khác
      (b) Feature Importance của RF (top features)
      (c) Các lớp mà RF dự đoán đúng nhưng các model khác sai
    """
    print("\n" + "=" * 65)
    print("  PHÂN TÍCH: TẠI SAO RANDOM FOREST VƯỢT TRỘI?")
    print("=" * 65)

    # (a) So sánh Accuracy
    acc_dict = {m["Model"]: m["Accuracy"] for m in all_metrics}
    rf_acc   = acc_dict.get("Random Forest", None)

    if rf_acc is None:
        print("[CẢNH BÁO] Không tìm thấy metrics của Random Forest.")
        return

    print("\n--- (a) So sánh Accuracy ---")
    for model_name, acc in acc_dict.items():
        diff = rf_acc - acc
        print(
            f"  {model_name:25s}: {acc:.4f}  "
            f"(Accuracy RF trừ mô hình này: {diff:+.4f})"
        )

    # (b) Feature Importance của RF
    print("\n--- (b) Feature Importance của Random Forest ---")
    rf_model = models.get("Random Forest")
    if rf_model is not None:
        try:
            # Lấy classifier từ pipeline
            rf_clf = rf_model.named_steps.get("classifier") or rf_model.steps[-1][1]
            importances = rf_clf.feature_importances_

            # Lấy tên feature sau pipeline (nếu không được thì dùng FEATURES gốc)
            try:
                feat_names = list(rf_model[:-1].get_feature_names_out())
            except Exception:
                feat_names = list(FEATURES)

            # Ghép tên và mức độ quan trọng, sắp xếp giảm dần
            feat_imp = sorted(zip(feat_names, importances),
                              key=lambda x: x[1], reverse=True)

            for rank, (fname, imp) in enumerate(feat_imp, start=1):
                bar = "█" * int(imp * 50)
                print(f"  {rank:2d}. {fname:35s}: {imp:.4f}  {bar}")
        except Exception as e:
            print(f"[LỖI] Không lấy được feature importance: {e}")
    else:
        print("[CẢNH BÁO] Model Random Forest không được load.")

    # (c) Các lớp mà RF đúng nhưng các model khác sai
    print("\n--- (c) Lớp mà RF đúng, các model khác sai ---")
    # Lấy y_pred của từng model
    pred_map = {m["Model"]: m["_y_pred"] for m in all_metrics}
    rf_pred  = pred_map.get("Random Forest")
    y_arr    = np.array(y_test)

    if rf_pred is None:
        print("[CẢNH BÁO] Không có dự đoán của Random Forest.")
        return

    # Với mỗi mẫu: RF đúng và ít nhất 1 model khác sai
    rf_correct   = (rf_pred == y_arr)
    other_models = {k: v for k, v in pred_map.items() if k != "Random Forest"}

    advantage_classes = {}
    for name, preds in other_models.items():
        other_wrong  = (preds != y_arr)
        advantage_mask = rf_correct & other_wrong    # RF đúng, model kia sai
        if advantage_mask.sum() == 0:
            continue
        # Thống kê theo lớp
        for cls in np.unique(y_arr[advantage_mask]):
            mask_cls = advantage_mask & (y_arr == cls)
            advantage_classes.setdefault(cls, {})
            advantage_classes[cls][name] = int(mask_cls.sum())

    if advantage_classes:
        print(f"  {'Lớp cây trồng':20s} | {'Số mẫu RF đúng – model khác sai':45s}")
        print("  " + "-" * 68)
        for cls, details in sorted(advantage_classes.items(),
                                   key=lambda x: sum(x[1].values()), reverse=True)[:15]:
            detail_str = "  ".join(f"{k}: {v}" for k, v in details.items())
            print(f"  {cls:20s} | {detail_str}")
    else:
        print("  Không tìm thấy lớp nào RF vượt trội rõ rệt so với tất cả các model khác.")

    print("\n--- Cấu hình Random Forest đã chốt qua CV ---")
    params = rf_model.named_steps["classifier"].get_params()
    for key in ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"):
        print(f"  {key}: {params[key]}")
    print("\n--- Permutation importance trên tập test (F1 macro, 10 lần) ---")
    importance = pd.read_csv(RESULTS_DIR / "final_permutation_importance.csv")
    importance = importance[
        (importance["source"] == "test_tuned") & (importance["model"] == "Random Forest")
    ]
    print(importance.to_string(index=False, float_format="{:.4f}".format))


# ==============================================================
# MAIN
# ==============================================================
def main():
    print("=" * 70)
    print("  ĐÁNH GIÁ CHUYÊN SÂU & PHÂN TÍCH LỖI – CROP RECOMMENDATION")
    print("=" * 70)

    # Reuse the exact fitted instances for tables, matrices, ROC and error analysis.
    models, X_test, y_test = evaluate_final_models()

    # --- Bước 2: Bảng so sánh đa chỉ số ---
    print("\n[Bước 2] Tính đa chỉ số cho tất cả models...")
    all_metrics = compare_all_models(models, X_test, y_test)

    # --- Bước 3: ROC Curve ---
    print("\n[Bước 3] Vẽ ROC Curve (macro OvR)...")
    plot_roc_all_models(models, X_test, y_test)

    # --- Bước 4: Top confusion pairs (RF) ---
    rf_model = models.get("Random Forest")
    if rf_model is not None:
        print("\n[Bước 4] Phân tích cặp nhầm lẫn nhiều nhất (RF)...")
        analyze_top_confusion_pairs(rf_model, X_test, y_test, top_n=10)
    else:
        print("\n[Bước 4] Bỏ qua: Random Forest không được load.")

    # --- Bước 5: Giải thích RF vượt trội ---
    print("\n[Bước 5] Giải thích tại sao RF vượt trội...")
    explain_rf_superiority(all_metrics, models, X_test, y_test)

    # --- Tóm tắt ---
    print("\n" + "=" * 70)
    print("  HOÀN TẤT PHÂN TÍCH LỖI")
    print("=" * 70)
    print(f"  Kết quả CSV → {RESULTS_DIR / 'model_comparison_full.csv'}")
    print(f"  Biểu đồ    → {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
