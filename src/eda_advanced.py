"""
==============================================================
  eda_advanced.py – Phân tích khám phá dữ liệu chuyên sâu (EDA)
  cho bài toán Crop Recommendation
==============================================================
Nội dung:
  1. Thống kê mô tả & kiểm tra dữ liệu khuyết thiếu / ngoại lai
  2. Phân tích ma trận tương quan giữa các đặc trưng (Correlation Heatmap)
  3. Phân tích ngoại lai bằng Boxplot (Outlier Detection)
  4. Phân tích phân phối của từng đặc trưng (KDE & Histogram)
  5. Xuất các biểu đồ trực quan vào thư mục figures/
==============================================================
"""

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import BASE_DIR, FEATURES, RAW_DATA_PATH, TARGET

warnings.filterwarnings("ignore")

FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("  KHÁM PHÁ VÀ PHÂN TÍCH DỮ LIỆU CHUYÊN SÂU (EDA)")
    print("=" * 60)

    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Dataset: {df.shape[0]} dòng, {df.shape[1]} cột")
    print(f"Số lớp cây trồng: {df[TARGET].nunique()}")
    print("\nThống kê mô tả các đặc trưng:")
    print(df[FEATURES].describe().T[["mean", "std", "min", "50%", "max"]].to_string())

    # 1. Ma trận tương quan (Correlation Heatmap)
    corr = df[FEATURES].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Ma Trận Tương Quan Pearson Giữa Các Đặc Trưng (Features Correlation)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    corr_path = FIGURES_DIR / "eda_correlation_matrix.png"
    fig.savefig(corr_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[EDA 1] Đã lưu ma trận tương quan: {corr_path}")

    # 2. Phân tích ngoại lai (Boxplot)
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    axes = axes.flatten()
    for i, col in enumerate(FEATURES):
        sns.boxplot(y=df[col], ax=axes[i], color="#4C72B0", flierprops={"marker": "o", "color": "red", "markersize": 4})
        axes[i].set_title(f"Boxplot: {col}", fontsize=11, fontweight="bold")
        axes[i].grid(axis="y", linestyle="--", alpha=0.5)

    # Ẩn subplot cuối cùng (thứ 8) vì chỉ có 7 đặc trưng
    fig.delaxes(axes[7])
    fig.suptitle("Phát Hiện Ngoại Lai Bằng Boxplot (Outlier Analysis)", fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()
    outlier_path = FIGURES_DIR / "eda_outliers_boxplot.png"
    fig.savefig(outlier_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA 2] Đã lưu biểu đồ ngoại lai: {outlier_path}")

    # 3. Phân phối đặc trưng (Histogram + KDE)
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    axes = axes.flatten()
    for i, col in enumerate(FEATURES):
        sns.histplot(df[col], kde=True, ax=axes[i], color="#2b5c8f", bins=25)
        axes[i].set_title(f"Phân phối: {col}", fontsize=11, fontweight="bold")
        axes[i].grid(axis="y", linestyle="--", alpha=0.5)

    fig.delaxes(axes[7])
    fig.suptitle("Phân Phối Của Từng Đặc Trưng (Feature Distributions)", fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()
    dist_path = FIGURES_DIR / "eda_feature_distributions.png"
    fig.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA 3] Đã lưu biểu đồ phân phối: {dist_path}")

    # Đếm số lượng ngoại lai theo chuẩn IQR
    print("\n--- Thống kê ngoại lai theo quy tắc 1.5*IQR ---")
    for col in FEATURES:
        q25, q75 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q75 - q25
        lower, upper = q25 - 1.5 * iqr, q75 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        print(f"  {col:<12}: {len(outliers):>3} mẫu ngoại lai ({len(outliers)/len(df)*100:.1f}%)")

    print("\n✅ Hoàn tất phân tích EDA chuyên sâu!")


if __name__ == "__main__":
    main()

