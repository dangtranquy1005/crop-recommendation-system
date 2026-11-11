"""
==============================================================
  predict_qda.py – Chạy thử mô hình QDA đã huấn luyện
  Chế độ 1: Đánh giá trên tập Test (X_test.csv / y_test.csv)
  Chế độ 2: Nhập tay thông số để dự đoán cây trồng
==============================================================
"""

import sys
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report

# Import cấu hình dự án
from config import BASE_DIR, PROCESSED_DATA_DIR, FEATURES

# ==============================================================
# Đường dẫn mô hình
# ==============================================================
MODEL_PATH = BASE_DIR / "models" / "qda_model.pkl"


def load_model():
    """Load mô hình QDA từ file .pkl."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy mô hình tại: {MODEL_PATH}\n"
            "Hãy chạy train_qda.py trước."
        )
    model = joblib.load(MODEL_PATH)
    print(f"✅ Đã load mô hình từ: {MODEL_PATH}\n")
    return model


# ==============================================================
# Chế độ 1: Đánh giá trên tập Test
# ==============================================================
def run_test_set(model):
    """Dự đoán toàn bộ tập Test và in kết quả."""
    x_path = PROCESSED_DATA_DIR / "X_test.csv"
    y_path = PROCESSED_DATA_DIR / "y_test.csv"

    if not x_path.exists() or not y_path.exists():
        print("⚠️  Không tìm thấy X_test.csv / y_test.csv. Bỏ qua chế độ này.")
        return

    X_test = pd.read_csv(x_path)[FEATURES]
    y_test = pd.read_csv(y_path)["label"]

    print("=" * 55)
    print("  CHẾ ĐỘ 1: ĐÁNH GIÁ TRÊN TẬP TEST")
    print("=" * 55)
    print(f"Số mẫu test : {len(X_test)}")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy trên Test Set: {acc:.4f} ({acc*100:.2f}%)\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # In 10 dự đoán mẫu
    print("── 10 dự đoán mẫu đầu tiên ──")
    sample = X_test.head(10).copy()
    sample["Thực tế"]  = y_test.values[:10]
    sample["Dự đoán"]  = y_pred[:10]
    sample["Đúng/Sai"] = sample.apply(
        lambda r: "✅" if r["Thực tế"] == r["Dự đoán"] else "❌", axis=1
    )
    print(sample[["Thực tế", "Dự đoán", "Đúng/Sai"]].to_string(index=False))
    print()


# ==============================================================
# Chế độ 2: Nhập tay để dự đoán
# ==============================================================
FEATURE_META = {
    "N":           ("Hàm lượng Nitơ (N) trong đất",          0,   140),
    "P":           ("Hàm lượng Phốt-pho (P) trong đất",       5,   145),
    "K":           ("Hàm lượng Kali (K) trong đất",           5,   205),
    "temperature": ("Nhiệt độ (°C)",                           8.8,  43.7),
    "humidity":    ("Độ ẩm tương đối (%)",                    14.3, 100.0),
    "ph":          ("Độ pH đất",                               3.5,   9.9),
    "rainfall":    ("Lượng mưa (mm)",                         20.2, 298.6),
}

SAMPLE_PRESETS = {
    "1": {"N": 90,  "P": 42,  "K": 43,  "temperature": 20.9, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9},
    "2": {"N": 0,   "P": 45,  "K": 20,  "temperature": 25.6, "humidity": 52.0, "ph": 7.0, "rainfall":  67.5},
    "3": {"N": 20,  "P": 68,  "K": 20,  "temperature": 25.0, "humidity": 80.0, "ph": 6.5, "rainfall": 150.0},
}


def get_float(prompt, lo, hi):
    """Nhận giá trị số thực từ người dùng với kiểm tra khoảng hợp lệ."""
    while True:
        raw = input(f"  {prompt} [{lo}–{hi}]: ").strip()
        try:
            val = float(raw)
            if lo <= val <= hi:
                return val
            print(f"    ⚠️  Vui lòng nhập giá trị trong khoảng [{lo}, {hi}]")
        except ValueError:
            print("    ⚠️  Giá trị không hợp lệ. Hãy nhập số.")


def run_interactive(model):
    """Vòng lặp nhập tay dự đoán cây trồng."""
    print("=" * 55)
    print("  CHẾ ĐỘ 2: NHẬP TAY DỰ ĐOÁN CÂY TRỒNG")
    print("=" * 55)

    while True:
        print("\nTùy chọn:")
        print("  [1] Dùng mẫu thử #1 (đất giàu N, ẩm cao → lúa?)")
        print("  [2] Dùng mẫu thử #2 (đất khô, N thấp → ?)")
        print("  [3] Dùng mẫu thử #3 (đất trung bình)")
        print("  [m] Nhập tay từng thông số")
        print("  [q] Thoát")

        choice = input("\nLựa chọn của bạn: ").strip().lower()

        if choice == "q":
            print("\n👋 Thoát chương trình.")
            break

        if choice in SAMPLE_PRESETS:
            data = SAMPLE_PRESETS[choice]
            print("\nThông số đầu vào:")
            for feat, val in data.items():
                desc = FEATURE_META[feat][0]
                print(f"  {desc}: {val}")
        elif choice == "m":
            print("\nNhập từng thông số (nhấn Enter sau mỗi giá trị):")
            data = {}
            for feat, (desc, lo, hi) in FEATURE_META.items():
                data[feat] = get_float(desc, lo, hi)
        else:
            print("  ⚠️  Lựa chọn không hợp lệ.")
            continue

        # Tạo DataFrame và dự đoán
        X_input = pd.DataFrame([data])[FEATURES]
        prediction  = model.predict(X_input)[0]
        proba       = model.predict_proba(X_input)[0]
        classes     = model.classes_
        top3_idx    = proba.argsort()[::-1][:3]

        print(f"\n🌾 Cây trồng được gợi ý: {prediction.upper()}")
        print("\nTop 3 khả năng cao nhất:")
        for rank, idx in enumerate(top3_idx, 1):
            print(f"  {rank}. {classes[idx]:<15} ({proba[idx]*100:.1f}%)")
        print()
        input("Nhấn Enter để tiếp tục...")


# ==============================================================
# MAIN
# ==============================================================
def main():
    print("\n" + "=" * 55)
    print("  CROP RECOMMENDATION – THỬ NGHIỆM MÔ HÌNH QDA")
    print("=" * 55)

    # Load mô hình
    model = load_model()

    # Chế độ 1: Test set
    run_test_set(model)

    # Chế độ 2: Nhập tay
    run_interactive(model)


if __name__ == "__main__":
    main()

