"""
==============================================================
  predict.py – Deployment script chính thức cho model tốt nhất
  Dự án: Crop Recommendation System
  Model: Random Forest (~98.79% accuracy trên tập Test)
==============================================================
Cấu trúc:
  1. load_best_model()       – Load model từ file .joblib
  2. validate_input()        – Kiểm tra và xác thực đầu vào
  3. predict_crop()          – Dự đoán cây trồng (đơn lẻ / batch)
  4. predict_from_csv()      – Dự đoán từ file CSV
  5. main()                  – Demo với 3 preset mẫu + batch test
==============================================================
"""

# ==============================================================
# 1. Import thư viện
# ==============================================================
import sys
import os

# Thêm thư mục src vào Python path để import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd

from config import BASE_DIR, FEATURES

# ==============================================================
# Hằng số – Range vật lý hợp lệ của từng feature
# ==============================================================
FEATURE_RANGES = {
    "N":           (0,    140),   # Hàm lượng Nitơ (mg/kg)
    "P":           (5,    145),   # Hàm lượng Phốt-pho (mg/kg)
    "K":           (5,    205),   # Hàm lượng Kali (mg/kg)
    "temperature": (8,     44),   # Nhiệt độ (°C)
    "humidity":    (14,   100),   # Độ ẩm (%)
    "ph":          (3.5,  9.9),   # Độ pH đất
    "rainfall":    (20,   299),   # Lượng mưa (mm)
}

# Đường dẫn model mặc định
MODEL_PATH = BASE_DIR / "models" / "random_forest_model.joblib"


# ==============================================================
# 2. load_best_model()
# ==============================================================
def load_best_model():
    """
    Load model Random Forest tốt nhất từ file .joblib.

    Returns
    -------
    model : sklearn Pipeline
        Pipeline đã được huấn luyện gồm:
        CropFeatureEngineer → ColumnTransformer(Imputer+Scaler) → RandomForestClassifier

    Raises
    ------
    FileNotFoundError
        Nếu file model không tồn tại.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file model tại: {MODEL_PATH}\n"
            "Hãy chạy train_random_forest.py trước."
        )

    print(f"[Model] Đang load model từ: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print("[Model] Load thành công! ✅")
    return model


# ==============================================================
# 3. validate_input()
# ==============================================================
def validate_input(input_dict):
    """
    Kiểm tra và xác thực dữ liệu đầu vào trước khi dự đoán.

    Parameters
    ----------
    input_dict : dict
        Dictionary chứa 7 features: N, P, K, temperature, humidity, ph, rainfall

    Returns
    -------
    input_dict : dict
        Dictionary đã được xác thực (giá trị được ép sang float)

    Raises
    ------
    ValueError
        Nếu thiếu features, giá trị không phải số, hoặc nằm ngoài range vật lý.
    """
    # Kiểm tra đủ 7 features
    missing = [f for f in FEATURES if f not in input_dict]
    if missing:
        raise ValueError(
            f"Thiếu {len(missing)} feature(s): {missing}\n"
            f"Yêu cầu đủ 7 features: {FEATURES}"
        )

    # Kiểm tra giá trị là số và nằm trong range vật lý
    validated = {}
    for feature in FEATURES:
        raw_val = input_dict[feature]

        # Kiểm tra kiểu số
        try:
            value = float(raw_val)
        except (TypeError, ValueError):
            raise ValueError(
                f"Feature '{feature}' phải là số, nhưng nhận được: '{raw_val}' "
                f"(kiểu {type(raw_val).__name__})"
            )

        # Kiểm tra NaN / Inf
        if np.isnan(value) or np.isinf(value):
            raise ValueError(
                f"Feature '{feature}' có giá trị không hợp lệ: {value}"
            )

        # Kiểm tra range vật lý
        lo, hi = FEATURE_RANGES[feature]
        if not (lo <= value <= hi):
            raise ValueError(
                f"Feature '{feature}' = {value} nằm ngoài khoảng hợp lệ [{lo}, {hi}].\n"
                f"  → Vui lòng kiểm tra lại giá trị đầu vào."
            )

        validated[feature] = value

    return validated


# ==============================================================
# 4. predict_crop()
# ==============================================================
def predict_crop(input_data, model=None):
    """
    Dự đoán loại cây trồng phù hợp cho điều kiện đất/khí hậu cho trước.

    Parameters
    ----------
    input_data : dict hoặc list of dict
        - dict   : dự đoán đơn lẻ 1 mẫu
        - list   : dự đoán batch nhiều mẫu cùng lúc
        Mỗi dict cần có 7 keys: N, P, K, temperature, humidity, ph, rainfall

    model : sklearn Pipeline, optional
        Model đã load sẵn. Nếu None thì tự động load từ file.

    Returns
    -------
    results : list of dict
        Mỗi phần tử là dict:
        {
            "prediction" : str,              # Loại cây trồng được dự đoán
            "confidence" : float,            # Xác suất của lớp được chọn (0-1)
            "top3"       : list of tuple,    # Top-3 (tên_cây, xác_suất)
        }

    Raises
    ------
    ValueError
        Nếu dữ liệu đầu vào không hợp lệ (qua validate_input).
    """
    # Load model nếu chưa có
    if model is None:
        model = load_best_model()

    # Chuẩn hoá đầu vào thành list of dicts
    is_single = isinstance(input_data, dict)
    if is_single:
        samples = [input_data]
    else:
        samples = list(input_data)

    # Xác thực từng mẫu
    validated_samples = []
    for i, sample in enumerate(samples):
        try:
            validated_samples.append(validate_input(sample))
        except ValueError as e:
            raise ValueError(f"Mẫu #{i}: {e}") from e

    # Chuyển thành DataFrame
    df_input = pd.DataFrame(validated_samples, columns=FEATURES)

    # Dự đoán – lấy nhãn và xác suất
    predictions = model.predict(df_input)
    proba_matrix = model.predict_proba(df_input)   # shape (n_samples, n_classes)
    class_labels = model.classes_                  # danh sách tên lớp theo thứ tự

    # Xây dựng kết quả
    results = []
    for pred_label, proba_row in zip(predictions, proba_matrix):
        # Confidence: xác suất của lớp được chọn
        confidence = float(proba_row.max())

        # Top-3: sắp xếp giảm dần theo xác suất
        top3_indices = np.argsort(proba_row)[::-1][:3]
        top3 = [(class_labels[idx], float(proba_row[idx])) for idx in top3_indices]

        results.append({
            "prediction": str(pred_label),
            "confidence": confidence,
            "top3":       top3,
        })

    return results


# ==============================================================
# 5. predict_from_csv()
# ==============================================================
def predict_from_csv(csv_path, output_path=None):
    """
    Đọc file CSV, chạy batch prediction, tuỳ chọn lưu kết quả ra file.

    Parameters
    ----------
    csv_path : str hoặc Path
        Đường dẫn file CSV đầu vào. File phải có các cột: N, P, K,
        temperature, humidity, ph, rainfall.
        Có thể có thêm cột 'label' để tính accuracy sau dự đoán.

    output_path : str hoặc Path, optional
        Đường dẫn file CSV đầu ra. Nếu None thì không lưu.

    Returns
    -------
    df_result : pd.DataFrame
        DataFrame gốc với 3 cột được thêm vào:
        'predicted_crop', 'confidence', 'top3_predictions'

    Raises
    ------
    FileNotFoundError
        Nếu file CSV đầu vào không tồn tại.
    ValueError
        Nếu CSV thiếu cột bắt buộc.
    """
    csv_path = str(csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file CSV: {csv_path}")

    print(f"\n[CSV] Đang đọc dữ liệu từ: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[CSV] Đọc được {len(df)} dòng, {len(df.columns)} cột.")

    # Kiểm tra cột bắt buộc
    missing_cols = [f for f in FEATURES if f not in df.columns]
    if missing_cols:
        raise ValueError(
            f"File CSV thiếu {len(missing_cols)} cột bắt buộc: {missing_cols}\n"
            f"Yêu cầu: {FEATURES}"
        )

    # Chuyển sang list of dicts để predict
    records = df[FEATURES].to_dict(orient="records")

    print(f"[CSV] Đang dự đoán cho {len(records)} mẫu...")
    model = load_best_model()
    results = predict_crop(records, model=model)

    # Gắn kết quả vào DataFrame
    df["predicted_crop"] = [r["prediction"]  for r in results]
    df["confidence"]     = [r["confidence"]  for r in results]
    df["top3_predictions"] = [
        str(r["top3"]) for r in results
    ]

    # Tính accuracy nếu có cột nhãn thật
    if "label" in df.columns:
        correct = (df["predicted_crop"] == df["label"]).sum()
        acc = correct / len(df)
        print(f"[CSV] Accuracy trên file này: {correct}/{len(df)} = {acc:.4f} ({acc*100:.2f}%)")

    # Lưu kết quả nếu được chỉ định
    if output_path is not None:
        df.to_csv(output_path, index=False)
        print(f"[CSV] Kết quả đã lưu tại: {output_path}")

    return df


# ==============================================================
# 6. Hàm tiện ích – In kết quả đẹp
# ==============================================================
def print_prediction_result(sample_name, input_dict, result):
    """
    In kết quả dự đoán theo định dạng dễ đọc.

    Parameters
    ----------
    sample_name : str
        Tên mô tả của mẫu (ví dụ: 'Đất lúa').
    input_dict : dict
        Dữ liệu đầu vào.
    result : dict
        Kết quả dự đoán từ predict_crop().
    """
    print(f"\n{'─'*55}")
    print(f"  🌱 MẪU: {sample_name}")
    print(f"{'─'*55}")
    print(f"  Đầu vào:")
    for feat in FEATURES:
        unit_map = {
            "N": "mg/kg", "P": "mg/kg", "K": "mg/kg",
            "temperature": "°C", "humidity": "%",
            "ph": "", "rainfall": "mm"
        }
        print(f"    {feat:12s} = {input_dict[feat]:>8.2f} {unit_map[feat]}")
    print()
    print(f"  🏆 DỰ ĐOÁN   : {result['prediction'].upper()}")
    print(f"  📊 ĐỘ TIN CẬY: {result['confidence']*100:.2f}%")
    print()
    print("  📋 Top-3 cây trồng phù hợp:")
    for rank, (crop, prob) in enumerate(result["top3"], start=1):
        bar = "█" * int(prob * 30)
        print(f"    {rank}. {crop:<18s} {prob*100:6.2f}%  {bar}")
    print(f"{'─'*55}")


# ==============================================================
# 7. main() – Demo
# ==============================================================
def main():
    """
    Hàm demo chính:
      1. Dự đoán cho 3 preset mẫu (đất lúa, đất khô, đất trung bình)
      2. Batch prediction trên tập Test và in accuracy
    """
    print("=" * 55)
    print("  CROP RECOMMENDATION – DEPLOYMENT DEMO")
    print("  Model: Random Forest (~98.79% Test Accuracy)")
    print("=" * 55)

    # ----------------------------------------------------------
    # Load model một lần, dùng lại cho tất cả
    # ----------------------------------------------------------
    try:
        model = load_best_model()
    except FileNotFoundError as e:
        print(f"\n[LỖI] {e}")
        sys.exit(1)

    # ----------------------------------------------------------
    # Phần 1: Dự đoán 3 preset mẫu
    # ----------------------------------------------------------
    print("\n" + "=" * 55)
    print("  PHẦN 1: DỰ ĐOÁN ĐƠN LẺ (3 MẪU PRESET)")
    print("=" * 55)

    sample_presets = [
        {
            "name": "Đất lúa (ẩm, nhiều mưa, pH trung tính)",
            "data": {
                "N":           82.0,    # Nitơ trung bình của rice
                "P":           48.0,    # Phốt-pho trung bình của rice
                "K":           40.0,    # Kali trung bình của rice
                "temperature": 23.5,   # Nhiệt độ ôn hòa
                "humidity":    82.0,    # Độ ẩm cao
                "ph":           6.5,   # pH gần trung tính
                "rainfall":   236.0,   # Lượng mưa cao
            },
        },
        {
            "name": "Đất khô (nghèo dinh dưỡng, ít mưa, nóng)",
            "data": {
                "N":           18.0,    # Nitơ thấp – điển hình đất khô
                "P":           20.0,
                "K":           25.0,
                "temperature": 38.0,   # Nhiệt độ cao
                "humidity":    30.0,   # Độ ẩm thấp
                "ph":           7.2,
                "rainfall":    35.0,   # Lượng mưa thấp
            },
        },
        {
            "name": "Đất trung bình (cân bằng dinh dưỡng)",
            "data": {
                "N":           50.0,
                "P":           53.0,
                "K":           48.0,
                "temperature": 25.0,
                "humidity":    65.0,
                "ph":           6.8,
                "rainfall":    103.0,
            },
        },
    ]

    for preset in sample_presets:
        try:
            results = predict_crop(preset["data"], model=model)
            print_prediction_result(preset["name"], preset["data"], results[0])
        except ValueError as e:
            print(f"\n[LỖI xác thực] {preset['name']}: {e}")

    # ----------------------------------------------------------
    # Phần 2: Batch prediction trên tập Test
    # ----------------------------------------------------------
    print("\n" + "=" * 55)
    print("  PHẦN 2: BATCH PREDICTION TRÊN TẬP TEST")
    print("=" * 55)

    x_test_path = BASE_DIR / "data" / "processed" / "X_test.csv"
    y_test_path = BASE_DIR / "data" / "processed" / "y_test.csv"

    if not x_test_path.exists() or not y_test_path.exists():
        print(f"[CẢNH BÁO] Không tìm thấy tập Test tại {x_test_path}")
        print("  Bỏ qua phần 2. Hãy chạy preprocessing.py trước.")
    else:
        try:
            X_test = pd.read_csv(x_test_path)
            y_test = pd.read_csv(y_test_path).squeeze()   # Series

            print(f"Tập Test: {X_test.shape[0]} mẫu, {X_test.shape[1]} features")

            # Dự đoán batch
            test_records = X_test[FEATURES].to_dict(orient="records")
            test_results = predict_crop(test_records, model=model)
            y_pred = [r["prediction"] for r in test_results]

            # Tính accuracy
            y_test_list = y_test.tolist()
            correct = sum(p == t for p, t in zip(y_pred, y_test_list))
            accuracy = correct / len(y_test_list)

            print(f"\nKết quả trên tập Test:")
            print(f"  Tổng mẫu     : {len(y_test_list)}")
            print(f"  Đúng         : {correct}")
            print(f"  Sai          : {len(y_test_list) - correct}")
            print(f"  Accuracy     : {accuracy:.4f}  ({accuracy*100:.2f}%)")

            # Thống kê theo lớp – những lớp bị sai
            errors = [(t, p) for t, p in zip(y_test_list, y_pred) if t != p]
            if errors:
                print(f"\nCác dự đoán sai ({len(errors)} mẫu):")
                error_df = pd.DataFrame(errors, columns=["actual", "predicted"])
                error_summary = (
                    error_df.groupby(["actual", "predicted"])
                    .size()
                    .reset_index(name="count")
                    .sort_values("count", ascending=False)
                )
                print(error_summary.to_string(index=False))
            else:
                print("\n✅ Không có lỗi nào trên tập Test!")

            # Lưu kết quả batch
            results_dir = BASE_DIR / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            output_path = results_dir / "test_predictions_rf.csv"

            df_out = X_test.copy()
            df_out["label"]          = y_test_list
            df_out["predicted_crop"] = y_pred
            df_out["confidence"]     = [r["confidence"] for r in test_results]
            df_out["correct"]        = df_out["label"] == df_out["predicted_crop"]
            df_out.to_csv(output_path, index=False)
            print(f"\n[Lưu] Kết quả batch đã lưu tại: {output_path}")

        except Exception as e:
            print(f"[LỖI] Không thể chạy batch prediction: {e}")
            import traceback
            traceback.print_exc()

    # ----------------------------------------------------------
    # Tổng kết
    # ----------------------------------------------------------
    print("\n" + "=" * 55)
    print("  DEMO HOÀN THÀNH")
    print("=" * 55)
    print("  Cách dùng trong code:")
    print("    from predict import predict_crop, load_best_model")
    print("    model = load_best_model()")
    print("    result = predict_crop({'N':82,'P':48,'K':40,")
    print("                           'temperature':23.5,'humidity':82,")
    print("                           'ph':6.5,'rainfall':236}, model)")
    print("    print(result[0]['prediction'])  # → 'rice'")
    print("=" * 55)


# ==============================================================
# Entry point
# ==============================================================
if __name__ == "__main__":
    main()
