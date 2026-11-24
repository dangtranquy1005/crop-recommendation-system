"""
==============================================================
  app.py – Giao diện Hệ thống Khuyến nghị Cây trồng
  Ứng dụng Machine Learning trong Nông nghiệp Chính xác
==============================================================
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from app_constants import CROP_INFO, SAMPLE_PRESETS
from config import FEATURES

# ==============================================================
# Thiết lập cấu hình trang
# ==============================================================
st.set_page_config(
    page_title="Crop Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tùy biến phong cách hiển thị (Clean, Neutral, Professional Enterprise UI)
st.markdown("""
<style>
    /* Tổng thể và Typography */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1e293b;
    }
    .app-header {
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 16px;
        margin-bottom: 24px;
    }
    .app-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Khối thẻ thông tin */
    .card-panel {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .card-header-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 6px;
    }
    .primary-crop-name {
        font-size: 2rem;
        font-weight: 800;
        color: #1b4332;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .scientific-name {
        font-size: 0.95rem;
        font-style: italic;
        color: #475569;
        margin-bottom: 14px;
    }

    /* Badge chỉ số */
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
        margin-right: 6px;
    }
    .badge-success {
        background-color: #dcfce7;
        color: #166534;
        border-color: #bbf7d0;
    }

    /* Tiêu đề nhóm thông số */
    .section-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #334155;
        border-left: 3px solid #2d6a4f;
        padding-left: 8px;
        margin: 16px 0 12px 0;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================
# Bộ nhớ đệm tải mô hình (Model Cache)
# ==============================================================
@st.cache_resource
def get_model_registry():
    """Tải danh sách mô hình từ thư mục models/."""
    models = {}
    path_mapping = {
        "Random Forest (Khuyến nghị - 99.70%)": [
            BASE_DIR / "models" / "best_tuned_model.joblib",
            BASE_DIR / "models" / "random_forest_model.joblib"
        ],
        "Quadratic Discriminant Analysis (97.27%)": [
            BASE_DIR / "models" / "qda_model.joblib",
            BASE_DIR / "models" / "qda_model.pkl"
        ],
        "Linear Discriminant Analysis (96.97%)": [
            BASE_DIR / "models" / "lda_model.joblib"
        ],
        "k-Nearest Neighbors (95.76%)": [
            BASE_DIR / "models" / "knn_model.joblib"
        ],
        "Logistic Regression (94.85%)": [
            BASE_DIR / "models" / "logistic_model.joblib"
        ]
    }

    for name, candidate_paths in path_mapping.items():
        for path in candidate_paths:
            if path.exists():
                try:
                    models[name] = joblib.load(path)
                    break
                except Exception:
                    pass
    return models


models_dict = get_model_registry()

# ==============================================================
# Header chính của ứng dụng
# ==============================================================
st.markdown("""
<div class="app-header">
    <h1 class="app-title">Hệ Thống Phân Tích & Khuyến Nghị Cây Trồng</h1>
    <p class="app-subtitle">Nền tảng hỗ trợ ra quyết định nông học dựa trên mô hình học máy và thông số môi trường thực tế</p>
</div>
""", unsafe_allow_html=True)

if not models_dict:
    st.error("Không tìm thấy tệp mô hình đã huấn luyện trong thư mục models/. Vui lòng kiểm tra lại quá trình huấn luyện.")
    st.stop()

# ==============================================================
# Thanh điều hướng Tab chính
# ==============================================================
tab_prediction, tab_batch, tab_eda, tab_benchmark = st.tabs([
    "Dự đoán cá thể",
    "Xử lý dữ liệu theo lô",
    "Phân tích dữ liệu gốc",
    "Đánh giá & Hiệu năng mô hình"
])


# ==============================================================
# TAB 1: DỰ ĐOÁN ĐƠN LẺ
# ==============================================================
with tab_prediction:
    col_config, col_output = st.columns([1.1, 1], gap="large")

    with col_config:
        st.markdown('<div class="section-title">Cấu hình mô hình & Dữ liệu đầu vào</div>', unsafe_allow_html=True)

        selected_model_key = st.selectbox(
            "Thuật toán phân loại:",
            options=list(models_dict.keys()),
            index=0
        )
        selected_model = models_dict[selected_model_key]

        preset_key = st.selectbox(
            "Tập thông số tham chiếu nhanh:",
            options=["-- Nhập thông số thủ công --"] + list(SAMPLE_PRESETS.keys())
        )

        preset_vals = SAMPLE_PRESETS.get(preset_key, {})

        st.markdown('<div class="section-title">Hàm lượng dinh dưỡng thổ nhưỡng (NPK)</div>', unsafe_allow_html=True)
        col_n, col_p, col_k = st.columns(3)
        with col_n:
            val_n = st.number_input(
                "Đạm tổng số - N (mg/kg)",
                min_value=0.0, max_value=140.0,
                value=float(preset_vals.get("N", 60.0)),
                step=1.0
            )
        with col_p:
            val_p = st.number_input(
                "Lân hữu dụng - P (mg/kg)",
                min_value=5.0, max_value=145.0,
                value=float(preset_vals.get("P", 40.0)),
                step=1.0
            )
        with col_k:
            val_k = st.number_input(
                "Kali trao đổi - K (mg/kg)",
                min_value=5.0, max_value=205.0,
                value=float(preset_vals.get("K", 40.0)),
                step=1.0
            )

        st.markdown('<div class="section-title">Điều kiện khí hậu và tính chất đất</div>', unsafe_allow_html=True)
        col_temp, col_hum = st.columns(2)
        with col_temp:
            val_temp = st.number_input(
                "Nhiệt độ không khí (°C)",
                min_value=8.0, max_value=44.0,
                value=float(preset_vals.get("temperature", 24.5)),
                step=0.5
            )
        with col_hum:
            val_hum = st.number_input(
                "Độ ẩm tương đối (%)",
                min_value=14.0, max_value=100.0,
                value=float(preset_vals.get("humidity", 78.0)),
                step=0.5
            )

        col_ph, col_rain = st.columns(2)
        with col_ph:
            val_ph = st.number_input(
                "Độ phản ứng của đất (pH)",
                min_value=3.5, max_value=9.9,
                value=float(preset_vals.get("ph", 6.5)),
                step=0.1
            )
        with col_rain:
            val_rain = st.number_input(
                "Lượng mưa tích lũy (mm)",
                min_value=20.0, max_value=300.0,
                value=float(preset_vals.get("rainfall", 160.0)),
                step=1.0
            )

        st.markdown("<br>", unsafe_allow_html=True)
        btn_run = st.button("Thực hiện phân tích khuyến nghị", type="primary", use_container_width=True)

    with col_output:
        st.markdown('<div class="section-title">Kết quả phân tích nông học</div>', unsafe_allow_html=True)

        sample_features = {
            "N": val_n, "P": val_p, "K": val_k,
            "temperature": val_temp, "humidity": val_hum,
            "ph": val_ph, "rainfall": val_rain
        }
        input_frame = pd.DataFrame([sample_features])[FEATURES]

        try:
            prediction_raw = selected_model.predict(input_frame)[0]
            crop_id = str(prediction_raw).lower()
            crop_data = CROP_INFO.get(crop_id, {
                "vi": crop_id.capitalize(),
                "scientific": "Chưa xác định",
                "category": "Nông sản",
                "soil_type": "Thông tin đang cập nhật",
                "climate_requirement": "Thông tin đang cập nhật"
            })

            # Tính xác suất phân loại nếu mô hình hỗ trợ
            has_proba = hasattr(selected_model, "predict_proba")
            if has_proba:
                probabilities = selected_model.predict_proba(input_frame)[0]
                classes = selected_model.classes_
                top_indices = np.argsort(probabilities)[::-1][:3]
                primary_confidence = probabilities[top_indices[0]] * 100
            else:
                top_indices = []
                primary_confidence = 100.0

            # Hiển thị thẻ kết quả chính
            st.markdown(f"""
            <div class="card-panel">
                <div class="card-header-label">Cây trồng thích nghi tối ưu</div>
                <div class="primary-crop-name">{crop_data['vi']}</div>
                <div class="scientific-name">Tên khoa học: {crop_data['scientific']} | Tên tiếng Anh: {crop_id.capitalize()}</div>
                <div>
                    <span class="status-badge badge-success">Độ tin cậy: {primary_confidence:.2f}%</span>
                    <span class="status-badge">Phân nhóm: {crop_data['category']}</span>
                    <span class="status-badge">Thuật toán: {selected_model_key.split('(')[0].strip()}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Đặc tả nông học
            st.markdown("**Yêu cầu thổ nhưỡng:**")
            st.write(crop_data["soil_type"])

            st.markdown("**Yêu cầu khí tượng & thủy văn:**")
            st.write(crop_data["climate_requirement"])

            # Biểu đồ Top 3 xác suất
            if has_proba and len(top_indices) > 1:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-title">Phân phối xác suất các phương án thay thế (Top 3)</div>', unsafe_allow_html=True)

                candidates = [f"{CROP_INFO.get(classes[i], {}).get('vi', classes[i])} ({classes[i]})" for i in top_indices]
                candidate_scores = [round(probabilities[i] * 100, 2) for i in top_indices]

                chart_data = pd.DataFrame({
                    "Cây trồng": candidates,
                    "Xác suất thích nghi (%)": candidate_scores
                }).set_index("Cây trồng")

                st.bar_chart(chart_data, color="#2d6a4f", use_container_width=True)

            # Đánh giá chỉ số đất thực tế
            st.markdown('<div class="section-title">Tổng hợp chỉ số dinh dưỡng đầu vào</div>', unsafe_allow_html=True)
            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            with kpi_col1:
                st.metric("Tổng dinh dưỡng NPK", f"{val_n + val_p + val_k:.0f} mg/kg")
            with kpi_col2:
                ph_eval = "Chua" if val_ph < 6.0 else ("Kiềm nhẹ" if val_ph > 7.5 else "Trung tính")
                st.metric("Độ phản ứng pH", f"{val_ph:.1f} ({ph_eval})")
            with kpi_col3:
                rain_eval = "Khô hạn" if val_rain < 80 else ("Mưa nhiều" if val_rain > 180 else "Ổn định")
                st.metric("Đặc tính nước tưới", f"{val_rain:.0f} mm ({rain_eval})")

        except Exception as ex:
            st.error(f"Đã xảy ra lỗi trong quá trình suy luận: {ex}")


# ==============================================================
# TAB 2: XỬ LÝ THEO LÔ (CSV BATCH)
# ==============================================================
with tab_batch:
    st.markdown('<div class="section-title">Phân loại dữ liệu theo lô từ tệp tin CSV</div>', unsafe_allow_html=True)
    st.write("Hỗ trợ phân tích đồng thời nhiều mẫu đất canh tác từ file dữ liệu có cấu trúc.")

    left_batch, right_batch = st.columns([2, 1], gap="large")

    with right_batch:
        st.markdown("**Cấu trúc tệp mẫu chuẩn:**")
        sample_template = pd.DataFrame([
            {"N": 90, "P": 42, "K": 43, "temperature": 20.8, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9},
            {"N": 20, "P": 68, "K": 20, "temperature": 25.0, "humidity": 80.0, "ph": 6.5, "rainfall": 150.0},
            {"N": 105, "P": 30, "K": 32, "temperature": 25.0, "humidity": 65.0, "ph": 6.8, "rainfall": 160.0},
        ])[FEATURES]

        st.dataframe(sample_template, use_container_width=True)
        csv_sample_bytes = sample_template.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Tải tệp mẫu (sample_input.csv)",
            data=csv_sample_bytes,
            file_name="crop_sample_input.csv",
            mime="text/csv",
            use_container_width=True
        )

    with left_batch:
        uploaded_csv = st.file_uploader("Tải tệp dữ liệu CSV cần xử lý:", type=["csv"])

    if uploaded_csv is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_csv)
            st.markdown(f"**Xem trước dữ liệu tải lên ({len(df_uploaded)} mẫu):**")
            st.dataframe(df_uploaded.head(5), use_container_width=True)

            absent_fields = [f for f in FEATURES if f not in df_uploaded.columns]
            if absent_fields:
                st.error(f"Tệp CSV thiếu các trường dữ liệu bắt buộc: {', '.join(absent_fields)}")
            else:
                if st.button("Bắt đầu suy luận danh mục cây trồng", type="primary"):
                    batch_preds = selected_model.predict(df_uploaded[FEATURES])
                    df_output = df_uploaded.copy()
                    df_output["Khuyen_Nghi_Tieng_Viet"] = [CROP_INFO.get(str(p).lower(), {}).get("vi", p) for p in batch_preds]
                    df_output["Ten_Khoa_Hoc"] = [CROP_INFO.get(str(p).lower(), {}).get("scientific", "") for p in batch_preds]
                    df_output["Phan_Loai_Chuan"] = batch_preds

                    st.success(f"Hoàn thành phân tích cho {len(df_output)} bản ghi.")
                    st.dataframe(df_output, use_container_width=True)

                    csv_result_data = df_output.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Tải về kết quả phân tích đầy đủ (.CSV)",
                        data=csv_result_data,
                        file_name="crop_classification_results.csv",
                        mime="text/csv"
                    )
        except Exception as err:
            st.error(f"Lỗi đọc tệp CSV: {err}")


# ==============================================================
# TAB 3: PHÂN TÍCH DỮ LIỆU GỐC (EDA)
# ==============================================================
with tab_eda:
    st.markdown('<div class="section-title">Thống kê mô tả & Phân tích tương quan đặc trưng</div>', unsafe_allow_html=True)
    st.write("Tổng hợp các phân tích định lượng trên bộ dữ liệu 2.200 mẫu nông nghiệp chuẩn hóa:")

    eda_sub1, eda_sub2, eda_sub3 = st.tabs([
        "Ma trận tương quan",
        "Phân bố ngoại lai",
        "Tần suất phân phối"
    ])

    with eda_sub1:
        st.markdown("**Ma trận hệ số tương quan Pearson giữa các yếu tố môi trường:**")
        path_corr = BASE_DIR / "figures" / "eda_correlation_matrix.png"
        if path_corr.exists():
            st.image(str(path_corr), use_container_width=True)
            st.caption("Ghi chú: Lân (P) và Kali (K) có tương quan tuyến tính thuận rõ nét (r = 0.74).")

    with eda_sub2:
        st.markdown("**Biểu đồ phân vị Boxplot và nhận diện điểm ngoại lai (1.5 * IQR):**")
        path_box = BASE_DIR / "figures" / "eda_outliers_boxplot.png"
        if path_box.exists():
            st.image(str(path_box), use_container_width=True)
            st.caption("Ghi chú: Giá trị ngoại lai ở Kali và Lân phản ánh nhu cầu dưỡng chất cá biệt của nhóm cây ăn quả ôn đới.")

    with eda_sub3:
        st.markdown("**Phân phối mật độ xác suất (Histogram & KDE):**")
        path_dist = BASE_DIR / "figures" / "eda_feature_distributions.png"
        if path_dist.exists():
            st.image(str(path_dist), use_container_width=True)
            st.caption("Ghi chú: Nhiệt độ và pH tuân theo phân phối tiệm cận chuẩn Gauss.")


# ==============================================================
# TAB 4: HIỆU NĂNG MÔ HÌNH (BENCHMARK)
# ==============================================================
with tab_benchmark:
    st.markdown('<div class="section-title">Kết quả đánh giá thực nghiệm các mô hình học máy</div>', unsafe_allow_html=True)
    st.write("Đánh giá chéo đa lớp trên tập Test độc lập (330 mẫu) với đầy đủ các chỉ số định lượng:")

    path_metrics_csv = BASE_DIR / "results" / "model_comparison_full.csv"
    if path_metrics_csv.exists():
        df_bench = pd.read_csv(path_metrics_csv)
        st.dataframe(
            df_bench.style.highlight_max(subset=["Accuracy", "F1 (macro)", "Cohen's Kappa", "MCC"], color="#dcfce7"),
            use_container_width=True
        )

    col_roc_view, col_fi_view = st.columns(2, gap="medium")
    with col_roc_view:
        st.markdown("**Đường cong ROC và chỉ số diện tích dưới đường cong (AUC):**")
        path_roc_fig = BASE_DIR / "figures" / "roc_curve_all_models.png"
        if path_roc_fig.exists():
            st.image(str(path_roc_fig), use_container_width=True)

    with col_fi_view:
        st.markdown("**Mức độ đóng góp của đặc trưng (Feature Importance - Random Forest):**")
        path_fi_fig = BASE_DIR / "figures" / "feature_importance_rf.png"
        if path_fi_fig.exists():
            st.image(str(path_fi_fig), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Ma trận nhầm lẫn đối chiếu (Confusion Matrix):**")
    path_cm_fig = BASE_DIR / "figures" / "compare_confusion_matrix_3models.png"
    if path_cm_fig.exists():
        st.image(str(path_cm_fig), use_container_width=True)


# ==============================================================
# Chân trang hệ thống
# ==============================================================
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.8rem; margin-top: 40px; border-top: 1px solid #e2e8f0; padding-top: 16px;">
    Crop Recommendation Decision Support System • Triển khai kỹ thuật: Scikit-learn Pipeline, Streamlit Framework
</div>
""", unsafe_allow_html=True)
