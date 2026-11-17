DỰ ÁN GỢI Ý LOẠI CÂY TRỒNG DỰA TRÊN ĐIỀU KIỆN KHÍ HẬU VÀ ĐẤT

---
---
---
Đề tài của nhóm là xây dựng hệ thống gợi ý cây trồng dựa trên điều kiện đất và khí hậu. Dữ liệu đầu vào bao gồm 7 đặc trưng: N, P, K, nhiệt độ, độ ẩm, pH và lượng mưa. Đầu ra là loại cây trồng được mô hình dự đoán.

Vì đầu ra là một trong nhiều loại cây trồng xác định nên đây là bài toán phân loại đa lớp. Nhóm so sánh bốn mô hình: Logistic Regression, LDA, QDA và Random Forest.


DATASET : https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset?

## Kỹ nghệ đặc trưng

Pipeline huấn luyện nhận 7 cột gốc. `CropFeatureEngineer` tạo thêm đặc trưng,
sau đó `ColumnTransformer` áp dụng `SimpleImputer` và `StandardScaler` cho
các cột số trước khi phân loại. Các đặc trưng mới gồm tổng N + P + K (`total_npk`), tỷ lệ N/P
(`np_ratio`), tỷ lệ N/K (`nk_ratio`) và khoảng cách đến pH trung tính
(`ph_distance_7 = |pH - 7|`). Tỷ lệ có mẫu số bằng 0 được để thiếu và điền
bằng trung vị học từ tập train. QDA bỏ `total_npk` vì cột này phụ thuộc tuyến
tính vào ba cột N, P, K, làm ma trận hiệp phương sai suy biến.

Sau khi cài thư viện, chạy từ thư mục gốc của dự án:

```powershell
python src/preprocessing.py
python src/evaluate_features.py
```

Script giữ nguyên tham số mô hình, huấn luyện hai cấu hình trên cùng tập train
và in Accuracy, F1 macro trên cùng tập validation. Permutation importance đo
mức giảm F1 macro khi xáo trộn từng cột đầu vào gốc. Khi xáo trộn N, P, K hoặc
pH, pipeline cũng tính lại các đặc trưng phụ thuộc vào cột đó.

Để tạo lại các tệp mô hình với bộ đặc trưng mới:

```powershell
python src/train_logistic_regression.py
python src/trainlda.py
python src/train_qda.py
python src/train_random_forest.py
```

Khi dự đoán, vẫn nhập 7 thông số gốc như trước.

## Kiểm định chéo và tinh chỉnh siêu tham số

```powershell
python src/preprocessing.py
python src/tune_models.py
```

`tune_models.py` gộp train và validation thành tập phát triển, rồi dùng cùng
5 fold phân tầng để chạy `GridSearchCV` cho Logistic Regression, LDA, QDA và
Random Forest. Mỗi ứng viên fit toàn bộ pipeline trong từng fold. Tập test
được giữ riêng cho đánh giá cuối. Kết quả được xếp theo F1 macro trung bình và
lưu tại `results/cv_tuning_summary.csv`, cùng độ lệch chuẩn, điểm từng fold và
bộ tham số tốt nhất của từng mô hình.



-  cài đặt các thư viện và cấu hình cần thiết ở file requirements :
   - b1: chạy lệnh này ở terminal/cmd : python -m venv .venv

   - b2: kích hoạt môi trường ảo  : .venv\Scripts\activate.bat

   - b3: cài thư viện : python -m pip install -r requirements.txt
     có thể kiểm tra bằng lệnh : python -m pip list

