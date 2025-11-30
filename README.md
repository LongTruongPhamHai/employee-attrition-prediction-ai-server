# AI Module - IBM HR Analytics Employee Attrition & Performance

## 1. Giới thiệu

Module này được xây dựng để phân tích và dự đoán **nhân viên rời công ty (Attrition)** dựa trên dataset IBM HR Analytics (fictional nhưng sát thực tế).  
Ngoài dự đoán Attrition, module còn hỗ trợ các tác vụ **ML preprocessing, training, inference và visualization**.

Cấu trúc thư mục:

```

ai-module/
├── venv/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── config/
│   └── settings.py
├── common/
│   ├── evaluate.py
│   ├── overview.py
│   ├── preprocess.py
│   ├── train_models.py
│   └── utils.py
├── services/
│   ├── overview_service.py
│   ├── train_service.py
│   └── predict_service.py
├── routes/
│   ├── overview_route.py
│   ├── predict_route.py
│   └── train_route.py
├── models/
├── main.py
└── requirements.txt

```

---

## 2. Mục đích

- Dự đoán **khả năng rời công ty (Attrition Prediction)** – supervised learning
- Phân nhóm nhân viên theo đặc điểm nghề nghiệp – unsupervised learning
- Trả lời các câu hỏi phân tích về Job Role, lương, khoảng cách nhà, Job Satisfaction, v.v.

---

## 3. Dataset

Dataset gồm các biến chính:

| Feature                  | Kiểu        | Mô tả                                |
| ------------------------ | ----------- | ------------------------------------ |
| Age                      | Numeric     | Tuổi nhân viên                       |
| Attrition                | Categorical | Nhân viên rời công ty (Yes/No)       |
| BusinessTravel           | Categorical | Tần suất đi công tác                 |
| DailyRate                | Numeric     | Lương theo ngày                      |
| Department               | Categorical | Bộ phận làm việc                     |
| DistanceFromHome         | Numeric     | Khoảng cách nhà → công ty (km)       |
| Education                | Ordinal     | 1=Below College → 5=Doctor           |
| EnvironmentSatisfaction  | Ordinal     | 1=Low → 4=Very High                  |
| JobInvolvement           | Ordinal     | 1=Low → 4=Very High                  |
| JobRole                  | Categorical | Vị trí công việc                     |
| JobSatisfaction          | Ordinal     | 1=Low → 4=Very High                  |
| MaritalStatus            | Categorical | Tình trạng hôn nhân                  |
| MonthlyIncome            | Numeric     | Lương tháng                          |
| NumCompaniesWorked       | Numeric     | Số công ty đã làm                    |
| OverTime                 | Categorical | Có làm thêm giờ hay không            |
| PercentSalaryHike        | Numeric     | Tỷ lệ tăng lương (%)                 |
| PerformanceRating        | Ordinal     | 1=Low → 4=Outstanding                |
| RelationshipSatisfaction | Ordinal     | 1=Low → 4=Very High                  |
| StockOptionLevel         | Numeric     | Cấp quyền chọn cổ phiếu              |
| TotalWorkingYears        | Numeric     | Tổng năm kinh nghiệm                 |
| TrainingTimesLastYear    | Numeric     | Số lần đào tạo năm trước             |
| WorkLifeBalance          | Ordinal     | 1=Bad → 4=Best                       |
| YearsAtCompany           | Numeric     | Số năm tại công ty                   |
| YearsInCurrentRole       | Numeric     | Số năm ở vị trí hiện tại             |
| YearsSinceLastPromotion  | Numeric     | Số năm kể từ lần thăng chức gần nhất |
| YearsWithCurrManager     | Numeric     | Số năm làm việc với quản lý hiện tại |

> Các biến Ordinal có thể dùng trực tiếp cho ML hoặc chuyển thành dummy variables nếu cần.

---

## 4. Ứng dụng

- **Supervised Learning:** Logistic Regression, Decision Tree, Random Forest → dự đoán Attrition
- **Unsupervised Learning:** K-means, Hierarchical Clustering → phân nhóm nhân viên
- **Visualization:** phân tích Age, MonthlyIncome, YearsAtCompany theo Attrition hoặc JobRole; PCA / t-SNE trực quan các cluster

---

## 5. Cài đặt

```bash
git clone <repo-url>
cd ai-module
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux / macOS
pip install -r requirements.txt
```

### **Dependencies chính**

- fastapi
- uvicorn
- numpy
- pandas
- scikit-learn
- joblib
- pydantic

---

## 6. Chạy API

```bash
uvicorn main:app --reload
```

- API endpoint ví dụ:

  - `/train-model` → huấn luyện model và trả về kết quả JSON
  - `/predict` → dự đoán nhân viên rời công ty
  - `/overview` → thống kê dữ liệu

Swagger UI: `http://127.0.0.1:8000/docs`

---

## 7. Lưu ý

- Dữ liệu **giả lập** nhưng cấu trúc giống thực tế → dùng cho học tập, demo ML
- Trước khi train model: xử lý categorical variables, missing values, scale numeric features nếu cần
- Có thể mở rộng module với các feature mới hoặc thuật toán ML khác

---
