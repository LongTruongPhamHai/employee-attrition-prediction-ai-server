# **IBM HR Analytics Employee Attrition & Performance**

## **1. Giới thiệu dataset**

Dataset này được tạo bởi các nhà khoa học dữ liệu của IBM nhằm mục đích **phân tích và dự đoán nhân viên rời công ty (attrition)** và khám phá các yếu tố liên quan đến hiệu suất, sự hài lòng và đặc điểm nghề nghiệp của nhân viên.

**Lưu ý:** Đây là dữ liệu **giả lập (fictional)** nhưng mô phỏng sát thực tế.

---

## **2. Mục đích sử dụng**

- Dự đoán khả năng nhân viên rời công ty (**Attrition Prediction – supervised learning**)
- Phân nhóm nhân viên theo đặc điểm nghề nghiệp và hành vi (**Employee Segmentation – unsupervised learning**)
- Trả lời các câu hỏi phân tích, ví dụ:

  - “So sánh khoảng cách từ nhà đến công ty theo Job Role và Attrition”
  - “So sánh mức lương trung bình theo trình độ học vấn và Attrition”
  - “Nhóm nhân viên theo thu nhập, tuổi, JobSatisfaction và YearsAtCompany”

---

## **3. Các biến chính (features)**

| Feature                  | Kiểu        | Mô tả                                                                 |
| ------------------------ | ----------- | --------------------------------------------------------------------- |
| Age                      | Numeric     | Tuổi của nhân viên                                                    |
| Attrition                | Categorical | Nhân viên rời công ty hay không (Yes/No)                              |
| BusinessTravel           | Categorical | Tần suất đi công tác (Non-Travel / Travel_Rarely / Travel_Frequently) |
| DailyRate                | Numeric     | Lương theo ngày                                                       |
| Department               | Categorical | Bộ phận làm việc (Sales, R&D, HR)                                     |
| DistanceFromHome         | Numeric     | Khoảng cách từ nhà đến công ty (km)                                   |
| Education                | Ordinal     | Trình độ học vấn (1=Below College → 5=Doctor)                         |
| EnvironmentSatisfaction  | Ordinal     | Mức độ hài lòng với môi trường làm việc (1=Low → 4=Very High)         |
| JobInvolvement           | Ordinal     | Mức độ tham gia công việc (1=Low → 4=Very High)                       |
| JobLevel                 | Numeric     | Cấp bậc công việc                                                     |
| JobRole                  | Categorical | Vị trí công việc                                                      |
| JobSatisfaction          | Ordinal     | Mức độ hài lòng với công việc (1=Low → 4=Very High)                   |
| MaritalStatus            | Categorical | Tình trạng hôn nhân                                                   |
| MonthlyIncome            | Numeric     | Lương hàng tháng                                                      |
| NumCompaniesWorked       | Numeric     | Số công ty đã làm trước đây                                           |
| OverTime                 | Categorical | Có làm thêm giờ hay không (Yes/No)                                    |
| PercentSalaryHike        | Numeric     | Tỷ lệ tăng lương (%)                                                  |
| PerformanceRating        | Ordinal     | 1=Low → 4=Outstanding                                                 |
| RelationshipSatisfaction | Ordinal     | 1=Low → 4=Very High                                                   |
| StockOptionLevel         | Numeric     | Cấp độ quyền chọn cổ phiếu                                            |
| TotalWorkingYears        | Numeric     | Tổng số năm kinh nghiệm                                               |
| TrainingTimesLastYear    | Numeric     | Số lần đào tạo năm trước                                              |
| WorkLifeBalance          | Ordinal     | 1=Bad → 4=Best                                                        |
| YearsAtCompany           | Numeric     | Số năm làm tại công ty                                                |
| YearsInCurrentRole       | Numeric     | Số năm ở vị trí hiện tại                                              |
| YearsSinceLastPromotion  | Numeric     | Số năm kể từ lần thăng chức gần nhất                                  |
| YearsWithCurrManager     | Numeric     | Số năm làm việc với quản lý hiện tại                                  |

---

## **4. Các giá trị mã hóa quan trọng**

- **Education:** 1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor
- **EnvironmentSatisfaction / JobInvolvement / JobSatisfaction / RelationshipSatisfaction:** 1=Low, 2=Medium, 3=High, 4=Very High
- **PerformanceRating:** 1=Low, 2=Good, 3=Excellent, 4=Outstanding
- **WorkLifeBalance:** 1=Bad, 2=Good, 3=Better, 4=Best

> Các biến Ordinal có thể dùng trực tiếp cho các thuật toán ML hoặc chuyển thành dummy variables nếu cần.

---

## **5. Ứng dụng gợi ý**

- **Supervised Learning:** Logistic Regression, Decision Tree, Random Forest → dự đoán Attrition
- **Unsupervised Learning:** K-means, Hierarchical Clustering → phân nhóm nhân viên
- **Visualization:** Biểu đồ phân phối Age, MonthlyIncome, YearsAtCompany theo Attrition hoặc JobRole; PCA / t-SNE trực quan các cluster

---

## **6. Lưu ý khi sử dụng**

- Dữ liệu **không thực** nhưng cấu trúc giống thực tế → thích hợp cho học tập, demo, thử nghiệm các mô hình AI/ML/DL
- Cần xử lý categorical variables, missing values trước khi train model
- Có thể mở rộng bằng các tính năng mới (ví dụ tổng thu nhập, tỉ lệ tăng lương theo năm…)
