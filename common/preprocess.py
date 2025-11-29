from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def data_preprocess(X_train):
    print("\n=== DATA PREPROCESS STARTED ===\n")

    df_temp = X_train.copy()

    num_cols = [
        "Age",
        "DailyRate",
        "DistanceFromHome",
        "Education",
        "EmployeeCount",
        "EmployeeNumber",
        "EnvironmentSatisfaction",
        "HourlyRate",
        "JobInvolvement",
        "JobLevel",
        "JobSatisfaction",
        "MonthlyIncome",
        "MonthlyRate",
        "NumCompaniesWorked",
        "PercentSalaryHike",
        "PerformanceRating",
        "RelationshipSatisfaction",
        "StandardHours",
        "StockOptionLevel",
        "TotalWorkingYears",
        "TrainingTimesLastYear",
        "WorkLifeBalance",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
    ]

    cat_map_cols = ["Gender", "OverTime"]
    cat_ohe_cols = [
        "BusinessTravel",
        "Department",
        "EducationField",
        "JobRole",
        "MaritalStatus",
    ]
    drop_cols = ["Over18"]

    df_temp["Gender"] = df_temp["Gender"].map({"Male": 1, "Female": 0})
    df_temp["OverTime"] = df_temp["OverTime"].map({"Yes": 1, "No": 0})

    num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
    )

    ohe_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transforms = ColumnTransformer(
        [
            ("num", num_pipeline, num_cols),
            ("cat_ohe", ohe_pipeline, cat_ohe_cols),
            ("drop", "drop", drop_cols),
        ]
    )

    preprocess = Pipeline([("preprocess", transforms)])

    return preprocess
