import matplotlib.pyplot as plt
import seaborn as sns
import json

"""
Age	    Attrition	    BusinessTravel	    DailyRate	    Department	    DistanceFromHome	    Education	    EducationField
EmployeeCount	        EmployeeNumber	    EnvironmentSatisfaction	        Gender	                HourlyRate	    JobInvolvement
JobLevel                JobRole	            JobSatisfaction	MaritalStatus	MonthlyIncome	        MonthlyRate	    NumCompaniesWorked
Over18	OverTime	    PercentSalaryHike	PerformanceRating	            RelationshipSatisfaction	            StandardHours
StockOptionLevel	    TotalWorkingYears	TrainingTimesLastYear	        WorkLifeBalance	        YearsAtCompany	YearsInCurrentRole
YearsSinceLastPromotion	YearsWithCurrManager
"""

import matplotlib.pyplot as plt
import seaborn as sns
import math


def raw_data_visualization(df, df_name, target_col=None):
    fig = plt.figure(figsize=(12, 7))
    fig.subplots_adjust(
        wspace=0.25,
        hspace=0.96,
        left=0.075,
        bottom=0.07,
        right=0.95,
        top=0.95,
    )
    fig.suptitle(f"{df_name} overview", fontsize=12)

    dtype_df = df.dtypes.reset_index()
    dtype_df.columns = ["column", "dtype"]
    dtype_df["dtype"] = dtype_df["dtype"].astype(str)

    mapping = {
        "int64": "number",
        "float64": "number",
        "int32": "number",
        "float32": "number",
        "object": "category",
    }
    dtype_df["dtype_group"] = dtype_df["dtype"].map(mapping).fillna(dtype_df["dtype"])

    numeric_cols = dtype_df[dtype_df["dtype_group"] == "number"]["column"].tolist()
    if numeric_cols:
        ax_box = fig.add_subplot(2, 2, 1)
        sns.boxplot(data=df[numeric_cols], ax=ax_box, showfliers=True)
        ax_box.set_title("Boxplot all numeric columns")
        ax_box.tick_params(axis="x", rotation=45)

    ax_dtype = fig.add_subplot(2, 2, 2)
    sns.countplot(data=dtype_df, x="dtype_group", ax=ax_dtype)
    ax_dtype.set_title("Dtype Distribution")
    ax_dtype.tick_params(axis="x", rotation=0)

    null_summary = df.isnull().sum().to_frame(name="Null Count")
    miss_val_df = null_summary.reset_index()
    miss_val_df.columns = ["columns", "count"]
    ax_missing = fig.add_subplot(2, 2, 3)
    sns.barplot(data=miss_val_df, x="columns", y="count", ax=ax_missing)
    ax_missing.set_title("Missing Value")
    ax_missing.tick_params(axis="x", rotation=45)

    ax_target = fig.add_subplot(2, 2, 4)
    if target_col and target_col in df.columns:
        class_counts = df[target_col].value_counts()
        class_percent = df[target_col].value_counts(normalize=True) * 100
        balance_df = class_counts.to_frame(name="Count").reset_index()
        balance_df.columns = [target_col, "Count"]
        balance_df["Percentage"] = class_percent.values
        sns.barplot(data=balance_df, x=target_col, y="Count", ax=ax_target)
        ax_target.set_title(f"Class distribution: {target_col}")
        for i, v in enumerate(balance_df["Count"]):
            ax_target.text(i, v + max(balance_df["Count"]) * 0.01, f"{v}", ha="center")
    else:
        ax_target.set_visible(False)

    plt.tight_layout()
    plt.show()

    cat_cols = dtype_df[dtype_df["dtype_group"] == "category"]["column"].tolist()
    n_cols = 3  # số cột trong grid
    n_rows = math.ceil(len(cat_cols) / n_cols)
    if cat_cols:
        fig_cat, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
        axes = axes.flatten()
        for i, col in enumerate(cat_cols):
            sns.countplot(data=df, x=col, ax=axes[i])
            axes[i].set_title(f"Count plot: {col}")
            axes[i].tick_params(axis="x", rotation=45)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        plt.show()


def raw_data_overview(df, df_name, target_col=None):
    print("\n=== DATA OVERVIEW ===\n")
    print(f"\nDataset: {df_name}")

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns)

    print("\nData types:")
    print(df.dtypes)

    print("\nInfo:")
    print(df.info())

    print("\nDescribe:")
    print(df.describe())

    print("\nMissing value:")
    null_summary = df.isnull().sum().to_frame(name="Null Count")
    null_summary["Null %"] = df.isna().mean() * 100
    print(null_summary)

    target_distribution = None
    if target_col and target_col in df.columns:
        class_counts = df[target_col].value_counts()
        class_percent = df[target_col].value_counts(normalize=True) * 100
        balance_df = class_counts.to_frame(name="Count")
        balance_df["Percentage"] = class_percent
        target_distribution = (
            balance_df.reset_index()
            .rename(columns={"index": target_col})
            .to_dict(orient="records")
        )

    overview_results = {
        "dataset": df_name,
        "shape": {"rows": df.shape[0], "cols": df.shape[1]},
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "describe": df.describe().reset_index().to_dict(orient="records"),
        "missing_values": null_summary.reset_index()
        .rename(columns={"index": "column"})
        .to_dict(orient="records"),
        "target_distribution": target_distribution,
    }

    raw_data_visualization(df, df_name, target_col)

    return overview_results
