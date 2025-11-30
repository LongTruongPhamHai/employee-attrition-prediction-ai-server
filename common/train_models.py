from config.settings import MODEL_DIR
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import os
import joblib
from common.evaluate import get_metrics


def train_models(preprocessor, X_train, y_train, X_val, y_val):
    models = {}

    models["Dummy (most_frequent)"] = DummyClassifier(strategy="most_frequent")
    models["Dummy (stratified)"] = DummyClassifier(strategy="stratified")
    models["Logistic Regression"] = LogisticRegression(max_iter=500, random_state=42)

    best_k, best_f1_knn = 1, 0
    print("\nFind the best K for KNN\n")
    for k in range(1, 21):
        knn_model = KNeighborsClassifier(n_neighbors=k, weights="distance")
        pipeline = Pipeline([("preprocess", preprocessor), ("classifier", knn_model)])
        f1_cv = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1").mean()
        if f1_cv > best_f1_knn:
            best_f1_knn, best_k = f1_cv, k
            print(f"  K={k} → F1-score CV improved: {f1_cv:.4f}")
    models["KNN"] = KNeighborsClassifier(n_neighbors=best_k, weights="distance")

    best_depth, best_f1_dt = 1, 0
    print("\nFind the best max_depth for Decision Tree\n")
    for depth in range(1, 11):
        dt_model = DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=5, random_state=42
        )
        pipeline = Pipeline([("preprocess", preprocessor), ("classifier", dt_model)])
        f1_cv = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1").mean()
        if f1_cv > best_f1_dt:
            best_f1_dt, best_depth = f1_cv, depth
            print(f"  max_depth={depth} → F1-score CV improved: {f1_cv:.4f}")
    models["Decision Tree"] = DecisionTreeClassifier(
        max_depth=best_depth, min_samples_leaf=5, random_state=42
    )

    results = {}
    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, model in models.items():
        pipeline = Pipeline([("preprocess", preprocessor), ("classifier", model)])
        pipeline.fit(X_train, y_train)
        y_val_pred = pipeline.predict(X_val)

        metrics = get_metrics(y_val, y_val_pred)

        results[name] = {
            "pipeline": pipeline,
            # "pred_val": y_val_pred,
            "val": metrics,
            "params": model.get_params(),
        }

        joblib.dump(pipeline, os.path.join(MODEL_DIR, f"{name}.pkl"))
        print(f"{name} model saved successfully!")

    return results
