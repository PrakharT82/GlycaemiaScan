import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate, cross_val_predict, GridSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix,
                             precision_recall_curve, ConfusionMatrixDisplay, RocCurveDisplay)

Path("models").mkdir(exist_ok=True)
Path("reports/figures").mkdir(parents=True, exist_ok=True)

# ---------- 1. Load and clean ----------
df = pd.read_csv("data/diabetes.csv")
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols] = df[cols].replace(0, np.nan)

X = df.drop(columns="Outcome")
y = df["Outcome"]
features = list(X.columns)

# ---------- 2. Stratified train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print("Train:", X_train.shape, "Test:", X_test.shape)

# ---------- 3. Pipeline: impute -> scale -> model (prevents data leakage) ----------
def make_pipe(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
}

# ---------- 4. Compare models with 5-fold CV ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for name, m in models.items():
    s = cross_validate(make_pipe(m), X_train, y_train, cv=cv,
                       scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
    rows.append({
        "Model": name,
        "Accuracy": s["test_accuracy"].mean(),
        "Precision": s["test_precision"].mean(),
        "Recall": s["test_recall"].mean(),
        "F1": s["test_f1"].mean(),
        "ROC_AUC": s["test_roc_auc"].mean(),
    })
comparison = pd.DataFrame(rows).round(3).sort_values("ROC_AUC", ascending=False)
print("\nCross-validation results:\n", comparison.to_string(index=False))
comparison.to_csv("reports/model_comparison.csv", index=False)

# ---------- 5. Hyperparameter tuning ----------
param_grids = {
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10]},
    "Random Forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5, 8, None],
        "model__min_samples_leaf": [1, 3, 5],
    },
}
tuned = {}
for name, grid in param_grids.items():
    gs = GridSearchCV(make_pipe(models[name]), grid, cv=cv, scoring="roc_auc")
    gs.fit(X_train, y_train)
    tuned[name] = gs
    print(f"\n{name} best params: {gs.best_params_} | CV ROC-AUC: {gs.best_score_:.3f}")

best_name = max(tuned, key=lambda k: tuned[k].best_score_)
final_model = tuned[best_name].best_estimator_
print("\nFinal model chosen:", best_name)

# ---------- 6. Threshold tuning (recall >= 80%), using train data only ----------
oof_proba = cross_val_predict(final_model, X_train, y_train, cv=cv,
                              method="predict_proba")[:, 1]
prec, rec, thr = precision_recall_curve(y_train, oof_proba)
ok = np.where(rec[:-1] >= 0.80)[0]
threshold = float(thr[ok[-1]]) if len(ok) else 0.5
print(f"Chosen threshold: {threshold:.2f}")

# ---------- 7. Final evaluation on the untouched test set ----------
proba = final_model.predict_proba(X_test)[:, 1]
final_pred = (proba >= threshold).astype(int)
for t in [0.5, threshold]:
    pred = (proba >= t).astype(int)
    print(f"\nThreshold {t:.2f} -> Acc {accuracy_score(y_test, pred):.3f} | "
          f"Prec {precision_score(y_test, pred):.3f} | Recall {recall_score(y_test, pred):.3f} | "
          f"F1 {f1_score(y_test, pred):.3f}")
print("ROC-AUC:", round(roc_auc_score(y_test, proba), 3))
print("\n", classification_report(y_test, final_pred))

ConfusionMatrixDisplay.from_predictions(y_test, final_pred, cmap="Blues")
plt.title("Confusion matrix (test set)")
plt.savefig("reports/figures/confusion_matrix.png", bbox_inches="tight")
plt.close()

RocCurveDisplay.from_predictions(y_test, proba)
plt.title("ROC curve (test set)")
plt.savefig("reports/figures/roc_curve.png", bbox_inches="tight")
plt.close()

# ---------- 8. Interpretability ----------
est = final_model.named_steps["model"]
raw = est.coef_[0] if hasattr(est, "coef_") else est.feature_importances_
imp = pd.Series(np.abs(raw), index=features)
imp = (imp / imp.max() * 100).sort_values(ascending=False)
print("\nFeature importance (0-100):\n", imp.round(1))
imp.sort_values().plot(kind="barh", figsize=(8, 5), color="#4C78A8")
plt.title("Feature importance (" + best_name + ")")
plt.savefig("reports/figures/feature_importance.png", bbox_inches="tight")
plt.close()

# ---------- 9. Save everything the web app needs ----------
joblib.dump({"pipeline": final_model, "threshold": threshold, "features": features,
             "model_name": best_name}, "models/diabetes_model.joblib")

tn, fp, fn, tp = confusion_matrix(y_test, final_pred).ravel()
metrics = {
    "model_name": best_name,
    "threshold": round(threshold, 2),
    "accuracy": float(accuracy_score(y_test, final_pred)),
    "precision": float(precision_score(y_test, final_pred)),
    "recall": float(recall_score(y_test, final_pred)),
    "f1": float(f1_score(y_test, final_pred)),
    "roc_auc": float(roc_auc_score(y_test, proba)),
    "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "importance": [{"feature": k, "value": float(v)} for k, v in imp.items()],
}
with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("\nSaved model and metrics.")