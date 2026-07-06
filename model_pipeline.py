"""
model_pipeline.py
====================
Ghosting Prediction — Full ML Training Pipeline
Project: Predictive Modeling of Sudden Communication Cessation in Romantic Relationships

Run:
    python model_pipeline.py

Outputs:
    ghosting_model.pkl          — Best trained model + preprocessing pipeline
    model_comparison.csv        — Cross-validation comparison of all classifiers
    feature_importance.csv      — SHAP mean absolute importance (for interface)
    training_report.txt         — Human-readable summary of all results
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
import shap
import json

from sklearn.pipeline          import Pipeline
from sklearn.compose           import ColumnTransformer
from sklearn.preprocessing     import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model      import LogisticRegression
from sklearn.tree              import DecisionTreeClassifier
from sklearn.ensemble          import RandomForestClassifier
from sklearn.model_selection   import StratifiedKFold, cross_validate
from sklearn.metrics           import (make_scorer, f1_score, precision_score,
                                        recall_score, roc_auc_score, accuracy_score,
                                        classification_report, confusion_matrix)
from sklearn.inspection        import permutation_importance
from imblearn.over_sampling    import SMOTE
from imblearn.pipeline         import Pipeline as ImbPipeline
from xgboost                   import XGBClassifier

print("=" * 65)
print("  GHOSTING PREDICTION — MODEL TRAINING PIPELINE")
print("=" * 65)

# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────

df = pd.read_csv("/mnt/user-data/outputs/ghosting_prediction_dataset.csv")
print(f"\n[1/6] Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"      Class balance — ghosted: {df['ghosted'].sum()} ({df['ghosted'].mean()*100:.1f}%)"
      f"  |  not ghosted: {(1-df['ghosted']).sum()} ({(1-df['ghosted']).mean()*100:.1f}%)")

X = df.drop(columns=["ghosted"])
y = df["ghosted"]

# ─── 2. DEFINE FEATURE GROUPS ─────────────────────────────────────────────────

# Ordinal — relationship stage has a natural order (must be encoded as 0–5)
ordinal_features  = ["relationship_stage"]
ordinal_categories = [["Unrequited", "Non-established", "Casual dating",
                        "Committed dating", "Cohabiting/Engaged", "Married"]]

# Nominal — no inherent order, one-hot encode
nominal_features  = ["gender", "platform"]

# Numeric — standardise
numeric_features  = [c for c in X.columns
                     if c not in ordinal_features + nominal_features]

print(f"\n      Features — numeric: {len(numeric_features)} | "
      f"ordinal: {len(ordinal_features)} | nominal: {len(nominal_features)}")

# ─── 3. PREPROCESSING PIPELINE ────────────────────────────────────────────────

preprocessor = ColumnTransformer(transformers=[
    ("num",  StandardScaler(),
             numeric_features),
    ("ord",  OrdinalEncoder(categories=ordinal_categories),
             ordinal_features),
    ("nom",  OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             nominal_features),
], remainder="drop")

# ─── 4. DEFINE CLASSIFIERS ────────────────────────────────────────────────────

classifiers = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42, C=1.0
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced", max_depth=8, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        max_depth=12, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        scale_pos_weight=(1 - y.mean()) / y.mean(),   # handles class imbalance
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0
    ),
}

# ─── 5. CROSS-VALIDATION COMPARISON ───────────────────────────────────────────

print(f"\n[2/6] Running 5-fold stratified cross-validation on all classifiers...")
print(f"      (SMOTE applied inside each training fold only — no data leakage)\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy":  "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall":    make_scorer(recall_score),
    "f1":        make_scorer(f1_score),
    "roc_auc":   "roc_auc",
}

results = []

for name, clf in classifiers.items():
    # SMOTE is applied INSIDE the pipeline → applied only to training folds
    pipe = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote",        SMOTE(random_state=42)),
        ("classifier",   clf),
    ])
    cv_results = cross_validate(pipe, X, y, cv=cv, scoring=scoring,
                                return_train_score=False, n_jobs=-1)
    row = {"Model": name}
    for metric in scoring:
        row[metric.capitalize()] = round(cv_results[f"test_{metric}"].mean(), 4)
        row[f"{metric.capitalize()}_std"] = round(cv_results[f"test_{metric}"].std(), 4)
    results.append(row)
    print(f"  {name:<22} | Acc={row['Accuracy']:.3f}  "
          f"F1={row['F1']:.3f}  AUC={row['Roc_auc']:.3f}  "
          f"Recall={row['Recall']:.3f}")

comparison_df = pd.DataFrame(results)

# ─── 6. SELECT BEST MODEL ─────────────────────────────────────────────────────

best_row  = comparison_df.loc[comparison_df["F1"].idxmax()]
best_name = best_row["Model"]
print(f"\n[3/6] Best model by F1-score: {best_name}  (F1={best_row['F1']:.4f})")

# Retrain best model on FULL dataset for deployment
best_clf  = classifiers[best_name]
best_pipe = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote",        SMOTE(random_state=42)),
    ("classifier",   best_clf),
])
best_pipe.fit(X, y)
print(f"      Retrained {best_name} on full dataset (N={len(X)})")

# ─── 7. FINAL EVALUATION ON FULL DATA (for report — not validation, just sanity) ───

y_pred     = best_pipe.predict(X)
y_proba    = best_pipe.predict_proba(X)[:, 1]

print(f"\n[4/6] In-sample classification report (training set — for sanity check only):")
print(classification_report(y, y_pred, target_names=["Not ghosted", "Ghosted"]))

cm = confusion_matrix(y, y_pred)
print(f"      Confusion matrix:\n      TN={cm[0,0]}  FP={cm[0,1]}\n"
      f"      FN={cm[1,0]}  TP={cm[1,1]}")

# ─── 8. SHAP FEATURE IMPORTANCE ───────────────────────────────────────────────

print(f"\n[5/6] Computing SHAP feature importance...")

# Extract the fitted preprocessor to transform X for SHAP
pre = best_pipe.named_steps["preprocessor"]
X_transformed = pre.transform(X)

# Build transformed feature names
num_names = numeric_features
ord_names = ordinal_features
nom_names = list(pre.named_transformers_["nom"].get_feature_names_out(nominal_features))
all_feature_names = num_names + ord_names + nom_names

# Get the fitted classifier
fitted_clf = best_pipe.named_steps["classifier"]

if best_name in ["Random Forest", "XGBoost"]:
    explainer = shap.TreeExplainer(fitted_clf)
    shap_values = explainer.shap_values(X_transformed)
    # For binary classification some models return list of arrays
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
else:
    explainer = shap.LinearExplainer(fitted_clf, X_transformed)
    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "feature":    all_feature_names,
    "importance": mean_abs_shap
}).sort_values("importance", ascending=False).reset_index(drop=True)

print(f"      Top 10 features by SHAP importance:")
for _, row in importance_df.head(10).iterrows():
    bar = "█" * int(row["importance"] / importance_df["importance"].max() * 20)
    print(f"      {row['feature']:<30} {bar} {row['importance']:.4f}")

# ─── 9. SAVE ARTEFACTS ────────────────────────────────────────────────────────

print(f"\n[6/6] Saving artefacts...")

BASE = "/mnt/user-data/outputs/"

# 9a — Trained pipeline
with open(BASE + "ghosting_model.pkl", "wb") as f:
    pickle.dump({
        "pipeline":      best_pipe,
        "best_model":    best_name,
        "feature_names": numeric_features + ordinal_features + nominal_features,
        "num_features":  numeric_features,
        "ord_features":  ordinal_features,
        "nom_features":  nominal_features,
        "ord_categories": ordinal_categories,
    }, f)
print(f"  ✓ ghosting_model.pkl")

# 9b — Model comparison CSV
comparison_df.to_csv(BASE + "model_comparison.csv", index=False)
print(f"  ✓ model_comparison.csv")

# 9c — Feature importance CSV
importance_df.to_csv(BASE + "feature_importance.csv", index=False)
print(f"  ✓ feature_importance.csv")

# 9d — Human-readable training report
report_lines = [
    "GHOSTING PREDICTION — TRAINING REPORT",
    "=" * 65,
    f"Dataset: 2,000 observations | 24 features | Target: ghosted (0/1)",
    f"Class balance: {y.sum()} ghosted ({y.mean()*100:.1f}%) | "
    f"{(1-y).sum()} not ghosted ({(1-y).mean()*100:.1f}%)",
    f"Validation: 5-fold stratified cross-validation with SMOTE (training folds only)",
    "",
    "MODEL COMPARISON (cross-validated, mean ± std):",
    "-" * 65,
]
for _, row in comparison_df.iterrows():
    report_lines.append(
        f"{row['Model']:<22} | Acc={row['Accuracy']:.3f}±{row['Accuracy_std']:.3f} | "
        f"F1={row['F1']:.3f}±{row['F1_std']:.3f} | "
        f"AUC={row['Roc_auc']:.3f}±{row['Roc_auc_std']:.3f} | "
        f"Recall={row['Recall']:.3f}±{row['Recall_std']:.3f}"
    )
report_lines += [
    "",
    f"BEST MODEL: {best_name} (by F1-score = {best_row['F1']:.4f})",
    "",
    "TOP 10 FEATURES BY SHAP IMPORTANCE:",
    "-" * 65,
]
for i, row in importance_df.head(10).iterrows():
    report_lines.append(f"  {i+1:2}. {row['feature']:<30} {row['importance']:.4f}")

with open(BASE + "training_report.txt", "w") as f:
    f.write("\n".join(report_lines))
print(f"  ✓ training_report.txt")

print(f"\n{'=' * 65}")
print(f"  Training complete. Best model: {best_name}")
print(f"  All artefacts saved to {BASE}")
print(f"{'=' * 65}\n")
