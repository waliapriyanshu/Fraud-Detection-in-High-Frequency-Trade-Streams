import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, precision_score,
                             recall_score, roc_curve, precision_recall_curve,
                             average_precision_score, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# =============================
# CONFIG
# =============================
DATA_PATH = "/Users/priyanshuwalia7/Downloads/raw"
OUT_DIR = "./ensemble_results_advanced_full"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================
# 1. Load Dataset
# =============================
train_trans = pd.read_csv(os.path.join(DATA_PATH, "train_transaction.csv"))
train_id = pd.read_csv(os.path.join(DATA_PATH, "train_identity.csv"))
train = train_trans.merge(train_id, on="TransactionID", how="left")

print(f"✅ Loaded dataset with shape: {train.shape}")

# =============================
# 2. Advanced Preprocessing
# =============================

# Drop columns with >60% missing
missing_ratio = train.isnull().mean()
train = train.loc[:, missing_ratio < 0.6]
print(f"After dropping high-missing cols: {train.shape}")

# Add missing indicators
for col in train.columns:
    if train[col].isnull().sum() > 0:
        train[f"{col}_missing"] = train[col].isnull().astype(int)

# Fill missing with sentinel
train.fillna(-999, inplace=True)

# Label encode categorical
le = LabelEncoder()
for col in train.select_dtypes('object').columns:
    train[col] = le.fit_transform(train[col].astype(str))

# Log-transform skewed numeric columns
log_cols = [c for c in train.columns if any(x in c for x in ['TransactionAmt', 'D', 'C'])]
for c in log_cols:
    if c in train.columns:
        train[c] = np.log1p(np.abs(train[c]))

# Clip outliers (1st–99th percentile)
for col in train.select_dtypes(include=np.number).columns:
    q1, q99 = train[col].quantile([0.01, 0.99])
    train[col] = np.clip(train[col], q1, q99)

# Remove low-variance features
low_var = train.var()[train.var() < 0.01].index
train.drop(columns=low_var, inplace=True)
print(f"After variance filtering: {train.shape}")

# Feature / target split
X = train.drop(['isFraud', 'TransactionID'], axis=1, errors='ignore')
y = train['isFraud']

# Save original class balance
class_before = y.value_counts(normalize=True)

# Apply SMOTE
sm = SMOTE(random_state=42, sampling_strategy=0.2)
X_res, y_res = sm.fit_resample(X, y)
class_after = pd.Series(y_res).value_counts(normalize=True)

# Scaling for NN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)
X_train_scaled, X_test_scaled, _, _ = train_test_split(X_scaled, y_res, test_size=0.2, random_state=42, stratify=y_res)

# =============================
# 3. Define Models
# =============================
models = {
    "RandomForest (Bagging)": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
    "ExtraTrees (Bagging)": ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42),
    "XGBoost (Boosting)": XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=8, subsample=0.8, eval_metric="logloss"),
    "LightGBM (Boosting)": LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=64),
    "CatBoost (Boosting)": CatBoostClassifier(iterations=400, learning_rate=0.05, depth=8, verbose=0)
}

# Stacking
estimators = [
    ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.05, eval_metric="logloss")),
    ('lgb', LGBMClassifier(n_estimators=200, learning_rate=0.05)),
    ('cat', CatBoostClassifier(iterations=200, learning_rate=0.05, verbose=0))
]
stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
models["Stacking (Meta-Ensemble)"] = stacking

# Hybrid Deep Ensemble (Neural)
def build_nn(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# =============================
# 4. Train & Evaluate
# =============================
results = {}
y_preds, y_probas = {}, {}

for name, model in models.items():
    print(f"\n🔹 Training {name} ...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]
    y_preds[name], y_probas[name] = preds, probas
    results[name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "AUC": roc_auc_score(y_test, probas),
        "AP": average_precision_score(y_test, probas)
    }

print("\n🔹 Training Hybrid Deep Ensemble (Neural Network) ...")
nn = build_nn(X_train_scaled.shape[1])
nn.fit(X_train_scaled, y_train, epochs=5, batch_size=512, verbose=1)
probas_nn = nn.predict(X_test_scaled).ravel()
preds_nn = (probas_nn > 0.5).astype(int)
y_preds["Hybrid Deep Ensemble"], y_probas["Hybrid Deep Ensemble"] = preds_nn, probas_nn
results["Hybrid Deep Ensemble"] = {
    "Accuracy": accuracy_score(y_test, preds_nn),
    "F1": f1_score(y_test, preds_nn),
    "Precision": precision_score(y_test, preds_nn),
    "Recall": recall_score(y_test, preds_nn),
    "AUC": roc_auc_score(y_test, probas_nn),
    "AP": average_precision_score(y_test, probas_nn)
}

# =============================
# 5. Results Table
# =============================
df = pd.DataFrame(results).T.sort_values("AUC", ascending=False)
df.to_csv(os.path.join(OUT_DIR, "model_metrics_advanced_full.csv"))
print("\n📊 Model Performance Summary:\n", df)

# =============================
# 6. Visualization
# =============================

# A. Heatmap
plt.figure(figsize=(10,5))
sns.heatmap(df, annot=True, cmap="Blues", fmt=".3f")
plt.title("Model Performance Heatmap — Advanced Preprocessing")
plt.savefig(os.path.join(OUT_DIR, "model_performance_heatmap.png"), dpi=200)
plt.close()

# B. Bar Chart (AUC/F1)
df[['AUC', 'F1']].plot(kind="bar", figsize=(8,5))
plt.title("Model Comparison (AUC vs F1)")
plt.ylabel("Score")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUT_DIR, "model_comparison_bar.png"), dpi=200)
plt.close()

# C. Confusion Matrices
for name, preds in y_preds.items():
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(OUT_DIR, f"cm_{name.replace(' ', '_')}.png"), dpi=200)
    plt.close()

# D. ROC Curves
plt.figure(figsize=(6,5))
for name, probas in y_probas.items():
    fpr, tpr, _ = roc_curve(y_test, probas)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, probas):.3f})")
plt.plot([0,1],[0,1],'--',color='gray')
plt.title("ROC Curves — Ensemble Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "roc_curves.png"), dpi=200)
plt.close()

# E. Precision–Recall Curves
plt.figure(figsize=(6,5))
for name, probas in y_probas.items():
    precision, recall, _ = precision_recall_curve(y_test, probas)
    plt.plot(recall, precision, label=f"{name} (AP={average_precision_score(y_test, probas):.3f})")
plt.title("Precision–Recall Curves — Ensemble Models")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "pr_curves.png"), dpi=200)
plt.close()

# =============================
# 7. EDA Plots
# =============================

# Class distribution before/after SMOTE
plt.figure(figsize=(6,4))
sns.barplot(x=class_before.index.astype(str), y=class_before.values*100, palette="Reds")
plt.title("Class Distribution Before SMOTE (%)")
plt.ylabel("Percentage")
plt.savefig(os.path.join(OUT_DIR, "eda_class_before.png"), dpi=200)
plt.close()

plt.figure(figsize=(6,4))
sns.barplot(x=class_after.index.astype(str), y=class_after.values*100, palette="Greens")
plt.title("Class Distribution After SMOTE (%)")
plt.ylabel("Percentage")
plt.savefig(os.path.join(OUT_DIR, "eda_class_after.png"), dpi=200)
plt.close()

# Correlation Heatmap (subset)
corr = X_res.corr().abs().iloc[:15, :15]
plt.figure(figsize=(8,6))
sns.heatmap(corr, cmap="coolwarm")
plt.title("Feature Correlation Heatmap (Sample Features)")
plt.savefig(os.path.join(OUT_DIR, "eda_corr_heatmap.png"), dpi=200)
plt.close()

# =============================
# 8. SHAP Explainability
# =============================
best_model_name = df.index[0]
best_model = models.get(best_model_name)
explainer = shap.Explainer(best_model, X_test.sample(1000, random_state=42))
shap_values = explainer(X_test.sample(1000, random_state=42))
shap.summary_plot(shap_values, show=False)
plt.savefig(os.path.join(OUT_DIR, f"shap_summary_{best_model_name}.png"), dpi=200, bbox_inches="tight")
plt.close()

print(f"\n✅ All visualizations, metrics, and SHAP results saved in: {OUT_DIR}")