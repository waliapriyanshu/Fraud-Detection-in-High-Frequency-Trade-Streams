import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ======================
# CONFIG
# ======================
DATA_PATH = "/Users/priyanshuwalia7/Downloads/raw"
OUT_DIR = "./eda_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ======================
# 1. Load Data
# ======================
train_trans = pd.read_csv(os.path.join(DATA_PATH, "train_transaction.csv"))
train_id = pd.read_csv(os.path.join(DATA_PATH, "train_identity.csv"))
train = train_trans.merge(train_id, on="TransactionID", how="left")

print(f"✅ Dataset loaded: {train.shape[0]} rows, {train.shape[1]} columns")

# ======================
# 2. Quick Overview
# ======================
summary = {
    "Rows": train.shape[0],
    "Columns": train.shape[1],
    "Missing (%)": round((train.isnull().sum().sum() / (train.shape[0]*train.shape[1])) * 100, 2),
    "Fraud Ratio (%)": round(train["isFraud"].mean() * 100, 3)
}
print("\n📊 Dataset Summary:")
for k,v in summary.items():
    print(f"{k}: {v}")

# ======================
# 3. Class Distribution
# ======================
plt.figure(figsize=(10,4))

# Bar plot
plt.subplot(1,2,1)
sns.countplot(x="isFraud", data=train, palette="viridis")
plt.title("Class Distribution (Fraud vs Non-Fraud)")
plt.xlabel("Class")
plt.ylabel("Count")

# Pie chart
plt.subplot(1,2,2)
counts = train["isFraud"].value_counts()
plt.pie(counts, labels=["Non-Fraud (0)", "Fraud (1)"], autopct='%1.2f%%', colors=["#5DADE2","#E74C3C"])
plt.title("Class Distribution Percentage")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_distribution.png"), dpi=200)
plt.close()

# ======================
# 4. Missing Value Analysis
# ======================
missing = train.isnull().mean().sort_values(ascending=False)
missing = missing[missing > 0]
plt.figure(figsize=(10,6))
sns.barplot(x=missing[:20]*100, y=missing.index[:20], palette="mako")
plt.title("Top 20 Columns by Missing Value Percentage")
plt.xlabel("Missing (%)")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "missing_values.png"), dpi=200)
plt.close()

# ======================
# 5. Transaction Amount Distribution
# ======================
plt.figure(figsize=(8,5))
sns.histplot(np.log1p(train["TransactionAmt"]), bins=100, color="#1ABC9C")
plt.title("Distribution of log(TransactionAmt)")
plt.xlabel("log(TransactionAmt + 1)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "transaction_amount_dist.png"), dpi=200)
plt.close()

# ======================
# 6. Correlation Heatmap
# ======================
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
corr = train[num_cols].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr.iloc[:20, :20], cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (Top 20 Features)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"), dpi=200)
plt.close()

# ======================
# 7. Feature Importance (Correlation with Target)
# ======================
corr_target = corr["isFraud"].drop("isFraud").sort_values(key=abs, ascending=False)[:15]
plt.figure(figsize=(8,5))
sns.barplot(x=corr_target.values, y=corr_target.index, palette="crest")
plt.title("Top 15 Features Correlated with Fraud Target")
plt.xlabel("Correlation with isFraud")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_target_corr.png"), dpi=200)
plt.close()

# ======================
# 8. Distribution by Fraud Class (key features)
# ======================
key_features = ["TransactionAmt", "card1", "card2", "C1", "C13"]
for feature in key_features:
    if feature in train.columns:
        plt.figure(figsize=(7,4))
        sns.kdeplot(data=train, x=feature, hue="isFraud", fill=True, common_norm=False, palette="coolwarm")
        plt.title(f"Distribution of {feature} by Fraud Class")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"feature_dist_{feature}.png"), dpi=200)
        plt.close()

# ======================
# 9. Pairwise Relationship (Sample Features)
# ======================
selected = [c for c in ["TransactionAmt", "C1", "C13", "V1", "V70", "isFraud"] if c in train.columns]
sns.pairplot(train[selected].sample(2000, random_state=42), hue="isFraud", palette="coolwarm")
plt.suptitle("Pairwise Relationships of Key Features", y=1.02)
plt.savefig(os.path.join(OUT_DIR, "pairwise_features.png"), dpi=200)
plt.close()

# ======================
# 10. Save Summary Table
# ======================
desc = train.describe(include='all').T
desc["missing_percent"] = (train.isnull().sum() / len(train) * 100)
desc.to_csv(os.path.join(OUT_DIR, "dataset_description.csv"))

print("\n✅ EDA completed successfully!")
print(f"All visualizations and summary files saved in: {OUT_DIR}")
