import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ─── LOAD DATA ──────────────────────────────────────────────────────────────

df = pd.read_csv("data/class_history.csv")

print("Data loaded. Shape:", df.shape)
print(df.head())

# ─── MODEL 1: CANCELLATION PREDICTOR (Random Forest) ───────────────────────

print("\n--- Training Model 1: Cancellation Predictor ---")

# Features = what the model uses to make predictions
# Label = what it's trying to predict
features = ["day_of_week", "subject", "faculty", "time_slot", "week", "days_to_exam"]
X = df[features]   # X is always the input features (convention)
y = df["class_held"]  # y is always the label (convention)

# Split data: 80% for training, 20% for testing
# The model never sees test data during training — that's how we evaluate honestly
# random_state=42 makes the split consistent every run
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Build and train the Random Forest
# n_estimators=100 means 100 trees will vote
# random_state=42 for reproducibility
# class_weight='balanced' tells the model to penalize mistakes on
# the minority class (cancelled) more heavily — fixes imbalanced data
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)  # This is where learning happens

# Evaluate on test data
y_pred = rf_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# This shows precision, recall, f1-score for each class

# Feature importance — which input matters most to the model?
importances = rf_model.feature_importances_
feat_importance = pd.Series(importances, index=features).sort_values(ascending=False)
print("\nFeature importances:")
print(feat_importance.round(3))

# Save the model
os.makedirs("models", exist_ok=True)
joblib.dump(rf_model, "models/cancellation_model.pkl")
print("Saved: models/cancellation_model.pkl")

# ─── MODEL 2: RISK SCORER (Logistic Regression) ─────────────────────────────

print("\n--- Training Model 2: Risk Scorer ---")

# For this model we need student-level attendance data
# We'll generate it synthetically — each row = one student's subject attendance
np.random.seed(42)
n_students = 300

# Simulate attendance % for each subject per student (values between 40 and 100)
# Generate students in three realistic groups
# Group 1 (100 students): good attendance — mostly 80-100%
# Group 2 (100 students): borderline — mostly 65-85%
# Group 3 (100 students): poor attendance — mostly 40-70%

def gen_group(low, high, n):
    return np.random.randint(low, high, n)

att_data = {
    "dbs_pct": np.concatenate([gen_group(80,101,100), gen_group(65,86,100), gen_group(40,71,100)]),
    "os_pct":  np.concatenate([gen_group(80,101,100), gen_group(65,86,100), gen_group(40,71,100)]),
    "ai_pct":  np.concatenate([gen_group(80,101,100), gen_group(65,86,100), gen_group(40,71,100)]),
    "po_pct":  np.concatenate([gen_group(80,101,100), gen_group(65,86,100), gen_group(40,71,100)]),
    "daa_pct": np.concatenate([gen_group(80,101,100), gen_group(65,86,100), gen_group(40,71,100)]),
}
att_df = pd.DataFrame(att_data)

# Overall average attendance across all subjects
att_df["avg_pct"] = att_df[["dbs_pct","os_pct","ai_pct","po_pct","daa_pct"]].mean(axis=1)

# Count how many subjects are below 75%
att_df["subjects_at_risk"] = (
    att_df[["dbs_pct","os_pct","ai_pct","po_pct","daa_pct"]] < 75
).sum(axis=1)

# Label: 0=Safe, 1=Warning, 2=Critical
def assign_risk(row):
    if row["avg_pct"] >= 85 and row["subjects_at_risk"] == 0:
        return 0  # Safe
    elif row["avg_pct"] >= 75 and row["subjects_at_risk"] <= 1:
        return 1  # Warning
    else:
        return 2  # Critical

att_df["risk_label"] = att_df.apply(assign_risk, axis=1)

print("\nRisk label distribution:")
print(att_df["risk_label"].value_counts())

# Features for risk model
X_risk = att_df[["avg_pct", "subjects_at_risk"]]
y_risk = att_df["risk_label"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_risk, y_risk, test_size=0.2, random_state=42
)

# Scale features — Logistic Regression works better when features are on same scale
# StandardScaler converts to mean=0, std=1
scaler = StandardScaler()
X_train_r_scaled = scaler.fit_transform(X_train_r)
X_test_r_scaled  = scaler.transform(X_test_r)
# Note: fit_transform on train, only transform on test
# You never fit the scaler on test data — that would leak information

risk_model = LogisticRegression(max_iter=1000, random_state=42)
risk_model.fit(X_train_r_scaled, y_train_r)

y_pred_r = risk_model.predict(X_test_r_scaled)
print("\nRisk Model Classification Report:")
print(classification_report(y_test_r, y_pred_r))

joblib.dump(risk_model, "models/risk_model.pkl")
joblib.dump(scaler, "models/risk_scaler.pkl")
print("Saved: models/risk_model.pkl")
print("Saved: models/risk_scaler.pkl")

# ─── MODEL 3: STUDENT CLUSTERING (K-Means) ──────────────────────────────────

print("\n--- Training Model 3: Student Clustering ---")

# Use the same attendance % data
X_cluster = att_df[["dbs_pct","os_pct","ai_pct","po_pct","daa_pct"]]

# K-Means needs scaled data too
cluster_scaler = StandardScaler()
X_cluster_scaled = cluster_scaler.fit_transform(X_cluster)

# 3 clusters: Consistent / Selective / At-risk
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_cluster_scaled)

att_df["cluster"] = kmeans.labels_

print("\nCluster sizes:")
print(att_df["cluster"].value_counts())

print("\nAverage attendance per cluster:")
print(att_df.groupby("cluster")[["dbs_pct","os_pct","ai_pct","po_pct","daa_pct"]].mean().round(1))

joblib.dump(kmeans, "models/cluster_model.pkl")
joblib.dump(cluster_scaler, "models/cluster_scaler.pkl")
print("Saved: models/cluster_model.pkl")
print("Saved: models/cluster_scaler.pkl")

print("\n All 3 models trained and saved successfully!")