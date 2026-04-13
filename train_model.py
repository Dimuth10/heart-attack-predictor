import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_PATH  = BASE_DIR / "data" / "heart_disease_uci.csv"
MODEL_DIR  = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# ── Load Data ───────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())

# ── Target Column ───────────────────────────────────────────────────────────
# 'num' column: 0 = no disease, 1-4 = disease → convert to binary
if 'num' in df.columns:
    df['target'] = (df['num'] > 0).astype(int)
    df.drop('num', axis=1, inplace=True)
elif 'target' in df.columns:
    pass
else:
    raise Exception("Target column not found!")

# ── Drop irrelevant columns ─────────────────────────────────────────────────
drop_cols = ['id', 'dataset']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# ── Features & Target ───────────────────────────────────────────────────────
X = df.drop('target', axis=1)
y = df['target']

print("\nTarget distribution:\n", y.value_counts())

# ── Handle categorical columns ───────────────────────────────────────────────
X = pd.get_dummies(X)

# ── Train / Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Logistic Regression Pipeline ────────────────────────────────────────────
lr_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   LogisticRegression(max_iter=1000))
])

lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_test)
lr_acc   = accuracy_score(y_test, lr_preds)
print("\n── Logistic Regression ──────────────────")
print(f"Accuracy: {lr_acc * 100:.2f}%")
print(classification_report(y_test, lr_preds))

# ── Random Forest Pipeline ───────────────────────────────────────────────────
rf_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model',   RandomForestClassifier(n_estimators=100, random_state=42))
])

rf_pipeline.fit(X_train, y_train)
rf_preds = rf_pipeline.predict(X_test)
rf_acc   = accuracy_score(y_test, rf_preds)
print("\n── Random Forest ────────────────────────")
print(f"Accuracy: {rf_acc * 100:.2f}%")
print(classification_report(y_test, rf_preds))

# ── Save Best Model ──────────────────────────────────────────────────────────
if rf_acc >= lr_acc:
    best_model     = rf_pipeline
    best_model_name = "Random Forest"
else:
    best_model     = lr_pipeline
    best_model_name = "Logistic Regression"

print(f"\nBest model: {best_model_name} ({max(rf_acc, lr_acc)*100:.2f}%)")

# Save model and feature columns
joblib.dump(best_model, MODEL_DIR / "heart_attack_model.pkl")
joblib.dump(X.columns.tolist(), MODEL_DIR / "feature_columns.pkl")

print("\nModel saved to model/heart_attack_model.pkl")
print("Feature columns saved to model/feature_columns.pkl")