import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# ── Standardizers ──────────────────────────────────────────────────────────
def standardize_cp(val):
    mapping = {
        'typical angina': 1, 'typical': 1, '1': 1,
        'atypical angina': 2, 'atypical': 2, '2': 2,
        'non-anginal': 3, 'non-anginal pain': 3, '3': 3,
        'asymptomatic': 0, '0': 0, '4': 0,
    }
    return mapping.get(str(val).strip().lower(), 0)

def standardize_sex(val):
    return 1 if str(val).strip().lower() in ['1', 'male', 'm'] else 0

def standardize_fbs(val):
    return 1 if str(val).strip().lower() in ['1', 'true', 'yes'] else 0

# ── Load Dataset 1: UCI Heart Disease ─────────────────────────────────────
df1 = pd.read_csv(DATA_DIR / "heart_disease_uci.csv")
print(f"UCI dataset: {df1.shape}")

if 'num' in df1.columns:
    df1['target'] = (df1['num'] > 0).astype(int)
    df1.drop(columns=['num'], inplace=True)

df1.drop(columns=[c for c in ['id', 'dataset'] if c in df1.columns], inplace=True)
df1.rename(columns={'thalch': 'thalach'}, inplace=True)
df1 = df1[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thalach', 'target']].copy()
df1['sex'] = df1['sex'].map(standardize_sex)
df1['fbs'] = df1['fbs'].map(standardize_fbs)
df1['cp']  = df1['cp'].map(standardize_cp)
df1.dropna(inplace=True)
print(f"UCI after cleaning: {df1.shape}")

# ── Load Dataset 2: Statlog ────────────────────────────────────────────────
df2 = pd.read_csv(DATA_DIR / "heart_statlog_cleveland_hungary_final.csv")
print(f"\nStatlog dataset: {df2.shape}")

df2.rename(columns={
    'chest pain type':     'cp',
    'resting bp s':        'trestbps',
    'cholesterol':         'chol',
    'fasting blood sugar': 'fbs',
    'max heart rate':      'thalach',
}, inplace=True)

df2 = df2[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thalach', 'target']].copy()
df2['sex'] = df2['sex'].map(standardize_sex)
df2['fbs'] = df2['fbs'].map(standardize_fbs)
df2['cp']  = df2['cp'].map(standardize_cp)
df2.dropna(inplace=True)
print(f"Statlog after cleaning: {df2.shape}")

# ── Combine WITHOUT dropping duplicates ────────────────────────────────────
df = pd.concat([df1, df2], ignore_index=True)
df.dropna(subset=['target'], inplace=True)
df = df.astype({'cp': int, 'sex': int, 'fbs': int, 'target': int})

print(f"\n✅ Combined dataset: {df.shape}")
print(f"Target: {df['target'].value_counts().to_dict()}")

# ── Features & Target ──────────────────────────────────────────────────────
X = df.drop('target', axis=1)
y = df['target']
print(f"Features: {X.columns.tolist()}")

# ── Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── Model 1: Logistic Regression ───────────────────────────────────────────
lr = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   LogisticRegression(max_iter=1000, C=0.1))
])
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))
print(f"\n── Logistic Regression:  {lr_acc*100:.2f}%")

# ── Model 2: Random Forest ─────────────────────────────────────────────────
rf = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model',   RandomForestClassifier(
        n_estimators=200, max_depth=10,
        min_samples_split=5, random_state=42
    ))
])
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"── Random Forest:        {rf_acc*100:.2f}%")

# ── Model 3: Gradient Boosting ─────────────────────────────────────────────
gb = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, random_state=42
    ))
])
gb.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test))
print(f"── Gradient Boosting:    {gb_acc*100:.2f}%")

# ── Model 4: Voting Ensemble ───────────────────────────────────────────────
voting = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000, C=0.1)),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42))
        ],
        voting='soft'
    ))
])
voting.fit(X_train, y_train)
voting_acc = accuracy_score(y_test, voting.predict(X_test))
print(f"── Voting Ensemble:      {voting_acc*100:.2f}%")

# ── Pick Best Model ────────────────────────────────────────────────────────
models = {
    'Logistic Regression': (lr,     lr_acc),
    'Random Forest':       (rf,     rf_acc),
    'Gradient Boosting':   (gb,     gb_acc),
    'Voting Ensemble':     (voting, voting_acc),
}

best_name, (best_model, best_acc) = max(models.items(), key=lambda x: x[1][1])
print(f"\n🏆 Best Model: {best_name} → {best_acc*100:.2f}%")
print(classification_report(y_test, best_model.predict(X_test)))

cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print(f"5-Fold CV: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ── Save ───────────────────────────────────────────────────────────────────
joblib.dump(best_model, MODEL_DIR / "heart_attack_model.pkl")
joblib.dump(X.columns.tolist(), MODEL_DIR / "feature_columns.pkl")
print(f"\n✅ Model saved → model/heart_attack_model.pkl")
print(f"✅ Features saved → model/feature_columns.pkl")