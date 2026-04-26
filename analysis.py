"""
Heart Attack Predict — Standalone Data Analysis Script
BSc (Hons) Software Engineering Final Year Project
NSBM Green University | University of Plymouth
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = BASE_DIR / "static" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  Heart Attack Predict — Data Analysis")
print("=" * 60)

# ── Load Data ────────────────────────────────────────────────────────────────
print("\n[1/6] Loading datasets...")

df1 = pd.read_csv(DATA_DIR / "heart_disease_uci.csv")
if 'num' in df1.columns:
    df1['target'] = (df1['num'] > 0).astype(int)
    df1.drop(columns=['num'], inplace=True)
df1.drop(columns=[c for c in ['id', 'dataset'] if c in df1.columns], inplace=True)
df1.rename(columns={'thalch': 'thalach'}, inplace=True)
df1 = df1[['age','sex','cp','trestbps','chol','fbs','thalach','target']].copy()

df2 = pd.read_csv(DATA_DIR / "heart_statlog_cleveland_hungary_final.csv")
df2.rename(columns={
    'chest pain type': 'cp', 'resting bp s': 'trestbps',
    'cholesterol': 'chol', 'fasting blood sugar': 'fbs',
    'max heart rate': 'thalach'
}, inplace=True)
df2 = df2[['age','sex','cp','trestbps','chol','fbs','thalach','target']].copy()

df = pd.concat([df1, df2], ignore_index=True)
df.dropna(subset=['target'], inplace=True)

print(f"    Dataset 1 records : {len(df1)}")
print(f"    Dataset 2 records : {len(df2)}")
print(f"    Combined records  : {len(df)}")

# ── Basic Stats ───────────────────────────────────────────────────────────────
print("\n[2/6] Basic statistics...")

high_risk = (df['target'] == 1).sum()
low_risk  = (df['target'] == 0).sum()

print(f"    High Risk : {high_risk} ({high_risk/len(df)*100:.1f}%)")
print(f"    Low Risk  : {low_risk}  ({low_risk/len(df)*100:.1f}%)")
print(f"    Age range : {int(df['age'].min())} – {int(df['age'].max())} years")
print(f"    Avg age   : {df['age'].mean():.1f} years")
print(f"    Avg chol  : {df['chol'].mean():.1f} mg/dl")
print(f"    Avg HR    : {df['thalach'].mean():.1f} bpm")

# ── Missing Values ────────────────────────────────────────────────────────────
print("\n[3/6] Missing value analysis...")
missing = df.isnull().sum()
for col, val in missing.items():
    if val > 0:
        print(f"    {col}: {val} missing ({val/len(df)*100:.1f}%)")
if missing.sum() == 0:
    print("    No missing values found.")

# ── Plot 1: Target Distribution ───────────────────────────────────────────────
print("\n[4/6] Generating plots...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.patch.set_facecolor('#020818')
for ax in axes.flat:
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='#888')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Chart 1 - Target distribution
labels = ['Low Risk', 'High Risk']
sizes  = [low_risk, high_risk]
colors = ['#198754', '#dc3545']
axes[0,0].bar(labels, sizes, color=colors, alpha=0.85, width=0.5)
axes[0,0].set_title('Risk Distribution', color='white', fontweight='bold', pad=12)
axes[0,0].set_ylabel('Count', color='#888')
for i, v in enumerate(sizes):
    axes[0,0].text(i, v + 5, str(v), ha='center', color='white', fontweight='bold')

# Chart 2 - Age distribution by risk
hr_ages = df[df['target']==1]['age']
lr_ages = df[df['target']==0]['age']
axes[0,1].hist(lr_ages, bins=20, alpha=0.7, color='#198754', label='Low Risk')
axes[0,1].hist(hr_ages, bins=20, alpha=0.7, color='#dc3545', label='High Risk')
axes[0,1].set_title('Age Distribution by Risk', color='white', fontweight='bold', pad=12)
axes[0,1].set_xlabel('Age', color='#888')
axes[0,1].set_ylabel('Count', color='#888')
axes[0,1].legend(facecolor='#0d1117', labelcolor='white', framealpha=0.5)

# Chart 3 - Cholesterol by risk
bp_data = [df[df['target']==0]['chol'].dropna(), df[df['target']==1]['chol'].dropna()]
bp = axes[0,2].boxplot(bp_data, patch_artist=True, labels=['Low Risk','High Risk'])
bp['boxes'][0].set_facecolor('#198754')
bp['boxes'][1].set_facecolor('#dc3545')
for element in ['whiskers','caps','medians']:
    for item in bp[element]:
        item.set_color('white')
axes[0,2].set_title('Cholesterol by Risk Level', color='white', fontweight='bold', pad=12)
axes[0,2].set_ylabel('Cholesterol (mg/dl)', color='#888')

# Chart 4 - Max Heart Rate by risk
hr_data = [df[df['target']==0]['thalach'].dropna(), df[df['target']==1]['thalach'].dropna()]
bp2 = axes[1,0].boxplot(hr_data, patch_artist=True, labels=['Low Risk','High Risk'])
bp2['boxes'][0].set_facecolor('#198754')
bp2['boxes'][1].set_facecolor('#dc3545')
for element in ['whiskers','caps','medians']:
    for item in bp2[element]:
        item.set_color('white')
axes[1,0].set_title('Max Heart Rate by Risk Level', color='white', fontweight='bold', pad=12)
axes[1,0].set_ylabel('Heart Rate (bpm)', color='#888')

# Chart 5 - Sex distribution
sex_risk = df.groupby(['sex','target']).size().unstack(fill_value=0)
x = np.arange(len(sex_risk.index))
axes[1,1].bar(x - 0.2, sex_risk[0], 0.4, label='Low Risk',  color='#198754', alpha=0.85)
axes[1,1].bar(x + 0.2, sex_risk[1], 0.4, label='High Risk', color='#dc3545', alpha=0.85)
axes[1,1].set_title('Risk by Sex', color='white', fontweight='bold', pad=12)
axes[1,1].set_xticks(x)
axes[1,1].set_xticks(x)
axes[1,1].set_xticklabels([f'Female ({i})' if i==0 else f'Male ({i})' for i in sex_risk.index], color='#888')
axes[1,1].set_ylabel('Count', color='#888')
axes[1,1].legend(facecolor='#0d1117', labelcolor='white', framealpha=0.5)

# Chart 6 - Blood Pressure by risk
bp_risk = df.groupby('target')['trestbps'].mean()
axes[1,2].bar(['Low Risk','High Risk'], bp_risk.values, color=['#198754','#dc3545'], alpha=0.85, width=0.5)
axes[1,2].set_title('Avg Blood Pressure by Risk', color='white', fontweight='bold', pad=12)
axes[1,2].set_ylabel('Blood Pressure (mmHg)', color='#888')
for i, v in enumerate(bp_risk.values):
    axes[1,2].text(i, v + 0.5, f'{v:.1f}', ha='center', color='white', fontweight='bold')

fig.suptitle('Heart Attack Predict — Dataset Analysis', color='white', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / 'analysis_charts.png', dpi=150, bbox_inches='tight', facecolor='#020818')
plt.close()
print("    Saved: static/analysis/analysis_charts.png")

# ── Correlation ───────────────────────────────────────────────────────────────
print("\n[5/6] Correlation analysis...")
numeric_df = df[['age','trestbps','chol','thalach','target']].dropna()
corr = numeric_df.corr()

fig2, ax2 = plt.subplots(figsize=(8, 6))
fig2.patch.set_facecolor('#020818')
ax2.set_facecolor('#0d1117')
im = ax2.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1)
ax2.set_xticks(range(len(corr.columns)))
ax2.set_yticks(range(len(corr.columns)))
ax2.set_xticklabels(corr.columns, color='white', rotation=45, ha='right')
ax2.set_yticklabels(corr.columns, color='white')
for i in range(len(corr)):
    for j in range(len(corr.columns)):
        ax2.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', color='black', fontsize=9, fontweight='bold')
plt.colorbar(im, ax=ax2)
ax2.set_title('Feature Correlation Matrix', color='white', fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(OUT_DIR / 'correlation_matrix.png', dpi=150, bbox_inches='tight', facecolor='#020818')
plt.close()
print("    Saved: static/analysis/correlation_matrix.png")

# ── Summary Report ────────────────────────────────────────────────────────────
print("\n[6/6] Summary report...")
print("\n" + "=" * 60)
print("  ANALYSIS SUMMARY")
print("=" * 60)
print(f"  Total Records     : {len(df)}")
print(f"  High Risk         : {high_risk} ({high_risk/len(df)*100:.1f}%)")
print(f"  Low Risk          : {low_risk} ({low_risk/len(df)*100:.1f}%)")
print(f"  Age Range         : {int(df['age'].min())} – {int(df['age'].max())} years")
print(f"  Mean Age          : {df['age'].mean():.1f} years")
print(f"  Mean Cholesterol  : {df['chol'].mean():.1f} mg/dl")
print(f"  Mean Heart Rate   : {df['thalach'].mean():.1f} bpm")
print(f"  Mean Blood Pressure: {df['trestbps'].mean():.1f} mmHg")
print(f"  Target Correlation:")
for col in ['age','trestbps','chol','thalach']:
    c = numeric_df[col].corr(numeric_df['target'])
    print(f"    {col:12s} → {c:+.3f}")
print("=" * 60)
print("\n  Analysis complete! Charts saved to static/analysis/")
print("=" * 60)
