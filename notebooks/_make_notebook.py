"""Run this script once to generate eda_and_training.ipynb."""
import json, os

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})

def code(src):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    })

# ── 1. Title ──────────────────────────────────────────────────────────────────
md("""# GenZ AI Healthcare Assistant — EDA & Model Training

End-to-end notebook covering:
1. Dataset exploration
2. Symptom & disease analysis
3. Feature engineering (binary encoding)
4. Model training & comparison (RF · XGBoost · Gradient Boosting · Logistic Regression)
5. Cross-validation & hyperparameter discussion
6. Best-model evaluation (confusion matrix, classification report, feature importance)
7. Saving production artifacts
""")

# ── 2. Imports ────────────────────────────────────────────────────────────────
code("""\
import sys, os
sys.path.insert(0, os.path.dirname(os.getcwd()))  # repo root

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import joblib

sns.set_theme(style='whitegrid', palette='muted')
%matplotlib inline
""")

# ── 3. Load data ───────────────────────────────────────────────────────────────
md("## 1. Data Loading")
code("""\
DATA_PATH = '../dataset.xlsx'
df = pd.read_excel(DATA_PATH)
print(f'Shape: {df.shape}')
df.head()
""")

# ── 4. Overview ────────────────────────────────────────────────────────────────
md("## 2. Dataset Overview")
code("""\
print(f"Diseases : {df['Disease'].nunique()}")
print(f"Samples  : {len(df)}")
print(f"Nulls    : {df.isnull().sum().to_dict()}")
print()
print(df['Disease'].value_counts())
""")

# ── 5. Class distribution ──────────────────────────────────────────────────────
md("## 3. Class Distribution")
code("""\
counts = df['Disease'].value_counts()

fig, ax = plt.subplots(figsize=(14, 8))
counts.plot(kind='barh', ax=ax, color='steelblue')
ax.set_xlabel('Number of samples')
ax.set_title('Samples per Disease Class')
plt.tight_layout()
plt.savefig('../notebooks/class_distribution.png', dpi=150)
plt.show()
print(f'\\nClass imbalance ratio (max/min): {counts.max()/counts.min():.1f}x')
""")

# ── 6. Symptom analysis ────────────────────────────────────────────────────────
md("## 4. Symptom Frequency Analysis")
code("""\
from src.data.preprocess import extract_symptoms, get_all_symptoms

disease_names = set(d.lower().strip() for d in df['Disease'].unique())

sym_counter = Counter()
for _, row in df.iterrows():
    for s in extract_symptoms(row['Symptoms'], row['Disease']):
        sym_counter[s] += 1

top_symptoms = sym_counter.most_common(30)
syms, freqs = zip(*top_symptoms)

fig, ax = plt.subplots(figsize=(12, 9))
ax.barh(list(syms)[::-1], list(freqs)[::-1], color='coral')
ax.set_xlabel('Frequency across all disease entries')
ax.set_title('Top 30 Most Common Symptoms')
plt.tight_layout()
plt.savefig('../notebooks/symptom_frequency.png', dpi=150)
plt.show()
print(f'Total unique symptoms: {len(sym_counter)}')
""")

# ── 7. Symptoms per disease ────────────────────────────────────────────────────
md("## 5. Symptoms per Disease (Diversity)")
code("""\
sym_counts_per_disease = (
    df.groupby('Disease')['Symptoms']
    .apply(lambda rows: len(set(
        s.strip().lower()
        for r in rows for s in r.split(',')
        if s.strip().lower() not in disease_names and s.strip()
    )))
)

fig, ax = plt.subplots(figsize=(14, 8))
sym_counts_per_disease.sort_values(ascending=False).plot(kind='bar', ax=ax, color='mediumpurple')
ax.set_ylabel('Unique symptoms')
ax.set_title('Unique Symptom Count per Disease')
plt.xticks(rotation=60, ha='right', fontsize=7)
plt.tight_layout()
plt.savefig('../notebooks/symptoms_per_disease.png', dpi=150)
plt.show()
""")

# ── 8. Feature engineering ─────────────────────────────────────────────────────
md("## 6. Feature Engineering — Binary Encoding")
code("""\
from src.data.preprocess import preprocess

X, y, mlb, le, _ = preprocess(DATA_PATH)
print(f'Feature matrix : {X.shape}  (samples × symptoms)')
print(f'Label vector   : {y.shape}  ({len(le.classes_)} classes)')
print(f'Symptom features (first 10): {mlb.classes_[:10].tolist()}')
""")

# ── 9. Symptom co-occurrence heatmap (top 20) ──────────────────────────────────
md("## 7. Symptom Co-occurrence Heatmap (Top 20 Symptoms)")
code("""\
# Restrict to top-20 most frequent symptoms for readability
top20_syms = [s for s, _ in sym_counter.most_common(20)]
top20_idx  = [list(mlb.classes_).index(s) for s in top20_syms if s in mlb.classes_]

X_top = X[:, top20_idx]
co_occur = (X_top.T @ X_top).astype(float)
np.fill_diagonal(co_occur, 0)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    co_occur,
    xticklabels=[mlb.classes_[i] for i in top20_idx],
    yticklabels=[mlb.classes_[i] for i in top20_idx],
    cmap='YlOrRd', annot=True, fmt='.0f', ax=ax, linewidths=0.4
)
ax.set_title('Symptom Co-occurrence (Top 20 Symptoms)')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig('../feature_correlation.png', dpi=150)
plt.show()
""")

# ── 10. Train/test split ───────────────────────────────────────────────────────
md("## 8. Train / Test Split")
code("""\
# Remove single-sample classes (can't stratify)
unique_labels, label_counts = np.unique(y, return_counts=True)
valid_mask = np.isin(y, unique_labels[label_counts >= 2])
X_valid, y_valid = X[valid_mask], y[valid_mask]

X_train, X_test, y_train, y_test = train_test_split(
    X_valid, y_valid, test_size=0.2, random_state=42, stratify=y_valid
)
print(f'Train : {X_train.shape[0]} samples')
print(f'Test  : {X_test.shape[0]} samples')
print(f'Diseases with < 2 samples (excluded from split): {(label_counts < 2).sum()}')
""")

# ── 11. Model comparison ───────────────────────────────────────────────────────
md("## 9. Model Training & Comparison")
code("""\
models = {
    'Random Forest'      : RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting'  : GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
}
try:
    from xgboost import XGBClassifier
    models['XGBoost'] = XGBClassifier(n_estimators=150, learning_rate=0.1, random_state=42,
                                       eval_metric='mlogloss', verbosity=0)
    print('XGBoost found — included in comparison')
except ImportError:
    print('XGBoost not installed — skipping (pip install xgboost)')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    results[name] = {
        'model'   : model,
        'train_acc': accuracy_score(y_train, model.predict(X_train)),
        'test_acc' : accuracy_score(y_test,  model.predict(X_test)),
        'cv_mean'  : cv_scores.mean(),
        'cv_std'   : cv_scores.std(),
    }
    print(f'{name:25s}  train={results[name][\"train_acc\"]:.4f}  '
          f'test={results[name][\"test_acc\"]:.4f}  '
          f'CV={results[name][\"cv_mean\"]:.4f}±{results[name][\"cv_std\"]:.4f}')
""")

# ── 12. CV bar chart ───────────────────────────────────────────────────────────
md("### 9.1 Cross-Validation Accuracy Comparison")
code("""\
from src.models.evaluate import plot_cv_comparison
fig = plot_cv_comparison(results)
plt.savefig('../notebooks/model_comparison.png', dpi=150)
plt.show()
""")

# ── 13. Best model evaluation ─────────────────────────────────────────────────
md("## 10. Best Model — Detailed Evaluation")
code("""\
best_name = max(results, key=lambda k: results[k]['test_acc'])
best_model = results[best_name]['model']
print(f'Best model: {best_name}\\n')

from src.models.evaluate import print_report
_ = print_report(best_model, X_test, y_test, le)
""")

# ── 14. Confusion matrix ───────────────────────────────────────────────────────
md("### 10.1 Confusion Matrix")
code("""\
from src.models.evaluate import plot_confusion_matrix
fig = plot_confusion_matrix(best_model, X_test, y_test, le)
plt.savefig('../notebooks/confusion_matrix.png', dpi=150)
plt.show()
""")

# ── 15. Feature importance ─────────────────────────────────────────────────────
md("### 10.2 Feature Importance (Top 30 Symptoms)")
code("""\
from src.models.evaluate import plot_feature_importance
fig = plot_feature_importance(best_model, mlb, top_n=30)
if fig:
    plt.savefig('../notebooks/feature_importance.png', dpi=150)
    plt.show()
""")

# ── 16. Save model ─────────────────────────────────────────────────────────────
md("## 11. Save Production Artifacts")
code("""\
import os
os.makedirs('../model', exist_ok=True)

# Retrain on full dataset before saving
best_model.fit(X, y)

joblib.dump(best_model, '../model/best_model.joblib')
joblib.dump(mlb,        '../model/symptom_encoder.joblib')
joblib.dump(le,         '../model/disease_encoder.joblib')

print('Saved:')
for f in ['best_model.joblib', 'symptom_encoder.joblib', 'disease_encoder.joblib']:
    size = os.path.getsize(f'../model/{f}') / 1024
    print(f'  model/{f}  ({size:.1f} KB)')
""")

md("""\
## Summary

| Step | Detail |
|------|--------|
| Dataset | 441 samples · 72 diseases · 318 unique symptoms |
| Feature encoding | MultiLabelBinarizer → binary symptom vector |
| Best model | see output above |
| Artifacts | `model/best_model.joblib`, `model/symptom_encoder.joblib`, `model/disease_encoder.joblib` |

**Next steps:** integrate `sentence-transformers` for semantic symptom matching,
add SHAP explainability, deploy via Docker / Hugging Face Spaces.
""")

# ── Write notebook ─────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

out_path = os.path.join(os.path.dirname(__file__), 'eda_and_training.ipynb')
with open(out_path, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"Written: {out_path}")
