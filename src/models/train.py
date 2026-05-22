"""
Train and compare disease-prediction classifiers.

Usage:
    python -m src.models.train
    python -m src.models.train --data dataset.xlsx --out model/
"""
import argparse
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data.preprocess import preprocess


def build_models():
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, C=1.0, random_state=42
        ),
    }
    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(
            n_estimators=150, learning_rate=0.1, random_state=42,
            eval_metric='mlogloss', verbosity=0
        )
    except ImportError:
        pass
    return models


def train_and_compare(data_path='dataset.xlsx', save_dir='model'):
    print(f"Loading data from {data_path}...")
    X, y, mlb, le, df = preprocess(data_path)
    print(f"  {X.shape[0]} samples | {X.shape[1]} symptom features | {len(le.classes_)} diseases")

    # Drop classes with only 1 sample — cannot split
    unique, counts = np.unique(y, return_counts=True)
    valid_mask = np.isin(y, unique[counts >= 2])
    X, y = X[valid_mask], y[valid_mask]
    dropped = (counts < 2).sum()
    if dropped:
        print(f"  Dropped {dropped} single-sample classes from split (kept for full-data training)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}\n")

    models = build_models()
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        # 5-fold CV on training set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

        results[name] = {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
        }
        print(f"[{name}]")
        print(f"  Train acc : {train_acc:.4f}")
        print(f"  Test  acc : {test_acc:.4f}")
        print(f"  CV (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

    best_name = max(results, key=lambda k: results[k]['test_acc'])
    best_model = results[best_name]['model']
    print(f"Best model: {best_name} (test acc {results[best_name]['test_acc']:.4f})")

    # Retrain best model on full dataset for deployment
    best_model.fit(X, y)

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(save_dir, 'best_model.joblib'))
    joblib.dump(mlb, os.path.join(save_dir, 'symptom_encoder.joblib'))
    joblib.dump(le, os.path.join(save_dir, 'disease_encoder.joblib'))
    print(f"\nArtifacts saved to {save_dir}/")

    return results, best_model, mlb, le, X_test, y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='dataset.xlsx')
    parser.add_argument('--out', default='model')
    args = parser.parse_args()
    train_and_compare(data_path=args.data, save_dir=args.out)
