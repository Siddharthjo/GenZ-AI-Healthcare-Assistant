"""
Central ML engine — loaded once at process startup.
Both app.py and the REST API blueprint import from here.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data.preprocess import get_all_symptoms
from src.data.symptom_matcher import SemanticSymptomMatcher

# ── Dataset ───────────────────────────────────────────────────────────────────

df = pd.read_excel('dataset.xlsx')

# ── Symptom matcher ───────────────────────────────────────────────────────────

all_symptoms: list[str] = get_all_symptoms(df)
matcher = SemanticSymptomMatcher(all_symptoms)

# ── Trained classifier ────────────────────────────────────────────────────────

_MODEL_DIR = 'model'
clf = mlb = le = None

if all(os.path.exists(os.path.join(_MODEL_DIR, f))
       for f in ('best_model.joblib', 'symptom_encoder.joblib', 'disease_encoder.joblib')):
    clf = joblib.load(os.path.join(_MODEL_DIR, 'best_model.joblib'))
    mlb = joblib.load(os.path.join(_MODEL_DIR, 'symptom_encoder.joblib'))
    le  = joblib.load(os.path.join(_MODEL_DIR, 'disease_encoder.joblib'))

# ── Public helpers ────────────────────────────────────────────────────────────

def predict_disease(symptom_list: list[str]) -> dict:
    """
    Map raw symptom strings → canonical → binary vector → classifier.

    Returns dict with keys: disease, confidence, symptoms_matched, html.
    Falls back to cosine-similarity text matching if model artifacts are absent.
    """
    matched = [matcher.match(s.strip()) for s in symptom_list if s.strip()]

    if clf is not None:
        X = mlb.transform([matched])
        y_pred = clf.predict(X)
        proba = clf.predict_proba(X)[0]
        disease = le.inverse_transform(y_pred)[0]
        confidence = float(proba.max())
        return {
            'disease': disease,
            'confidence': round(confidence, 4),
            'symptoms_matched': matched,
            'html': f'<b>{disease}</b>',
        }

    # Fallback
    vectorizer = CountVectorizer()
    X_corpus = vectorizer.fit_transform(df['Symptoms'])
    user_X = vectorizer.transform([', '.join(matched)])
    scores = cosine_similarity(X_corpus, user_X)
    max_score = float(scores.max())
    max_idx = int(scores.argmax())
    disease = df.iloc[max_idx]['Disease']
    return {
        'disease': disease,
        'confidence': round(max_score, 4),
        'symptoms_matched': matched,
        'html': f'<b>{disease}</b>',
    }


def get_disease_symptoms(disease_name: str) -> tuple[bool, list[str] | str]:
    """Return (found, symptom_list) for a given disease name."""
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Disease'])
    user_X = vectorizer.transform([disease_name])
    scores = cosine_similarity(X, user_X)
    max_score = float(scores.max())
    if max_score < 0.7:
        return False, 'No matching diseases found'
    max_indices = scores.argmax(axis=0)
    matched_symptoms: set[str] = set()
    for i in max_indices:
        if scores[i] == max_score:
            matched_symptoms.update(df.iloc[i]['Symptoms'].split(','))
    return True, sorted(s.strip() for s in matched_symptoms if s.strip())


_explainer = None  # lazy-init (TreeExplainer takes ~1s to build)


def explain_prediction(symptom_list: list[str], top_n: int = 6) -> dict:
    """
    Return SHAP-based explanation for which symptoms most influenced the prediction.

    Positive shap_value → symptom supports the diagnosis.
    Negative shap_value → symptom works against it (expected but absent, or misleading).
    """
    if clf is None:
        return {'error': 'Model not loaded — run src/models/train.py first'}

    global _explainer
    if _explainer is None:
        import shap
        _explainer = shap.TreeExplainer(clf)

    matched = [matcher.match(s.strip()) for s in symptom_list if s.strip()]
    X = mlb.transform([matched])
    y_pred = clf.predict(X)[0]
    disease = le.inverse_transform([y_pred])[0]

    import shap
    sv = _explainer.shap_values(X)           # (n_samples, n_features, n_clf_classes)
    model_class_idx = int(np.where(clf.classes_ == y_pred)[0][0])
    sv_for_pred = sv[0, :, model_class_idx]  # (n_features,)

    top_idx = np.argsort(np.abs(sv_for_pred))[::-1][:top_n]
    contributors = [
        {
            'symptom': mlb.classes_[i],
            'shap_value': round(float(sv_for_pred[i]), 4),
            'direction': 'supports' if sv_for_pred[i] > 0 else 'against',
        }
        for i in top_idx
    ]

    return {
        'disease': disease,
        'top_contributors': contributors,
        'note': (
            'Positive SHAP values push the model toward this diagnosis; '
            'negative values push against it.'
        ),
    }


def get_disease_info(keywords: str) -> str:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(keywords, max_results=1))
            if results:
                return results[0].get('body', '')
    except Exception:
        pass
    try:
        from duckduckgo_search import ddg
        results = ddg(keywords, region='wt-wt', safesearch='Off', time='y')
        if results:
            return results[0]['body']
    except Exception:
        pass
    return ''
