import pytest
from src.ml_engine import (
    all_symptoms,
    clf,
    explain_prediction,
    get_disease_symptoms,
    le,
    matcher,
    mlb,
    predict_disease,
)


def test_predict_returns_required_keys():
    result = predict_disease(['skin rash', 'itching'])
    assert 'disease' in result
    assert 'confidence' in result
    assert 'symptoms_matched' in result
    assert 'html' in result


def test_predict_confidence_in_range():
    result = predict_disease(['fever', 'cough'])
    assert 0.0 <= result['confidence'] <= 1.0


def test_predict_disease_is_non_empty_string():
    result = predict_disease(['fever'])
    assert isinstance(result['disease'], str)
    assert len(result['disease']) > 0


def test_predict_symptoms_matched_is_list():
    result = predict_disease(['FEVER', 'HEADACHE'])
    assert isinstance(result['symptoms_matched'], list)


def test_predict_empty_list_still_returns():
    # Edge case: empty input should not crash
    result = predict_disease([])
    assert 'disease' in result


def test_predict_fungal_infection():
    result = predict_disease(['skin rash', 'itching', 'nodal skin eruptions'])
    assert result['disease'] == 'Fungal infection'


def test_get_disease_symptoms_found():
    found, syms = get_disease_symptoms('Diabetes')
    assert found is True
    assert isinstance(syms, list)
    assert len(syms) > 0


def test_get_disease_symptoms_all_strings():
    found, syms = get_disease_symptoms('Malaria')
    assert found is True
    for s in syms:
        assert isinstance(s, str)


def test_get_disease_symptoms_not_found():
    found, msg = get_disease_symptoms('xyznonexistent_disease_zzz')
    assert found is False
    assert isinstance(msg, str)


def test_matcher_exact_match():
    assert matcher.match('fever') == 'fever'


def test_matcher_handles_typo():
    result = matcher.match('joint pian')
    assert 'joint' in result.lower() or 'pain' in result.lower()


def test_matcher_case_insensitive():
    lower = matcher.match('fever')
    upper = matcher.match('FEVER')
    assert lower == upper


def test_matcher_top_k_count():
    results = matcher.match_many('fever', top_k=3)
    assert len(results) == 3


def test_matcher_top_k_contains_best():
    results = matcher.match_many('fever', top_k=5)
    assert 'fever' in results


def test_explain_prediction_structure():
    result = explain_prediction(['skin rash', 'itching'])
    assert 'disease' in result
    assert 'top_contributors' in result
    assert isinstance(result['top_contributors'], list)


def test_explain_prediction_contributor_fields():
    result = explain_prediction(['skin rash', 'itching', 'fever'])
    for c in result['top_contributors']:
        assert 'symptom' in c
        assert 'shap_value' in c
        assert 'direction' in c
        assert c['direction'] in ('supports', 'against')


def test_explain_prediction_top_n():
    result = explain_prediction(['skin rash', 'itching'], top_n=3)
    assert len(result['top_contributors']) == 3


def test_clf_knows_all_classes():
    assert len(clf.classes_) == len(le.classes_)


def test_mlb_symmetric_with_clf():
    assert mlb.classes_.shape[0] == clf.n_features_in_
