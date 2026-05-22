import pandas as pd
import numpy as np
import pytest
from src.data.preprocess import (
    build_feature_matrix,
    extract_symptoms,
    get_all_symptoms,
    load_dataset,
)


@pytest.fixture(scope='module')
def df():
    return load_dataset('dataset.xlsx')


def test_extract_symptoms_removes_disease_name():
    syms = extract_symptoms("Fungal infection, skin rash, itching", "Fungal infection")
    assert "fungal infection" not in syms
    assert "skin rash" in syms
    assert "itching" in syms


def test_extract_symptoms_keeps_all_non_disease():
    syms = extract_symptoms("fever, headache, fatigue", "Malaria")
    assert "fever" in syms
    assert "headache" in syms
    assert "fatigue" in syms


def test_extract_symptoms_empty_string():
    syms = extract_symptoms("", "Diabetes")
    assert syms == []


def test_get_all_symptoms_no_disease_names(df):
    disease_names = {d.lower().strip() for d in df['Disease'].unique()}
    syms = get_all_symptoms(df)
    for s in syms:
        assert s not in disease_names, f"Disease name leaked into symptoms: {s}"


def test_get_all_symptoms_sorted(df):
    syms = get_all_symptoms(df)
    assert syms == sorted(syms)


def test_get_all_symptoms_non_empty(df):
    syms = get_all_symptoms(df)
    assert len(syms) > 0


def test_build_feature_matrix_row_count(df):
    X, y, mlb, le = build_feature_matrix(df)
    assert X.shape[0] == len(df)
    assert len(y) == len(df)


def test_build_feature_matrix_binary(df):
    X, _, _, _ = build_feature_matrix(df)
    unique_vals = set(X.flatten().tolist())
    assert unique_vals.issubset({0, 1})


def test_build_feature_matrix_label_count(df):
    _, y, _, le = build_feature_matrix(df)
    assert len(le.classes_) == df['Disease'].nunique()


def test_feature_matrix_no_all_zero_rows(df):
    X, _, _, _ = build_feature_matrix(df)
    row_sums = X.sum(axis=1)
    assert (row_sums > 0).all(), "Some rows have no symptoms encoded"
