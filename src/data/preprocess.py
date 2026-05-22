import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer


def load_dataset(path='dataset.xlsx'):
    return pd.read_excel(path)


def extract_symptoms(symptom_str, disease_name):
    """Parse comma-separated symptom string, stripping the disease name if echoed as first token."""
    disease_lower = disease_name.lower().strip()
    symptoms = []
    for s in symptom_str.split(','):
        s = s.strip().lower()
        if s and s != disease_lower:
            symptoms.append(s)
    return symptoms


def get_all_symptoms(df):
    """Return sorted list of every unique symptom in the dataset."""
    disease_names = {d.lower().strip() for d in df['Disease'].unique()}
    all_syms = set()
    for _, row in df.iterrows():
        for s in row['Symptoms'].split(','):
            s = s.strip().lower()
            if s and s not in disease_names:
                all_syms.add(s)
    return sorted(all_syms)


def build_feature_matrix(df):
    """
    Returns:
        X   : binary ndarray (n_samples, n_symptoms)
        y   : integer label array
        mlb : fitted MultiLabelBinarizer (symptom columns)
        le  : fitted LabelEncoder (disease classes)
    """
    symptom_lists = [
        extract_symptoms(row['Symptoms'], row['Disease'])
        for _, row in df.iterrows()
    ]

    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(symptom_lists)

    le = LabelEncoder()
    y = le.fit_transform(df['Disease'])

    return X, y, mlb, le


def preprocess(data_path='dataset.xlsx'):
    df = load_dataset(data_path)
    X, y, mlb, le = build_feature_matrix(df)
    return X, y, mlb, le, df
