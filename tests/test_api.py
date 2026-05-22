import pytest


# ── System ─────────────────────────────────────────────────────────────────────

def test_health_ok(client):
    r = client.get('/api/system/health')
    assert r.status_code == 200
    data = r.get_json()
    assert data['status'] == 'ok'
    assert data['model_loaded'] is True
    assert 'symptom_matcher_backend' in data


# ── Reference — symptoms ───────────────────────────────────────────────────────

def test_list_symptoms_returns_list(client):
    r = client.get('/api/reference/symptoms')
    assert r.status_code == 200
    data = r.get_json()
    assert isinstance(data['symptoms'], list)
    assert data['count'] > 100


def test_list_symptoms_filter(client):
    r = client.get('/api/reference/symptoms?q=fever')
    data = r.get_json()
    assert all('fever' in s for s in data['symptoms'])
    assert data['count'] == len(data['symptoms'])


def test_list_symptoms_filter_no_results(client):
    r = client.get('/api/reference/symptoms?q=zzznomatch')
    data = r.get_json()
    assert data['count'] == 0
    assert data['symptoms'] == []


# ── Reference — diseases ───────────────────────────────────────────────────────

def test_list_diseases(client):
    r = client.get('/api/reference/diseases')
    assert r.status_code == 200
    data = r.get_json()
    assert data['count'] >= 70


def test_list_diseases_filter(client):
    r = client.get('/api/reference/diseases?q=hepatitis')
    data = r.get_json()
    assert data['count'] > 0
    assert all('hepatitis' in d.lower() for d in data['diseases'])


def test_disease_symptoms_diabetes(client):
    r = client.get('/api/reference/diseases/Diabetes/symptoms')
    assert r.status_code == 200
    data = r.get_json()
    assert data['count'] > 0
    assert isinstance(data['symptoms'], list)


def test_disease_symptoms_not_found(client):
    r = client.get('/api/reference/diseases/xyznonexistent123abc/symptoms')
    assert r.status_code == 404


# ── Prediction ─────────────────────────────────────────────────────────────────

def test_predict_valid_request(client):
    r = client.post('/api/predict/', json={'symptoms': ['skin rash', 'itching', 'fever']})
    assert r.status_code == 200
    data = r.get_json()
    assert isinstance(data['disease'], str)
    assert 0.0 <= data['confidence'] <= 1.0
    assert 'disclaimer' in data


def test_predict_returns_explanation(client):
    r = client.post('/api/predict/', json={'symptoms': ['skin rash', 'itching']})
    assert r.status_code == 200
    data = r.get_json()
    assert 'explanation' in data
    assert 'top_contributors' in data['explanation']
    contributors = data['explanation']['top_contributors']
    assert len(contributors) > 0
    assert 'symptom' in contributors[0]
    assert 'shap_value' in contributors[0]
    assert contributors[0]['direction'] in ('supports', 'against')


def test_predict_missing_body(client):
    r = client.post('/api/predict/', json={})
    assert r.status_code == 400


def test_predict_empty_symptoms(client):
    r = client.post('/api/predict/', json={'symptoms': []})
    assert r.status_code == 400


def test_predict_fungal_infection(client):
    r = client.post('/api/predict/',
                    json={'symptoms': ['skin rash', 'itching', 'nodal skin eruptions']})
    data = r.get_json()
    assert data['disease'] == 'Fungal infection'


def test_predict_symptoms_matched_present(client):
    r = client.post('/api/predict/', json={'symptoms': ['fever', 'cough']})
    data = r.get_json()
    assert isinstance(data['symptoms_matched'], list)
    assert len(data['symptoms_matched']) == 2


# ── Symptom matching ───────────────────────────────────────────────────────────

def test_match_symptom_basic(client):
    r = client.post('/api/predict/match-symptom', json={'text': 'joint pian'})
    assert r.status_code == 200
    data = r.get_json()
    assert 'matches' in data
    assert len(data['matches']) > 0


def test_match_symptom_top_k(client):
    r = client.post('/api/predict/match-symptom', json={'text': 'fever', 'top_k': 5})
    data = r.get_json()
    assert len(data['matches']) == 5


def test_match_symptom_contains_exact(client):
    r = client.post('/api/predict/match-symptom', json={'text': 'fever', 'top_k': 3})
    data = r.get_json()
    assert 'fever' in data['matches']


def test_match_symptom_missing_text(client):
    r = client.post('/api/predict/match-symptom', json={})
    assert r.status_code == 400
