import os
import random
import re
import secrets

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

import msgConstant as msgCons
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data.preprocess import get_all_symptoms, preprocess
from src.data.symptom_matcher import SemanticSymptomMatcher

load_dotenv()

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///database.db')
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

db = SQLAlchemy(app)


def make_token():
    return secrets.token_urlsafe(16)


class user(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    email = db.Column(db.String(120))
    password = db.Column(db.String(256))


# ── ML artefacts (loaded once at startup) ─────────────────────────────────────

df = pd.read_excel('dataset.xlsx')

_MODEL_DIR = 'model'
_clf = _mlb = _le = None

if all(os.path.exists(os.path.join(_MODEL_DIR, f))
       for f in ('best_model.joblib', 'symptom_encoder.joblib', 'disease_encoder.joblib')):
    _clf = joblib.load(os.path.join(_MODEL_DIR, 'best_model.joblib'))
    _mlb = joblib.load(os.path.join(_MODEL_DIR, 'symptom_encoder.joblib'))
    _le  = joblib.load(os.path.join(_MODEL_DIR, 'disease_encoder.joblib'))

_all_symptoms = get_all_symptoms(df)
_matcher = SemanticSymptomMatcher(_all_symptoms)
print(f"Symptom matcher backend: {_matcher.backend_name}")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/user")
def index_auth():
    my_id = make_token()
    session[f'chat_{my_id}'] = {'state': -1, 'name': '', 'age': 0, 'gender': '', 'symptoms': []}
    return render_template("index_auth.html", sessionId=my_id)


@app.route("/instruct")
def instruct():
    return render_template("instructions.html")


@app.route("/upload")
def bmi():
    return render_template("bmi.html")


@app.route("/diseases")
def diseases():
    return render_template("diseases.html")


@app.route('/pred_page')
def pred_page():
    pred = session.get('pred_label', None)
    f_name = session.get('filename', None)
    return render_template('pred.html', pred=pred, f_name=f_name)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uname = request.form["uname"]
        passw = request.form["passw"]
        existing_user = user.query.filter_by(username=uname).first()
        if existing_user and check_password_hash(existing_user.password, passw):
            return redirect(url_for("index_auth"))
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        uname = request.form['uname']
        mail = request.form['mail']
        passw = request.form['passw']
        hashed_pw = generate_password_hash(passw)
        new_user = user(username=uname, email=mail, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for("login"))
    return render_template("register.html")


# ── ML helpers ─────────────────────────────────────────────────────────────────

def predict_disease_from_symptom(symptom_list: list[str]) -> tuple[str, str]:
    """
    Returns (html_response, disease_name).

    Uses the trained classifier when artefacts are present; falls back to
    cosine-similarity text matching on the raw dataset otherwise.
    """
    # Normalise user symptoms via semantic matcher
    matched = [_matcher.match(s.strip()) for s in symptom_list if s.strip()]

    if _clf is not None:
        X = _mlb.transform([matched])
        y_pred = _clf.predict(X)
        disease = _le.inverse_transform(y_pred)[0]
        details = getDiseaseInfo(disease)
        return f"<b>{disease}</b><br>{details}", disease

    # Fallback: cosine similarity on symptom strings
    vectorizer = CountVectorizer()
    X_corpus = vectorizer.fit_transform(df['Symptoms'])
    user_X = vectorizer.transform([', '.join(matched)])
    scores = cosine_similarity(X_corpus, user_X)
    max_score = scores.max()
    max_indices = scores.argmax(axis=0)
    diseases_found = {df.iloc[i]['Disease'] for i in max_indices if scores[i] == max_score}

    if not diseases_found:
        return "<b>No matching diseases found</b>", ""
    if len(diseases_found) == 1:
        disease = list(diseases_found)[0]
        details = getDiseaseInfo(disease)
        return f"<b>{disease}</b><br>{details}", disease
    return "The most likely diseases are<br><b>" + ', '.join(diseases_found) + "</b>", ""


def get_symtoms(user_disease: str) -> tuple[bool, set | str]:
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Disease'])
    user_X = vectorizer.transform([user_disease])
    scores = cosine_similarity(X, user_X)
    max_score = scores.max()
    if max_score < 0.7:
        return False, "No matching diseases found"
    max_indices = scores.argmax(axis=0)
    matched_symptoms = set()
    for i in max_indices:
        if scores[i] == max_score:
            matched_symptoms.update(df.iloc[i]['Symptoms'].split(','))
    return True, matched_symptoms


def getDiseaseInfo(keywords: str) -> str:
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
    return ""


# ── Chat state machine ─────────────────────────────────────────────────────────

_DEFAULT_CHAT = {'state': -1, 'name': '', 'age': 0, 'gender': '', 'symptoms': []}

_FOLLOW_UP_QUESTIONS = {
    4: "Could you describe the symptoms you're suffering from?",
    5: "What are the symptoms you're currently dealing with?",
    6: "What symptoms have you been experiencing lately?",
    7: "What are the symptoms that you're currently dealing with?",
    8: "What symptoms have you been experiencing lately?",
}


@app.route('/ask', methods=['GET', 'POST'])
def chat_msg():
    user_message = request.args.get("message", "").lower()
    session_id = request.args.get("sessionId", "")
    chat_key = f'chat_{session_id}'

    response = []

    if not user_message or user_message == "undefined":
        rand_num = random.randint(0, 4)
        response.append(msgCons.WELCOME_GREET[rand_num])
        response.append("What is your good name?")
        return jsonify({'status': 'OK', 'answer': response})

    chat = dict(session.get(chat_key, _DEFAULT_CHAT.copy()))
    if 'symptoms' not in chat:
        chat['symptoms'] = []
    current_state = chat['state']

    if current_state == -1:
        response.append(f"Hi {user_message}, to predict your disease based on symptoms, "
                        "we need some information about you. Please provide accordingly.")
        chat['name'] = user_message
        chat['state'] = 0

    elif current_state == 0:
        response.append(f"{chat['name']}, what is your age?")
        chat['state'] = 1

    elif current_state == 1:
        result = re.findall(r'\d+', user_message)
        if not result or not (0 < float(result[0]) < 130):
            response.append("Invalid input, please provide a valid age.")
        else:
            chat['age'] = float(result[0])
            response.append(f"{chat['name']}, choose an option:")
            response.append("1. Predict Disease")
            response.append("2. Check Disease Symptoms")
            chat['state'] = 2

    elif current_state == 2:
        if '2' in user_message or 'check' in user_message:
            response.append(f"{chat['name']}, what's the disease name?")
            chat['state'] = 20
        else:
            response.append(f"{chat['name']}, what symptoms are you experiencing?")
            response.append('<a href="/diseases" target="_blank">Symptoms List</a>')
            chat['state'] = 3

    elif current_state == 3:
        chat['symptoms'].extend([s.strip() for s in user_message.split(",")])
        response.append(f"{chat['name']}, what kind of symptoms are you currently experiencing?")
        response.append("1. Check Disease")
        response.append('<a href="/diseases" target="_blank">Symptoms List</a>')
        chat['state'] = 4

    elif current_state in range(4, 9):
        if '1' in user_message or 'disease' in user_message:
            disease, disease_type = predict_disease_from_symptom(chat['symptoms'])
            response.append("<b>The following disease may be causing your discomfort</b>")
            response.append(disease)
            if disease_type:
                response.append(
                    f'<a href="https://www.google.com/search?q={disease_type} disease hospital near me"'
                    ' target="_blank">Search Nearby Hospitals</a>'
                )
            chat['state'] = 10
        else:
            chat['symptoms'].extend([s.strip() for s in user_message.split(",")])
            next_state = min(current_state + 1, 8)
            question = _FOLLOW_UP_QUESTIONS.get(next_state, _FOLLOW_UP_QUESTIONS[8])
            response.append(f"{chat['name']}, {question}")
            response.append("1. Check Disease")
            response.append('<a href="/diseases" target="_blank">Symptoms List</a>')
            chat['state'] = next_state

    elif current_state == 10:
        response.append('<a href="/user" target="_blank">Predict Again</a>')

    elif current_state == 20:
        found, data = get_symtoms(user_message)
        if found:
            response.append(f"The symptoms of {user_message} are:")
            for sym in data:
                response.append(sym.strip().capitalize())
        else:
            response.append(data)
        chat['state'] = 2
        response.extend(["", "Choose an option:", "1. Predict Disease", "2. Check Disease Symptoms"])

    session[chat_key] = chat
    session.modified = True
    return jsonify({'status': 'OK', 'answer': response})


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=False, port=3000)
