import os
import random
import re
import secrets

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

import msgConstant as msgCons
from src.ml_engine import (
    get_disease_info,
    get_disease_symptoms,
    matcher,
    predict_disease,
)

load_dotenv()

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///database.db')
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

db = SQLAlchemy(app)

# ── REST API + Swagger UI (/api/docs) ──────────────────────────────────────────
from api.routes import api            # noqa: E402 (after app init)
api.init_app(app)

print(f"Symptom matcher backend: {matcher.backend_name}")


# ── DB model ───────────────────────────────────────────────────────────────────

def make_token():
    return secrets.token_urlsafe(16)


class user(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    email = db.Column(db.String(120))
    password = db.Column(db.String(256))


# ── Page routes ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/user')
def index_auth():
    my_id = make_token()
    session[f'chat_{my_id}'] = {'state': -1, 'name': '', 'age': 0, 'gender': '', 'symptoms': []}
    return render_template('index_auth.html', sessionId=my_id)


@app.route('/instruct')
def instruct():
    return render_template('instructions.html')


@app.route('/upload')
def bmi():
    return render_template('bmi.html')


@app.route('/diseases')
def diseases():
    return render_template('diseases.html')


@app.route('/pred_page')
def pred_page():
    pred = session.get('pred_label', None)
    f_name = session.get('filename', None)
    return render_template('pred.html', pred=pred, f_name=f_name)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form['uname']
        passw = request.form['passw']
        existing_user = user.query.filter_by(username=uname).first()
        if existing_user and check_password_hash(existing_user.password, passw):
            return redirect(url_for('index_auth'))
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        uname = request.form['uname']
        mail = request.form['mail']
        passw = request.form['passw']
        hashed_pw = generate_password_hash(passw)
        new_user = user(username=uname, email=mail, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')


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
    user_message = request.args.get('message', '').lower()
    session_id = request.args.get('sessionId', '')
    chat_key = f'chat_{session_id}'

    response = []

    if not user_message or user_message == 'undefined':
        rand_num = random.randint(0, 4)
        response.append(msgCons.WELCOME_GREET[rand_num])
        response.append('What is your good name?')
        return jsonify({'status': 'OK', 'answer': response})

    chat = dict(session.get(chat_key, _DEFAULT_CHAT.copy()))
    if 'symptoms' not in chat:
        chat['symptoms'] = []
    current_state = chat['state']

    if current_state == -1:
        response.append(
            f"Hi {user_message}, to predict your disease based on symptoms, "
            "we need some information about you. Please provide accordingly."
        )
        chat['name'] = user_message
        chat['state'] = 0

    elif current_state == 0:
        response.append(f"{chat['name']}, what is your age?")
        chat['state'] = 1

    elif current_state == 1:
        result = re.findall(r'\d+', user_message)
        if not result or not (0 < float(result[0]) < 130):
            response.append('Invalid input, please provide a valid age.')
        else:
            chat['age'] = float(result[0])
            response.append(f"{chat['name']}, choose an option:")
            response.append('1. Predict Disease')
            response.append('2. Check Disease Symptoms')
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
        chat['symptoms'].extend([s.strip() for s in user_message.split(',')])
        response.append(f"{chat['name']}, what kind of symptoms are you currently experiencing?")
        response.append('1. Check Disease')
        response.append('<a href="/diseases" target="_blank">Symptoms List</a>')
        chat['state'] = 4

    elif current_state in range(4, 9):
        if '1' in user_message or 'disease' in user_message:
            result = predict_disease(chat['symptoms'])
            disease = result['disease']
            info = get_disease_info(disease)
            response.append('<b>The following disease may be causing your discomfort</b>')
            response.append(f"<b>{disease}</b><br>{info}")
            if disease:
                response.append(
                    f'<a href="https://www.google.com/search?q={disease} disease hospital near me"'
                    ' target="_blank">Search Nearby Hospitals</a>'
                )
            chat['state'] = 10
        else:
            chat['symptoms'].extend([s.strip() for s in user_message.split(',')])
            next_state = min(current_state + 1, 8)
            question = _FOLLOW_UP_QUESTIONS.get(next_state, _FOLLOW_UP_QUESTIONS[8])
            response.append(f"{chat['name']}, {question}")
            response.append('1. Check Disease')
            response.append('<a href="/diseases" target="_blank">Symptoms List</a>')
            chat['state'] = next_state

    elif current_state == 10:
        response.append('<a href="/user" target="_blank">Predict Again</a>')

    elif current_state == 20:
        found, data = get_disease_symptoms(user_message)
        if found:
            response.append(f"The symptoms of {user_message} are:")
            for sym in data:
                response.append(sym.capitalize())
        else:
            response.append(data)
        chat['state'] = 2
        response.extend(['', 'Choose an option:', '1. Predict Disease', '2. Check Disease Symptoms'])

    session[chat_key] = chat
    session.modified = True
    return jsonify({'status': 'OK', 'answer': response})


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=False, port=3000)
