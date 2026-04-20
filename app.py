from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd

# ── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'cardiopredict_secret_2025'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cardiopredict.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db      = SQLAlchemy(app)
bcrypt  = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "heart_attack_model.pkl"
COLS_PATH  = BASE_DIR / "model" / "feature_columns.pkl"

model    = joblib.load(MODEL_PATH)
features = joblib.load(COLS_PATH)

# ── Database Models ──────────────────────────────────────────────────────────
class User(db.Model, UserMixin):
    id          = db.Column(db.Integer, primary_key=True)
    full_name   = db.Column(db.String(100), nullable=False)
    email       = db.Column(db.String(120), unique=True, nullable=False)
    age         = db.Column(db.Integer, nullable=False)
    password    = db.Column(db.String(200), nullable=False)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    probability = db.Column(db.Float, nullable=False)
    risk_level  = db.Column(db.String(20), nullable=False)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)
    age         = db.Column(db.Integer)
    sex         = db.Column(db.String(10))
    cp          = db.Column(db.String(30))
    trestbps    = db.Column(db.Float)
    chol        = db.Column(db.Float)
    fbs         = db.Column(db.String(10))
    thalach     = db.Column(db.Float)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ── CP label for display ─────────────────────────────────────────────────────
CP_LABELS = {
    '0': 'Asymptomatic',
    '1': 'Typical Angina',
    '2': 'Atypical Angina',
    '3': 'Non-Anginal',
}

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        full_name = request.form['full_name']
        email     = request.form['email']
        age       = request.form['age']
        password  = request.form['password']
        confirm   = request.form['confirm_password']

        if password != confirm:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered!', 'danger')
            return redirect(url_for('register'))

        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(full_name=full_name, email=email, age=age, password=hashed_pw)
        db.session.add(user)
        db.session.commit()
        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email    = request.form['email']
        password = request.form['password']
        user     = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Welcome back, ' + user.full_name + '!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password!', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).all()
    total     = len(predictions)
    high_risk = sum(1 for p in predictions if p.risk_level == 'High Risk')
    low_risk  = sum(1 for p in predictions if p.risk_level == 'Low Risk')
    latest    = predictions[0] if predictions else None
    return render_template('dashboard.html',
                           predictions=predictions,
                           total=total,
                           high_risk=high_risk,
                           low_risk=low_risk,
                           latest=latest)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            # cp comes in as numeric string '0','1','2','3' from the form
            cp_val  = int(request.form['cp'])
            sex_val = int(request.form['sex'])
            fbs_val = int(request.form['fbs'])

            user_input = {
                'age':      float(request.form['age']),
                'sex':      sex_val,
                'cp':       cp_val,
                'trestbps': float(request.form['trestbps']),
                'chol':     float(request.form['chol']),
                'fbs':      fbs_val,
                'thalach':  float(request.form['thalach']),
            }

            input_df = pd.DataFrame([user_input])
            input_df = input_df.reindex(columns=features, fill_value=0)

            probability     = model.predict_proba(input_df)[0][1]
            probability_pct = round(probability * 100, 2)
            risk_level      = "High Risk" if probability_pct >= 50 else "Low Risk"

            # Save to database
            pred = Prediction(
                user_id     = current_user.id,
                probability = probability_pct,
                risk_level  = risk_level,
                age         = int(request.form['age']),
                sex         = 'Male' if sex_val == 1 else 'Female',
                cp          = CP_LABELS.get(request.form['cp'], 'Unknown'),
                trestbps    = float(request.form['trestbps']),
                chol        = float(request.form['chol']),
                fbs         = 'Yes' if fbs_val == 1 else 'No',
                thalach     = float(request.form['thalach']),
            )
            db.session.add(pred)
            db.session.commit()

            return render_template('result.html',
                                   probability=probability_pct,
                                   risk_level=risk_level)
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
            return redirect(url_for('predict'))

    return render_template('predict.html')

@app.route('/history')
@login_required
def history():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.asc()).all()
    return render_template('history.html', predictions=predictions)

# ── Init DB & Run ────────────────────────────────────────────────────────────
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)