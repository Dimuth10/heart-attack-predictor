from flask import Flask, render_template, request, redirect, url_for, flash, session, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import joblib
import pandas as pd
import numpy as np
import json
import io
from functools import wraps
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer

# ── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'cardiopredict_secret_2025'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cardiopredict.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# ── Mail Config ──────────────────────────────────────────────────────────────
app.config['MAIL_SERVER']   = 'smtp.gmail.com'
app.config['MAIL_PORT']     = 587
app.config['MAIL_USE_TLS']  = True
app.config['MAIL_USERNAME'] = 'dimuth.thakshila.2003@gmail.com'
app.config['MAIL_PASSWORD'] = 'usaq nobt xadd hikh'
app.config['MAIL_DEFAULT_SENDER'] = ('CardioPredict AI', 'dimuth.thakshila.2003@gmail.com')

db      = SQLAlchemy(app)
bcrypt  = Bcrypt(app)
login_manager = LoginManager(app)
mail       = Mail(app)
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
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
    is_admin    = db.Column(db.Boolean, default=False)
    gender      = db.Column(db.String(10), nullable=True)
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

# ── Helper: Standardize data ─────────────────────────────────────────────────
def standardize_cp(val):
    mapping = {
        'typical angina': 1, 'typical': 1, '1': 1,
        'atypical angina': 2, 'atypical': 2, '2': 2,
        'non-anginal': 3, 'non-anginal pain': 3, '3': 3,
        'asymptomatic': 0, '0': 0, '4': 0,
    }
    return mapping.get(str(val).strip().lower(), 0)

def std_sex(val):
    return 1 if str(val).strip().lower() in ['1', 'male', 'm'] else 0

def std_fbs(val):
    return 1 if str(val).strip().lower() in ['1', 'true', 'yes'] else 0

# ── Admin required decorator ─────────────────────────────────────────────────
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('Admin access required!', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function


# ── Global Template Variables ─────────────────────────────────────────────────
from config import MODEL_STATS

@app.context_processor
def inject_model_stats():
    return dict(MODEL_STATS=MODEL_STATS)

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

        gender    = request.form.get('gender', '')
        is_admin  = User.query.count() == 0
        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(full_name=full_name, email=email, age=age,
                    password=hashed_pw, is_admin=is_admin, gender=gender)
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
            cp_val  = int(request.form['cp'])
            sex_val = int(request.form['sex'])
            fbs_val = int(request.form['fbs'])


            user_input = {
                'age':              float(request.form['age']),
                'sex':              sex_val,
                'chest pain type':  cp_val,
                'resting bp s':     float(request.form['trestbps']),
                'cholesterol':      float(request.form['chol']),
                'fasting blood sugar': fbs_val,
                'resting ecg':      int(request.form.get('resting_ecg', 0)),
                'max heart rate':   float(request.form['thalach']),
                'exercise angina':  int(request.form.get('exercise_angina', 0)),
                'oldpeak':          float(request.form.get('oldpeak', 0.0)),
                'ST slope':         int(request.form.get('st_slope', 1)),
            }
            # keep short-key aliases for factors section below
            user_input['cp']       = user_input['chest pain type']
            user_input['trestbps'] = user_input['resting bp s']
            user_input['chol']     = user_input['cholesterol']
            user_input['fbs']      = user_input['fasting blood sugar']
            user_input['thalach']  = user_input['max heart rate']

            input_df = pd.DataFrame([user_input])
            input_df = input_df.reindex(columns=features, fill_value=0)

            probability     = model.predict_proba(input_df)[0][1]
            probability_pct = round(probability * 100, 2)
            risk_level      = "High Risk" if probability_pct >= 50 else "Low Risk"

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

            session['last_prediction'] = {
                'probability': probability_pct,
                'risk_level':  risk_level,
                'age':         int(request.form['age']),
                'sex':         'Male' if sex_val == 1 else 'Female',
                'cp':          CP_LABELS.get(request.form['cp'], 'Unknown'),
                'trestbps':    float(request.form['trestbps']),
                'chol':        float(request.form['chol']),
                'fbs':         'Yes' if fbs_val == 1 else 'No',
                'thalach':     float(request.form['thalach']),
                'date':        datetime.now().strftime('%Y-%m-%d %H:%M'),
            }

            # ── Health Factors ────────────────────────────────────────────
            factors = []

            if user_input['chol'] >= 240:
                factors.append({'name': 'Cholesterol', 'value': f"{int(user_input['chol'])} mg/dl", 'status': 'danger', 'msg': f"Your cholesterol is {int(user_input['chol'])} mg/dl which is above 240 mg/dl — classified as high. High cholesterol causes fatty deposits (plaque) to build up inside your arteries, narrowing them and reducing blood flow to the heart. This significantly increases your risk of heart attack and stroke. Consider reducing saturated fats, red meat, and processed foods in your diet."})
            elif user_input['chol'] >= 200:
                factors.append({'name': 'Cholesterol', 'value': f"{int(user_input['chol'])} mg/dl", 'status': 'warning', 'msg': f"Your cholesterol is {int(user_input['chol'])} mg/dl which is borderline (200–239 mg/dl). While not yet high, this level can gradually contribute to plaque buildup in your arteries. Increasing physical activity, reducing fried and fatty foods, and eating more fibre-rich foods like oats and vegetables can help bring this down to the healthy range below 200 mg/dl."})
            else:
                factors.append({'name': 'Cholesterol', 'value': f"{int(user_input['chol'])} mg/dl", 'status': 'success', 'msg': f"Your cholesterol is {int(user_input['chol'])} mg/dl which is within the healthy range (below 200 mg/dl). This means your arteries are less likely to have dangerous plaque buildup. Continue maintaining a balanced diet, regular exercise, and routine check-ups to keep it at this level."})

            if user_input['trestbps'] >= 140:
                factors.append({'name': 'Blood Pressure', 'value': f"{int(user_input['trestbps'])} mmHg", 'status': 'danger', 'msg': f"Your resting blood pressure is {int(user_input['trestbps'])} mmHg which is above 140 mmHg — this is hypertension (high blood pressure). High blood pressure forces your heart to work harder than normal, damaging artery walls over time and dramatically increasing your risk of heart attack and stroke. Reducing salt intake, avoiding alcohol, managing stress, and regular exercise can help lower it. Please consult a doctor."})
            elif user_input['trestbps'] >= 120:
                factors.append({'name': 'Blood Pressure', 'value': f"{int(user_input['trestbps'])} mmHg", 'status': 'warning', 'msg': f"Your resting blood pressure is {int(user_input['trestbps'])} mmHg which is elevated (120–139 mmHg). While not yet hypertension, this level puts extra strain on your heart and blood vessels over time. Lifestyle changes such as reducing sodium, increasing potassium-rich foods like bananas, regular aerobic exercise, and stress management can help bring it back to the healthy range below 120 mmHg."})
            else:
                factors.append({'name': 'Blood Pressure', 'value': f"{int(user_input['trestbps'])} mmHg", 'status': 'success', 'msg': f"Your resting blood pressure is {int(user_input['trestbps'])} mmHg which is within the normal healthy range (below 120 mmHg). This means your heart is not under excessive strain from blood pressure. Continue staying active, maintaining a low-sodium diet, and getting regular check-ups."})

            expected_max = 220 - user_input['age']
            if user_input['thalach'] < expected_max * 0.5:
                factors.append({'name': 'Maximum Heart Rate', 'value': f"{int(user_input['thalach'])} bpm", 'status': 'danger', 'msg': f"Your maximum heart rate is {int(user_input['thalach'])} bpm which is very low compared to the expected {int(expected_max * 0.85)} bpm for your age. A very low maximum heart rate can indicate that your heart muscle is weakened or your arteries are narrowed, preventing your heart from pumping efficiently during activity. This is a significant warning sign. Please consult a cardiologist for further evaluation such as a stress test."})
            elif user_input['thalach'] < expected_max * 0.7:
                factors.append({'name': 'Maximum Heart Rate', 'value': f"{int(user_input['thalach'])} bpm", 'status': 'warning', 'msg': f"Your maximum heart rate is {int(user_input['thalach'])} bpm which is below the expected {int(expected_max * 0.85)} bpm for your age. A lower than expected heart rate during activity may suggest reduced cardiovascular fitness or early signs of reduced heart efficiency. Regular aerobic exercise such as brisk walking, swimming, or cycling can gradually improve your heart rate capacity."})
            else:
                factors.append({'name': 'Maximum Heart Rate', 'value': f"{int(user_input['thalach'])} bpm", 'status': 'success', 'msg': f"Your maximum heart rate is {int(user_input['thalach'])} bpm which is within the expected range for your age (target: {int(expected_max * 0.85)} bpm). This suggests your heart is pumping efficiently during physical activity. Maintaining regular exercise and a healthy lifestyle will help preserve this good cardiovascular fitness."})

            if user_input['fbs'] == 1:
                factors.append({'name': 'Fasting Blood Sugar', 'value': 'Above 120 mg/dl', 'status': 'warning', 'msg': 'Your fasting blood sugar is above 120 mg/dl which is elevated. High blood sugar over time damages blood vessel walls and nerves that control the heart. People with diabetes or pre-diabetes are 2–4 times more likely to develop heart disease. Reducing sugary foods, refined carbohydrates, and staying physically active can help manage blood sugar levels. A doctor check-up is recommended.'})
            else:
                factors.append({'name': 'Fasting Blood Sugar', 'value': 'Below 120 mg/dl', 'status': 'success', 'msg': 'Your fasting blood sugar is below 120 mg/dl which is within the normal healthy range. Normal blood sugar means your body is processing glucose effectively, which reduces the risk of damage to your blood vessels and heart. Maintain this by limiting sugary drinks, processed foods, and staying physically active.'})

            # ── Sex Factor ───────────────────────────────────────────────────────────
            if user_input['sex'] == 1:
                factors.append({'name': 'Sex', 'value': 'Male', 'status': 'warning',
                    'msg': 'Being male is a clinically recognised risk factor for heart disease. Men tend to develop heart disease 10–15 years earlier than women on average. Male hormones such as testosterone can contribute to higher LDL (bad) cholesterol and lower HDL (good) cholesterol levels compared to pre-menopausal women. This does not mean heart disease is inevitable — maintaining a healthy lifestyle, regular check-ups, and managing other risk factors can significantly reduce this risk.'})
            else:
                factors.append({'name': 'Sex', 'value': 'Female', 'status': 'success',
                    'msg': 'Being female is generally associated with a lower risk of heart disease before menopause, due to the protective effect of oestrogen which helps maintain healthy cholesterol levels and flexible blood vessels. However, after menopause this protective effect reduces and womens risk increases significantly. Women also tend to experience different heart attack symptoms such as fatigue, nausea, and jaw pain rather than classic chest pain, so awareness remains important.'})

            cp_analysis = {
                0: ('Asymptomatic', 'danger', 'You reported no chest pain at all. While this may seem reassuring, medically it can be a significant warning sign. When heart disease is severe, damaged nerves may stop sending pain signals — a condition known as silent ischemia. This means serious blockages can exist without any discomfort. This is especially common in people with diabetes or older adults. This is why the model assigns higher risk to this pattern.'),
                1: ('Typical Angina', 'warning', 'You experience chest tightness or pressure during physical activity or stress that goes away with rest. This is called typical angina and is caused by reduced blood flow to the heart during exertion. While this is a warning sign of narrowed arteries, it also means your heart is still communicating warning signals — which is medically better than silent ischemia. A cardiology evaluation including a stress test is recommended.'),
                2: ('Atypical Angina', 'warning', 'You experience unusual chest discomfort that does not follow the typical pattern of heart-related pain. Atypical angina may feel like burning, aching, or pressure that occurs at rest or in unexpected situations. While it does not always indicate heart disease, it can sometimes be an early or atypical presentation of reduced blood flow. Monitoring and a medical check-up are advisable.'),
                3: ('Non-Anginal', 'success', 'Your chest pain has been assessed as non-anginal, meaning it is not directly related to the heart. This type of pain is often caused by musculoskeletal issues, acid reflux, or anxiety. From a cardiac perspective this is a more reassuring pattern and the model assigns lower risk contribution to this finding. However always consult a doctor to confirm the cause of any chest pain.'),
            }
            cp_info = cp_analysis.get(user_input['cp'], ('Unknown', 'warning', ''))
            factors.append({'name': 'Chest Pain Type', 'value': cp_info[0], 'status': cp_info[1], 'msg': cp_info[2]})

            # ── SHAP Explainability ───────────────────────────────────────
            shap_data = []
            try:
                rf_model      = model
                input_imputed = input_df.reindex(columns=features, fill_value=0)

                explainer   = shap.TreeExplainer(rf_model)
                shap_values = explainer.shap_values(input_imputed)

                sv = np.array(shap_values)
                if len(sv.shape) == 3:
                    sv = sv[0, :, 1]
                elif len(sv.shape) == 2:
                    sv = sv[0]
                else:
                    sv = sv

                feature_labels = {
                    'age': 'Age', 'sex': 'Sex',
                    'cp': 'Chest Pain Type',
                    'trestbps': 'Blood Pressure',
                    'chol': 'Cholesterol',
                    'fbs': 'Blood Sugar',
                    'thalach': 'Heart Rate'
                }

                for fname, fval in zip(features, sv):
                    label = feature_labels.get(fname, fname)
                    shap_data.append({
                        'feature':  label,
                        'value':    round(float(fval), 4),
                        'positive': float(fval) > 0
                    })

                shap_data.sort(key=lambda x: abs(x['value']), reverse=True)
                shap_data = shap_data[:7]

            except Exception as shap_err:
                print(f"SHAP Error: {shap_err}")
                shap_data = []

            return render_template('result.html',
                                   probability=probability_pct,
                                   risk_level=risk_level,
                                   factors=factors,
                                   shap_data=shap_data)

        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
            return redirect(url_for('predict'))

    return render_template('predict.html')


# ── PDF Download Route ───────────────────────────────────────────────────────
@app.route('/download-report')
@login_required
def download_report():
    data = session.get('last_prediction')
    if not data:
        flash('No prediction found. Please run a prediction first.', 'warning')
        return redirect(url_for('predict'))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    elements = []

    title_style = ParagraphStyle('title',
        fontSize=24, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#9b2335'),
        alignment=TA_CENTER, spaceAfter=5)
    subtitle_style = ParagraphStyle('subtitle',
        fontSize=11, fontName='Helvetica',
        textColor=colors.HexColor('#888888'),
        alignment=TA_CENTER, spaceAfter=20)
    heading_style = ParagraphStyle('heading',
        fontSize=13, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#222222'),
        spaceBefore=15, spaceAfter=8)
    body_style = ParagraphStyle('body',
        fontSize=10, fontName='Helvetica',
        textColor=colors.HexColor('#444444'),
        spaceAfter=5, leading=16)
    small_style = ParagraphStyle('small',
        fontSize=8, fontName='Helvetica',
        textColor=colors.HexColor('#888888'),
        alignment=TA_CENTER, spaceAfter=5)

    elements.append(Paragraph('CardioPredict AI', title_style))
    elements.append(Paragraph('Heart Attack Risk Assessment Report', subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1,
                               color=colors.HexColor('#9b2335'), spaceAfter=20))

    elements.append(Paragraph('Patient Information', heading_style))
    patient_data = [
        ['Full Name',    current_user.full_name],
        ['Email',        current_user.email],
        ['Report Date',  data['date']],
    ]
    patient_table = Table(patient_data, colWidths=[5*cm, 12*cm])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR',  (0, 0), (0, -1), colors.HexColor('#555555')),
        ('FONTNAME',   (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME',   (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE',   (0, 0), (-1, -1), 10),
        ('PADDING',    (0, 0), (-1, -1), 8),
        ('GRID',       (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1),
         [colors.HexColor('#ffffff'), colors.HexColor('#f8f9fa')]),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 0.5*cm))

    elements.append(Paragraph('Clinical Input Parameters', heading_style))
    input_data = [
        ['Parameter',                 'Value',              'Unit'],
        ['Age',                       str(data['age']),     'Years'],
        ['Sex',                       data['sex'],          '—'],
        ['Chest Pain Type',           data['cp'],           '—'],
        ['Resting Blood Pressure',    str(data['trestbps']), 'mmHg'],
        ['Serum Cholesterol',         str(data['chol']),    'mg/dl'],
        ['Fasting Blood Sugar > 120', data['fbs'],          '—'],
        ['Maximum Heart Rate',        str(data['thalach']), 'bpm'],
    ]
    input_table = Table(input_data, colWidths=[7*cm, 5*cm, 5*cm])
    input_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9b2335')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME',   (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',   (0, 0), (-1, -1), 10),
        ('PADDING',    (0, 0), (-1, -1), 8),
        ('GRID',       (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.HexColor('#ffffff'), colors.HexColor('#f8f9fa')]),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ]))
    elements.append(input_table)
    elements.append(Spacer(1, 0.5*cm))

    elements.append(Paragraph('Prediction Result', heading_style))
    risk_color = '#9b2335' if data['risk_level'] == 'High Risk' else '#198754'
    result_data = [
        ['Risk Level',            data['risk_level']],
        ['Predicted Probability', f"{data['probability']}%"],
        ['Model Used',            'Random Forest Classifier'],
        ['Dataset',               'UCI Heart Disease + Statlog Cleveland Hungary'],
        ['Model Accuracy',        '89.14%'],
    ]
    result_table = Table(result_data, colWidths=[7*cm, 10*cm])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR',  (0, 0), (0, -1), colors.HexColor('#555555')),
        ('FONTNAME',   (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME',   (1, 0), (1, -1), 'Helvetica'),
        ('TEXTCOLOR',  (1, 0), (1, 0),  colors.HexColor(risk_color)),
        ('FONTNAME',   (1, 0), (1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 10),
        ('PADDING',    (0, 0), (-1, -1), 8),
        ('GRID',       (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1),
         [colors.HexColor('#ffffff'), colors.HexColor('#f8f9fa')]),
    ]))
    elements.append(result_table)
    elements.append(Spacer(1, 0.5*cm))

    elements.append(Paragraph('Recommended Next Steps', heading_style))
    if data['risk_level'] == 'High Risk':
        recommendations = [
            '• Consult a cardiologist or medical professional immediately',
            '• Consider tests such as ECG, blood profile, stress test or echocardiography',
            '• Monitor blood pressure and cholesterol levels regularly',
            '• Adopt lifestyle improvements: healthy diet, regular exercise, quit smoking',
            '• Reduce stress and maintain a healthy body weight',
        ]
    else:
        recommendations = [
            '• Maintain a healthy diet and regular physical activity',
            '• Schedule periodic check-ups for preventive heart screening',
            '• Monitor key indicators such as BP, cholesterol and glucose levels',
            '• Avoid smoking, excessive alcohol and high-stress lifestyle',
        ]
    for rec in recommendations:
        elements.append(Paragraph(rec, body_style))

    elements.append(Spacer(1, 0.5*cm))
    elements.append(HRFlowable(width="100%", thickness=0.5,
                               color=colors.HexColor('#dddddd'), spaceAfter=10))
    disclaimer = (
        'DISCLAIMER: This report is generated by a machine learning model for educational '
        'and research purposes only. It does not replace professional medical diagnosis or '
        'clinical advice. Always consult a qualified healthcare professional for medical decisions. '
        'CardioPredict AI — BSc (Hons) Software Engineering Final Year Project, '
        'NSBM Green University | University of Plymouth.'
    )
    elements.append(Paragraph(disclaimer, small_style))

    doc.build(elements)
    buffer.seek(0)

    filename = f"CardioPredict_Report_{current_user.full_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf"
    response = make_response(buffer.read())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename={filename}'
    return response


# ── Model Evaluation Route ───────────────────────────────────────────────────
@app.route('/evaluation')
@login_required
def evaluation():
    DATA_DIR = BASE_DIR / "data"

    df1 = pd.read_csv(DATA_DIR / "heart_disease_uci.csv")
    if 'num' in df1.columns:
        df1['target'] = (df1['num'] > 0).astype(int)
        df1.drop(columns=['num'], inplace=True)
    df1.drop(columns=[c for c in ['id', 'dataset'] if c in df1.columns], inplace=True)
    df1.rename(columns={'thalch': 'thalach'}, inplace=True)
    df1 = df1[['age','sex','cp','trestbps','chol','fbs','thalach','target']].copy()
    df1['sex'] = df1['sex'].map(std_sex)
    df1['fbs'] = df1['fbs'].map(std_fbs)
    df1['cp']  = df1['cp'].map(standardize_cp)
    df1.dropna(inplace=True)

    df2 = pd.read_csv(DATA_DIR / "heart_statlog_cleveland_hungary_final.csv")
    df2.rename(columns={
        'chest pain type': 'cp', 'resting bp s': 'trestbps',
        'cholesterol': 'chol', 'fasting blood sugar': 'fbs',
        'max heart rate': 'thalach'
    }, inplace=True)
    df2 = df2[['age','sex','cp','trestbps','chol','fbs','thalach','target']].copy()
    df2['sex'] = df2['sex'].map(std_sex)
    df2['fbs'] = df2['fbs'].map(std_fbs)
    df2['cp']  = df2['cp'].map(standardize_cp)
    df2.dropna(inplace=True)

    df = pd.concat([df1, df2], ignore_index=True)
    df.dropna(subset=['target'], inplace=True)
    df = df.astype({'cp': int, 'sex': int, 'fbs': int, 'target': int})

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    imputer    = SimpleImputer(strategy='median')
    X_test_imp = imputer.fit(X_train).transform(X_test)

    y_pred      = model.predict(X_test_imp)
    y_pred_prob = model.predict_proba(X_test_imp)[:, 1]

    acc       = round(accuracy_score(y_test, y_pred) * 100, 2)
    precision = round(precision_score(y_test, y_pred) * 100, 2)
    recall    = round(recall_score(y_test, y_pred) * 100, 2)
    f1        = round(f1_score(y_test, y_pred) * 100, 2)

    cm_vals = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm_vals.ravel()

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc     = round(auc(fpr, tpr), 4)
    roc_data    = {
        'fpr': [round(x, 4) for x in fpr.tolist()],
        'tpr': [round(x, 4) for x in tpr.tolist()],
        'auc': roc_auc
    }

    return render_template('evaluation.html',
                           acc=acc, precision=precision,
                           recall=recall, f1=f1,
                           tn=int(tn), fp=int(fp),
                           fn=int(fn), tp=int(tp),
                           roc_data=json.dumps(roc_data),
                           total_records=len(df),
                           test_records=len(X_test))


# ── Feature Importance Route ─────────────────────────────────────────────────
@app.route('/feature-importance')
@login_required
def feature_importance():
    try:
        rf_model    = model
        importances = rf_model.feature_importances_

        feature_labels = {
            'age':      'Age',
            'sex':      'Sex',
            'cp':       'Chest Pain Type',
            'trestbps': 'Blood Pressure',
            'chol':     'Cholesterol',
            'fbs':      'Fasting Blood Sugar',
            'thalach':  'Max Heart Rate'
        }

        feature_descriptions = {
            'age':      'Patient age in years. Older age is associated with higher cardiovascular risk.',
            'sex':      'Biological sex (Male/Female). Males tend to have higher heart disease prevalence.',
            'cp':       'Type of chest pain experienced. Asymptomatic type is a strong predictor.',
            'trestbps': 'Resting blood pressure in mmHg. High BP strains the heart over time.',
            'chol':     'Serum cholesterol in mg/dl. High levels lead to arterial plaque buildup.',
            'fbs':      'Fasting blood sugar above 120 mg/dl. Elevated levels indicate diabetes risk.',
            'thalach':  'Maximum heart rate achieved during exercise. Lower rates can signal risk.'
        }

        feature_ranges = {
            'age':      'Typical: 29 – 77 years',
            'sex':      '0 = Female, 1 = Male',
            'cp':       '0 = Asymptomatic, 1–3 = Various pain types',
            'trestbps': 'Normal: < 120 mmHg',
            'chol':     'Normal: < 200 mg/dl',
            'fbs':      '0 = Normal, 1 = Above 120 mg/dl',
            'thalach':  'Normal: 60 – 202 bpm'
        }

        importance_data = []
        total = sum(importances)
        for fname, imp in zip(features, importances):
            pct = round((imp / total) * 100, 2)
            importance_data.append({
                'feature':     feature_labels.get(fname, fname),
                'key':         fname,
                'importance':  round(float(imp), 4),
                'percentage':  pct,
                'description': feature_descriptions.get(fname, ''),
                'range':       feature_ranges.get(fname, ''),
            })

        importance_data.sort(key=lambda x: x['importance'], reverse=True)

        bar_colors = ['#9b2335', '#e05a2b', '#e07b20', '#c99a18', '#a0a818',
                      '#6aaa20', '#20aa6a']
        for i, item in enumerate(importance_data):
            item['color'] = bar_colors[i] if i < len(bar_colors) else '#888888'
            item['rank']  = i + 1

        return render_template('feature_importance.html',
                               importance_data=importance_data,
                               total_features=len(importance_data))

    except Exception as e:
        flash(f'Error loading feature importance: {str(e)}', 'danger')
        return redirect(url_for('dashboard'))


# ── Profile Route ────────────────────────────────────────────────────────────
@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    predictions = Prediction.query.filter_by(user_id=current_user.id).all()
    total     = len(predictions)
    high_risk = sum(1 for p in predictions if p.risk_level == 'High Risk')
    low_risk  = sum(1 for p in predictions if p.risk_level == 'Low Risk')

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'update_profile':
            current_user.full_name = request.form['full_name']
            current_user.age       = request.form['age']
            current_user.gender    = request.form.get('gender', current_user.gender)
            db.session.commit()
            flash('Profile updated successfully!', 'success')
            return redirect(url_for('profile'))

        elif action == 'change_password':
            current_pw  = request.form['current_password']
            new_pw      = request.form['new_password']
            confirm_pw  = request.form['confirm_password']

            if not bcrypt.check_password_hash(current_user.password, current_pw):
                flash('Current password is incorrect!', 'danger')
                return redirect(url_for('profile'))

            if new_pw != confirm_pw:
                flash('New passwords do not match!', 'danger')
                return redirect(url_for('profile'))

            if len(new_pw) < 6:
                flash('Password must be at least 6 characters!', 'danger')
                return redirect(url_for('profile'))

            current_user.password = bcrypt.generate_password_hash(new_pw).decode('utf-8')
            db.session.commit()
            flash('Password changed successfully!', 'success')
            return redirect(url_for('profile'))

    return render_template('profile.html',
                           total=total,
                           high_risk=high_risk,
                           low_risk=low_risk)


# ── Admin Dashboard Route ────────────────────────────────────────────────────
@app.route('/admin')
@login_required
@admin_required
def admin():
    all_users       = User.query.order_by(User.created_at.desc()).all()
    all_predictions = Prediction.query.order_by(Prediction.created_at.desc()).all()

    total_users       = len(all_users)
    total_predictions = len(all_predictions)
    total_high_risk   = sum(1 for p in all_predictions if p.risk_level == 'High Risk')
    total_low_risk    = sum(1 for p in all_predictions if p.risk_level == 'Low Risk')

    return render_template('admin.html',
                           all_users=all_users,
                           all_predictions=all_predictions,
                           total_users=total_users,
                           total_predictions=total_predictions,
                           total_high_risk=total_high_risk,
                           total_low_risk=total_low_risk)


# ── History Route ────────────────────────────────────────────────────────────
@app.route('/history')
@login_required
def history():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.asc()).all()
    return render_template('history.html', predictions=predictions)


# ── Forgot Password ─────────────────────────────────────────────────────────
@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form['email']
        user  = User.query.filter_by(email=email).first()
        if user:
            token = serializer.dumps(email, salt='password-reset')
            reset_url = url_for('reset_password', token=token, _external=True)
            try:
                msg = Message('Reset Your CardioPredict AI Password', recipients=[email])
                msg.html = f"""
                <!DOCTYPE html>
                <html>
                <head><meta charset="UTF-8"></head>
                <body style="margin:0;padding:0;background:#020818;font-family:'Segoe UI',Arial,sans-serif;">
                  <div style="max-width:580px;margin:40px auto;background:#0d1117;border:1px solid rgba(220,53,69,0.2);border-radius:16px;overflow:hidden;">
                    <!-- Header -->
                    <div style="background:linear-gradient(135deg,rgba(220,53,69,0.2),rgba(139,0,0,0.1));padding:36px 40px;text-align:center;border-bottom:1px solid rgba(220,53,69,0.15);">
                      <div style="font-size:2rem;margin-bottom:8px;">❤️</div>
                      <h1 style="color:white;font-size:1.6rem;font-weight:800;margin:0;">CardioPredict AI</h1>
                      <p style="color:#888;margin:6px 0 0;font-size:0.9rem;">Heart Attack Risk Prediction System</p>
                    </div>
                    <!-- Body -->
                    <div style="padding:40px;">
                      <h2 style="color:white;font-size:1.3rem;font-weight:700;margin:0 0 12px;">Password Reset Request</h2>
                      <p style="color:#888;font-size:0.95rem;line-height:1.7;margin:0 0 24px;">
                        Hi <strong style="color:#e0e0e0;">{user.full_name}</strong>,<br><br>
                        We received a request to reset your CardioPredict AI password.
                        Click the button below to create a new password.
                        This link will expire in <strong style="color:#9b2335;">30 minutes</strong>.
                      </p>
                      <!-- Button -->
                      <div style="text-align:center;margin:32px 0;">
                        <a href="{reset_url}"
                           style="display:inline-block;background:linear-gradient(135deg,#9b2335,#b02a37);
                                  color:white;text-decoration:none;padding:14px 36px;
                                  border-radius:8px;font-weight:700;font-size:1rem;
                                  box-shadow:0 4px 20px rgba(220,53,69,0.4);">
                          Reset My Password
                        </a>
                      </div>
                      <p style="color:#555;font-size:0.82rem;line-height:1.6;margin:0;">
                        If the button doesn't work, copy and paste this link into your browser:<br>
                        <a href="{reset_url}" style="color:#9b2335;word-break:break-all;">{reset_url}</a>
                      </p>
                      <hr style="border:none;border-top:1px solid rgba(255,255,255,0.06);margin:28px 0;">
                      <p style="color:#444;font-size:0.8rem;margin:0;">
                        If you didn't request a password reset, you can safely ignore this email.
                        Your password will not be changed.
                      </p>
                    </div>
                    <!-- Footer -->
                    <div style="background:rgba(255,255,255,0.02);padding:20px 40px;text-align:center;border-top:1px solid rgba(255,255,255,0.04);">
                      <p style="color:#333;font-size:0.78rem;margin:0;">
                        © 2025 CardioPredict AI — BSc (Hons) Software Engineering Final Year Project<br>
                        NSBM Green University | University of Plymouth
                      </p>
                    </div>
                  </div>
                </body>
                </html>
                """
                mail.send(msg)
                flash('Password reset email sent! Check your inbox.', 'success')
            except Exception as e:
                flash(f'Email could not be sent. Error: {str(e)}', 'danger')
        else:
            flash('If that email exists in our system, a reset link has been sent.', 'info')
        return redirect(url_for('forgot_password'))
    return render_template('forgot_password.html')


@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    try:
        email = serializer.loads(token, salt='password-reset', max_age=1800)
    except Exception:
        flash('The reset link is invalid or has expired. Please request a new one.', 'danger')
        return redirect(url_for('forgot_password'))

    user = User.query.filter_by(email=email).first()
    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        new_pw     = request.form['new_password']
        confirm_pw = request.form['confirm_password']

        if new_pw != confirm_pw:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('reset_password', token=token))
        if len(new_pw) < 6:
            flash('Password must be at least 6 characters!', 'danger')
            return redirect(url_for('reset_password', token=token))

        user.password = bcrypt.generate_password_hash(new_pw).decode('utf-8')
        db.session.commit()
        flash('Password reset successfully! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('reset_password.html', token=token)



# ── Export History CSV ───────────────────────────────────────────────────────
@app.route('/export-history')
@login_required
def export_history():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.asc()).all()
    if not predictions:
        flash('No predictions to export.', 'warning')
        return redirect(url_for('history'))
    output = io.StringIO()
    output.write('No,Date,Time,Age,Sex,Chest Pain Type,Blood Pressure (mmHg),Cholesterol (mg/dl),Fasting Blood Sugar,Max Heart Rate (bpm),Probability (%),Risk Level\n')
    for i, p in enumerate(predictions, 1):
        d = p.created_at.strftime('%d/%m/%Y')
        t = p.created_at.strftime('%H:%M')
        row = ','.join([str(i), d, t, str(p.age), str(p.sex), str(p.cp), str(p.trestbps), str(p.chol), str(p.fbs), str(p.thalach), str(p.probability), p.risk_level])
        output.write(row + '\n')
    output.seek(0)
    n = current_user.full_name.replace(' ', '_')
    dt = datetime.now().strftime('%Y%m%d')
    filename = 'HeartAttackPredict_History_' + n + '_' + dt + '.csv'
    response = make_response(output.getvalue())
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = 'attachment; filename=' + filename
    return response

# ── Init DB & Run ────────────────────────────────────────────────────────────
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)