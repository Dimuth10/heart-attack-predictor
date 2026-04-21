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
import joblib
import pandas as pd
import numpy as np
import json
import io

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

# ── Helper: Standardize data for evaluation ──────────────────────────────────
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

            factors = []

            if user_input['chol'] >= 240:
                factors.append({'name': 'Cholesterol', 'value': f"{int(user_input['chol'])} mg/dl", 'status': 'danger', 'msg': 'High — above 240 mg/dl is considered high risk'})
            elif user_input['chol'] >= 200:
                factors.append({'name': 'Cholesterol', 'value': f"{int(user_input['chol'])} mg/dl", 'status': 'warning', 'msg': 'Borderline — healthy range is below 200 mg/dl'})
            else:
                factors.append({'name': 'Cholesterol', 'value': f"{int(user_input['chol'])} mg/dl", 'status': 'success', 'msg': 'Normal — within healthy range'})

            if user_input['trestbps'] >= 140:
                factors.append({'name': 'Blood Pressure', 'value': f"{int(user_input['trestbps'])} mmHg", 'status': 'danger', 'msg': 'High — above 140 mmHg indicates hypertension'})
            elif user_input['trestbps'] >= 120:
                factors.append({'name': 'Blood Pressure', 'value': f"{int(user_input['trestbps'])} mmHg", 'status': 'warning', 'msg': 'Elevated — healthy range is below 120 mmHg'})
            else:
                factors.append({'name': 'Blood Pressure', 'value': f"{int(user_input['trestbps'])} mmHg", 'status': 'success', 'msg': 'Normal — within healthy range'})

            expected_max = 220 - user_input['age']
            if user_input['thalach'] < expected_max * 0.5:
                factors.append({'name': 'Maximum Heart Rate', 'value': f"{int(user_input['thalach'])} bpm", 'status': 'danger', 'msg': f"Very low — expected around {int(expected_max * 0.85)} bpm for your age"})
            elif user_input['thalach'] < expected_max * 0.7:
                factors.append({'name': 'Maximum Heart Rate', 'value': f"{int(user_input['thalach'])} bpm", 'status': 'warning', 'msg': f"Below average — expected around {int(expected_max * 0.85)} bpm for your age"})
            else:
                factors.append({'name': 'Maximum Heart Rate', 'value': f"{int(user_input['thalach'])} bpm", 'status': 'success', 'msg': 'Normal — within expected range for your age'})

            if user_input['fbs'] == 1:
                factors.append({'name': 'Fasting Blood Sugar', 'value': 'Above 120 mg/dl', 'status': 'warning', 'msg': 'Elevated — may indicate diabetes risk which affects heart health'})
            else:
                factors.append({'name': 'Fasting Blood Sugar', 'value': 'Below 120 mg/dl', 'status': 'success', 'msg': 'Normal — within healthy range'})

            cp_analysis = {
                0: ('Asymptomatic',  'danger',  'No chest pain felt — often associated with higher cardiac risk'),
                1: ('Typical Angina',  'warning', 'Chest pain triggered by activity — may indicate reduced blood flow'),
                2: ('Atypical Angina', 'warning', 'Unusual chest discomfort — worth monitoring'),
                3: ('Non-Anginal',     'success', 'Chest pain unrelated to heart — lower cardiac concern'),
            }
            cp_info = cp_analysis.get(user_input['cp'], ('Unknown', 'warning', ''))
            factors.append({'name': 'Chest Pain Type', 'value': cp_info[0], 'status': cp_info[1], 'msg': cp_info[2]})

            return render_template('result.html',
                                   probability=probability_pct,
                                   risk_level=risk_level,
                                   factors=factors)

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
        textColor=colors.HexColor('#dc3545'),
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
                               color=colors.HexColor('#dc3545'), spaceAfter=20))

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
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc3545')),
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
    risk_color = '#dc3545' if data['risk_level'] == 'High Risk' else '#198754'
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


# ── History Route ────────────────────────────────────────────────────────────
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