import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, db, User, Prediction
from flask_bcrypt import Bcrypt

bcrypt = Bcrypt(app)

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['WTF_CSRF_ENABLED'] = False
    app.config['SECRET_KEY'] = 'test_secret'
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            yield client
            db.drop_all()

@pytest.fixture
def auth_client(client):
    with app.app_context():
        hashed = bcrypt.generate_password_hash('testpass123').decode('utf-8')
        user = User(full_name='Test User', email='test@test.com', age=30, password=hashed, is_admin=False)
        db.session.add(user)
        db.session.commit()
    client.post('/login', data={'email': 'test@test.com', 'password': 'testpass123'})
    return client

@pytest.fixture
def admin_client(client):
    with app.app_context():
        hashed = bcrypt.generate_password_hash('adminpass123').decode('utf-8')
        admin = User(full_name='Admin User', email='admin@test.com', age=35, password=hashed, is_admin=True)
        db.session.add(admin)
        db.session.commit()
    client.post('/login', data={'email': 'admin@test.com', 'password': 'adminpass123'})
    return admin_client

def test_home_page_loads(client):
    response = client.get('/')
    assert response.status_code == 200

def test_login_page_loads(client):
    response = client.get('/login')
    assert response.status_code == 200

def test_register_page_loads(client):
    response = client.get('/register')
    assert response.status_code == 200

def test_about_page_loads(client):
    response = client.get('/about')
    assert response.status_code == 200

def test_register_new_user(client):
    response = client.post('/register', data={'full_name': 'John Doe', 'email': 'john@test.com', 'age': 25, 'password': 'password123', 'confirm_password': 'password123'}, follow_redirects=True)
    assert response.status_code == 200
    with app.app_context():
        user = User.query.filter_by(email='john@test.com').first()
        assert user is not None

def test_register_duplicate_email(client):
    with app.app_context():
        hashed = bcrypt.generate_password_hash('pass123').decode('utf-8')
        user = User(full_name='Existing', email='existing@test.com', age=25, password=hashed)
        db.session.add(user)
        db.session.commit()
    response = client.post('/register', data={'full_name': 'New', 'email': 'existing@test.com', 'age': 25, 'password': 'pass123', 'confirm_password': 'pass123'}, follow_redirects=True)
    assert b'already registered' in response.data

def test_register_password_mismatch(client):
    response = client.post('/register', data={'full_name': 'Test', 'email': 'test@test.com', 'age': 25, 'password': 'pass123', 'confirm_password': 'different'}, follow_redirects=True)
    assert b'do not match' in response.data

def test_login_valid(client):
    with app.app_context():
        hashed = bcrypt.generate_password_hash('pass123').decode('utf-8')
        user = User(full_name='Test', email='test@test.com', age=25, password=hashed)
        db.session.add(user)
        db.session.commit()
    response = client.post('/login', data={'email': 'test@test.com', 'password': 'pass123'}, follow_redirects=True)
    assert response.status_code == 200

def test_login_invalid(client):
    response = client.post('/login', data={'email': 'wrong@test.com', 'password': 'wrong'}, follow_redirects=True)
    assert b'Invalid email or password' in response.data

def test_dashboard_requires_login(client):
    response = client.get('/dashboard')
    assert response.status_code == 302

def test_predict_requires_login(client):
    response = client.get('/predict')
    assert response.status_code == 302

def test_history_requires_login(client):
    response = client.get('/history')
    assert response.status_code == 302

def test_profile_requires_login(client):
    response = client.get('/profile')
    assert response.status_code == 302

def test_model_loaded():
    with app.app_context():
        from app import model, features
        assert model is not None
        assert features is not None

def test_model_prediction_output():
    import pandas as pd
    from app import model, features
    test_input = {'age': 55.0, 'sex': 1, 'cp': 0, 'trestbps': 140.0, 'chol': 250.0, 'fbs': 1, 'thalach': 120.0}
    df = pd.DataFrame([test_input])
    df = df.reindex(columns=features, fill_value=0)
    prob = model.predict_proba(df)[0][1]
    assert 0.0 <= prob <= 1.0

def test_prediction_probability_range():
    import pandas as pd
    from app import model, features
    test_input = {'age': 25.0, 'sex': 0, 'cp': 3, 'trestbps': 110.0, 'chol': 170.0, 'fbs': 0, 'thalach': 190.0}
    df = pd.DataFrame([test_input])
    df = df.reindex(columns=features, fill_value=0)
    prob = model.predict_proba(df)[0][1]
    assert 0.0 <= prob <= 1.0

def test_risk_level_logic():
    probability = 75.0
    risk_level = "High Risk" if probability >= 50 else "Low Risk"
    assert risk_level == "High Risk"

def test_risk_level_low_logic():
    probability = 30.0
    risk_level = "High Risk" if probability >= 50 else "Low Risk"
    assert risk_level == "Low Risk"
