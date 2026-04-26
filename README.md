# ❤️ Heart Attack Predict

> AI-powered heart attack risk prediction web application — BSc (Hons) Software Engineering Final Year Project

![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=flat-square&logo=flask)
![ML](https://img.shields.io/badge/ML-Random%20Forest-red?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-89.14%25-green?style=flat-square)
![AUC](https://img.shields.io/badge/AUC-0.9652-brightgreen?style=flat-square)

---

## 📋 Overview

**Heart Attack Predict** is a machine learning web application that predicts the risk of a heart attack based on 7 clinical parameters. Built using Python, Flask, and a Random Forest classifier trained on 2,024 clinical records from the UCI Heart Disease and Statlog Cleveland Hungary datasets.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 ML Prediction | Random Forest with 89.14% accuracy |
| 🧠 SHAP Explainability | Shows why the model made each prediction |
| 📊 Dashboard | Animated stats, 3D heart, ECG, risk trend chart |
| 📄 PDF Report | Professional downloadable clinical report |
| 📁 CSV Export | Download full prediction history as spreadsheet |
| 🔐 Authentication | Register, login, logout with bcrypt encryption |
| 📧 Password Reset | Email-based reset with 30-minute token expiry |
| 📈 Model Evaluation | ROC curve, confusion matrix, animated metrics |
| 🔍 Feature Importance | Bar chart, donut chart, feature breakdown |
| 🕒 History Timeline | Full prediction history with risk trend chart |
| 👤 Profile Page | Update details, change password with strength meter |
| 🛡️ Admin Dashboard | User management, predictions table, analytics charts |
| 🧪 Unit Tests | 18 tests covering routes, auth, prediction and more |
| 📊 Data Analysis | Standalone analysis script with chart generation |

---

## 🤖 ML Model

| Metric | Score |
|---|---|
| Algorithm | Random Forest Classifier (100 trees) |
| Accuracy | 89.14% |
| AUC Score | 0.9652 |
| Precision | 87.83% |
| Recall | 92.66% |
| F1 Score | 90.18% |
| Training Records | 2,024 |
| Features | 7 clinical parameters |

### Input Features
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Serum Cholesterol
- Fasting Blood Sugar
- Maximum Heart Rate

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Dimuth10/heart-attack-predictor.git
cd heart-attack-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Visit **http://127.0.0.1:5000**

---

## 📁 Project Structure
---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📊 Data Analysis

```bash
python analysis.py
```

Generates charts saved to `static/analysis/`.

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask, SQLAlchemy, Flask-Login, Flask-Bcrypt, Flask-Mail
- **ML:** scikit-learn, SHAP, pandas, numpy
- **Frontend:** HTML5, CSS3, Bootstrap 5, Chart.js, JavaScript
- **Database:** SQLite
- **PDF:** ReportLab
- **Testing:** pytest

---

## ⚠️ Disclaimer

This application is built for **educational purposes only** as part of a final year project. It does not replace professional medical diagnosis. Always consult a qualified healthcare professional.

---

## 👨‍💻 Developer

**Dimuth Fonseka**
BSc (Hons) Software Engineering
NSBM Green University | University of Plymouth — 2025
