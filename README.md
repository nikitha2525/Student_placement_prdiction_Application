# 🎓 Day 52 — Full-Stack AI: Student Placement Prediction with Flask + PostgreSQL + TensorFlow

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=flat-square&logo=flask&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![HTML](https://img.shields.io/badge/HTML5-Frontend-E34F26?style=flat-square&logo=html5&logoColor=white)
![CSS](https://img.shields.io/badge/CSS3-Styling-1572B6?style=flat-square&logo=css3&logoColor=white)
![DataGrip](https://img.shields.io/badge/DataGrip-DB%20Management-000000?style=flat-square&logo=datagrip&logoColor=white)
![Challenge](https://img.shields.io/badge/100%20Days%20AI%2FML-Day%2052-blueviolet?style=flat-square)

**Training a model is only one step. The real challenge is deploying it, managing user data, and delivering predictions through an application people can actually use.**

</div>

---

## 📌 Overview

Day 51 built the ANN model. Day 52 makes it real — wrapping that model inside a **production-style full-stack web application** where:

- Users fill out a placement form on a clean frontend
- Flask validates inputs, preprocesses features, and runs the ANN
- Predictions and student records are **persisted in PostgreSQL**
- Results are displayed instantly in the browser with placement probability

This is the full loop: AI model + web backend + relational database + frontend UI.

> **Hard truth learned today:** Training a model is only one step. The real challenge is deploying it, managing user data, and delivering predictions through an application that people can actually use.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    User Browser                          │
│                  (HTML / CSS UI)                         │
│                                                          │
│   ┌─────────────────────────────────────────────────┐   │
│   │         Student Details Form                    │   │
│   │  Name | CGPA | Aptitude | Communication | ...   │   │
│   └──────────────────────┬──────────────────────────┘   │
└─────────────────────────-│──────────────────────────────┘
                           │  POST /predict
                           ▼
┌──────────────────────────────────────────────────────────┐
│                   Flask Backend                          │
│                                                          │
│   ① Receive & validate form data                        │
│   ② Preprocess inputs (encode + scale)                  │
│   ③ Run ANN model → P(placed)                           │
│   ④ Save student record + prediction to PostgreSQL      │
│   ⑤ Return result to frontend                           │
└───────────┬────────────────────────┬─────────────────────┘
            │                        │
            ▼                        ▼
┌───────────────────┐     ┌──────────────────────────┐
│   TensorFlow ANN  │     │      PostgreSQL DB        │
│   (model.h5)      │     │                          │
│                   │     │  Table: student_records  │
│  Input(6)         │     │  ─────────────────────   │
│  Dense(64, ReLU)  │     │  id · name · cgpa        │
│  Dense(32, ReLU)  │     │  aptitude · comm         │
│  Dense(1, Sigmoid)│     │  internship · projects   │
│                   │     │  technical · prediction  │
│  → P(placed)      │     │  probability · timestamp │
└───────────────────┘     └──────────────────────────┘
            │
            ▼
┌───────────────────────────┐
│     Prediction Result     │
│  ✅ Placed  /  ❌ Not Placed │
│  Confidence: 87.3%        │
└───────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Deep Learning | TensorFlow / Keras ANN | Placement probability prediction |
| Backend | Flask (Python) | API routes, business logic, model serving |
| Database | PostgreSQL | Persistent storage of student records |
| DB Management | DataGrip | Schema design, query inspection, data browsing |
| Frontend | HTML + CSS | Input form + results display |

---

## 🗄️ Database Schema

```sql
-- PostgreSQL: student_records table
CREATE TABLE student_records (
    id                  SERIAL PRIMARY KEY,
    name                VARCHAR(100)    NOT NULL,
    cgpa                DECIMAL(4, 2)   NOT NULL,
    aptitude_score      INTEGER         NOT NULL,
    communication       INTEGER         NOT NULL,   -- scale 1–10
    internship          BOOLEAN         NOT NULL,
    project_experience  INTEGER         NOT NULL,   -- number of projects
    technical_skills    INTEGER         NOT NULL,   -- scale 1–10
    prediction          VARCHAR(20)     NOT NULL,   -- 'Placed' / 'Not Placed'
    probability         DECIMAL(5, 4)   NOT NULL,   -- e.g. 0.8732
    submitted_at        TIMESTAMP       DEFAULT NOW()
);

-- Index for fast lookup by prediction outcome
CREATE INDEX idx_prediction ON student_records(prediction);
```

---

## 🔬 What I Implemented

### Flask App + PostgreSQL Integration

```python
# app.py
import numpy as np
import psycopg2
import joblib
import tensorflow as tf
from flask import Flask, render_template, request
from datetime import datetime

app = Flask(__name__)

# ─── Load Model & Scaler ────────────────────────────────────────────────────────
model  = tf.keras.models.load_model('models/placement_model.h5')
scaler = joblib.load('models/scaler.pkl')

# ─── PostgreSQL Connection ───────────────────────────────────────────────────────
def get_db_connection():
    return psycopg2.connect(
        host     = "localhost",
        database = "placement_db",
        user     = "your_user",
        password = "your_password",
        port     = 5432
    )

# ─── Home Route ─────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

# ─── Predict Route ──────────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ① Collect inputs
        name          = request.form['name']
        cgpa          = float(request.form['cgpa'])
        aptitude      = int(request.form['aptitude_score'])
        communication = int(request.form['communication'])
        internship    = int(request.form['internship'])          # 1 or 0
        projects      = int(request.form['project_experience'])
        technical     = int(request.form['technical_skills'])

        # ② Preprocess
        features = np.array([[cgpa, aptitude, communication,
                               internship, projects, technical]])
        features_scaled = scaler.transform(features)

        # ③ Predict
        prob        = float(model.predict(features_scaled, verbose=0)[0][0])
        placed      = prob >= 0.5
        prediction  = "Placed" if placed else "Not Placed"
        confidence  = f"{prob:.1%}" if placed else f"{(1 - prob):.1%}"

        # ④ Save to PostgreSQL
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO student_records
              (name, cgpa, aptitude_score, communication,
               internship, project_experience, technical_skills,
               prediction, probability)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (name, cgpa, aptitude, communication,
             bool(internship), projects, technical,
             prediction, round(prob, 4))
        )
        conn.commit()
        cursor.close()
        conn.close()

        # ⑤ Return result
        return render_template(
            'result.html',
            name       = name,
            prediction = prediction,
            confidence = confidence,
            placed     = placed
        )

    except Exception as e:
        return render_template('error.html', error=str(e))

# ─── Records Route (view all submissions) ───────────────────────────────────────
@app.route('/records')
def records():
    conn   = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name, cgpa, prediction, probability, submitted_at "
        "FROM student_records ORDER BY submitted_at DESC LIMIT 50"
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('records.html', records=rows)

if __name__ == '__main__':
    app.run(debug=True)
```

---

### ANN Model (from Day 51)

```python
# train.py
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

df = pd.read_csv('data/placement_data.csv')
X  = df[['cgpa', 'aptitude_score', 'communication',
          'internship', 'project_experience', 'technical_skills']].values
y  = df['placed'].values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'models/scaler.pkl')

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(6,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1,  activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=100, batch_size=16,
          validation_split=0.15, verbose=1)

model.save('models/placement_model.h5')
print("✅ Model saved")
```

---

### Frontend — Input Form (index.html)

```html
<!-- templates/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Student Placement Predictor</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
  <div class="container">
    <h1>🎓 Student Placement Predictor</h1>
    <p class="subtitle">Enter your academic details to predict placement probability</p>

    <form action="/predict" method="POST">
      <div class="form-group">
        <label>Full Name</label>
        <input type="text" name="name" placeholder="Your name" required>
      </div>
      <div class="form-group">
        <label>CGPA (0.0 – 10.0)</label>
        <input type="number" name="cgpa" min="0" max="10"
               step="0.01" placeholder="e.g. 8.5" required>
      </div>
      <div class="form-group">
        <label>Aptitude Score (0 – 100)</label>
        <input type="number" name="aptitude_score"
               min="0" max="100" placeholder="e.g. 76" required>
      </div>
      <div class="form-group">
        <label>Communication Skills (1 – 10)</label>
        <input type="number" name="communication"
               min="1" max="10" placeholder="e.g. 7" required>
      </div>
      <div class="form-group">
        <label>Internship Experience</label>
        <select name="internship" required>
          <option value="1">Yes</option>
          <option value="0">No</option>
        </select>
      </div>
      <div class="form-group">
        <label>Projects Completed</label>
        <input type="number" name="project_experience"
               min="0" placeholder="e.g. 3" required>
      </div>
      <div class="form-group">
        <label>Technical Skills (1 – 10)</label>
        <input type="number" name="technical_skills"
               min="1" max="10" placeholder="e.g. 8" required>
      </div>
      <button type="submit" class="btn-predict">Predict Placement</button>
    </form>
  </div>
</body>
</html>
```

---

## 🔁 Request Lifecycle

```
User clicks "Predict Placement"
           │
           ▼
    POST /predict
    Form data sent to Flask
           │
           ▼
    Input validation
    (type checks, range validation)
           │
           ▼
    Feature array built
    → StandardScaler.transform()
           │
           ▼
    model.predict(features_scaled)
    → probability ∈ (0.0, 1.0)
           │
           ├── prob ≥ 0.5 → "Placed"   ✅
           └── prob < 0.5 → "Not Placed" ❌
           │
           ▼
    INSERT INTO student_records (...)
    (name, features, prediction, probability, timestamp)
           │
           ▼
    render_template('result.html')
    → Show prediction + confidence to user
```

---

## 💡 Key Learnings

- **A model becomes useful only when integrated into an application** — a `.h5` file alone helps no one
- **Flask simplifies connecting AI models with user interfaces** — routes, forms, and `render_template` make the bridge easy to build
- **PostgreSQL provides reliable, queryable storage** — every prediction is logged and retrievable for auditing, retraining, or analytics
- **Prediction probabilities offer more insight than class labels** — "Placed with 91% confidence" is more actionable than just "Placed"
- **End-to-end projects demand both ML and software engineering** — model accuracy matters, but so do schema design, error handling, and UX

---

## ⚠️ Limitations & Next Steps

| Current Limitation | Improvement |
|---|---|
| Local PostgreSQL only | Migrate to cloud DB (Supabase, Railway, AWS RDS) |
| No authentication | Add student login + admin dashboard |
| Raw confidence scores | Add calibration (Platt scaling / isotonic regression) |
| No input sanitization | Add server-side validation and SQL injection protection |
| Single model | Add model versioning and A/B testing capability |
| No retraining pipeline | Log predictions → retrain periodically on new data |

---

## 🗂️ Project Structure

```
day-52-fullstack-placement/
├── app.py                        # Flask routes + DB + prediction logic
├── train.py                      # ANN training script
├── requirements.txt
│
├── models/
│   ├── placement_model.h5        # Trained TensorFlow ANN
│   └── scaler.pkl                # Fitted StandardScaler
│
├── templates/
│   ├── index.html                # Student input form
│   ├── result.html               # Prediction result page
│   ├── records.html              # View all submissions
│   └── error.html                # Error display
│
├── static/
│   └── css/
│       └── style.css
│
├── data/
│   └── placement_data.csv
│
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/your-username/day-52-fullstack-placement
cd day-52-fullstack-placement
pip install -r requirements.txt

# Set up PostgreSQL
psql -U postgres -c "CREATE DATABASE placement_db;"
psql -U postgres -d placement_db -f schema.sql

# Train the ANN model
python train.py

# Run Flask
python app.py
# → Open http://localhost:5000
```

**Requirements:**
```
tensorflow>=2.10
flask
psycopg2-binary
scikit-learn
numpy
pandas
joblib
```

---

## 🔗 Part of the 100 Days AI/ML Engineer Challenge

> Day 52 of 100 — Full-Stack AI: Deep Learning + Flask + PostgreSQL

| ← Previous | Current | Next → |
|---|---|---|
| [Day 51 — ANN Placement Model](#) | **Day 52 — Full-Stack AI App** | [Day 53](#) |


---

<div align="center">
<sub>Built with curiosity · Part of #100DaysOfAIML · #FullStack #Flask #PostgreSQL #TensorFlow #ANN #PlacementPrediction #EndToEnd</sub>
</div>
