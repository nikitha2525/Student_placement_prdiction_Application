from tensorflow.keras.models import load_model
from flask import Flask, render_template, url_for, request, redirect
import pickle
import psycopg2
import numpy as np
import os
import logging




app = Flask(__name__)

# Load model
model = load_model('placement_model.keras')

# Load scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)




DB_CONFIG = dict(
    host="localhost",
    database="postgres",
    user="postgres",
    password="0925",
    port="5432"
)

def get_db():
    return psycopg2.connect(**DB_CONFIG)




@app.route('/')
def index():
    return render_template('pl.html')


@app.route('/pl', methods=['GET','POST'])
def pl():
    field = (
        request.form.get('Name'),
        request.form.get('email'),
        request.form.get('password')
    )
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO Use_r(name,email,password) VALUES (%s,%s,%s)", field
                )
        conn.commit()
        cur.close()
        conn.close()
        
    except Exception as e:
        logging.exception("DB insert Use_r failed: %s", e)
    return render_template('st.html')   


@app.route('/st')
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

       
        name   = request.form.get('Name',   '').strip()
        Gender = request.form.get('Gender', '').strip()

        try:
            CGPA = float(request.form.get('CGPA', 0))
        except (TypeError, ValueError):
            CGPA = 0.0

        try:
            Apptitude_score = float(request.form.get('Apptitude_score', 0))
        except (TypeError, ValueError):
            Apptitude_score = 0.0

        try:
            communication_score = float(
                request.form.get('Communication_score') or
                request.form.get('communication_score') or 0
            )
        except (TypeError, ValueError):
            communication_score = 0.0

       
        Internship_Experience = request.form.get('Internship_Experience', '0')
        raw = Internship_Experience.strip()

        if raw in ['1', 'Yes', 'yes', 'YES']:
            internship_encoded = 1
            Internship_Experience_clean = 'Yes'
        else:
            internship_encoded = 0
            Internship_Experience_clean = 'No'

        try:
            projects_completed = int(
                request.form.get('Projects_completed') or
                request.form.get('projects_completed') or 0
            )
        except (TypeError, ValueError):
            projects_completed = 0

        try:
            IQ = int(request.form.get('IQ', 0))
        except (TypeError, ValueError):
            IQ = 0

        try:
            Prev_Sem_Result = float(request.form.get('Prev_Sem_Result', 0))
        except (TypeError, ValueError):
            Prev_Sem_Result = 0.0

        try:
            Academic_Performance = int(request.form.get('Academic_Performance', 0))
        except (TypeError, ValueError):
            Academic_Performance = 0
        try :
            Extra_Curricular_Score = int(request.form.get('Extra_Curricular_Score',0))
        except(TypeError,ValueError):
            Extra_Curricular_Score = 0
        

        print("=" * 60)
        print(f"Name                  : {name}")
        print(f"CGPA                  : {CGPA}")
        print(f"Aptitude              : {Apptitude_score}")
        print(f"Communication         : {communication_score}")
        print(f"Internship            : {Internship_Experience_clean} → {internship_encoded}")
        print(f"Projects              : {projects_completed}")
        print(f"IQ                    : {IQ}")
        print(f"Prev Sem Result       : {Prev_Sem_Result}")
        print(f"Academic Performance  : {Academic_Performance}")
        print(f"Extra curricular score :{Extra_Curricular_Score}")
        print("=" * 60)

       
        features = np.array([[
            CGPA,
            Apptitude_score,
            communication_score,
            internship_encoded,
            projects_completed,
            IQ,
            Prev_Sem_Result,
            Academic_Performance
        ]])
        print(f"Features : {features}")

        prob   = 0.0
        status = "Model not loaded"

        if model is not None:
            try:
                if hasattr(model, 'predict_proba'):
                    proba   = model.predict_proba(features)[0]
                    

                    print(f"All probs : {proba}") 

                else:
                    # Neural network
                    pred = model.predict(features)
                    prob = float(np.array(pred).ravel()[0])

                print(f"Probability : {prob}")
                status = "Placed !!!" if prob >= 0.50 else "Not Placed"
                print(f"Status      : {status}")

            except Exception as e:
                logging.exception("Prediction error: %s", e)
                status = f"Prediction error: {str(e)}"

        
        
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO student("
            "  name, Gender, CGPA, Apptitude_score,"
            "  Communication_Score, Internship_Experience,"
            "  projects_completed, IQ, Prev_Sem_Result,"
            "  Academic_Performance, Extra_Curricular_Score"
            ") VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                name, Gender, CGPA, Apptitude_score,
                communication_score, Internship_Experience,
                projects_completed, IQ, Prev_Sem_Result,
                Academic_Performance,Extra_Curricular_Score))
        conn.commit()
        cur.close()
        conn.close()
        logging.info("student insert successful")

        return render_template('st.html', status=status)

    return render_template('st.html')


if __name__ == "__main__":
    app.run(debug=True)