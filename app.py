from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("nfl_touchdown_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("playtype_encoder.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        qtr = int(request.form['qtr'])
        down = int(request.form['down'])
        time_secs = float(request.form['time_secs'])
        yrdline100 = float(request.form['yrdline100'])
        ydstogo = float(request.form['ydstogo'])
        goal_to_go = int(request.form['goal_to_go'])
        score_diff = float(request.form['score_diff'])
        abs_score_diff = abs(score_diff)
        posteam_timeouts = int(request.form['posteam_timeouts'])
        home_timeouts = int(request.form['home_timeouts'])
        away_timeouts = int(request.form['away_timeouts'])
        play_type = request.form['play_type']
        play_type_enc = le.transform([play_type])[0]

        features = np.array([[qtr, down, time_secs, yrdline100, ydstogo,
                               goal_to_go, score_diff, abs_score_diff,
                               posteam_timeouts, home_timeouts, away_timeouts,
                               play_type_enc]])

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        result = "TOUCHDOWN! 🏈" if prediction == 1 else "No Touchdown"
        confidence = f"{probability * 100:.1f}%"

        return render_template('index.html', prediction=result, confidence=confidence)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}", confidence="—")

if __name__ == '__main__':
    app.run(debug=True)
