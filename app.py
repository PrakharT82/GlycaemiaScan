import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained pipeline once when the server starts
bundle = joblib.load("models/diabetes_model.joblib")
pipe, threshold, features = bundle["pipeline"], bundle["threshold"], bundle["features"]


@app.route("/")
def home():
    with open("reports/metrics.json") as f:
        metrics = json.load(f)
    comparison = pd.read_csv("reports/model_comparison.csv").to_dict("records")
    stats = pd.read_csv("reports/stat_tests.csv").to_dict("records")
    return render_template("index.html", m=metrics, comparison=comparison, stats=stats)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    try:
        # None (unknown) becomes NaN; the pipeline's imputer fills it
        row = pd.DataFrame([[data.get(f) for f in features]], columns=features).astype(float)
    except (TypeError, ValueError):
        return jsonify(error="Invalid input values"), 400
    prob = float(pipe.predict_proba(row)[0, 1])
    return jsonify(probability=prob, high_risk=prob >= threshold, threshold=threshold)


if __name__ == "__main__":
    app.run(debug=True)