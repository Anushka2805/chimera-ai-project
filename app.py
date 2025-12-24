import os, json, joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from haversine import haversine, Unit

app = Flask(__name__, static_folder="frontend")
CORS(app)

# 🚫 Disable caching (IMPORTANT)
@app.after_request
def no_cache(res):
    res.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    res.headers["Pragma"] = "no-cache"
    res.headers["Expires"] = "0"
    return res

# Load assets
try:
    model = joblib.load("chimera_model.pkl")
    all_assets_df = pd.read_csv("master_assets.csv")
    with open("model_columns.json") as f:
        MODEL_COLUMNS = json.load(f)
    print("✅ Model & data loaded")
except:
    model = None
    all_assets_df = pd.DataFrame()
    MODEL_COLUMNS = []

def action_plan(direct, cascade):
    return f"""
AI Simulation Complete.
Total assets at risk: {len(direct)+len(cascade)}

• PRIORITY 1: Secure substations
• PRIORITY 2: Protect hospitals
• PRIORITY 3: Reroute traffic
"""

@app.route("/api/predict-failure", methods=["POST"])
def predict():
    data = request.get_json() or {}
    lat, lng = data.get("lat"), data.get("lng")

    failing = pd.DataFrame()
    if model is not None:
        preds = model.predict(all_assets_df[MODEL_COLUMNS])
        failing = all_assets_df[preds == 1]

    direct = failing[failing["asset_type"] == 1]
    cascade = failing[failing["asset_type"] == 0]

    return jsonify({
        "failing_assets": pd.concat([direct,cascade]).to_json(orient="records"),
        "action_plan": action_plan(direct, cascade)
    })

@app.route("/")
def home():
    return send_from_directory("frontend", "Dashboard.html")

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT",5000)),
        debug=False
    )
