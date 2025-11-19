import os
import joblib
import pandas as pd
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
from haversine import haversine, Unit 

# --- 1. SETUP & LOAD MODELS ---
app = Flask(__name__)
CORS(app)

# Load the Scikit-learn (Random Forest) model
try:
    model = joblib.load('chimera_model.pkl')
    all_assets_df = pd.read_csv('master_assets.csv')
    with open('model_columns.json', 'r') as f:
        MODEL_COLUMNS = json.load(f)
    print("✅ Scikit-learn model and asset database loaded.")
except FileNotFoundError:
    print("❌ ERROR: Missing model files. Did you download all 3 files from Colab?")
    model = None
    all_assets_df = pd.DataFrame()
    MODEL_COLUMNS = []

# --- 2. DUMMY ACTION PLAN (Gemini ki jagah) ---
def get_dummy_action_plan(direct_failures, cascade_failures):
    """
    Gemini API ki jagah ek hardcoded plan bhejta hai.
    """
    plan = f"""
AI Simulation Complete. {len(direct_failures) + len(cascade_failures)} assets at risk.

• PRIORITY 1: Dispatch emergency generators and mobile units to all {len(cascade_failures)} predicted hospital failures.
• PRIORITY 2: Secure the {len(direct_failures)} failing substations to prevent further damage.
• PRIORITY 3: Reroute all non-emergency traffic away from affected zones.
"""
    return plan

# --- 3. RESUME-READY PREDICTION API (Wahi logic) ---
@app.route("/api/predict-failure", methods=['POST'])
def predict_failure():
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    # --- PART A: AI PREDICTION (Direct Failures) ---
    assets_to_predict_df = all_assets_df[MODEL_COLUMNS]
    predictions = model.predict(assets_to_predict_df)
    all_failing_assets_df = all_assets_df[predictions == 1].copy()
    
    direct_failures = all_failing_assets_df[
        all_failing_assets_df['asset_type'] == 1 # 1 = Substation
    ]

    # --- PART B: LOGIC PREDICTION (Cascading Failures) ---
    all_hospitals = all_assets_df[all_assets_df['asset_type'] == 0].copy()
    failing_substation_coords = [
        (row['latitude'], row['longitude']) for index, row in direct_failures.iterrows()
    ]
    cascade_failures_list = []

    if failing_substation_coords: # Check if list is not empty
        for index, hospital in all_hospitals.iterrows():
            hospital_coord = (hospital['latitude'], hospital['longitude'])
            min_dist_km = 99999
            
            for sub_coord in failing_substation_coords:
                dist = haversine(hospital_coord, sub_coord, unit=Unit.KILOMETERS)
                if dist < min_dist_km:
                    min_dist_km = dist
            
            # RULE: If a hospital is within 10km of a *failed* substation, it fails
            if min_dist_km <= 10:
                cascade_failures_list.append(hospital)

    cascade_failures = pd.DataFrame(cascade_failures_list)

    # --- PART C: GET DUMMY ACTION PLAN ---
    action_plan_text = get_dummy_action_plan(direct_failures, cascade_failures)

    # --- PART D: Send ALL results back to the frontend ---
    # Combine both lists for the frontend to render
    all_failing_for_frontend = pd.concat([direct_failures, cascade_failures]).drop_duplicates()

    return jsonify({
        # Send the combined list
        "failing_assets": all_failing_for_frontend.to_json(orient='records'),
        "action_plan": action_plan_text
    })

# --- 4. HEALTH CHECK & RUN SERVER ---
@app.route("/")
def health_check():
    return jsonify({"status": "ok", "scikit-model-loaded": model is not None})

if __name__ == '__main__':
    print("🚀 Starting Flask server (Resume-Ready Plan - NO GEMINI)...")
    app.run(debug=True, port=5000)