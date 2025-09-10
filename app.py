from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pandas as pd
from catboost import CatBoostRegressor
import numpy as np
import os


# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Load Model and Columns (once, on startup) ---
MODEL_FILENAME = "catboost_food_model.cbm"
COLUMNS_FILENAME = "model_columns.txt"
model = None
model_columns = None

try:
    model = CatBoostRegressor()
    model.load_model(MODEL_FILENAME)
    with open(COLUMNS_FILENAME, 'r') as f:
        model_columns = [line.strip() for line in f.readlines()]
    print("✅ Model and columns loaded successfully.")
except Exception as e:
    print(f"❌ ERROR: Could not load model or columns file: {e}")
    print("Please make sure you have run 'python model_trainer.py' successfully.")

# --- Helper Functions ---
def get_product_data(barcode):
    """Fetches only the necessary data from Open Food Facts API for scoring."""
    url = f"https://world.openfoodfacts.org/api/v2/product/{barcode}"
    params = {"fields": "nutriments,ingredients_text_en"}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == 1 and "product" in data:
            return data["product"]
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
    return None

def scale_nutriscore_to_10(score):
    """Scales Nutri-Score (-15 to 40) to a 1-10 health rating using buckets."""
    if score <= -2: return 10.0
    if score <= 0: return 9.5
    if score <= 2: return 9.0
    if score <= 5: return 8.0
    if score <= 8: return 7.0
    if score <= 11: return 6.0
    if score <= 14: return 5.0
    if score <= 17: return 4.0
    if score <= 20: return 3.0
    if score > 20: return 2.0
    return 1.0

# --- Main Prediction Route ---
@app.route('/predict_score', methods=['GET'])
def predict_score():
    barcode = request.args.get('barcode')
    if not barcode:
        return jsonify({"error": "Barcode parameter is missing"}), 400

    product_data = get_product_data(barcode)
    if not product_data:
        return jsonify({"error": f"Could not find product data for barcode {barcode}"}), 404
    
    # --- Hybrid Model Scoring ---
    nutriments = product_data.get("nutriments", {})
    input_df = pd.DataFrame([nutriments])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    for col in model_columns:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

    catboost_score = model.predict(input_df)[0]
    base_scaled_score = scale_nutriscore_to_10(catboost_score)

    ingredients_text = product_data.get('ingredients_text_en', '').lower()
    
    RULE_BASED_ADJUSTMENTS = {
        "aspartame": -1.5, "acesulfame": -1.5, "sucralose": -1.0,
        "high fructose corn syrup": -2.0, "hydrogenated": -1.5,
        "monosodium glutamate": -1.0, "msg": -1.0,
        "whole grain": 0.5, "organic": 0.5, "fiber": 0.2
    }
    
    final_score = base_scaled_score
    for ingredient, adjustment in RULE_BASED_ADJUSTMENTS.items():
        if ingredient in ingredients_text:
            final_score += adjustment

    final_hybrid_score = np.clip(final_score, 1, 10)

    # --- Final Response (Score Only) ---
    return jsonify({"health_score": final_hybrid_score})

if __name__ == '__main__':
    if not model or not model_columns:
        print("Model not loaded. Exiting.")
    else:
        app.run(debug=True)