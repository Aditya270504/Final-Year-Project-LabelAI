import pandas as pd
import requests
from catboost import CatBoostRegressor
import argparse
import os

# --- Configuration ---
MODEL_FILENAME = "catboost_food_model.cbm"
COLUMNS_FILENAME = "model_columns.txt"

# --- Rule-Based Model Configuration ---
RULE_BASED_ADJUSTMENTS = {
    # Negative adjustments (penalties) for certain ingredients.
    # The key is the ingredient text (lowercase), the value is the points to subtract from the Nutri-Score.
    "harmful": {
        "hydrogenated oil": 2.5,
        "partially hydrogenated oil": 2.5,
        "high fructose corn syrup": 2.0,
        "aspartame": 1.5,
        "monosodium glutamate": 1.0,
        "msg": 1.0,
        "artificial color": 0.5,
        "sodium nitrate": 1.0,
    },
    # Positive adjustments (bonuses) for beneficial ingredients.
    # The value is the points to subtract (improve) from the Nutri-Score.
    "beneficial": {
        "whole grain": -1.5,
        "whole wheat": -1.5,
        "probiotic": -1.0,
        "organic": -0.5, # Small bonus for being organic
    }
}


# --- 1. Load Model and Columns ---
def load_model_and_columns():
    """Loads the trained CatBoost model and the list of feature columns."""
    if not os.path.exists(MODEL_FILENAME) or not os.path.exists(COLUMNS_FILENAME):
        print(f"Error: Model ('{MODEL_FILENAME}') or columns file ('{COLUMNS_FILENAME}') not found.")
        print("Please run the 'model_trainer.py' script first to train and save the model.")
        return None, None

    # Load the model
    print("Loading trained model...")
    model = CatBoostRegressor()
    model.load_model(MODEL_FILENAME)
    
    # Load the columns
    with open(COLUMNS_FILENAME, 'r') as f:
        columns = [line.strip() for line in f]
        
    print("Model and columns loaded successfully.")
    return model, columns

# --- 2. Fetch Product Data ---
def get_product_data_from_api(barcode):
    """Fetches product data for a given barcode from the Open Food Facts API."""
    print(f"Fetching data for barcode: {barcode}...")
    url = f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == 1 and "product" in data:
            return data["product"]
        else:
            print(f"Product with barcode '{barcode}' not found.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None

# --- 3. Preprocess Input for Prediction ---
def preprocess_for_prediction(product_data, columns):
    """Prepares the fetched product data to match the model's input format."""
    if not product_data or "nutriments" not in product_data:
        return None
        
    nutriments = product_data["nutriments"]
    
    # Create a dictionary with the features required by the model
    # Use .get() to handle cases where a nutrient might be missing
    input_data = {col: nutriments.get(col, 0) for col in columns}
    
    # Convert to a pandas DataFrame
    df = pd.DataFrame([input_data])
    
    # Ensure all columns are numeric, filling any potential non-numeric with 0
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    return df[columns] # Ensure column order is correct


# --- 4. Hybrid Model Prediction ---
def apply_rule_based_adjustment(product_data, initial_score):
    """Applies adjustments to the CatBoost score based on the ingredient list."""
    ingredients_text = product_data.get('ingredients_text', '').lower()
    if not ingredients_text:
        return initial_score, []

    adjustment = 0
    reasons = []

    # Apply penalties for harmful ingredients
    for ingredient, penalty in RULE_BASED_ADJUSTMENTS["harmful"].items():
        if ingredient in ingredients_text:
            adjustment += penalty
            reasons.append(f"  - Penalty of {penalty:.1f} points for containing '{ingredient}'.")

    # Apply bonuses for beneficial ingredients
    for ingredient, bonus in RULE_BASED_ADJUSTMENTS["beneficial"].items():
        if ingredient in ingredients_text:
            adjustment += bonus
            reasons.append(f"  - Bonus of {abs(bonus):.1f} points for containing '{ingredient}'.")
    
    adjusted_score = initial_score + adjustment
    return adjusted_score, reasons

def get_hybrid_health_rating(model, data_df, product_data):
    """
    Predicts a score using CatBoost, adjusts it with a rule-based model,
    and converts it to a 1-10 health rating.
    """
    # 1. Get base prediction from CatBoost model
    predicted_nutriscore = model.predict(data_df)[0]
    
    # 2. Apply rule-based adjustments based on ingredients
    adjusted_score, adjustment_reasons = apply_rule_based_adjustment(product_data, predicted_nutriscore)
    
    print(f"\n- CatBoost Model Base Score: {predicted_nutriscore:.2f}")
    if adjustment_reasons:
        print(f"- Rule-Based Adjusted Score: {adjusted_score:.2f}")
    
    # 3. Scale the final score to a 1-10 rating
    # Nutri-Score ranges from -15 (best) to 40 (worst).
    clipped_score = max(-15, min(40, adjusted_score))
    
    # Normalize the score from 0 (worst) to 1 (best)
    normalized_score = (clipped_score - 40) / (-15 - 40)

    # Scale to a 1-9 range and add 1 to make it 1-10
    health_rating = normalized_score * 9 + 1
    
    return round(health_rating, 1), adjustment_reasons

def display_scoring_rubric():
    """Prints the explanation of the scoring model."""
    print("\n" + "="*50)
    print("Hybrid Model Scoring Rubric")
    print("="*50)
    print("The final 1-10 rating is calculated in three steps:\n")
    print("1. CatBoost Model Score:")
    print("   - A machine learning model predicts a Nutri-Score based on nutritional")
    print("     values (fat, sugar, salt, etc.).")
    print("   - This score ranges from -15 (healthiest) to 40 (least healthy).\n")
    print("2. Rule-Based Ingredient Analysis:")
    print("   - The ingredient list is scanned for specific keywords.")
    print("   - Penalties are applied for 'harmful' ingredients (e.g., hydrogenated oil).")
    print("   - Bonuses are given for 'beneficial' ingredients (e.g., whole grain).\n")
    print("3. Final Rating Conversion:")
    print("   - The adjusted score is scaled to a user-friendly 1 to 10 rating,")
    print("     where 10 is the best possible score.\n")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Predict a health rating for a food product using its barcode.")
    parser.add_argument("barcode", type=str, help="The barcode of the product to analyze.")
    args = parser.parse_args()
    
    model, columns = load_model_and_columns()
    
    if model and columns:
        display_scoring_rubric()
        product_data = get_product_data_from_api(args.barcode)
        if product_data:
            processed_df = preprocess_for_prediction(product_data, columns)
            if processed_df is not None:
                rating, reasons = get_hybrid_health_rating(model, processed_df, product_data)
                
                print("\n" + "="*30)
                print(f"Analysis for: {product_data.get('product_name', 'N/A')}")
                print("="*30)
                
                if reasons:
                    print("Score Adjustments from Ingredients:")
                    for reason in reasons:
                        print(reason)
                else:
                    print("No specific ingredient adjustments applied.")

                print("\n" + "-"*30)
                print(f"⭐ Final Hybrid Health Rating: {rating} / 10")
                print("-"*30)

if __name__ == "__main__":
    main()