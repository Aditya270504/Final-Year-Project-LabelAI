import pandas as pd
import requests
import os
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import tqdm
import gzip
import shutil

# --- Configuration ---
# Updated URL to the compressed CSV (which is actually tab-separated)
DATA_URL = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
MODEL_FILENAME = "catboost_food_model.cbm"
COLUMNS_FILENAME = "model_columns.txt"
CHUNK_SIZE = 8192
COMPRESSED_FILENAME = "en.openfoodfacts.org.products.csv.gz"
DATA_FILENAME = "en.openfoodfacts.org.products.csv"


# --- 1. Data Fetching and Decompression ---
def download_and_decompress_data(url, compressed_filename, decompressed_filename):
    """Downloads and decompresses the Open Food Facts dataset."""
    
    # Step 1: Download the compressed file if it doesn't exist
    if not os.path.exists(compressed_filename):
        print(f"Downloading data from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(compressed_filename, "wb") as f, tqdm(
                desc=f"Downloading {compressed_filename}",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    bar.update(len(chunk))
        except requests.exceptions.RequestException as e:
            print(f"Error downloading data: {e}")
            return None
    else:
        print(f"'{compressed_filename}' already exists. Skipping download.")

    # Step 2: Decompress the file if the output doesn't exist
    if not os.path.exists(decompressed_filename):
        print(f"Decompressing '{compressed_filename}'...")
        try:
            with gzip.open(compressed_filename, 'rb') as f_in:
                with open(decompressed_filename, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Decompressed data saved to '{decompressed_filename}'")
        except Exception as e:
            print(f"Error decompressing file: {e}")
            return None
    else:
         print(f"'{decompressed_filename}' already exists. Skipping decompression.")

    return decompressed_filename


# --- 2. Data Preprocessing & Feature Engineering ---
def preprocess_data(filepath):
    """Loads, cleans, and prepares the data for training by processing it in chunks."""
    print("Loading and preprocessing data in chunks to conserve memory...")
    
    features = [
        'sugars_100g', 'fat_100g', 'saturated-fat_100g', 
        'salt_100g', 'proteins_100g', 'fiber_100g', 
        'energy-kcal_100g'
    ]
    target = 'nutriscore_score'
    columns_to_keep = features + [target]
    
    chunk_list = []
    # Create an iterator that reads the CSV in chunks of 100,000 rows
    # This avoids loading the entire massive file into memory at once.
    chunk_iter = pd.read_csv(
        filepath, 
        sep='\t', 
        low_memory=False, 
        on_bad_lines='warn',
        chunksize=100000,
        usecols=columns_to_keep # Optimization: Only load the columns we need
    )

    for chunk in tqdm(chunk_iter, desc="Processing file chunks"):
        # --- Data Cleaning on the current chunk ---
        chunk.dropna(subset=[target], inplace=True)
        
        for col in features:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
            
        for col in ['sugars_100g', 'fat_100g', 'saturated-fat_100g', 'salt_100g', 'proteins_100g', 'fiber_100g']:
           # Use .loc to avoid SettingWithCopyWarning
           chunk = chunk.loc[(chunk[col] <= 100) & (chunk[col] >= 0)]

        # Impute missing values before appending
        for col in features:
            if chunk[col].isnull().any():
                median_val = chunk[col].median()
                chunk[col].fillna(median_val, inplace=True)
            
        chunk_list.append(chunk)

    print("Concatenating processed chunks into final DataFrame...")
    df = pd.concat(chunk_list, ignore_index=True)

    # Define features (X) and target (y)
    X = df[features]
    y = df[target]
    
    print("Preprocessing complete.")
    print(f"Final dataset shape after preprocessing: {X.shape}")
    
    return X, y, features

# --- 3. Model Training ---
def train_and_save_model(X, y, feature_names):
    """Trains the CatBoost model and saves it to a file."""
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training CatBoost model...")
    # Initialize CatBoost Regressor
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        loss_function='RMSE',
        verbose=100  # Print progress every 100 iterations
    )

    # Train the model
    model.fit(X_train, y_train, eval_set=(X_test, y_test))

    # Evaluate the model
    print("\nEvaluating model performance...")
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE) on test set: {rmse:.4f}")

    # Save the model
    print(f"Saving model to '{MODEL_FILENAME}'...")
    model.save_model(MODEL_FILENAME)
    
    # Save the column names used for training
    print(f"Saving model columns to '{COLUMNS_FILENAME}'...")
    with open(COLUMNS_FILENAME, 'w') as f:
        for col in feature_names:
            f.write(f"{col}\n")

    print("\nTraining complete and model saved successfully!")


if __name__ == "__main__":
    data_path = download_and_decompress_data(DATA_URL, COMPRESSED_FILENAME, DATA_FILENAME)
    if data_path:
        X, y, features = preprocess_data(data_path)
        train_and_save_model(X, y, features)