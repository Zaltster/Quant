# Filename: backend_main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib # For loading the scaler
import os
import yfinance as yf
from fastapi import FastAPI, HTTPException
from datetime import datetime
import uvicorn # For running the app
from fastapi.middleware.cors import CORSMiddleware
import random # For random seed selection

# --- Configuration ---

GAME_STOCK_TICKERS = ['BTC-USD', 'NVDA', 'TSLA', 'sp500', 'MSFT']
MODEL_PATH_TEMPLATE = "saved_models/best_{}_model.pth"
SCALER_PATH_TEMPLATE = "saved_scalers/scaler_{}.joblib"
DATA_PATH_TEMPLATE = "data/{}_max_history.csv"

# Model & Generation Parameters
sequence_length = 30
input_size = 1
hidden_size = 64
num_layers = 1
output_size = 1
dropout_prob = 0.4

# Generation parameters
NUM_STEPS_TO_GENERATE = 100
# --- Noise Injection Parameter ---
# Standard deviation of noise added to scaled prediction at each step
# *** This is a key parameter to TUNE! Start small. ***
# Value depends on the typical variance of your SCALED data's daily changes.
# Maybe estimate std dev of scaled_train_data.diff().std() during training.
GENERATION_NOISE_STD_DEV = 0.008 # Example value, adjust based on results
# --- End Noise Injection Parameter ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- LSTM Model Definition ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        lstm_dropout = dropout_prob if num_layers > 1 else 0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=lstm_dropout)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# --- Utility Functions ---

def load_model_and_scaler(ticker):
    """Loads the saved model state and scaler for a given ticker."""
    model_path = MODEL_PATH_TEMPLATE.format(ticker)
    scaler_path = SCALER_PATH_TEMPLATE.format(ticker)
    model = None
    scaler = None
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            print(f"Loaded scaler for {ticker} from {scaler_path}")
        except Exception as e:
            print(f"Error loading scaler for {ticker}: {e}")
            return None, None
    else:
        print(f"Error: Scaler file not found at {scaler_path}")
        return None, None
    try:
        model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob).to(device)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)) # Set weights_only=False if needed
            model.eval()
            print(f"Loaded model for {ticker} from {model_path}")
        else:
            print(f"Error: Model file not found at {model_path}")
            return None, scaler
    except Exception as e:
        print(f"Error loading model state_dict for {ticker}: {e}")
        return None, scaler
    return model, scaler

def prepare_seed_sequence(ticker, scaler, sequence_length):
    """Loads data, selects random sequence, returns scaled seed tensor."""
    data_path = DATA_PATH_TEMPLATE.format(ticker)
    feature_col = 'Close'
    print(f"Loading full historical data for random seed selection from: {data_path}")
    try:
        # Using robust read_csv needed for user's files
        df = pd.read_csv(
            data_path, index_col='Date', parse_dates=True,
            skiprows=3, names=['Date', feature_col]
        )
        if feature_col not in df.columns: raise KeyError(f"Column '{feature_col}' not found.")
        df = df.dropna()
    except FileNotFoundError:
        print(f"Error: File not found {data_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV {data_path}: {e}")
        return None
    if len(df) < sequence_length:
        print(f"Error: Not enough data for {ticker} (need {sequence_length}, got {len(df)})")
        return None
    max_start_index = len(df) - sequence_length
    if max_start_index < 0: return None
    random_start_index = random.randint(0, max_start_index)
    try:
        seed_start_date = df.index[random_start_index]
        seed_end_date = df.index[random_start_index + sequence_length - 1]
        print(f"Selected random seed period: {seed_start_date.strftime('%Y-%m-%d')} to {seed_end_date.strftime('%Y-%m-%d')}")
    except IndexError: print("Warning: Could not retrieve dates for index.")
    seed_sequence_original = df.iloc[random_start_index : random_start_index + sequence_length].values
    try:
        if scaler is None or not hasattr(scaler, 'scale_'): return None
        seed_sequence_scaled = scaler.transform(seed_sequence_original)
    except Exception as e:
        print(f"Error scaling data for {ticker} seed: {e}")
        return None
    seed_tensor = torch.tensor(seed_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    print(f"Prepared random seed sequence tensor for {ticker} with shape: {seed_tensor.shape}")
    return seed_tensor

def generate_sequence(model, scaler, seed_tensor, n_steps, device, noise_std=0.0):
    """Generates a sequence autoregressively, optionally adding noise."""
    if model is None or scaler is None or seed_tensor is None:
        print("Error: Missing model, scaler, or seed tensor.")
        return None

    model.eval()
    generated_scaled = []
    current_sequence_tensor = seed_tensor.clone()

    print(f"Generating {n_steps} steps with noise_std={noise_std}...")
    with torch.no_grad():
        for i in range(n_steps):
            prediction_scaled_tensor = model(current_sequence_tensor) # Original prediction tensor

            # --- Add Gaussian Noise ---
            if noise_std > 0:
                noise = torch.randn(prediction_scaled_tensor.shape, device=device) * noise_std
                noisy_prediction_tensor = prediction_scaled_tensor + noise
                # Optional: Clamp to [0, 1] - prevents extreme values from noise
                noisy_prediction_tensor.clamp_(0.0, 1.0)
                final_prediction_tensor = noisy_prediction_tensor
            else:
                # No noise added if noise_std is 0 or less
                final_prediction_tensor = prediction_scaled_tensor
            # --- End Noise Addition ---

            predicted_value_to_store = final_prediction_tensor.item() # Get scalar value
            generated_scaled.append(predicted_value_to_store)

            # Prepare prediction tensor for appending (shape [1, 1, 1])
            next_step_tensor_for_seq = final_prediction_tensor.unsqueeze(1)

            # Update sequence: remove oldest, append prediction
            current_sequence_tensor = torch.cat(
                (current_sequence_tensor[:, 1:, :], next_step_tensor_for_seq), dim=1
            )
            # Progress indicator
            if (i + 1) % 20 == 0: print(f"  Step {i+1}/{n_steps}")

    # Inverse transform
    generated_scaled_np = np.array(generated_scaled).reshape(-1, 1)
    try:
        generated_original = scaler.inverse_transform(generated_scaled_np)
    except Exception as e:
        print(f"Error during inverse transform: {e}")
        return None

    print("Sequence generation complete.")
    return generated_original.flatten()

# --- Main Orchestration Function ---
loaded_models_cache = {}
loaded_scalers_cache = {}

def generate_game_data(game_tickers, num_steps, noise_level):
    """Generates data for all stocks needed for a game session."""
    all_generated_data = {}
    generation_successful = True

    print("\nLoading models and scalers (using cache if available)...")
    for ticker in game_tickers:
        if ticker not in loaded_models_cache or ticker not in loaded_scalers_cache:
            model, scaler = load_model_and_scaler(ticker)
            if model is None or scaler is None:
                print(f"FATAL: Failed to load components for {ticker}.")
                return None
            loaded_models_cache[ticker] = model
            loaded_scalers_cache[ticker] = scaler
        else:
            print(f"Using cached model and scaler for {ticker}.")

    print("\nGenerating sequences...")
    for ticker in game_tickers:
        print(f"--- Generating for {ticker} ---")
        model = loaded_models_cache[ticker]
        scaler = loaded_scalers_cache[ticker]
        seed_tensor = prepare_seed_sequence(ticker, scaler, sequence_length)
        if seed_tensor is None:
            print(f"Failed to prepare seed for {ticker}. Skipping...")
            all_generated_data[ticker] = None
            generation_successful = False
            continue

        # Pass noise level to generate_sequence
        generated_prices = generate_sequence(model, scaler, seed_tensor, num_steps, device, noise_std=noise_level)

        if generated_prices is None:
             print(f"Failed to generate sequence for {ticker}. Skipping.")
             all_generated_data[ticker] = None
             generation_successful = False
             continue
        all_generated_data[ticker] = generated_prices.tolist()
        print(f"Generated {len(all_generated_data[ticker])} steps for {ticker}.")

    if not generation_successful:
        print("\nWarning: Generation failed for one or more tickers.")
        return None

    print("\nFinished generating data for all tickers successfully.")
    return all_generated_data

# --- FastAPI Application ---
app = FastAPI()
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/game_simulation_data")
async def get_simulation_data():
    """API endpoint for game data generation with noise injection."""
    print(f"\nAPI Request received for /game_simulation_data")
    print(f"Attempting to generate {NUM_STEPS_TO_GENERATE} steps with noise_std={GENERATION_NOISE_STD_DEV}...")

    # Pass noise level from config to generation function
    generated_data = generate_game_data(GAME_STOCK_TICKERS, NUM_STEPS_TO_GENERATE, GENERATION_NOISE_STD_DEV)

    if generated_data:
        print("Data generation successful. Sending API response.")
        return {"status": "success", "stock_data": generated_data}
    else:
        print("Error during data generation process for API request.")
        raise HTTPException(status_code=500, detail="Failed to generate simulation data.")

@app.get("/")
async def root():
    return {"message": "LSTM Stock Generation Backend is running. Use /game_simulation_data"}

# --- Run the Backend Server ---
if __name__ == "__main__":
    print("Starting backend server with CORS enabled...")
    uvicorn.run("backend_main:app", host="127.0.0.1", port=8000, reload=True)