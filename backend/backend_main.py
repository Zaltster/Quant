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

# List of stock tickers used in the game (Matching your saved files)
GAME_STOCK_TICKERS = ['BTC-USD', 'NVDA', 'TSLA', 'sp500', 'MSFT'] # Using 'sp500'

# Paths where models and scalers are saved (relative to script location)
MODEL_PATH_TEMPLATE = "saved_models/best_{}_model.pth"
SCALER_PATH_TEMPLATE = "saved_scalers/scaler_{}.joblib"
DATA_PATH_TEMPLATE = "data/{}_max_history.csv"

# Model & Generation Parameters (MUST match the parameters used for training saved models)
sequence_length = 30  # Sequence length model was trained with
input_size = 1        # Single feature input
hidden_size = 64      # Hidden size model was trained with
num_layers = 1        # Number of layers model was trained with
output_size = 1       # Single output prediction
dropout_prob = 0.4    # Dropout used during training

# Generation parameters
NUM_STEPS_TO_GENERATE = 100 # Generate 100 steps (days) for the frontend
# --- Noise Injection Parameter ---
# Standard deviation of noise added to scaled prediction at each step
# *** Tune this value based on desired output volatility ***
GENERATION_NOISE_STD_DEV = 0.008 # Example value
# --- End Noise Injection Parameter ---

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- LSTM Model Definition ---
# (Must be the same architecture as the saved models)
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

    # Load Scaler
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            print(f"Loaded scaler for {ticker} from {scaler_path}")
        except Exception as e:
            print(f"Error loading scaler for {ticker} from {scaler_path}: {e}")
            return None, None
    else:
        print(f"Error: Scaler file not found at {scaler_path}")
        return None, None

    # Load Model
    try:
        # Ensure model architecture matches saved file
        model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob).to(device)
        if os.path.exists(model_path):
            # Using weights_only=False based on previous runs, adjust if needed/possible
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
            model.eval() # Set to evaluation mode
            print(f"Loaded model for {ticker} from {model_path}")
        else:
            print(f"Error: Model file not found at {model_path}")
            return None, scaler
    except Exception as e:
        print(f"Error loading model state_dict for {ticker} from {model_path}: {e}")
        return None, scaler

    return model, scaler

def prepare_seed_sequence(ticker, scaler, sequence_length):
    """
    Loads historical data, selects a RANDOM valid sequence,
    and prepares the scaled seed sequence.
    Uses robust CSV reading.
    """
    data_path = DATA_PATH_TEMPLATE.format(ticker)
    feature_col = 'Close' # Assuming 'Close' is the desired column name after skipping

    print(f"Loading full historical data for random seed selection from: {data_path}")
    try:
        # Use the robust read_csv call needed for potentially malformed CSVs
        df = pd.read_csv(
            data_path,
            index_col='Date',
            parse_dates=True,
            skiprows=3,                # Skip potentially problematic header lines
            names=['Date', feature_col] # Assign names explicitly
        )

        # Check if the specified feature_col exists after loading/naming
        if feature_col not in df.columns:
             raise KeyError(f"Column '{feature_col}' not found after loading/naming.")

        df = df.dropna() # Drop NaNs

    except FileNotFoundError:
        print(f"Error: Historical data file not found for {ticker} at {data_path}")
        return None
    except Exception as e:
        print(f"Error reading or processing CSV {data_path}: {e}")
        return None

    # Basic check for sufficient data length
    if len(df) < sequence_length:
        print(f"Error: Not enough historical data for {ticker} (need {sequence_length}, got {len(df)})")
        return None

    # --- Random Seed Selection ---
    max_start_index = len(df) - sequence_length
    if max_start_index < 0:
        print(f"Error: Data length {len(df)} insufficient for sequence {sequence_length}.")
        return None

    random_start_index = random.randint(0, max_start_index)
    try: # Print selected seed range for verification
        seed_start_date = df.index[random_start_index]
        seed_end_date = df.index[random_start_index + sequence_length - 1]
        print(f"Selected random seed period: {seed_start_date.strftime('%Y-%m-%d')} to {seed_end_date.strftime('%Y-%m-%d')}")
    except IndexError:
        print("Warning: Could not retrieve dates for selected random index.")

    seed_sequence_original = df.iloc[random_start_index : random_start_index + sequence_length].values
    # --- End Random Seed Selection ---

    # Scale the selected sequence
    try:
        if scaler is None or not hasattr(scaler, 'scale_'):
             print(f"Error: Invalid or unfit scaler provided for {ticker}.")
             return None
        seed_sequence_scaled = scaler.transform(seed_sequence_original)
    except Exception as e:
        print(f"Error scaling data for {ticker} seed: {e}")
        return None

    # Convert to PyTorch tensor
    seed_tensor = torch.tensor(seed_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    print(f"Prepared random seed sequence tensor for {ticker} with shape: {seed_tensor.shape}")
    return seed_tensor

def generate_sequence(model, scaler, seed_tensor, n_steps, device, noise_std=0.0):
    """Generates a sequence autoregressively, adding Gaussian noise."""
    if model is None or scaler is None or seed_tensor is None:
        print("Error: Missing model, scaler, or seed tensor.")
        return None

    model.eval()
    generated_scaled = []
    current_sequence_tensor = seed_tensor.clone()

    print(f"Generating {n_steps} steps with noise_std={noise_std}...")
    with torch.no_grad():
        for i in range(n_steps):
            prediction_scaled_tensor = model(current_sequence_tensor)

            # Add Gaussian Noise
            if noise_std > 0:
                noise = torch.randn(prediction_scaled_tensor.shape, device=device) * noise_std
                noisy_prediction_tensor = prediction_scaled_tensor + noise
                noisy_prediction_tensor.clamp_(0.0, 1.0) # Clamp to [0, 1]
                final_prediction_tensor = noisy_prediction_tensor
            else:
                final_prediction_tensor = prediction_scaled_tensor

            predicted_value_to_store = final_prediction_tensor.item()
            generated_scaled.append(predicted_value_to_store)

            next_step_tensor_for_seq = final_prediction_tensor.unsqueeze(1)

            current_sequence_tensor = torch.cat(
                (current_sequence_tensor[:, 1:, :], next_step_tensor_for_seq), dim=1
            )
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
# Simple cache for loaded models/scalers to avoid reloading on every request
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
                # Clear cache in case only one part loaded, force reload next time
                if ticker in loaded_models_cache: del loaded_models_cache[ticker]
                if ticker in loaded_scalers_cache: del loaded_scalers_cache[ticker]
                return None # Abort generation if any component fails loading
            loaded_models_cache[ticker] = model
            loaded_scalers_cache[ticker] = scaler
        else:
            print(f"Using cached model and scaler for {ticker}.")

    print("\nGenerating sequences...")
    for ticker in game_tickers:
        print(f"--- Generating for {ticker} ---")
        # Ensure components are loaded (might have failed previously but not aborted)
        if ticker not in loaded_models_cache or ticker not in loaded_scalers_cache:
             print(f"Components for {ticker} not loaded correctly. Skipping...")
             all_generated_data[ticker] = None
             generation_successful = False
             continue

        model = loaded_models_cache[ticker]
        scaler = loaded_scalers_cache[ticker]
        seed_tensor = prepare_seed_sequence(ticker, scaler, sequence_length)
        if seed_tensor is None:
            print(f"Failed to prepare seed for {ticker}. Skipping...")
            all_generated_data[ticker] = None
            generation_successful = False
            continue

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
        # Return None if any generation failed, forcing a 500 error in API
        return None

    print("\nFinished generating data for all tickers successfully.")
    return all_generated_data

# --- FastAPI Application ---
app = FastAPI()

# Define allowed origins for CORS
origins = [
    "http://localhost:3000", # Default Next.js dev port
    # Add other frontend origins if needed (e.g., deployed URL)
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---
@app.get("/game_simulation_data")
async def get_simulation_data():
    """
    API endpoint to generate and return stock simulation data for the game.
    Generates NUM_STEPS_TO_GENERATE steps for the predefined game tickers using random seeds and noise.
    """
    print(f"\nAPI Request received for /game_simulation_data")
    print(f"Attempting to generate {NUM_STEPS_TO_GENERATE} steps with noise_std={GENERATION_NOISE_STD_DEV}...")

    # Call the main generation function, passing the configured noise level
    generated_data = generate_game_data(
        GAME_STOCK_TICKERS,
        NUM_STEPS_TO_GENERATE,
        GENERATION_NOISE_STD_DEV
    )

    if generated_data:
        print("Data generation successful. Sending API response.")
        return {"status": "success", "stock_data": generated_data}
    else:
        print("Error during data generation process for API request.")
        # Raise HTTPException which FastAPI turns into a 500 error
        raise HTTPException(status_code=500, detail="Failed to generate simulation data.")

@app.get("/")
async def root():
    """Root endpoint for basic check."""
    return {"message": "LSTM Stock Generation Backend is running. Use /game_simulation_data"}

# --- Run the Backend Server ---
if __name__ == "__main__":
    print("Starting backend server with CORS enabled...")
    # Ensure prerequisites (.pth, .joblib, .csv files) are in correct subdirs defined by templates.
    # Ensure hyperparameters in this script match saved models.
    uvicorn.run(
        "backend_main:app", # Tells uvicorn where to find the FastAPI app instance
        host="127.0.0.1",   # Listen only on localhost
        port=8000,          # Standard port for development
        reload=True         # Automatically restart server on code changes
        )