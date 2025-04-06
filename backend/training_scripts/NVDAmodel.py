# Filename: train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import joblib # Make sure joblib is imported
import yfinance as yf # Import yfinance for download part

# --- Configuration & Hyperparameters ---
# >>> CHANGE THIS LINE FOR EACH STOCK YOU WANT TO TRAIN <<<
TARGET_TICKER_INPUT = 'NVDA' # Options: 'MSFT', 'JNJ', 'NEE', 'TSLA', 'NVDA', '^GSPC'
# >>>--------------------------------------------------<<<

# Use consistent naming for files, handling ^GSPC case
TARGET_TICKER_FILENAME = 'sp500' if TARGET_TICKER_INPUT == '^GSPC' else TARGET_TICKER_INPUT

# Construct filenames based on the target ticker
data_csv_path = f"data/{TARGET_TICKER_FILENAME}_max_history.csv"
best_model_path = f"saved_models/best_{TARGET_TICKER_FILENAME}_model.pth"
scaler_path = f"saved_scalers/scaler_{TARGET_TICKER_FILENAME}.joblib"

feature_col = 'Close'         # Feature column name

# --- Model & Training Hyperparameters ---
sequence_length = 30
input_size = 1
hidden_size = 64
num_layers = 1
output_size = 1
dropout_prob = 0.4
learning_rate = 0.0005
weight_decay = 1e-5
num_epochs = 500
batch_size = 32
patience = 20
# ----------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- Training for Ticker: {TARGET_TICKER_INPUT} (Files: {TARGET_TICKER_FILENAME}) ---")
print(f"Using device: {device}")


# --- 1. Data Loading and Preprocessing Function ---
def load_and_process_single_stock_data(ticker_input, ticker_filename, csv_path, scaler_path, feature_col, sequence_length):
    """Downloads if needed, loads, processes, scales, saves scaler, creates sequences, splits."""
    print(f"Processing data for: {ticker_input}...")

    # --- Download/Load Data ---
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}. Downloading data for {ticker_input}...")
        try:
            df_download = yf.download(ticker_input, period="max", interval="1d", progress=False) # Suppress progress bar inside function
            if df_download.empty: raise ValueError(f"No data downloaded for {ticker_input}.")

            if feature_col not in df_download.columns:
                if 'Adj Close' in df_download.columns: feature_col = 'Adj Close'
                else: raise KeyError(f"Required column ('Close' or 'Adj Close') not found.")

            df = df_download[[feature_col]].copy()
            df = df.sort_index().dropna()
            if df.empty: raise ValueError("No data after download/processing.")
            df.to_csv(csv_path, index=True)
            print(f"Data downloaded and saved to {csv_path}")
        except Exception as e:
            print(f"Error downloading or saving data for {ticker_input}: {e}")
            exit()
    else:
         print(f"Loading existing data from: {csv_path}...")
         try:
             # *** CORRECTED pd.read_csv ***
             # Reads the standard CSV saved by download script or previous run
             df = pd.read_csv(
                 csv_path,
                 index_col='Date',          # Use 'Date' column after naming
                 parse_dates=True,          # Parse the index as dates
                 skiprows=3,                # Skip the first 3 rows
                 names=['Date', feature_col] # Provide names for columns from row 4 onwards
             )
             # ******************************

             if feature_col not in df.columns: # Check if expected column is there
                 # Try 'Adj Close' if 'Close' is missing in existing file
                 if 'Adj Close' in df.columns: feature_col = 'Adj Close'
                 else: raise KeyError(f"Column '{feature_col}' or 'Adj Close' not found in {csv_path}")
             df = df[[feature_col]] # Select only the target column
             df = df.sort_index().dropna()
             if df.empty: raise ValueError("CSV file loaded but contains no usable data.")
         except Exception as e:
             print(f"Error loading or processing data from {csv_path}: {e}")
             exit()

    print(f"Loaded data shape: {df.shape}")

    # --- Train/Val/Test Split ---
    total_days = len(df)
    if total_days < sequence_length * 3: # Basic check for enough data
         print(f"Error: Not enough data ({total_days} days) to create train/val/test sequences of length {sequence_length}")
         exit()
    train_end_idx = int(total_days * 0.70)
    val_end_idx = train_end_idx + int(total_days * 0.15)
    df_train = df[:train_end_idx]
    df_val = df[train_end_idx:val_end_idx]
    df_test = df[val_end_idx:]
    print(f"\nData Splits: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    # --- Scaling & Saving Scaler ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(df_train) # Fit ONLY on training data
    try:
        # *** ADDED SCALER SAVING ***
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
        # ***************************
    except Exception as e:
        print(f"Error saving scaler to {scaler_path}: {e}")
        # Decide if you want to exit or continue without saving scaler
        exit()
    scaled_train_data = scaler.transform(df_train)
    scaled_val_data = scaler.transform(df_val)
    scaled_test_data = scaler.transform(df_test)

    # --- Create Sequences ---
    def create_sequences(input_data, seq_length):
        X_seq, y_seq = [], []
        for i in range(len(input_data) - seq_length):
            X_seq.append(input_data[i:(i + seq_length), :])
            y_seq.append(input_data[i + seq_length, 0])
        return np.array(X_seq), np.array(y_seq)

    X_train, y_train = create_sequences(scaled_train_data, sequence_length)
    X_val, y_val = create_sequences(scaled_val_data, sequence_length)
    X_test, y_test = create_sequences(scaled_test_data, sequence_length)
    print(f"\nSequence Splits:")
    print(f"Training sequences: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation sequences: X={X_val.shape}, y={y_val.shape}")
    print(f"Test sequences: X={X_test.shape}, y={y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, df_train, df_val, df_test, scaler

# --- 2. PyTorch Dataset Class ---
# (Keep TimeSeriesDataset class definition as before)
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- 3. LSTM Model Architecture ---
# (Keep LSTMModel class definition as before)
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

# --- Main Execution ---
if __name__ == "__main__": # Ensure this block runs when script is executed
    # Load and process data
    X_train, y_train, X_val, y_val, X_test, y_test, \
    df_train, df_val, df_test, loaded_scaler = \
        load_and_process_single_stock_data(
            TARGET_TICKER_INPUT, TARGET_TICKER_FILENAME, data_csv_path, scaler_path, feature_col, sequence_length
        )

    # Create Datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the model
    model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob).to(device)
    print("\nModel Architecture:")
    print(model)

    # Define Loss function and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # --- Training Loop ---
    print("\nStarting Training with Early Stopping...")
    train_losses = []
    val_losses = []
    best_val_loss = np.Inf
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}')

        # Early Stopping Check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f'Validation loss decreased ({best_val_loss:.6f}). Saving model to {best_model_path}...')
        else:
            epochs_no_improve += 1
            # Optional: add print statement here if desired
            # print(f'Validation loss did not improve for {epochs_no_improve} epoch(s).')

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            break

    print("Training Finished.")

    # --- Load Best Model ---
    if os.path.exists(best_model_path):
        print(f"\nLoading best model weights from {best_model_path} for final evaluation...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("\nWarning: No best model checkpoint found. Evaluating the final model state (from last epoch).")

    # --- Plot Losses ---
    plt.figure(figsize=(10, 5))
    epochs_trained = len(train_losses)
    plt.plot(range(epochs_trained), train_losses, label='Training Loss')
    plt.plot(range(epochs_trained), val_losses, label='Validation Loss')
    plt.title(f'Model Training & Validation Loss ({TARGET_TICKER_INPUT} - Stopped at Epoch {epochs_trained})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    if epochs_trained > patience and epochs_no_improve >= patience:
        best_epoch = epochs_trained - patience -1
        if best_epoch >= 0: # Ensure index is valid
             plt.axvline(best_epoch, color='r', linestyle='--', label=f'Best Val Epoch ~{best_epoch+1}')
    plt.legend()
    plt.show()


    # --- Final Evaluation on Test Set ---
    print("\nEvaluating on Test Set with Best Model...")
    model.eval()
    test_loss = 0.0
    predictions_scaled = []
    actuals_scaled = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            predictions_scaled.append(outputs.cpu().numpy())
            actuals_scaled.append(targets.cpu().numpy())

    final_test_loss = test_loss / len(test_loader.dataset)
    print(f'Final Test Loss (MSE - Scaled): {final_test_loss:.6f}')

    # --- Inverse Transform ---
    predictions_scaled = np.concatenate(predictions_scaled, axis=0)
    actuals_scaled = np.concatenate(actuals_scaled, axis=0)
    try:
        predictions_original = loaded_scaler.inverse_transform(predictions_scaled)
        actuals_original = loaded_scaler.inverse_transform(actuals_scaled)
        mae = mean_absolute_error(actuals_original, predictions_original)
        print(f"Test MAE (Original Scale, Target: {feature_col}): {mae:.4f}")

        # --- Plot Test Set Predictions ---
        test_dates = df_test.index[sequence_length:]
        if len(test_dates) == len(actuals_original) == len(predictions_original):
            plt.figure(figsize=(14, 7))
            plt.plot(test_dates, actuals_original, label='Actual Prices', linewidth=2)
            plt.plot(test_dates, predictions_original, label='Predicted Prices', alpha=0.8, linestyle='--')
            plt.title(f'Test Set: Actual vs Predicted Prices ({TARGET_TICKER_INPUT}) - Best Model')
            plt.xlabel('Date')
            plt.ylabel(f'{feature_col} Price')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print("\nWarning: Length mismatch plotting test results.")

    except Exception as e:
        print(f"\nError during inverse transform or MAE calculation: {e}")
        print("Ensure the scaler was loaded correctly and matches the data.")


    # --- Generation Placeholder ---
    print("\nModel training and evaluation complete. Ready for generation using the best model.")