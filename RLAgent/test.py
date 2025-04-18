import torch
import os

# Define the current directory and model path
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "lstm_transformer_dqn_model.pth")

# Try to load the model file
try:
    print(f"Attempting to load model from: {model_path}")
    model_state = torch.load(model_path, map_location="cpu")
    
    # Print the keys in the state dict to see its structure
    print(f"Model state type: {type(model_state)}")
    
    if isinstance(model_state, dict):
        print(f"Model state keys: {model_state.keys()}")
        
        # Check if it has the 'model' key that's causing issues
        if 'model' in model_state:
            print("Found 'model' key! Content structure:", type(model_state['model']))
        else:
            print("No 'model' key found. Available keys:", model_state.keys())
    else:
        print("Model state is not a dictionary, it's a:", type(model_state))
        
    print("Model loaded successfully!")
    
except Exception as e:
    print(f"Error loading model: {e}")