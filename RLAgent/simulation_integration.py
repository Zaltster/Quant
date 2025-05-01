"""
Integration module for LSTM-Transformer-DQN agent with the trading simulation
"""
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import uvicorn
import torch
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from pydantic import BaseModel
from typing import Dict, List, Optional

# Import the agent from dqn_agent module
from dqn_agent import SimulationAgent

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Create FastAPI app
app = FastAPI(title="LSTM-Transformer-DQN Trading Bot API")

# Define allowed origins for CORS
origins = [
    "http://localhost:3000",  # Default Next.js dev port
    "http://127.0.0.1:3000"
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
active_connections = []
dqn_agent = None
import os
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lstm_transformer_dqn_model_active.pth")

# Default simulation parameters - used if backend_main.py is not available
GAME_STOCK_TICKERS = ['BTC-USD', 'NVDA', 'TSLA', 'sp500', 'MSFT']
NUM_STEPS_TO_GENERATE = 100
GENERATION_NOISE_STD_DEV = 0.008

# Load the DQN agent if model exists
@app.on_event("startup")
async def startup_event():
    global dqn_agent
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for model at: {os.path.abspath(model_path)}")
    
    try:
        # Always create the agent, even if model loading fails
        # The SimulationAgent class now has fallback behavior for random actions
        dqn_agent = SimulationAgent(model_path)
        print(f"Agent initialized. Model loaded: {getattr(dqn_agent, 'model_loaded', False)}")
    except Exception as e:
        print(f"Error initializing agent: {e}")
        import traceback
        traceback.print_exc()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# WebSocket endpoint for real-time trading bot interaction
@app.websocket("/ws/trading_bot")
async def trading_bot_websocket(websocket: WebSocket):
    await manager.connect(websocket)
    global dqn_agent
    
    # Reset agent state at the start of a new connection
    if dqn_agent:
        dqn_agent.reset()
    
    try:
        while True:
            # Receive message from frontend
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Process message based on type
            if message["type"] == "PRICE_UPDATE":
                # New price data from simulation
                current_price = message["data"]["price"]
                current_step = message["data"]["step"]
                agent_balance = message["data"]["agent_balance"]
                agent_shares = message["data"]["agent_shares"]
                
                # Only run agent if it's loaded
                if dqn_agent is not None:
                    try:
                        # Get agent's decision
                        action = dqn_agent.decide_action(
                            current_price, 
                            agent_balance, 
                            agent_shares
                        )
                        
                        # Map action to name
                        action_names = ["HOLD", "BUY", "SELL"]
                        
                        # Send agent's decision back to frontend
                        await websocket.send_json({
                            "type": "AGENT_DECISION",
                            "data": {
                                "action": int(action),
                                "action_name": action_names[action],
                                "step": current_step,
                                "price": current_price
                            }
                        })
                        
                        # Log the decision
                        print(f"Step {current_step}: Agent decided to {action_names[action]} at price ${current_price:.2f}")
                    
                    except Exception as e:
                        print(f"Error in agent decision: {e}")
                        await websocket.send_json({
                            "type": "AGENT_ERROR",
                            "data": {
                                "message": f"Error in agent decision: {str(e)}"
                            }
                        })
                else:
                    await websocket.send_json({
                        "type": "AGENT_ERROR",
                        "data": {
                            "message": "Agent not loaded. Please train the model first."
                        }
                    })
            
            elif message["type"] == "SIMULATION_COMPLETE":
                # Simulation has finished, calculate performance
                agent_final_value = message["data"]["agent_portfolio_value"]
                user_final_value = message["data"]["user_portfolio_value"]
                initial_value = message["data"]["initial_balance"]
                
                agent_roi = (agent_final_value - initial_value) / initial_value * 100
                user_roi = (user_final_value - initial_value) / initial_value * 100
                
                # Reset agent state for next session
                if dqn_agent:
                    dqn_agent.reset()
                
                await websocket.send_json({
                    "type": "PERFORMANCE_SUMMARY",
                    "data": {
                        "agent_final_value": agent_final_value,
                        "user_final_value": user_final_value,
                        "agent_roi": agent_roi,
                        "user_roi": user_roi,
                        "winner": "agent" if agent_roi > user_roi else "user" if user_roi > agent_roi else "tie"
                    }
                })
                
                # Log the performance
                print(f"\nSimulation Complete:")
                print(f"Agent ROI: {agent_roi:.2f}%, Final Value: ${agent_final_value:.2f}")
                print(f"User ROI: {user_roi:.2f}%, Final Value: ${user_final_value:.2f}")
                print(f"Winner: {'Agent' if agent_roi > user_roi else 'User' if user_roi > agent_roi else 'Tie'}")
            
            elif message["type"] == "RESET_AGENT":
                # Reset agent state
                if dqn_agent:
                    dqn_agent.reset()
                    await websocket.send_json({
                        "type": "AGENT_RESET",
                        "data": {
                            "message": "Agent state reset successfully"
                        }
                    })
                else:
                    await websocket.send_json({
                        "type": "AGENT_ERROR",
                        "data": {
                            "message": "No agent loaded to reset"
                        }
                    })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in manager.active_connections:
            manager.disconnect(websocket)

# API endpoint to check if agent is ready
@app.get("/agent_status")
async def get_agent_status():
    """Check if the LSTM-Transformer-DQN agent is loaded and ready"""
    return {
        "agent_loaded": dqn_agent is not None,
        "model_exists": os.path.exists(model_path),
        "model_path": model_path
    }

# Data model for training request
class TrainingRequest(BaseModel):
    ticker: str = "BTC-USD"
    train_start: str = "2020-01-01"
    train_end: str = "2021-12-31"
    episodes: int = 50
    window_size: int = 10
    initial_balance: float = 100000.0
    transaction_fee: float = 0.001

# API endpoint to train agent
@app.post("/train_agent")
async def train_agent(request: TrainingRequest):
    """Train a new LSTM-Transformer-DQN agent"""
    try:
        # Import here to avoid circular imports
        from dqn_training import run_dqn_trading
        
        # Run training with parameters from request
        agent, train_metrics, test_results, trade_history = run_dqn_trading(
            ticker=request.ticker,
            train_start=request.train_start,
            train_end=request.train_end,
            episodes=request.episodes,
            window_size=request.window_size,
            initial_balance=request.initial_balance,
            transaction_fee=request.transaction_fee
        )
        
        # Load the newly trained agent
        global dqn_agent, model_path
        
        # Find the saved model path from the output
        result_dirs = [d for d in os.listdir() if d.startswith(f"results_{request.ticker.replace('-', '_')}")]
        if result_dirs:
            # Get the most recent result directory
            latest_dir = max(result_dirs)
            model_path = f"{latest_dir}/lstm_transformer_dqn_model.pth"
            dqn_agent = SimulationAgent(model_path)
            
            return {
                "success": True,
                "message": f"Agent trained successfully! ROI: {test_results['roi']:.2f}%",
                "model_path": model_path,
                "results_dir": latest_dir,
                "test_results": test_results
            }
        else:
            return {
                "success": False,
                "message": "Training completed but couldn't find the saved model directory."
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training agent: {str(e)}")

# API endpoint to get training results if available
@app.get("/training_results")
async def get_training_results():
    """Get results from the most recent training run if available"""
    result_dirs = [d for d in os.listdir() if d.startswith("results_")]
    
    if not result_dirs:
        return {
            "available": False,
            "message": "No training results found"
        }
    
    # Get the most recent result directory
    latest_dir = max(result_dirs)
    
    # Check for results files
    training_metrics_path = f"{latest_dir}/training_metrics.json"
    test_results_path = f"{latest_dir}/test_results.json"
    
    results = {"available": True, "directory": latest_dir}
    
    if os.path.exists(training_metrics_path):
        with open(training_metrics_path, 'r') as f:
            results["training_metrics"] = json.load(f)
            
    if os.path.exists(test_results_path):
        with open(test_results_path, 'r') as f:
            results["test_results"] = json.load(f)
    
    return results

# Function to generate stock data for simulation
def generate_stock_data(tickers, num_steps, noise_level):
    """Generate simulated stock data for the frontend"""
    stock_data = {}
    
    for ticker in tickers:
        # Try to get real data as a starting point
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if len(data) > num_steps:
                # Use a random segment of real data
                start_idx = random.randint(0, len(data) - num_steps)
                prices = data['Close'].values[start_idx:start_idx + num_steps].tolist()
            else:
                # Not enough data, generate synthetic
                last_price = data['Close'].iloc[-1] if len(data) > 0 else 100.0
                prices = [last_price]
                for _ in range(num_steps - 1):
                    # Random walk with drift
                    change = random.normalvariate(0.0001, noise_level)  # Small positive drift
                    new_price = prices[-1] * (1 + change)
                    prices.append(new_price)
        except Exception as e:
            print(f"Error fetching {ticker} data: {e}. Generating synthetic data.")
            # Generate synthetic data
            if ticker == "BTC-USD":
                base_price = 40000.0
            elif ticker == "NVDA":
                base_price = 800.0
            elif ticker == "TSLA": 
                base_price = 200.0
            elif ticker == "sp500":
                base_price = 5000.0
            elif ticker == "MSFT":
                base_price = 400.0
            else:
                base_price = 100.0
                
            prices = [base_price]
            for _ in range(num_steps - 1):
                change = random.normalvariate(0.0001, noise_level)
                new_price = prices[-1] * (1 + change)
                prices.append(max(0.01, new_price))  # Ensure price doesn't go negative
        
        stock_data[ticker] = prices
    
    return stock_data

# API endpoint to integrate agent with the frontend simulation
@app.get("/game_simulation_data")
async def get_simulation_data():
    """API endpoint compatible with the original backend to generate simulation data"""
    try:
        # Generate our own data
        print("Generating simulation data for the frontend")
        stock_data = generate_stock_data(
            GAME_STOCK_TICKERS,
            NUM_STEPS_TO_GENERATE,
            GENERATION_NOISE_STD_DEV
        )
        return {"status": "success", "stock_data": stock_data}
    except Exception as e:
        print(f"Error generating simulation data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate simulation data: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "LSTM-Transformer-DQN Trading Bot API",
        "version": "1.0.0",
        "description": "API for training and using an LSTM-Transformer-DQN agent for stock trading",
        "endpoints": {
            "/": "This information",
            "/agent_status": "Check if the agent is loaded",
            "/train_agent": "Train a new agent (POST)",
            "/training_results": "Get results from the most recent training",
            "/game_simulation_data": "Get simulation data compatible with the frontend",
            "/ws/trading_bot": "WebSocket for real-time agent interaction"
        }
    }

# Main entry point
if __name__ == "__main__":
    print("Starting LSTM-Transformer-DQN Trading Bot API...")
    uvicorn.run(
        "simulation_integration:app",
        host="127.0.0.1",
        port=8001,
        reload=True
    )