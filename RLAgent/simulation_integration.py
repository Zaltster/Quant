# simulation_integration.py
# Module to integrate the DQN agent with your simulation
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import uvicorn
import torch
import os

from dqn_agent import SimulationAgent

# Create FastAPI app
app = FastAPI()

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
simulation_running = False
dqn_agent = None
current_balance = 100000.0
current_shares = 0.0
trade_history = []
step = 0
agent_performance = 0.0
user_performance = 0.0

# Load the DQN agent if model exists
@app.on_event("startup")
async def startup_event():
    global dqn_agent
    if os.path.exists("dqn_trading_model.pth"):
        try:
            dqn_agent = SimulationAgent("dqn_trading_model.pth")
            print("DQN agent loaded successfully!")
        except Exception as e:
            print(f"Error loading DQN agent: {e}")
    else:
        print("No DQN model found. Please train the model first.")

# WebSocket connection for real-time communication with frontend
@app.websocket("/ws/trading_bot")
async def trading_bot_websocket(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
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
                
                # Only run agent if it's loaded
                if dqn_agent is not None:
                    # Get agent's decision
                    action = dqn_agent.decide_action(
                        current_price, 
                        message["data"]["agent_balance"], 
                        message["data"]["agent_shares"]
                    )
                    
                    # Send agent's decision back to frontend
                    await websocket.send_json({
                        "type": "AGENT_DECISION",
                        "data": {
                            "action": int(action),
                            "action_name": ["HOLD", "BUY", "SELL"][action],
                            "step": current_step,
                            "price": current_price
                        }
                    })
            
            elif message["type"] == "SIMULATION_COMPLETE":
                # Simulation has finished, calculate performance
                agent_final_value = message["data"]["agent_portfolio_value"]
                user_final_value = message["data"]["user_portfolio_value"]
                initial_value = message["data"]["initial_balance"]
                
                agent_roi = (agent_final_value - initial_value) / initial_value * 100
                user_roi = (user_final_value - initial_value) / initial_value * 100
                
                await websocket.send_json({
                    "type": "PERFORMANCE_SUMMARY",
                    "data": {
                        "agent_roi": agent_roi,
                        "user_roi": user_roi,
                        "winner": "agent" if agent_roi > user_roi else "user" if user_roi > agent_roi else "tie"
                    }
                })
    
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

# API endpoint to check if agent is ready
@app.get("/agent_status")
async def get_agent_status():
    return {
        "agent_loaded": dqn_agent is not None,
        "model_exists": os.path.exists("dqn_trading_model.pth")
    }

# API endpoint to train agent with default parameters (will run the training script)
@app.post("/train_agent")
async def train_agent():
    try:
        # This would typically be a background task, but for simplicity:
        from dqn_training import run_dqn_trading
        
        # Run training with default parameters
        _, _, final_value, roi = run_dqn_trading(episodes=50)
        
        # Load the newly trained agent
        global dqn_agent
        dqn_agent = SimulationAgent("dqn_trading_model.pth")
        
        return {
            "success": True,
            "message": f"Agent trained successfully! ROI: {roi:.2f}%",
            "final_value": final_value,
            "roi": roi
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training agent: {str(e)}")

# Main entrypoint
if __name__ == "__main__":
    uvicorn.run("simulation_integration:app", host="127.0.0.1", port=8001, reload=True)