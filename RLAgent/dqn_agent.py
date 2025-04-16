"""
DQN Agent with LSTM and Transformer components for trading
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import math

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformer Position Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Self-Attention Module
class AttentionModule(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super(AttentionModule, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=num_heads,
            batch_first=True
        )
        
        # Layer normalization and feedforward
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, x):
        # Apply multi-head attention with residual connection and layer norm
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Apply feedforward with residual connection and layer norm
        ff_output = self.feedforward(x)
        x = self.norm2(x + ff_output)
        
        return x

# LSTM-Transformer-DQN Model
class LSTMTransformerDQNModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128, lstm_layers=1, 
                 transformer_layers=2, num_heads=4, dropout_prob=0.1):
        super(LSTMTransformerDQNModel, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        
        # Input embedding
        self.embedding = nn.Linear(state_size, hidden_size)
        
        # LSTM for sequential processing and memory
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_prob if lstm_layers > 1 else 0
        )
        
        # Positional encoding for transformer
        self.pos_encoder = PositionalEncoding(hidden_size)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            AttentionModule(hidden_size, num_heads) 
            for _ in range(transformer_layers)
        ])
        
        # Output layers
        self.dropout = nn.Dropout(dropout_prob)
        self.output = nn.Linear(hidden_size, action_size)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # If input is 2D (batch_size, state_size), reshape to 3D for sequence processing
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, state_size)
        
        # Initial embedding
        x = self.embedding(x)  # (batch_size, seq_len, hidden_size)
        
        # LSTM processing - this maintains long and short-term memory
        if hidden is None:
            h0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(x.device)
            hidden = (h0, c0)
        
        lstm_out, hidden = self.lstm(x, hidden)  # lstm_out: (batch_size, seq_len, hidden_size)
        
        # Add positional encoding for transformer
        pos_encoded = self.pos_encoder(lstm_out)
        
        # Pass through transformer blocks - this applies attention
        transformer_out = pos_encoded
        for transformer_block in self.transformer_blocks:
            transformer_out = transformer_block(transformer_out)
        
        # Take the output of the last step in the sequence
        out = transformer_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply dropout and output layer
        out = self.dropout(out)
        q_values = self.output(out)  # (batch_size, action_size)
        
        return q_values, hidden

# Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Trading Environment
class TradingEnvironment:
    def __init__(self, data, initial_balance=100000, transaction_fee_percent=0.001, window_size=10):
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.window_size = window_size
        self.reset()
        
    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.trade_history = []
        
        # Use a window of historical prices for the state
        price_window = min(self.window_size, len(self.data))
        price_history = []
        
        # Fill price history with first price for padding if needed
        first_price = self.data[0]
        if isinstance(first_price, (list, np.ndarray)):
            first_price = float(first_price[0])
        
        # Create price history with proper padding
        for i in range(self.window_size - price_window):
            price_history.append(first_price)
            
        # Add actual prices
        for i in range(price_window):
            price = self.data[i]
            if isinstance(price, (list, np.ndarray)):
                price = float(price[0])
            price_history.append(price)
        
        # State representation: [normalized balance, normalized shares_held, normalized prices]
        self.state = self._get_normalized_state(self.balance, self.shares_held, price_history)
        return self.state
    
    def _get_normalized_state(self, balance, shares_held, price_history):
        # Normalize balance: 0 to 2x initial balance range becomes 0 to 1
        norm_balance = balance / (2 * self.initial_balance)
        
        # Normalize shares: 0 to balance/current_price range becomes 0 to 1
        current_price = price_history[-1]
        
        # Make sure current_price is a scalar
        if isinstance(current_price, (list, np.ndarray)):
            current_price = float(current_price[0])
            
        max_shares = self.initial_balance / current_price if current_price > 0 else 1
        norm_shares = shares_held / max_shares if max_shares > 0 else 0
        
        # Normalize prices: use min-max scaling over price history
        # Ensure all prices are scalar values
        price_history_scalar = []
        for p in price_history:
            if isinstance(p, (list, np.ndarray)):
                price_history_scalar.append(float(p[0]))
            else:
                price_history_scalar.append(float(p))
                
        min_price = min(price_history_scalar)
        max_price = max(price_history_scalar)
        price_range = max_price - min_price
        
        norm_prices = [(p - min_price) / price_range if price_range > 0 else 0.5 for p in price_history_scalar]
        
        return [norm_balance, norm_shares] + norm_prices
    
    def step(self, action):
        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.current_step += 1
        current_price = self.data[self.current_step - 1]
        
        # Ensure current_price is a scalar
        if isinstance(current_price, (list, np.ndarray)):
            current_price = float(current_price[0])
            
        prev_portfolio_value = self.balance + self.shares_held * current_price
        
        # Execute action
        if action == 1:  # Buy
            # Calculate maximum shares that can be bought
            max_shares = self.balance / (current_price * (1 + self.transaction_fee_percent))
            # Buy all possible shares (simplified strategy)
            shares_to_buy = max_shares
            cost = shares_to_buy * current_price * (1 + self.transaction_fee_percent)
            
            if shares_to_buy > 0:
                self.shares_held += shares_to_buy
                self.balance -= cost
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'buy',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'cost': cost,
                    'balance': self.balance
                })
        
        elif action == 2:  # Sell
            # Sell all shares (simplified strategy)
            if self.shares_held > 0:
                proceeds = self.shares_held * current_price * (1 - self.transaction_fee_percent)
                self.balance += proceeds
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'shares': self.shares_held,
                    'price': current_price,
                    'proceeds': proceeds,
                    'balance': self.balance
                })
                self.shares_held = 0
        
        # Calculate reward (percentage change in portfolio value)
        next_price = self.data[self.current_step] if self.current_step < len(self.data) else current_price
        
        # Ensure next_price is a scalar
        if isinstance(next_price, (list, np.ndarray)):
            next_price = float(next_price[0])
            
        # Get new portfolio value
        new_portfolio_value = self.balance + self.shares_held * next_price
        
        # Reward is the percentage change in portfolio value
        reward = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > 0 else 0
        
        # Check if done
        done = self.current_step >= len(self.data) - 1
        
        # Update state
        if not done:
            start_idx = max(0, self.current_step - self.window_size + 1)
            end_idx = self.current_step + 1
            
            price_history = []
            for i in range(start_idx, end_idx):
                price = self.data[i]
                if isinstance(price, (list, np.ndarray)):
                    price = float(price[0])
                price_history.append(price)
                
            # Pad if needed
            if len(price_history) < self.window_size:
                first_price = price_history[0] if price_history else 0
                padding = [first_price] * (self.window_size - len(price_history))
                price_history = padding + price_history
                
            self.state = self._get_normalized_state(self.balance, self.shares_held, price_history)
            
        return self.state, reward, done, {"portfolio_value": new_portfolio_value}

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=128, window_size=10,
                lstm_layers=1, transformer_layers=2, num_heads=4,
                learning_rate=0.001, gamma=0.99, epsilon=1.0, 
                epsilon_min=0.01, epsilon_decay=0.995, 
                memory_size=10000, batch_size=64):
        
        self.state_size = state_size
        self.action_size = action_size
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Create main network and target network
        self.model = LSTMTransformerDQNModel(
            state_size=state_size, 
            action_size=action_size,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            transformer_layers=transformer_layers,
            num_heads=num_heads
        ).to(device)
        
        self.target_model = LSTMTransformerDQNModel(
            state_size=state_size, 
            action_size=action_size,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            transformer_layers=transformer_layers,
            num_heads=num_heads
        ).to(device)
        
        self.update_target_model()
        
        # LSTM states for sequence processing
        self.lstm_hidden = None
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Setup replay memory
        self.memory = ReplayMemory(memory_size)
        self.train_start = batch_size
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def get_action(self, state, is_eval=False):
        # Epsilon-greedy action selection
        if not is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Convert state to sequence form if it's not already
        if isinstance(state, list):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)  # (1, state_size)
        
        # Reshape for sequence processing if needed
        if len(state.shape) == 2:
            state = state.unsqueeze(1)  # (batch_size, 1, state_size)
        
        with torch.no_grad():
            q_values, self.lstm_hidden = self.model(state, self.lstm_hidden)
            
        return q_values.cpu().data.numpy()[0].argmax()
    
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        
        # Sample batch from replay memory
        batch = self.memory.sample(self.batch_size)
        
        # Extract batch components and convert to tensors
        states = torch.FloatTensor([batch[i][0] for i in range(len(batch))]).to(device)
        actions = torch.LongTensor([batch[i][1] for i in range(len(batch))]).to(device)
        next_states = torch.FloatTensor([batch[i][2] for i in range(len(batch))]).to(device)
        rewards = torch.FloatTensor([batch[i][3] for i in range(len(batch))]).to(device)
        dones = torch.FloatTensor([batch[i][4] for i in range(len(batch))]).to(device)
        
        # Reshape for sequence processing
        states = states.unsqueeze(1)  # (batch_size, 1, state_size)
        next_states = next_states.unsqueeze(1)  # (batch_size, 1, state_size)
        
        # Get current Q values
        current_q, _ = self.model(states)
        current_q = current_q.gather(1, actions.unsqueeze(1))
        
        # Get max Q' values for next states from target model
        with torch.no_grad():
            next_q, _ = self.target_model(next_states)
            max_next_q = next_q.max(1)[0]
        
        # Calculate target Q values
        target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Calculate loss and update model
        loss = self.criterion(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients (common for LSTM)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()
    
    def reset_lstm_state(self):
        """Reset the LSTM hidden state (for episode boundaries)"""
        self.lstm_hidden = None
    
    def save(self, filename):
        """Save model parameters"""
        model_state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(model_state, filename)
        
    def load(self, filename):
        """Load model parameters"""
        model_state = torch.load(filename, map_location=device)
        self.model.load_state_dict(model_state['model'])
        self.target_model.load_state_dict(model_state['model'])
        self.optimizer.load_state_dict(model_state['optimizer'])
        self.epsilon = model_state['epsilon']

# SimulationAgent - For direct integration with the trading simulation
class SimulationAgent:
    def __init__(self, model_path="lstm_transformer_dqn_model.pth", window_size=10):
        # Define model parameters
        self.state_size = window_size + 2  # prices + balance + shares
        self.action_size = 3  # buy, hold, sell
        self.window_size = window_size
        
        # Initialize model
        self.model = LSTMTransformerDQNModel(
            state_size=self.state_size,
            action_size=self.action_size
        ).to(device)
        
        # Load trained model
        try:
            model_state = torch.load(model_path, map_location=device)
            self.model.load_state_dict(model_state['model'])
            self.model.eval()
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
        
        # Initialize state tracking
        self.price_history = []
        self.lstm_hidden = None
        
    def normalize_state(self, balance, shares_held, price_history):
        """Normalize the state for the agent"""
        # Normalize balance (0 to 2x initial balance)
        initial_balance = 100000
        norm_balance = balance / (2 * initial_balance)
        
        # Normalize shares (0 to initial_balance/current_price)
        current_price = price_history[-1]
        max_shares = initial_balance / current_price if current_price > 0 else 1
        norm_shares = shares_held / max_shares if max_shares > 0 else 0
        
        # Normalize prices
        min_price = min(price_history)
        max_price = max(price_history)
        price_range = max_price - min_price
        norm_prices = [(p - min_price) / price_range if price_range > 0 else 0.5 for p in price_history]
        
        return [norm_balance, norm_shares] + norm_prices
    
    def decide_action(self, current_price, balance, shares_held):
        """Decide what action to take given the current state"""
        # Update price history
        self.price_history.append(current_price)
        if len(self.price_history) > self.window_size:
            self.price_history = self.price_history[-self.window_size:]
        
        # Pad price history if needed
        if len(self.price_history) < self.window_size:
            self.price_history = [current_price] * (self.window_size - len(self.price_history)) + self.price_history
        
        # Create normalized state
        state = self.normalize_state(balance, shares_held, self.price_history)
        
        # Convert to tensor and reshape for sequence processing
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
        
        # Get action from model
        with torch.no_grad():
            q_values, self.lstm_hidden = self.model(state_tensor, self.lstm_hidden)
            
        action = q_values.cpu().data.numpy()[0].argmax()
        
        return action  # 0: hold, 1: buy, 2: sell
    
    def reset(self):
        """Reset agent state between episodes"""
        self.price_history = []
        self.lstm_hidden = None