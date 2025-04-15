# dqn_agent.py
# Core DQN agent implementation

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN Neural Network Model
class DQNModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

# Replay Memory for experience replay
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
    def __init__(self, data, initial_balance=100000, transaction_fee_percent=0.001):
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.reset()
        
    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.trade_history = []
        
        # Initial state: [balance, shares_held, current_price, price_history]
        # Use a window of 10 previous prices
        price_window = min(10, len(self.data))
        price_history = [self.data[0]] * (10 - price_window) + self.data[:price_window].tolist()
        
        # State representation: [normalized balance, normalized shares_held, normalized prices]
        self.state = self._get_normalized_state(self.balance, self.shares_held, price_history)
        return self.state
    
    def _get_normalized_state(self, balance, shares_held, price_history):
        # Normalize balance: 0 to 2x initial balance range becomes 0 to 1
        norm_balance = balance / (2 * self.initial_balance)
        
        # Normalize shares: 0 to balance/current_price range becomes 0 to 1
        current_price = price_history[-1]
        max_shares = self.initial_balance / current_price if current_price > 0 else 1
        norm_shares = shares_held / max_shares if max_shares > 0 else 0
        
        # Normalize prices: use min-max scaling over price history
        min_price = min(price_history)
        max_price = max(price_history)
        price_range = max_price - min_price
        norm_prices = [(p - min_price) / price_range if price_range > 0 else 0.5 for p in price_history]
        
        return [norm_balance, norm_shares] + norm_prices
    
    def step(self, action):
        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.current_step += 1
        current_price = self.data[self.current_step - 1]
        prev_portfolio_value = self.balance + self.shares_held * current_price
        
        # Execute action
        reward = 0
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
        
        # Calculate reward using current price
        if self.current_step < len(self.data):
            next_price = self.data[self.current_step]
        else:
            next_price = current_price
            
        # Get new portfolio value
        new_portfolio_value = self.balance + self.shares_held * next_price
        
        # Reward is the percentage change in portfolio value
        reward = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Create new state
        done = self.current_step >= len(self.data) - 1
        
        # Get price history for state
        if not done:
            start_idx = max(0, self.current_step - 9)
            end_idx = self.current_step + 1
            price_history = self.data[start_idx:end_idx].tolist()
            # Pad if needed
            price_history = [self.data[0]] * (10 - len(price_history)) + price_history
            self.state = self._get_normalized_state(self.balance, self.shares_held, price_history)
            
        return self.state, reward, done, {"portfolio_value": new_portfolio_value}

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, 
                epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, 
                memory_size=10000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Create main network and target network
        self.model = DQNModel(state_size, action_size).to(device)
        self.target_model = DQNModel(state_size, action_size).to(device)
        self.update_target_model()
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Setup replay memory
        self.memory = ReplayMemory(memory_size)
        self.train_start = batch_size
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.cpu().data.numpy()[0].argmax()
    
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        
        batch = self.memory.sample(self.batch_size)
        states = torch.FloatTensor([batch[i][0] for i in range(len(batch))]).to(device)
        actions = torch.LongTensor([batch[i][1] for i in range(len(batch))]).to(device)
        next_states = torch.FloatTensor([batch[i][2] for i in range(len(batch))]).to(device)
        rewards = torch.FloatTensor([batch[i][3] for i in range(len(batch))]).to(device)
        dones = torch.FloatTensor([batch[i][4] for i in range(len(batch))]).to(device)
        
        # Get current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Get max Q' values for next states using target model
        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1)[0]
        
        # Calculate target Q values
        target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Calculate loss and update model
        loss = self.criterion(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filename):
        torch.save(self.model.state_dict(), filename)
        
    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=device))
        self.target_model.load_state_dict(torch.load(filename, map_location=device))

# SimulationAgent - For direct integration with your stock simulation
class SimulationAgent:
    def __init__(self, model_path="dqn_trading_model.pth"):
        self.state_size = 12
        self.action_size = 3
        self.model = DQNModel(self.state_size, self.action_size).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        self.price_history = []
        self.max_history_length = 10
        
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
        if len(self.price_history) > self.max_history_length:
            self.price_history = self.price_history[-self.max_history_length:]
        
        # Pad price history if needed
        if len(self.price_history) < self.max_history_length:
            self.price_history = [current_price] * (self.max_history_length - len(self.price_history)) + self.price_history
        
        # Create state
        state = self.normalize_state(balance, shares_held, self.price_history)
        
        # Get action from model
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        action = q_values.cpu().data.numpy()[0].argmax()
        
        return action  # 0: hold, 1: buy, 2: sell