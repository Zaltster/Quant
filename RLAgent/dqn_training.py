"""
Training module for LSTM-Transformer-DQN model
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import yfinance as yf
import os
import json
from datetime import datetime
import argparse

# Import the agent and environment from the dqn_agent module
from dqn_agent import DQNAgent, TradingEnvironment

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Function to train the agent
def train_dqn_agent(stock_data, episodes=100, window_size=10, batch_size=64, 
                   initial_balance=100000, transaction_fee=0.001,
                   learning_rate=0.001, gamma=0.99, epsilon=1.0, 
                   epsilon_min=0.01, epsilon_decay=0.995, target_update=10,
                   hidden_size=128, lstm_layers=1, transformer_layers=2, num_heads=4,
                   model_save_path="lstm_transformer_dqn_model.pth", verbose=True):
    """
    Train the LSTM-Transformer-DQN agent on historical stock data
    
    Args:
        stock_data: numpy array of stock prices
        episodes: number of training episodes
        window_size: number of past prices to include in state
        batch_size: size of batches sampled from replay memory
        initial_balance: starting portfolio balance
        transaction_fee: transaction fee as a percentage
        learning_rate: learning rate for the optimizer
        gamma: discount factor for future rewards
        epsilon: initial exploration rate
        epsilon_min: minimum exploration rate
        epsilon_decay: rate at which exploration decays
        target_update: frequency of target network updates
        hidden_size: size of hidden layers
        lstm_layers: number of LSTM layers
        transformer_layers: number of transformer layers
        num_heads: number of attention heads in transformer
        model_save_path: where to save the trained model
        verbose: whether to print progress
        
    Returns:
        agent: trained DQN agent
        metrics: dictionary of training metrics (rewards, portfolio values, etc.)
    """
    # Environment setup
    env = TradingEnvironment(stock_data, initial_balance, transaction_fee, window_size)
    state_size = window_size + 2  # prices + balance + shares
    action_size = 3  # hold, buy, sell
    
    # Agent setup
    agent = DQNAgent(
        state_size=state_size, 
        action_size=action_size,
        hidden_size=hidden_size,
        window_size=window_size,
        lstm_layers=lstm_layers,
        transformer_layers=transformer_layers,
        num_heads=num_heads,
        learning_rate=learning_rate, 
        gamma=gamma, 
        epsilon=epsilon, 
        epsilon_min=epsilon_min, 
        epsilon_decay=epsilon_decay,
        batch_size=batch_size
    )
    
    # Metrics to track
    metrics = {
        'episode_rewards': [],
        'portfolio_values': [],
        'losses': [],
        'epsilons': []
    }
    
    # Training loop
    for episode in range(episodes):
        state = env.reset()
        agent.reset_lstm_state()  # Reset LSTM state at the start of each episode
        
        done = False
        total_reward = 0
        episode_losses = []
        
        # Progress bar for steps within episode
        step_progress = tqdm(total=len(stock_data)-1, desc=f"Episode {episode+1}/{episodes}", 
                          disable=not verbose)
        
        while not done:
            # Get action
            action = agent.get_action(state)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store experience in replay memory
            agent.memory.push(state, action, next_state, reward, done)
            
            # Train model and get loss
            loss = agent.train_model()
            if loss is not None:
                episode_losses.append(loss)
            
            # Update state and accumulate reward
            state = next_state
            total_reward += reward
            
            step_progress.update(1)
        
        step_progress.close()
        
        # Update target model
        if (episode + 1) % target_update == 0:
            agent.update_target_model()
        
        # Track metrics
        metrics['episode_rewards'].append(total_reward)
        metrics['portfolio_values'].append(info["portfolio_value"])
        metrics['epsilons'].append(agent.epsilon)
        
        if episode_losses:
            avg_loss = sum(episode_losses) / len(episode_losses)
            metrics['losses'].append(avg_loss)
        else:
            metrics['losses'].append(None)
        
        # Print progress
        if verbose and (episode + 1) % max(1, episodes // 10) == 0:
            print(f"Episode: {episode+1}/{episodes}, Reward: {total_reward:.4f}, Portfolio: ${info['portfolio_value']:.2f}, Epsilon: {agent.epsilon:.4f}, Avg Loss: {avg_loss if episode_losses else 'N/A'}")
    
    # Save the trained model
    agent.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Calculate Sharpe ratio
    if len(metrics['portfolio_values']) > 2:
        returns = np.diff(metrics['portfolio_values']) / metrics['portfolio_values'][:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        print(f"Training Sharpe Ratio: {sharpe_ratio:.4f}")
        metrics['sharpe_ratio'] = sharpe_ratio
    
    return agent, metrics

# Function to test the agent
def test_dqn_agent(stock_data, model_path, window_size=10, 
                  initial_balance=100000, transaction_fee=0.001, verbose=True):
    """
    Test the trained LSTM-Transformer-DQN agent on unseen stock data
    
    Args:
        stock_data: numpy array of stock prices
        model_path: path to the saved model
        window_size: number of past prices in state
        initial_balance: starting portfolio balance
        transaction_fee: transaction fee as a percentage
        verbose: whether to print progress
        
    Returns:
        results: dictionary with test results
        trade_history: list of trades made by the agent
    """
    # Environment setup
    env = TradingEnvironment(stock_data, initial_balance, transaction_fee, window_size)
    state_size = window_size + 2  # prices + balance + shares
    action_size = 3  # hold, buy, sell
    
    # Create agent and load trained model
    agent = DQNAgent(
        state_size=state_size, 
        action_size=action_size,
        window_size=window_size,
        epsilon=0  # No exploration during testing
    )
    agent.load(model_path)
    
    # Test loop
    state = env.reset()
    agent.reset_lstm_state()  # Reset LSTM state at the start
    
    done = False
    total_reward = 0
    
    # Progress bar
    step_progress = tqdm(total=len(stock_data)-1, desc="Testing", disable=not verbose)
    
    while not done:
        # Get action (no exploration)
        action = agent.get_action(state, is_eval=True)
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state
        
        step_progress.update(1)
    
    step_progress.close()
    
    # Calculate final performance metrics
    final_portfolio_value = info["portfolio_value"]
    roi = (final_portfolio_value - env.initial_balance) / env.initial_balance * 100
    
    # Calculate Sharpe ratio based on daily returns
    daily_returns = []
    prev_value = initial_balance
    
    for trade in env.trade_history:
        # Calculate portfolio value at this step
        if trade['action'] == 'buy':
            current_value = trade['balance'] + trade['shares'] * trade['price']
        else:  # sell
            current_value = trade['balance']
        
        # Calculate return
        daily_return = (current_value - prev_value) / prev_value
        daily_returns.append(daily_return)
        prev_value = current_value
    
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) if len(daily_returns) > 1 and np.std(daily_returns) > 0 else 0
    
    # Print results
    if verbose:
        print(f"Initial Balance: ${env.initial_balance:.2f}")
        print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
        print(f"Return on Investment: {roi:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Total trades: {len(env.trade_history)}")
    
    # Compile results
    results = {
        'initial_balance': env.initial_balance,
        'final_value': final_portfolio_value,
        'roi': roi,
        'sharpe_ratio': sharpe_ratio,
        'total_reward': total_reward,
        'total_trades': len(env.trade_history)
    }
    
    return results, env.trade_history

# Plot training metrics
def plot_training_metrics(metrics, save_path="training_metrics.png"):
    """
    Plot and save training metrics
    
    Args:
        metrics: dictionary of training metrics
        save_path: where to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(metrics['episode_rewards'])
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # Plot portfolio values
    plt.subplot(2, 2, 2)
    plt.plot(metrics['portfolio_values'])
    plt.title('Portfolio Value')
    plt.xlabel('Episode')
    plt.ylabel('Value ($)')
    plt.grid(True)
    
    # Plot losses
    plt.subplot(2, 2, 3)
    # Filter out None values
    episodes = [i for i, loss in enumerate(metrics['losses']) if loss is not None]
    losses = [loss for loss in metrics['losses'] if loss is not None]
    
    if losses:
        plt.plot(episodes, losses)
        plt.title('Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Average Loss')
        plt.grid(True)
    
    # Plot epsilons
    plt.subplot(2, 2, 4)
    plt.plot(metrics['epsilons'])
    plt.title('Exploration Rate (Epsilon)')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved training metrics plot to {save_path}")
    
    # Show the plot (can be commented out for headless environments)
    plt.show()

# Plot test performance
def plot_test_performance(stock_data, trade_history, save_path="test_performance.png"):
    """
    Plot test performance with buy/sell markers
    
    Args:
        stock_data: numpy array of stock prices
        trade_history: list of trades made by the agent
        save_path: where to save the plot
    """
    plt.figure(figsize=(15, 7))
    
    # Create time axis (just using indices for simplicity)
    time_axis = range(len(stock_data))
    
    # Plot stock price
    plt.plot(time_axis, stock_data, label='Stock Price')
    
    # Add buy/sell markers
    for trade in trade_history:
        step = trade['step'] - 1  # Adjust for 0-indexing
        if step < len(time_axis):
            if trade['action'] == 'buy':
                plt.scatter(step, stock_data[step], color='green', marker='^', s=100, label='_nolegend_')
            elif trade['action'] == 'sell':
                plt.scatter(step, stock_data[step], color='red', marker='v', s=100, label='_nolegend_')
    
    # Add legend
    handles, labels = plt.gca().get_legend_handles_labels()
    buy_marker = plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='g', markersize=10, label='Buy')
    sell_marker = plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='r', markersize=10, label='Sell')
    plt.legend(handles + [buy_marker, sell_marker])
    
    plt.title('LSTM-Transformer-DQN Trading Performance')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved test performance plot to {save_path}")
    
    # Show the plot (can be commented out for headless environments)
    plt.show()

# Function to fetch data and run training/testing
def run_dqn_trading(ticker="BTC-USD", train_start='2020-01-01', train_end='2021-12-31', 
                   test_start='2022-01-01', test_end='2022-12-31', episodes=50,
                   window_size=10, initial_balance=100000, transaction_fee=0.001):
    """
    End-to-end pipeline to fetch data, train and test the LSTM-Transformer-DQN agent
    
    Args:
        ticker: stock/crypto ticker symbol
        train_start/end: training data date range
        test_start/end: testing data date range
        episodes: number of training episodes
        window_size: number of past prices to include in state
        initial_balance: starting portfolio balance
        transaction_fee: transaction fee as a percentage
        
    Returns:
        agent: trained DQN agent
        train_metrics: training metrics
        test_results: test results
        trade_history: trades made during testing
    """
    # Create output directory
    output_dir = f"results_{ticker.replace('-', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Fetch data using yfinance
    print(f"Fetching {ticker} data...")
    train_data = yf.download(ticker, start=train_start, end=train_end)
    test_data = yf.download(ticker, start=test_start, end=test_end)
    
    # Extract closing prices
    train_prices = train_data['Close'].values
    test_prices = test_data['Close'].values
    
    # Save the data for reference
    np.save(f"{output_dir}/train_prices.npy", train_prices)
    np.save(f"{output_dir}/test_prices.npy", test_prices)
    
    # Model save path
    model_path = f"{output_dir}/lstm_transformer_dqn_model.pth"
    
    # Train the agent
    print(f"Training LSTM-Transformer-DQN agent on {ticker} data...")
    agent, train_metrics = train_dqn_agent(
        stock_data=train_prices, 
        episodes=episodes,
        window_size=window_size,
        initial_balance=initial_balance,
        transaction_fee=transaction_fee,
        model_save_path=model_path
    )
    
    # Plot and save training metrics
    plot_training_metrics(train_metrics, save_path=f"{output_dir}/training_metrics.png")
    
    # Save training metrics to JSON
    with open(f"{output_dir}/training_metrics.json", 'w') as f:
        json.dump({k: v if not isinstance(v, np.ndarray) else v.tolist() 
                  for k, v in train_metrics.items()}, f, indent=4)
    
    # Test the agent
    print(f"Testing LSTM-Transformer-DQN agent on {ticker} data...")
    test_results, trade_history = test_dqn_agent(
        stock_data=test_prices,
        model_path=model_path,
        window_size=window_size,
        initial_balance=initial_balance,
        transaction_fee=transaction_fee
    )
    
    # Plot and save test performance
    plot_test_performance(test_prices, trade_history, save_path=f"{output_dir}/test_performance.png")
    
    # Save test results to JSON
    with open(f"{output_dir}/test_results.json", 'w') as f:
        json.dump(test_results, f, indent=4)
    
    # Save trade history to JSON
    with open(f"{output_dir}/trade_history.json", 'w') as f:
        json.dump(trade_history, f, indent=4)
    
    print(f"All results saved to directory: {output_dir}")
    
    return agent, train_metrics, test_results, trade_history

# Main function - Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and test LSTM-Transformer-DQN trading agent')
    
    # Data params
    parser.add_argument('--ticker', type=str, default='BTC-USD', help='Stock/crypto ticker symbol')
    parser.add_argument('--train_start', type=str, default='2020-01-01', help='Training start date')
    parser.add_argument('--train_end', type=str, default='2021-12-31', help='Training end date')
    parser.add_argument('--test_start', type=str, default='2022-01-01', help='Testing start date')
    parser.add_argument('--test_end', type=str, default='2022-12-31', help='Testing end date')
    
    # Model params
    parser.add_argument('--window_size', type=int, default=10, help='Number of past prices in state')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--lstm_layers', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--transformer_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    
    # Training params
    parser.add_argument('--episodes', type=int, default=50, help='Number of training episodes')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial exploration rate')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='Minimum exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Exploration decay rate')
    
    # Environment params
    parser.add_argument('--initial_balance', type=float, default=100000, help='Initial balance')
    parser.add_argument('--transaction_fee', type=float, default=0.001, help='Transaction fee percentage')
    
    args = parser.parse_args()
    
    # Run the training and testing pipeline
    run_dqn_trading(
        ticker=args.ticker,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        episodes=args.episodes,
        window_size=args.window_size,
        initial_balance=args.initial_balance,
        transaction_fee=args.transaction_fee
    )