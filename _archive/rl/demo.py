"""
RL Trading System Demo
======================

Quick demo to test the RL trading system.
Run this to verify installation and basic functionality.
"""

import os
import sys

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

import numpy as np


def check_dependencies():
    """Check if all dependencies are installed"""
    print("\n" + "="*60)
    print("Checking Dependencies")
    print("="*60)

    dependencies = {
        'numpy': None,
        'pandas': None,
        'torch': None,
        'gymnasium': None,
        'stable_baselines3': None,
        'SmartApi.smartConnect': None,  # SmartApi needs submodule import
        'pyotp': None,
    }

    for dep in dependencies:
        try:
            # Handle submodule imports like SmartApi.smartConnect
            if '.' in dep:
                parts = dep.split('.')
                module = __import__(dep)
                for part in parts[1:]:
                    module = getattr(module, part)
                display_name = parts[0]
            else:
                module = __import__(dep)
                display_name = dep
            version = getattr(module, '__version__', 'installed')
            dependencies[dep] = version
            print(f"  [OK] {display_name}: {version}")
        except (ImportError, AttributeError) as e:
            display_name = dep.split('.')[0] if '.' in dep else dep
            print(f"  [MISSING] {display_name}")

    missing = [k for k, v in dependencies.items() if v is None]
    if missing:
        display_missing = [k.split('.')[0] if '.' in k else k for k in missing]
        print(f"\nMissing dependencies: {', '.join(display_missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    print("\nAll dependencies installed!")
    return True


def demo_environment():
    """Demo the trading environment"""
    print("\n" + "="*60)
    print("Demo: Trading Environment")
    print("="*60)

    try:
        from src.rl.environments.single_stock_env import SingleStockEnv
        from src.rl.utils.state_builder import StateBuilder, create_empty_portfolio

        # Create a simple test environment with mock data
        import pandas as pd

        # Create mock prediction data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        mock_data = pd.DataFrame({
            'timestamp': dates,
            'open': 100 + np.random.randn(100).cumsum(),
            'high': 101 + np.random.randn(100).cumsum(),
            'low': 99 + np.random.randn(100).cumsum(),
            'close': 100 + np.random.randn(100).cumsum(),
            'volume': np.random.randint(1000000, 5000000, 100),
            'XGBoost_direction_pred': np.random.randint(0, 2, 100),
            'XGBoost_close_pred': np.random.randn(100) * 0.02,
            'XGBoost_high_pred': np.abs(np.random.randn(100) * 0.02),
            'XGBoost_low_pred': -np.abs(np.random.randn(100) * 0.02),
        })
        mock_data.set_index('timestamp', inplace=True)

        # Make high > low > 0 consistent
        mock_data['high'] = mock_data[['open', 'high', 'close']].max(axis=1) + 0.1
        mock_data['low'] = mock_data[['open', 'low', 'close']].min(axis=1) - 0.1

        print("Creating environment with mock data...")

        # Pre-cache mock data in state builder to avoid file loading
        state_builder = StateBuilder()
        state_builder._prediction_cache['TEST'] = mock_data

        # Create environment
        env = SingleStockEnv(
            symbol='TEST',
            predictions_df=mock_data,
            initial_capital=100000,
            reward_type='simple_pnl'
        )
        # Inject pre-cached state builder
        env.state_builder = state_builder

        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")

        # Run a few steps
        print("\nRunning environment simulation...")
        obs, info = env.reset()
        total_reward = 0

        for step in range(10):
            action = env.action_space.sample()  # Random action
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            action_names = ['Hold', 'Buy', 'Sell']
            print(f"  Step {step+1}: Action={action_names[action]}, Reward={reward:.2f}, Value={info['portfolio_value']:.2f}")

            if done or truncated:
                break

        print(f"\nTotal reward over 10 steps: {total_reward:.2f}")
        print("Environment demo successful!")
        return True

    except Exception as e:
        print(f"Environment demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_agents():
    """Demo the RL agents"""
    print("\n" + "="*60)
    print("Demo: RL Agents")
    print("="*60)

    try:
        import gymnasium as gym
        from stable_baselines3 import DQN, PPO

        # Create a simple test environment
        env = gym.make('CartPole-v1')

        print("\nTesting DQN agent...")
        dqn_model = DQN('MlpPolicy', env, verbose=0)
        print("  DQN model created successfully")

        print("\nTesting PPO agent...")
        ppo_model = PPO('MlpPolicy', env, verbose=0)
        print("  PPO model created successfully")

        # Quick training test
        print("\nQuick training test (100 steps)...")
        dqn_model.learn(total_timesteps=100, progress_bar=False)
        print("  Training completed!")

        print("\nAgents demo successful!")
        return True

    except Exception as e:
        print(f"Agents demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_paper_trading():
    """Demo paper trading"""
    print("\n" + "="*60)
    print("Demo: Paper Trading")
    print("="*60)

    try:
        from src.rl.brokers.angel_one.api import PaperTradingAPI

        # Create paper trading API
        api = PaperTradingAPI(initial_capital=100000)

        print(f"Initial capital: Rs {api.capital:,.2f}")

        # Simulate some trades
        print("\nSimulating trades...")

        # Buy HDFCBANK
        order_id = api.place_market_order('HDFCBANK', 10, 'BUY')
        print(f"  Bought 10 HDFCBANK: Order {order_id}")

        # Check positions
        positions = api.get_positions()
        print(f"  Positions: {len(positions)}")

        # Check funds
        funds = api.get_funds()
        print(f"  Cash: Rs {funds['cash']:,.2f}")
        print(f"  Total Value: Rs {funds['total_value']:,.2f}")

        # Sell HDFCBANK
        order_id = api.place_market_order('HDFCBANK', 10, 'SELL')
        print(f"  Sold 10 HDFCBANK: Order {order_id}")

        # Final funds
        funds = api.get_funds()
        print(f"  Final Cash: Rs {funds['cash']:,.2f}")

        print("\nPaper trading demo successful!")
        return True

    except Exception as e:
        print(f"Paper trading demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_config():
    """Demo configuration"""
    print("\n" + "="*60)
    print("Demo: Configuration")
    print("="*60)

    try:
        from src.rl.config.rl_config import RLConfig, rl_config
        from src.rl.config.trading_config import TradingConfig, trading_config

        print("\nRL Configuration:")
        print(f"  Default agent: {rl_config.default_agent}")
        print(f"  Total timesteps: {rl_config.total_timesteps:,}")
        print(f"  Seed: {rl_config.seed}")

        print("\nDQN Config:")
        print(f"  Hidden layers: {rl_config.dqn.hidden_layers}")
        print(f"  Learning rate: {rl_config.dqn.learning_rate}")
        print(f"  Gamma: {rl_config.dqn.gamma}")

        print("\nTrading Configuration:")
        print(f"  Initial capital: Rs {trading_config.capital.initial_capital:,.2f}")
        print(f"  Max positions: {trading_config.capital.max_positions}")
        print(f"  Position size: {trading_config.capital.position_size_pct*100:.0f}%")
        print(f"  Max drawdown: {trading_config.risk.max_drawdown_pct*100:.0f}%")
        print(f"  Paper trading: {trading_config.paper_trading}")

        print("\nTradeable stocks:")
        print(f"  {', '.join(trading_config.tradeable_stocks[:5])}...")

        print("\nConfiguration demo successful!")
        return True

    except Exception as e:
        print(f"Configuration demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all demos"""
    print("\n" + "#"*60)
    print("RL Trading System - Demo")
    print("#"*60)

    results = {}

    # Check dependencies
    results['dependencies'] = check_dependencies()

    if results['dependencies']:
        # Run demos
        results['config'] = demo_config()
        results['environment'] = demo_environment()
        results['agents'] = demo_agents()
        results['paper_trading'] = demo_paper_trading()

    # Summary
    print("\n" + "="*60)
    print("Demo Summary")
    print("="*60)

    for name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n" + "="*60)
        print("All demos passed! System is ready.")
        print("="*60)
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Train an agent: python src/rl/training/train_agent.py --symbol HDFCBANK")
        print("  3. Set up Angel One API: See docs/ANGEL_ONE_SETUP.md")
    else:
        print("\nSome demos failed. Check the errors above.")

    return all_passed


if __name__ == '__main__':
    main()
