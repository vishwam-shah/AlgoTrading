"""
RL Agent Training Script
========================

Unified training script for DQN, PPO, and SAC agents.
Includes training, evaluation, and model selection.
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, BASE_DIR)

from src.rl.environments.single_stock_env import SingleStockEnv, create_trading_env
from src.rl.agents.dqn_agent import DQNAgent
from src.rl.agents.ppo_agent import PPOAgent
from src.rl.agents.sac_agent import SACAgent
from src.rl.config.rl_config import RLConfig, rl_config
from src.rl.config.trading_config import TradingConfig, trading_config


class AgentTrainer:
    """
    Unified trainer for RL trading agents.

    Handles:
    - Training multiple agent types
    - Evaluation and comparison
    - Model selection based on performance
    - Saving best models
    """

    def __init__(
        self,
        symbol: str,
        config: RLConfig = None,
        trading_cfg: TradingConfig = None,
        save_dir: str = None
    ):
        """
        Initialize trainer.

        Args:
            symbol: Stock symbol to train on
            config: RL configuration
            trading_cfg: Trading configuration
            save_dir: Directory to save models
        """
        self.symbol = symbol
        self.config = config or rl_config
        self.trading_config = trading_cfg or trading_config

        self.save_dir = save_dir or os.path.join(BASE_DIR, 'models', 'rl')
        os.makedirs(self.save_dir, exist_ok=True)

        # Results tracking
        self.training_results: Dict[str, Dict] = {}
        self.evaluation_results: Dict[str, Dict] = {}

        # Create environment
        self.env = self._create_environment()

    def _create_environment(self) -> SingleStockEnv:
        """Create trading environment"""
        return create_trading_env(
            symbol=self.symbol,
            config=self.trading_config,
            initial_capital=self.trading_config.capital.initial_capital,
            reward_type='risk_adjusted'
        )

    def train_agent(
        self,
        agent_type: str,
        total_timesteps: int = None,
        model_name: str = None
    ) -> Dict:
        """
        Train a single agent.

        Args:
            agent_type: 'dqn', 'ppo', or 'sac'
            total_timesteps: Training timesteps
            model_name: Name for saving

        Returns:
            Training results
        """
        total_timesteps = total_timesteps or self.config.total_timesteps
        model_name = model_name or f"{self.symbol}_{agent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"\n{'='*60}")
        print(f"Training {agent_type.upper()} agent for {self.symbol}")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"{'='*60}\n")

        # Create agent
        agent = self._create_agent(agent_type, model_name)

        # Train
        start_time = datetime.now()
        training_stats = agent.train(
            total_timesteps=total_timesteps,
            log_interval=100
        )
        training_time = (datetime.now() - start_time).total_seconds()

        # Evaluate
        eval_stats = agent.evaluate(n_episodes=10)

        # Save model
        agent.save()

        # Compile results
        results = {
            'agent_type': agent_type,
            'symbol': self.symbol,
            'model_name': model_name,
            'total_timesteps': total_timesteps,
            'training_time_seconds': training_time,
            'training_stats': training_stats,
            'evaluation': eval_stats
        }

        self.training_results[agent_type] = results
        self.evaluation_results[agent_type] = eval_stats

        print(f"\n{agent_type.upper()} Training Complete:")
        print(f"  Mean Reward: {eval_stats['mean_reward']:.2f}")
        print(f"  Mean Return: {eval_stats.get('mean_return', 0)*100:.2f}%")
        print(f"  Win Rate: {eval_stats.get('mean_win_rate', 0)*100:.1f}%")
        print(f"  Training Time: {training_time:.1f}s")

        return results

    def _create_agent(self, agent_type: str, model_name: str):
        """Create agent instance"""
        agent_config = self.config.get_agent_config(agent_type)

        if agent_type.lower() == 'dqn':
            return DQNAgent(
                env=self.env,
                config=agent_config,
                model_name=model_name,
                seed=self.config.seed,
                device=self.config.device
            )
        elif agent_type.lower() == 'ppo':
            return PPOAgent(
                env=self.env,
                config=agent_config,
                model_name=model_name,
                seed=self.config.seed,
                device=self.config.device
            )
        elif agent_type.lower() == 'sac':
            return SACAgent(
                env=self.env,
                config=agent_config,
                model_name=model_name,
                seed=self.config.seed,
                device=self.config.device
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def train_all_agents(
        self,
        total_timesteps: int = None,
        agents: List[str] = None
    ) -> Dict:
        """
        Train all agent types and compare.

        Args:
            total_timesteps: Training timesteps per agent
            agents: List of agent types to train

        Returns:
            Comparison results
        """
        agents = agents or ['dqn', 'ppo', 'sac']
        total_timesteps = total_timesteps or self.config.total_timesteps

        print(f"\n{'#'*60}")
        print(f"Training All Agents for {self.symbol}")
        print(f"Agents: {', '.join(agents)}")
        print(f"{'#'*60}\n")

        for agent_type in agents:
            try:
                self.train_agent(agent_type, total_timesteps)
            except Exception as e:
                print(f"Error training {agent_type}: {e}")
                self.training_results[agent_type] = {'error': str(e)}

        # Compare and select best
        comparison = self.compare_agents()

        return comparison

    def compare_agents(self) -> Dict:
        """
        Compare trained agents and select best.

        Returns:
            Comparison results with best agent
        """
        if not self.evaluation_results:
            return {}

        comparison = {
            'symbol': self.symbol,
            'agents': {},
            'best_agent': None,
            'best_metric': None
        }

        # Collect metrics
        best_sharpe = -np.inf
        best_agent = None

        for agent_type, eval_stats in self.evaluation_results.items():
            metrics = {
                'mean_reward': eval_stats.get('mean_reward', 0),
                'mean_return': eval_stats.get('mean_return', 0),
                'win_rate': eval_stats.get('mean_win_rate', 0),
                'std_reward': eval_stats.get('std_reward', 0)
            }

            # Calculate Sharpe-like metric
            if metrics['std_reward'] > 0:
                sharpe = metrics['mean_reward'] / metrics['std_reward']
            else:
                sharpe = metrics['mean_reward']

            metrics['sharpe_approx'] = sharpe
            comparison['agents'][agent_type] = metrics

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_agent = agent_type

        comparison['best_agent'] = best_agent
        comparison['best_metric'] = best_sharpe

        # Print comparison
        print(f"\n{'='*60}")
        print("Agent Comparison Results")
        print(f"{'='*60}")
        print(f"{'Agent':<10} {'Reward':>12} {'Return':>12} {'Win Rate':>12} {'Sharpe':>12}")
        print(f"{'-'*60}")

        for agent_type, metrics in comparison['agents'].items():
            marker = " *" if agent_type == best_agent else ""
            print(f"{agent_type:<10} {metrics['mean_reward']:>12.2f} "
                  f"{metrics['mean_return']*100:>11.2f}% "
                  f"{metrics['win_rate']*100:>11.1f}% "
                  f"{metrics['sharpe_approx']:>12.2f}{marker}")

        print(f"{'-'*60}")
        print(f"Best Agent: {best_agent}")
        print(f"{'='*60}\n")

        # Save comparison
        self._save_comparison(comparison)

        return comparison

    def _save_comparison(self, comparison: Dict):
        """Save comparison results to file"""
        results_dir = os.path.join(BASE_DIR, 'results', 'rl')
        os.makedirs(results_dir, exist_ok=True)

        filename = f"{self.symbol}_agent_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(results_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)

        print(f"Comparison saved to {filepath}")

    def load_best_agent(self):
        """Load the best performing agent"""
        comparison = self.compare_agents()
        best_agent_type = comparison.get('best_agent')

        if best_agent_type is None:
            raise ValueError("No trained agents found")

        # Find the model file
        model_dir = os.path.join(self.save_dir, best_agent_type)
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Get latest model for this symbol
        model_files = [f for f in os.listdir(model_dir) if f.startswith(self.symbol)]
        if not model_files:
            raise FileNotFoundError(f"No models found for {self.symbol}")

        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(model_dir, latest_model)

        # Create and load agent
        agent = self._create_agent(best_agent_type, latest_model.replace('.zip', ''))
        agent.load(model_path)

        return agent


def train_single_stock(
    symbol: str,
    agent_type: str = 'dqn',
    total_timesteps: int = 50000,
    **kwargs
):
    """
    Train a single agent for a single stock.

    Args:
        symbol: Stock symbol
        agent_type: Agent type ('dqn', 'ppo', 'sac')
        total_timesteps: Training timesteps
    """
    trainer = AgentTrainer(symbol)
    results = trainer.train_agent(agent_type, total_timesteps)
    return results


def train_all_stocks(
    symbols: List[str] = None,
    agents: List[str] = None,
    total_timesteps: int = 50000,
    **kwargs
):
    """
    Train agents for multiple stocks.

    Args:
        symbols: List of stock symbols
        agents: List of agent types
        total_timesteps: Training timesteps per agent
    """
    symbols = symbols or trading_config.tradeable_stocks
    agents = agents or ['dqn', 'ppo', 'sac']

    all_results = {}

    for symbol in symbols:
        print(f"\n{'#'*70}")
        print(f"Processing {symbol}")
        print(f"{'#'*70}")

        try:
            trainer = AgentTrainer(symbol)
            comparison = trainer.train_all_agents(total_timesteps, agents)
            all_results[symbol] = comparison
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            all_results[symbol] = {'error': str(e)}

    # Save overall results
    results_dir = os.path.join(BASE_DIR, 'results', 'rl')
    os.makedirs(results_dir, exist_ok=True)

    filename = f"all_stocks_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(results_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAll results saved to {filepath}")

    return all_results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train RL Trading Agents')

    parser.add_argument('--symbol', type=str, default='HDFCBANK',
                        help='Stock symbol to train on')
    parser.add_argument('--agent', type=str, default='all',
                        choices=['dqn', 'ppo', 'sac', 'all'],
                        help='Agent type to train')
    parser.add_argument('--timesteps', type=int, default=50000,
                        help='Total training timesteps')
    parser.add_argument('--all-stocks', action='store_true',
                        help='Train on all tradeable stocks')

    args = parser.parse_args()

    if args.all_stocks:
        train_all_stocks(
            agents=['dqn', 'ppo', 'sac'] if args.agent == 'all' else [args.agent],
            total_timesteps=args.timesteps
        )
    elif args.agent == 'all':
        trainer = AgentTrainer(args.symbol)
        trainer.train_all_agents(args.timesteps)
    else:
        train_single_stock(
            args.symbol,
            args.agent,
            args.timesteps
        )


if __name__ == '__main__':
    main()
