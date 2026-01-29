"""
Base Agent Class for RL Trading Agents
======================================

Abstract base class that all RL agents inherit from.
"""

import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class BaseAgent(ABC):
    """
    Abstract base class for RL trading agents.

    All agents (DQN, PPO, SAC) inherit from this class and implement
    the required methods.
    """

    def __init__(
        self,
        env,
        config: Any = None,
        model_name: str = None,
        seed: int = 42,
        device: str = 'auto'
    ):
        """
        Initialize base agent.

        Args:
            env: Gymnasium environment
            config: Agent-specific configuration
            model_name: Name for saving/loading
            seed: Random seed
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.env = env
        self.config = config
        self.model_name = model_name or f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.seed = seed
        self.device = device

        # Training state
        self.total_timesteps = 0
        self.episodes = 0
        self.training_history: List[Dict] = []

        # Model (to be set by subclass)
        self.model = None

        # Save directory
        self.save_dir = os.path.join(BASE_DIR, 'models', 'rl', self.agent_type)
        os.makedirs(self.save_dir, exist_ok=True)

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Return agent type identifier"""
        pass

    @abstractmethod
    def train(
        self,
        total_timesteps: int,
        callback = None,
        log_interval: int = 100
    ) -> Dict:
        """
        Train the agent.

        Args:
            total_timesteps: Total timesteps to train
            callback: Training callback
            log_interval: Logging frequency

        Returns:
            Training statistics
        """
        pass

    @abstractmethod
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[int, Optional[Dict]]:
        """
        Predict action given observation.

        Args:
            observation: Current state
            deterministic: Use deterministic policy

        Returns:
            Tuple of (action, additional_info)
        """
        pass

    @abstractmethod
    def save(self, path: str = None):
        """
        Save agent to disk.

        Args:
            path: Save path (uses default if None)
        """
        pass

    @abstractmethod
    def load(self, path: str = None):
        """
        Load agent from disk.

        Args:
            path: Load path (uses default if None)
        """
        pass

    def evaluate(
        self,
        env = None,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict:
        """
        Evaluate agent performance.

        Args:
            env: Environment to evaluate on (uses training env if None)
            n_episodes: Number of episodes
            deterministic: Use deterministic policy

        Returns:
            Evaluation statistics
        """
        eval_env = env or self.env

        episode_rewards = []
        episode_lengths = []
        episode_returns = []
        win_rates = []

        for _ in range(n_episodes):
            obs, info = eval_env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0

            while not done and not truncated:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, done, truncated, info = eval_env.step(action)
                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Get episode summary from environment
            if hasattr(eval_env, 'get_episode_summary'):
                summary = eval_env.get_episode_summary()
                episode_returns.append(summary.get('total_return', 0))
                win_rates.append(summary.get('win_rate', 0))

        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_return': np.mean(episode_returns) if episode_returns else 0,
            'mean_win_rate': np.mean(win_rates) if win_rates else 0,
            'n_episodes': n_episodes
        }

    def get_action_probabilities(self, observation: np.ndarray) -> np.ndarray:
        """
        Get action probabilities for current state.

        Args:
            observation: Current state

        Returns:
            Array of action probabilities
        """
        # Default implementation - override in subclasses
        return np.array([0.33, 0.33, 0.34])  # Equal probabilities

    def set_training_mode(self, mode: bool = True):
        """Set training mode"""
        if self.model is not None and hasattr(self.model, 'set_training_mode'):
            self.model.set_training_mode(mode)

    def get_training_stats(self) -> Dict:
        """Get training statistics"""
        return {
            'total_timesteps': self.total_timesteps,
            'episodes': self.episodes,
            'agent_type': self.agent_type,
            'model_name': self.model_name
        }

    def _save_metadata(self, path: str):
        """Save training metadata"""
        metadata = {
            'agent_type': self.agent_type,
            'model_name': self.model_name,
            'total_timesteps': self.total_timesteps,
            'episodes': self.episodes,
            'seed': self.seed,
            'config': str(self.config),
            'saved_at': datetime.now().isoformat()
        }

        metadata_path = path.replace('.zip', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _load_metadata(self, path: str) -> Dict:
        """Load training metadata"""
        metadata_path = path.replace('.zip', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
