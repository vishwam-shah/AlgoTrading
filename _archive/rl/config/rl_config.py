"""
RL Configuration - Hyperparameters for Reinforcement Learning Agents
=====================================================================

Contains all RL-specific hyperparameters for DQN, PPO, and SAC agents.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class DQNConfig:
    """Deep Q-Network Configuration"""
    # Network architecture
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    activation: str = 'relu'

    # Training parameters
    learning_rate: float = 0.0003
    gamma: float = 0.99  # Discount factor
    batch_size: int = 64
    buffer_size: int = 100000

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    epsilon_decay_steps: int = 10000

    # Target network
    target_update_freq: int = 100
    tau: float = 0.005  # Soft update coefficient

    # Prioritized Experience Replay
    use_per: bool = True
    per_alpha: float = 0.6  # Priority exponent
    per_beta_start: float = 0.4  # IS weight exponent
    per_beta_end: float = 1.0

    # Double DQN
    use_double_dqn: bool = True

    # Dueling DQN
    use_dueling: bool = True


@dataclass
class PPOConfig:
    """Proximal Policy Optimization Configuration"""
    # Network architecture
    policy_hidden_layers: List[int] = field(default_factory=lambda: [128, 64])
    value_hidden_layers: List[int] = field(default_factory=lambda: [128, 64])
    activation: str = 'tanh'

    # Training parameters
    learning_rate: float = 0.0003
    gamma: float = 0.99
    batch_size: int = 64
    n_epochs: int = 10

    # PPO-specific
    clip_range: float = 0.2
    clip_range_vf: float = None  # None = no clipping
    gae_lambda: float = 0.95  # GAE lambda

    # Entropy
    ent_coef: float = 0.01
    ent_coef_decay: float = 0.999

    # Value function
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Rollout
    n_steps: int = 2048
    n_envs: int = 1


@dataclass
class SACConfig:
    """Soft Actor-Critic Configuration"""
    # Network architecture
    hidden_layers: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = 'relu'

    # Training parameters
    learning_rate: float = 0.0003
    gamma: float = 0.99
    batch_size: int = 256
    buffer_size: int = 1000000

    # SAC-specific
    tau: float = 0.005  # Soft update coefficient

    # Entropy
    ent_coef: str = 'auto'  # 'auto' = automatic tuning
    target_entropy: str = 'auto'  # 'auto' = -dim(action_space)

    # Learning starts
    learning_starts: int = 1000
    train_freq: int = 1
    gradient_steps: int = 1


@dataclass
class RLConfig:
    """Master RL Configuration"""

    # Random seed
    seed: int = 42

    # Device
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'

    # Agent configurations
    dqn: DQNConfig = field(default_factory=DQNConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    sac: SACConfig = field(default_factory=SACConfig)

    # Default agent type
    default_agent: str = 'dqn'  # 'dqn', 'ppo', 'sac'

    # Training settings
    total_timesteps: int = 100000
    eval_freq: int = 5000
    n_eval_episodes: int = 10

    # Logging
    log_interval: int = 100
    verbose: int = 1
    tensorboard_log: str = os.path.join(BASE_DIR, 'logs', 'tensorboard')

    # Model saving
    save_freq: int = 10000
    model_save_dir: str = os.path.join(BASE_DIR, 'models', 'rl')

    # Checkpointing
    save_best_only: bool = True
    monitor_metric: str = 'sharpe_ratio'  # Metric to monitor for best model

    # Environment settings
    normalize_obs: bool = True
    normalize_reward: bool = True
    clip_obs: float = 10.0
    clip_reward: float = 10.0

    # Action space
    action_space_type: str = 'discrete'  # 'discrete' or 'continuous'

    def get_agent_config(self, agent_type: str) -> Any:
        """Get configuration for specific agent type"""
        configs = {
            'dqn': self.dqn,
            'ppo': self.ppo,
            'sac': self.sac
        }
        return configs.get(agent_type.lower(), self.dqn)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            'seed': self.seed,
            'device': self.device,
            'default_agent': self.default_agent,
            'total_timesteps': self.total_timesteps,
            'action_space_type': self.action_space_type
        }


# Global instance
rl_config = RLConfig()
