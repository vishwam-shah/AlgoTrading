"""
Deep Q-Network (DQN) Agent
==========================

DQN agent with:
- Double DQN
- Prioritized Experience Replay
- Dueling architecture
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from .base_agent import BaseAgent

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Try to import stable-baselines3, fall back to custom implementation
try:
    from stable_baselines3 import DQN as SB3_DQN
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN architecture.

    Separates value and advantage streams for better learning.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: List[int] = [128, 64, 32]
    ):
        super().__init__()

        # Feature extractor
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_layers[:-1]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1], 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1], action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage (dueling architecture)
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q_values


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.

    Samples experiences based on TD-error priority.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add experience to buffer"""
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        """Sample batch with prioritization"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        # Calculate importance sampling weights
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        # Get experiences
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])

        self.frame += 1

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities based on TD-errors"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small epsilon for stability

    def __len__(self):
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """
    Deep Q-Network Agent.

    Features:
    - Double DQN for more stable learning
    - Prioritized Experience Replay for efficient sampling
    - Dueling architecture for better value estimation
    - Epsilon-greedy exploration with decay
    """

    def __init__(
        self,
        env,
        config = None,
        model_name: str = None,
        seed: int = 42,
        device: str = 'auto',
        use_sb3: bool = True
    ):
        """
        Initialize DQN agent.

        Args:
            env: Gymnasium environment
            config: DQN configuration
            model_name: Model name for saving
            seed: Random seed
            device: Device ('auto', 'cpu', 'cuda')
            use_sb3: Use stable-baselines3 if available
        """
        super().__init__(env, config, model_name, seed, device)

        self.use_sb3 = use_sb3 and SB3_AVAILABLE

        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Initialize model
        if self.use_sb3:
            self._init_sb3_model()
        else:
            self._init_custom_model()

    @property
    def agent_type(self) -> str:
        return 'dqn'

    def _init_sb3_model(self):
        """Initialize stable-baselines3 DQN"""
        policy_kwargs = {
            'net_arch': self.config.hidden_layers if self.config else [128, 64, 32]
        }

        self.model = SB3_DQN(
            policy='MlpPolicy',
            env=self.env,
            learning_rate=self.config.learning_rate if self.config else 0.0003,
            buffer_size=self.config.buffer_size if self.config else 100000,
            learning_starts=1000,
            batch_size=self.config.batch_size if self.config else 64,
            tau=self.config.tau if self.config else 0.005,
            gamma=self.config.gamma if self.config else 0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=self.config.target_update_freq if self.config else 100,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.01,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=self.seed,
            device=self.device
        )

    def _init_custom_model(self):
        """Initialize custom DQN implementation"""
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        hidden_layers = self.config.hidden_layers if self.config else [128, 64, 32]

        # Networks
        self.q_network = DuelingQNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_network = DuelingQNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        lr = self.config.learning_rate if self.config else 0.0003
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        buffer_size = self.config.buffer_size if self.config else 100000
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)

        # Exploration
        self.epsilon = self.config.epsilon_start if self.config else 1.0
        self.epsilon_end = self.config.epsilon_end if self.config else 0.01
        self.epsilon_decay = self.config.epsilon_decay if self.config else 0.995

        # Hyperparameters
        self.gamma = self.config.gamma if self.config else 0.99
        self.batch_size = self.config.batch_size if self.config else 64
        self.target_update_freq = self.config.target_update_freq if self.config else 100
        self.tau = self.config.tau if self.config else 0.005

    def train(
        self,
        total_timesteps: int,
        callback = None,
        log_interval: int = 100
    ) -> Dict:
        """Train the DQN agent"""
        if self.use_sb3:
            return self._train_sb3(total_timesteps, callback, log_interval)
        else:
            return self._train_custom(total_timesteps, log_interval)

    def _train_sb3(self, total_timesteps: int, callback, log_interval: int) -> Dict:
        """Train using stable-baselines3"""
        # Create callbacks
        callbacks = []

        if callback:
            callbacks.append(callback)

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=self.save_dir,
            name_prefix=self.model_name
        )
        callbacks.append(checkpoint_callback)

        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=log_interval
        )

        self.total_timesteps += total_timesteps

        return {'total_timesteps': self.total_timesteps}

    def _train_custom(self, total_timesteps: int, log_interval: int) -> Dict:
        """Train using custom implementation"""
        episode_rewards = []
        episode_lengths = []

        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(total_timesteps):
            # Select action
            action = self._select_action(obs)

            # Take step
            next_obs, reward, done, truncated, info = self.env.step(action)

            # Store transition
            self.replay_buffer.push(obs, action, reward, next_obs, done or truncated)

            episode_reward += reward
            episode_length += 1

            # Learn
            if len(self.replay_buffer) >= self.batch_size:
                self._learn()

            # Update target network
            if step % self.target_update_freq == 0:
                self._soft_update_target()

            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            obs = next_obs

            if done or truncated:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                self.episodes += 1

                if self.episodes % log_interval == 0:
                    mean_reward = np.mean(episode_rewards[-100:])
                    print(f"Episode {self.episodes} | Mean Reward: {mean_reward:.2f} | Epsilon: {self.epsilon:.3f}")

                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0

        self.total_timesteps += total_timesteps

        return {
            'total_timesteps': self.total_timesteps,
            'episodes': self.episodes,
            'mean_reward': np.mean(episode_rewards[-100:]) if episode_rewards else 0
        }

    def _select_action(self, state: np.ndarray) -> int:
        """Select action with epsilon-greedy exploration"""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def _learn(self):
        """Perform one learning step"""
        # Sample batch
        states, actions, rewards, next_states, dones, indices, weights = \
            self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Double DQN: use online network to select action, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + self.gamma * next_q * (1 - dones.unsqueeze(1))

        # TD error for priority update
        td_errors = (current_q - target_q).abs().detach().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(indices, td_errors)

        # Loss with importance sampling weights
        loss = (weights.unsqueeze(1) * (current_q - target_q) ** 2).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()

    def _soft_update_target(self):
        """Soft update target network"""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[int, Optional[Dict]]:
        """Predict action given observation"""
        if self.use_sb3:
            action, _ = self.model.predict(observation, deterministic=deterministic)
            return int(action), None
        else:
            if not deterministic and random.random() < self.epsilon:
                return self.env.action_space.sample(), None

            with torch.no_grad():
                state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                q_values = self.q_network(state)
                return q_values.argmax().item(), {'q_values': q_values.cpu().numpy()}

    def get_action_probabilities(self, observation: np.ndarray) -> np.ndarray:
        """Get softmax probabilities over actions"""
        with torch.no_grad():
            if self.use_sb3:
                state = torch.FloatTensor(observation).unsqueeze(0)
                q_values = self.model.q_net(state)
            else:
                state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                q_values = self.q_network(state)

            # Softmax over Q-values
            probs = torch.softmax(q_values, dim=-1)
            return probs.cpu().numpy().flatten()

    def save(self, path: str = None):
        """Save model to disk"""
        if path is None:
            path = os.path.join(self.save_dir, f"{self.model_name}.zip")

        if self.use_sb3:
            self.model.save(path.replace('.zip', ''))
        else:
            torch.save({
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'total_timesteps': self.total_timesteps,
                'episodes': self.episodes
            }, path)

        self._save_metadata(path)
        print(f"Model saved to {path}")

    def load(self, path: str = None):
        """Load model from disk"""
        if path is None:
            path = os.path.join(self.save_dir, f"{self.model_name}.zip")

        if self.use_sb3:
            self.model = SB3_DQN.load(path.replace('.zip', ''), env=self.env)
        else:
            checkpoint = torch.load(path, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.total_timesteps = checkpoint['total_timesteps']
            self.episodes = checkpoint['episodes']

        metadata = self._load_metadata(path)
        print(f"Model loaded from {path}")

        return metadata
