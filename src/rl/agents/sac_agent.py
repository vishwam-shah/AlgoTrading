"""
Soft Actor-Critic (SAC) Agent
=============================

SAC agent with:
- Maximum entropy framework
- Automatic temperature tuning
- Twin Q-networks for stability
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from .base_agent import BaseAgent

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Try to import stable-baselines3
try:
    from stable_baselines3 import SAC as SB3_SAC
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


class SoftQNetwork(nn.Module):
    """Soft Q-Network for SAC"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: List[int] = [256, 256]
    ):
        super().__init__()

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))
        self.q_net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.q_net(state)


class DiscretePolicy(nn.Module):
    """Policy network for discrete actions with SAC"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: List[int] = [256, 256]
    ):
        super().__init__()

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))
        self.policy_net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.policy_net(state)
        return F.softmax(logits, dim=-1)

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        probs = self.forward(state)
        dist = Categorical(probs)

        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy


class ReplayBuffer:
    """Simple replay buffer for SAC"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


class SACAgent(BaseAgent):
    """
    Soft Actor-Critic Agent for Discrete Actions.

    Features:
    - Maximum entropy RL for better exploration
    - Automatic temperature (alpha) tuning
    - Twin Q-networks for reduced overestimation
    - Soft policy updates
    """

    def __init__(
        self,
        env,
        config = None,
        model_name: str = None,
        seed: int = 42,
        device: str = 'auto',
        use_sb3: bool = False  # SAC in SB3 is for continuous actions
    ):
        """
        Initialize SAC agent.

        Args:
            env: Gymnasium environment
            config: SAC configuration
            model_name: Model name for saving
            seed: Random seed
            device: Device ('auto', 'cpu', 'cuda')
            use_sb3: Use stable-baselines3 (only for continuous actions)
        """
        super().__init__(env, config, model_name, seed, device)

        # SAC in stable-baselines3 is for continuous actions only
        # We implement discrete SAC ourselves
        self.use_sb3 = False

        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self._init_custom_model()

    @property
    def agent_type(self) -> str:
        return 'sac'

    def _init_custom_model(self):
        """Initialize custom Discrete SAC implementation"""
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        hidden_layers = self.config.hidden_layers if self.config else [256, 256]

        # Policy network
        self.policy = DiscretePolicy(state_dim, action_dim, hidden_layers).to(self.device)

        # Twin Q-networks
        self.q1 = SoftQNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.q2 = SoftQNetwork(state_dim, action_dim, hidden_layers).to(self.device)

        # Target Q-networks
        self.q1_target = SoftQNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.q2_target = SoftQNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        lr = self.config.learning_rate if self.config else 0.0003
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        # Automatic temperature tuning
        self.target_entropy = -np.log(1.0 / action_dim) * 0.98  # Target entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # Replay buffer
        buffer_size = self.config.buffer_size if self.config else 1000000
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Hyperparameters
        self.gamma = self.config.gamma if self.config else 0.99
        self.tau = self.config.tau if self.config else 0.005
        self.batch_size = self.config.batch_size if self.config else 256
        self.learning_starts = self.config.learning_starts if self.config else 1000
        self.train_freq = self.config.train_freq if self.config else 1
        self.gradient_steps = self.config.gradient_steps if self.config else 1

    @property
    def alpha(self):
        """Current temperature parameter"""
        return self.log_alpha.exp()

    def train(
        self,
        total_timesteps: int,
        callback = None,
        log_interval: int = 100
    ) -> Dict:
        """Train the SAC agent"""
        episode_rewards = []
        episode_lengths = []

        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(total_timesteps):
            # Select action
            if step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action, _, _ = self._select_action(obs)

            # Take step
            next_obs, reward, done, truncated, info = self.env.step(action)

            # Store transition
            self.replay_buffer.push(obs, action, reward, next_obs, done or truncated)

            episode_reward += reward
            episode_length += 1
            obs = next_obs

            # Learn
            if step >= self.learning_starts and step % self.train_freq == 0:
                for _ in range(self.gradient_steps):
                    self._learn()

            if done or truncated:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                self.episodes += 1

                if self.episodes % log_interval == 0:
                    mean_reward = np.mean(episode_rewards[-100:])
                    print(f"Episode {self.episodes} | Mean Reward: {mean_reward:.2f} | Alpha: {self.alpha.item():.4f}")

                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0

        self.total_timesteps += total_timesteps

        return {
            'total_timesteps': self.total_timesteps,
            'episodes': self.episodes,
            'mean_reward': np.mean(episode_rewards[-100:]) if episode_rewards else 0
        }

    def _select_action(self, state: np.ndarray, deterministic: bool = False):
        """Select action using policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, entropy = self.policy.get_action(state_tensor, deterministic)
            return action.item(), log_prob.item(), entropy.item()

    def _learn(self):
        """Perform one learning step"""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Update Q-networks
        with torch.no_grad():
            # Get next action probabilities
            next_probs = self.policy(next_states)

            # Compute next Q-values
            next_q1 = self.q1_target(next_states)
            next_q2 = self.q2_target(next_states)
            next_q = torch.min(next_q1, next_q2)

            # Compute V(s') = sum_a pi(a|s') * (Q(s',a) - alpha * log(pi(a|s')))
            next_log_probs = torch.log(next_probs + 1e-8)
            next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)

            # Compute target Q-value
            target_q = rewards + self.gamma * (1 - dones) * next_v

        # Current Q-values
        current_q1 = self.q1(states).gather(1, actions.unsqueeze(1))
        current_q2 = self.q2(states).gather(1, actions.unsqueeze(1))

        # Q-network losses
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)

        # Update Q-networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update policy
        probs = self.policy(states)
        log_probs = torch.log(probs + 1e-8)

        with torch.no_grad():
            q1_values = self.q1(states)
            q2_values = self.q2(states)
            min_q = torch.min(q1_values, q2_values)

        # Policy loss: maximize E[Q(s,a) - alpha * log(pi(a|s))]
        policy_loss = (probs * (self.alpha * log_probs - min_q)).sum(dim=1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update temperature (alpha)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        alpha_loss = (self.log_alpha * (entropy - self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update target networks
        self._soft_update()

    def _soft_update(self):
        """Soft update target networks"""
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[int, Optional[Dict]]:
        """Predict action given observation"""
        action, log_prob, entropy = self._select_action(observation, deterministic)
        return action, {'log_prob': log_prob, 'entropy': entropy}

    def get_action_probabilities(self, observation: np.ndarray) -> np.ndarray:
        """Get action probabilities"""
        with torch.no_grad():
            state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            probs = self.policy(state)
            return probs.cpu().numpy().flatten()

    def save(self, path: str = None):
        """Save model to disk"""
        if path is None:
            path = os.path.join(self.save_dir, f"{self.model_name}.zip")

        torch.save({
            'policy': self.policy.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'q1_optimizer': self.q1_optimizer.state_dict(),
            'q2_optimizer': self.q2_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'total_timesteps': self.total_timesteps,
            'episodes': self.episodes
        }, path)

        self._save_metadata(path)
        print(f"Model saved to {path}")

    def load(self, path: str = None):
        """Load model from disk"""
        if path is None:
            path = os.path.join(self.save_dir, f"{self.model_name}.zip")

        checkpoint = torch.load(path, map_location=self.device)

        self.policy.load_state_dict(checkpoint['policy'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.q1_target.load_state_dict(checkpoint['q1_target'])
        self.q2_target.load_state_dict(checkpoint['q2_target'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.total_timesteps = checkpoint['total_timesteps']
        self.episodes = checkpoint['episodes']

        metadata = self._load_metadata(path)
        print(f"Model loaded from {path}")

        return metadata
