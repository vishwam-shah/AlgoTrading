"""
Proximal Policy Optimization (PPO) Agent
========================================

PPO agent with:
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Value function clipping
- Entropy bonus for exploration
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .base_agent import BaseAgent

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Try to import stable-baselines3
try:
    from stable_baselines3 import PPO as SB3_PPO
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.

    Shared feature extractor with separate policy (actor) and value (critic) heads.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: List[int] = [128, 64]
    ):
        super().__init__()

        # Shared feature extractor
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh(),
            ])
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*layers)

        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(prev_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic (value) head
        self.critic = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

    def get_action(self, x: torch.Tensor, deterministic: bool = False):
        action_probs, value = self.forward(x)
        dist = Categorical(action_probs)

        if deterministic:
            action = action_probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value


class RolloutBuffer:
    """Buffer for storing rollout experiences"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def get(self):
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.values),
            np.array(self.log_probs),
            np.array(self.dones)
        )

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def __len__(self):
        return len(self.states)


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization Agent.

    Features:
    - Clipped surrogate objective for stable policy updates
    - Generalized Advantage Estimation (GAE)
    - Entropy bonus for exploration
    - Value function clipping
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
        Initialize PPO agent.

        Args:
            env: Gymnasium environment
            config: PPO configuration
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
        return 'ppo'

    def _init_sb3_model(self):
        """Initialize stable-baselines3 PPO"""
        policy_kwargs = {
            'net_arch': {
                'pi': self.config.policy_hidden_layers if self.config else [128, 64],
                'vf': self.config.value_hidden_layers if self.config else [128, 64]
            }
        }

        self.model = SB3_PPO(
            policy='MlpPolicy',
            env=self.env,
            learning_rate=self.config.learning_rate if self.config else 0.0003,
            n_steps=self.config.n_steps if self.config else 2048,
            batch_size=self.config.batch_size if self.config else 64,
            n_epochs=self.config.n_epochs if self.config else 10,
            gamma=self.config.gamma if self.config else 0.99,
            gae_lambda=self.config.gae_lambda if self.config else 0.95,
            clip_range=self.config.clip_range if self.config else 0.2,
            ent_coef=self.config.ent_coef if self.config else 0.01,
            vf_coef=self.config.vf_coef if self.config else 0.5,
            max_grad_norm=self.config.max_grad_norm if self.config else 0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=self.seed,
            device=self.device
        )

    def _init_custom_model(self):
        """Initialize custom PPO implementation"""
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        hidden_layers = self.config.policy_hidden_layers if self.config else [128, 64]

        # Actor-Critic network
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_layers).to(self.device)

        # Optimizer
        lr = self.config.learning_rate if self.config else 0.0003
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer()

        # Hyperparameters
        self.gamma = self.config.gamma if self.config else 0.99
        self.gae_lambda = self.config.gae_lambda if self.config else 0.95
        self.clip_range = self.config.clip_range if self.config else 0.2
        self.ent_coef = self.config.ent_coef if self.config else 0.01
        self.vf_coef = self.config.vf_coef if self.config else 0.5
        self.max_grad_norm = self.config.max_grad_norm if self.config else 0.5
        self.n_epochs = self.config.n_epochs if self.config else 10
        self.batch_size = self.config.batch_size if self.config else 64
        self.n_steps = self.config.n_steps if self.config else 2048

    def train(
        self,
        total_timesteps: int,
        callback = None,
        log_interval: int = 1
    ) -> Dict:
        """Train the PPO agent"""
        if self.use_sb3:
            return self._train_sb3(total_timesteps, callback, log_interval)
        else:
            return self._train_custom(total_timesteps, log_interval)

    def _train_sb3(self, total_timesteps: int, callback, log_interval: int) -> Dict:
        """Train using stable-baselines3"""
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
            # Collect rollout
            with torch.no_grad():
                state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action, log_prob, _, value = self.actor_critic.get_action(state)
                action = action.item()
                log_prob = log_prob.item()
                value = value.item()

            next_obs, reward, done, truncated, info = self.env.step(action)

            self.rollout_buffer.add(obs, action, reward, value, log_prob, done or truncated)

            episode_reward += reward
            episode_length += 1
            obs = next_obs

            if done or truncated:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                self.episodes += 1

                if self.episodes % log_interval == 0:
                    mean_reward = np.mean(episode_rewards[-100:])
                    print(f"Episode {self.episodes} | Mean Reward: {mean_reward:.2f}")

                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0

            # Update policy after n_steps
            if len(self.rollout_buffer) >= self.n_steps:
                self._update_policy()
                self.rollout_buffer.clear()

        self.total_timesteps += total_timesteps

        return {
            'total_timesteps': self.total_timesteps,
            'episodes': self.episodes,
            'mean_reward': np.mean(episode_rewards[-100:]) if episode_rewards else 0
        }

    def _compute_gae(self, rewards, values, dones, last_value):
        """Compute Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def _update_policy(self):
        """Update policy using collected rollout"""
        states, actions, rewards, values, old_log_probs, dones = self.rollout_buffer.get()

        # Get last value for GAE computation
        with torch.no_grad():
            last_state = torch.FloatTensor(states[-1]).unsqueeze(0).to(self.device)
            _, last_value = self.actor_critic(last_state)
            last_value = last_value.item()

        # Compute advantages and returns
        advantages, returns = self._compute_gae(rewards, values, dones, last_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Multiple epochs of updates
        for _ in range(self.n_epochs):
            # Mini-batch updates
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Forward pass
                action_probs, values = self.actor_critic(batch_states)
                dist = Categorical(action_probs)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy()

                # Ratio for clipping
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * (values.squeeze() - batch_returns).pow(2).mean()

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

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
            with torch.no_grad():
                state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                action, _, _, value = self.actor_critic.get_action(state, deterministic=deterministic)
                return action.item(), {'value': value.item()}

    def get_action_probabilities(self, observation: np.ndarray) -> np.ndarray:
        """Get action probabilities"""
        with torch.no_grad():
            if self.use_sb3:
                obs = torch.FloatTensor(observation).unsqueeze(0)
                action_probs = self.model.policy.get_distribution(obs).distribution.probs
            else:
                state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                action_probs, _ = self.actor_critic(state)

            return action_probs.cpu().numpy().flatten()

    def save(self, path: str = None):
        """Save model to disk"""
        if path is None:
            path = os.path.join(self.save_dir, f"{self.model_name}.zip")

        if self.use_sb3:
            self.model.save(path.replace('.zip', ''))
        else:
            torch.save({
                'actor_critic': self.actor_critic.state_dict(),
                'optimizer': self.optimizer.state_dict(),
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
            self.model = SB3_PPO.load(path.replace('.zip', ''), env=self.env)
        else:
            checkpoint = torch.load(path, map_location=self.device)
            self.actor_critic.load_state_dict(checkpoint['actor_critic'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.total_timesteps = checkpoint['total_timesteps']
            self.episodes = checkpoint['episodes']

        metadata = self._load_metadata(path)
        print(f"Model loaded from {path}")

        return metadata
