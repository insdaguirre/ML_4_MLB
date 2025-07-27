"""
PPO Training Script for MLB Betting System
Phase 3: Train RL agent on CPU with tiny network
"""

import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os
import logging
from pathlib import Path
import time
from typing import Dict, Any

from .baseball_env import create_env

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force CPU-only and single thread for optimal performance
torch.set_num_threads(1)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

class TinyMLPNetwork(nn.Module):
    """
    Tiny network architecture as specified in roadmap:
    [Linear(128) → ReLU → Linear(64) → Tanh]
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        
        # Policy head (actor)
        self.policy_net = nn.Linear(64, output_dim)
        
        # Value head (critic)
        self.value_net = nn.Linear(64, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through shared layers"""
        shared_features = self.shared_net(x)
        return shared_features
    
    def get_policy(self, x: torch.Tensor) -> torch.Tensor:
        """Get policy output"""
        shared_features = self.forward(x)
        return self.policy_net(shared_features)
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get value output"""
        shared_features = self.forward(x)
        return self.value_net(shared_features)

class CustomPPO(PPO):
    """Custom PPO with tiny network architecture"""
    
    def __init__(self, *args, **kwargs):
        # Override network architecture
        kwargs['policy_kwargs'] = {
            'net_arch': [dict(pi=[128, 64], vf=[128, 64])],
            'activation_fn': nn.ReLU,
            'final_activation_fn': nn.Tanh
        }
        super().__init__(*args, **kwargs)

def create_training_env(data_dir: str = "data", num_envs: int = 6):
    """Create vectorized training environment"""
    # Create base environment
    env = create_env(data_dir, num_envs)
    
    # Wrap with normalization
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    return env

def create_eval_env(data_dir: str = "data", num_envs: int = 2):
    """Create evaluation environment"""
    env = create_env(data_dir, num_envs)
    
    # Wrap with normalization (but don't update stats during eval)
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize rewards during eval
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    return env

def train_ppo_model(
    total_timesteps: int = 2_000_000,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_steps: int = 2048,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    data_dir: str = "data",
    model_dir: str = "models",
    log_dir: str = "logs",
    num_envs: int = 6
):
    """
    Train PPO model with CPU optimization
    """
    logger.info("Starting PPO training...")
    
    # Create directories
    Path(model_dir).mkdir(exist_ok=True)
    Path(log_dir).mkdir(exist_ok=True)
    
    # Create environments
    train_env = create_training_env(data_dir, num_envs)
    eval_env = create_eval_env(data_dir, 2)
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=200_000,  # Save every 200k steps as specified
        save_path=model_dir,
        name_prefix="mlb_ppo"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_dir}/best_model",
        log_path=log_dir,
        eval_freq=50_000,
        deterministic=True,
        render=False
    )
    
    # Create model
    model = CustomPPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
        tensorboard_log=log_dir
    )
    
    # Train model
    logger.info(f"Training for {total_timesteps} timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Average speed: {total_timesteps / training_time:.0f} steps/second")
    
    # Save final model
    final_model_path = f"{model_dir}/mlb_ppo_final"
    model.save(final_model_path)
    
    # Save environment normalization
    train_env.save(f"{model_dir}/vec_normalize.pkl")
    
    logger.info(f"Model saved to {final_model_path}")
    
    return model, train_env

def evaluate_model(model_path: str, data_dir: str = "data", num_episodes: int = 100):
    """Evaluate trained model"""
    logger.info(f"Evaluating model from {model_path}")
    
    # Load model
    model = CustomPPO.load(model_path)
    
    # Create evaluation environment
    eval_env = create_eval_env(data_dir, 1)
    
    # Load normalization stats
    vec_normalize_path = model_path.replace("mlb_ppo", "vec_normalize")
    if os.path.exists(vec_normalize_path + ".pkl"):
        eval_env = VecNormalize.load(vec_normalize_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    
    # Run evaluation
    episode_rewards = []
    episode_stats = []
    
    for episode in range(num_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            episode_reward += reward[0]
        
        episode_rewards.append(episode_reward)
        
        # Get episode stats
        if hasattr(eval_env.envs[0], 'get_episode_stats'):
            stats = eval_env.envs[0].get_episode_stats()
            episode_stats.append(stats)
    
    # Calculate metrics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    logger.info(f"Evaluation Results:")
    logger.info(f"Average episode reward: {avg_reward:.2f} ± {std_reward:.2f}")
    logger.info(f"Min reward: {np.min(episode_rewards):.2f}")
    logger.info(f"Max reward: {np.max(episode_rewards):.2f}")
    
    if episode_stats:
        avg_roi = np.mean([stats.get('roi', 0) for stats in episode_stats])
        avg_sharpe = np.mean([stats.get('sharpe_ratio', 0) for stats in episode_stats])
        avg_drawdown = np.mean([stats.get('max_drawdown', 0) for stats in episode_stats])
        
        logger.info(f"Average ROI: {avg_roi:.3f}")
        logger.info(f"Average Sharpe: {avg_sharpe:.3f}")
        logger.info(f"Average Max Drawdown: {avg_drawdown:.3f}")
    
    return episode_rewards, episode_stats

def main():
    """Main training function"""
    # Training parameters
    config = {
        'total_timesteps': 2_000_000,  # 2M steps as specified
        'learning_rate': 3e-4,
        'batch_size': 64,
        'n_steps': 2048,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'num_envs': 6  # Exactly 6 cores as specified
    }
    
    logger.info("MLB Betting RL Training")
    logger.info(f"Config: {config}")
    
    # Train model
    model, train_env = train_ppo_model(**config)
    
    # Evaluate model
    evaluate_model("models/mlb_ppo_final")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 