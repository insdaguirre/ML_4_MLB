#!/usr/bin/env python3
"""
Production PPO Training Script for MLB Betting System
Full training with 2M timesteps and optimized hyperparameters
"""

import os
import torch
import numpy as np
import logging
from pathlib import Path

# CPU optimization
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.rl.baseball_env import BaseballBetEnv
from src.data.data_pipeline import MLBDataPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_env():
    """Environment factory function"""
    data_pipeline = MLBDataPipeline("data")
    env = BaseballBetEnv(
        data_pipeline=data_pipeline,
        initial_bankroll=10000.0,
        num_sims=1000,  # Full simulation
        max_games=162   # Full season
    )
    return Monitor(env)

def main():
    """Production training script"""
    logger.info("🚀 Starting production MLB RL training...")
    
    # Create directories
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("eval").mkdir(exist_ok=True)
    
    # Create training environment
    logger.info("Creating training environment...")
    train_env = DummyVecEnv([make_env for _ in range(6)])  # 6 parallel environments
    
    # Create evaluation environment
    logger.info("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env for _ in range(2)])
    
    # Create PPO model with production configuration
    logger.info("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,    # Larger batch for stability
        batch_size=64,    # Larger batch size
        n_epochs=10,      # More epochs per update
        gamma=0.99,       # Discount factor
        gae_lambda=0.95,  # GAE lambda
        clip_range=0.2,   # PPO clip range
        ent_coef=0.01,   # Entropy coefficient
        vf_coef=0.5,      # Value function coefficient
        max_grad_norm=0.5, # Gradient clipping
        verbose=1,
        device='cpu'
    )
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="models/",
        name_prefix="mlb_ppo_production"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/best_model/",
        log_path="logs/",
        eval_freq=25000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    # Train for full 2M timesteps
    logger.info("Starting production training for 2,000,000 timesteps...")
    logger.info("This will take approximately 2-4 hours...")
    
    try:
        model.learn(
            total_timesteps=2000000,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
        
        # Save final model
        model.save("models/mlb_ppo_production_final")
        logger.info("✅ Production training completed successfully!")
        
        # Test the trained model
        logger.info("Testing production model...")
        obs = eval_env.reset()
        total_reward = 0
        for i in range(20):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            total_reward += float(reward[0])
            logger.info(f"Step {i}: action={float(action[0]):.4f}, reward={float(reward[0]):.4f}")
            if done[0]:
                obs = eval_env.reset()
                logger.info(f"Episode finished, total reward: {total_reward:.2f}")
                total_reward = 0
        
        logger.info("✅ Production model test completed!")
        
    except Exception as e:
        logger.error(f"❌ Production training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 