#!/usr/bin/env python3
"""
Simplified PPO Training Script for MLB Betting System
Focus on getting the basic training loop working
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
from stable_baselines3.common.callbacks import CheckpointCallback

from src.rl.baseball_env import BaseballBetEnv
from src.data.data_pipeline import MLBDataPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_env():
    """Environment factory function"""
    data_pipeline = MLBDataPipeline("data")
    return BaseballBetEnv(
        data_pipeline=data_pipeline,
        initial_bankroll=10000.0,
        num_sims=100,  # Reduced for faster training
        max_games=50   # Reduced for faster episodes
    )

def main():
    """Simple training script"""
    logger.info("🚀 Starting simplified MLB RL training...")
    
    # Create directories
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Create single environment first (simpler)
    logger.info("Creating environment...")
    env = DummyVecEnv([make_env])
    
    # Create PPO model with simple configuration
    logger.info("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,     # Reduced for faster training
        batch_size=32,   # Reduced for faster training
        n_epochs=5,      # Reduced for faster training
        verbose=1,
        device='cpu'
    )
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="models/",
        name_prefix="simple_mlb_ppo"
    )
    
    # Train for shorter time to test
    logger.info("Starting training for 50,000 timesteps...")
    try:
        model.learn(
            total_timesteps=50000,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # Save final model
        model.save("models/simple_mlb_ppo_final")
        logger.info("✅ Training completed successfully!")
        
        # Test the trained model
        logger.info("Testing trained model...")
        obs = env.reset()
        for i in range(10):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            logger.info(f"Step {i}: action={float(action[0]):.4f}, reward={float(reward[0]):.4f}")
            if done[0]:
                obs = env.reset()
                logger.info("Episode finished, resetting...")
        
        logger.info("✅ Model test completed!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 