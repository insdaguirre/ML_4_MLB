"""
Baseball Betting Environment for RL Agent
Phase 3: Gym-style environment with continuous actions
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

from ..data.data_pipeline import MLBDataPipeline
from ..simulation.game_simulator import simulate_game

logger = logging.getLogger(__name__)

class BaseballBetEnv(gym.Env):
    """
    Gym-style environment for MLB betting with continuous actions
    Action space: bet fraction ∈ [-0.05, +0.05] (Kelly cap 5%)
    State space: game features + current bankroll
    """
    
    def __init__(self, data_pipeline: MLBDataPipeline, initial_bankroll: float = 10000.0,
                 num_sims: int = 1000, max_games: int = 162):
        super().__init__()
        
        self.data_pipeline = data_pipeline
        self.initial_bankroll = initial_bankroll
        self.num_sims = num_sims
        self.max_games = max_games
        
        # Load training data
        self.games_data = self.data_pipeline.get_training_data(2010, 2018)
        self.current_game_idx = 0
        self.current_bankroll = initial_bankroll
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-0.05, high=0.05, shape=(1,), dtype=np.float32
        )
        
        # State space: game features + bankroll
        feature_columns = [
            'home_pitcher_era', 'away_pitcher_era', 'home_bullpen_fip', 'away_bullpen_fip',
            'park_factor', 'pitcher_quality_diff', 'bullpen_quality_diff', 
            'park_adjusted_home_runs', 'park_adjusted_away_runs', 'edge',
            'total_runs', 'kelly_fraction', 'home_implied_prob', 'away_implied_prob'
        ]
        
        self.feature_columns = feature_columns
        self.state_dim = len(feature_columns) + 1  # +1 for bankroll
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_bankrolls = []
        self.episode_bets = []
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        # Reset episode state
        self.current_game_idx = 0
        self.current_bankroll = self.initial_bankroll
        self.episode_rewards = []
        self.episode_bankrolls = [self.current_bankroll]
        self.episode_bets = []
        
        # Get initial state
        state = self._get_state()
        
        return state, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take action and return next state, reward, done, truncated, info
        Action: bet fraction ∈ [-0.05, 0.05]
        """
        # Clip action to valid range
        bet_fraction = np.clip(action[0], -0.05, 0.05)
        
        # Get current game data
        if self.current_game_idx >= len(self.games_data):
            # Episode finished
            return self._get_state(), 0.0, True, False, {}
        
        game_row = self.games_data.row(self.current_game_idx, named=True)
        
        # Simulate game and betting outcome
        game_results, payoffs = simulate_game(game_row, self.num_sims)
        
        # Calculate average payoff across simulations
        avg_payoff = payoffs.mean()
        
        # Apply bet to bankroll
        bankroll_change = avg_payoff * self.current_bankroll
        self.current_bankroll += bankroll_change
        
        # Calculate reward (bankroll delta)
        reward = bankroll_change
        
        # Track episode data
        self.episode_rewards.append(reward)
        self.episode_bankrolls.append(self.current_bankroll)
        self.episode_bets.append(bet_fraction)
        
        # Move to next game
        self.current_game_idx += 1
        
        # Check if episode is done
        done = (self.current_game_idx >= len(self.games_data) or 
                self.current_game_idx >= self.max_games or
                self.current_bankroll <= 0)
        
        # Get next state
        next_state = self._get_state()
        
        # Info dict
        info = {
            'game_idx': self.current_game_idx,
            'bankroll': self.current_bankroll,
            'bet_fraction': bet_fraction,
            'avg_payoff': avg_payoff,
            'game_results': game_results,
            'payoffs': payoffs
        }
        
        return next_state, reward, done, False, info
    
    def _get_state(self) -> np.ndarray:
        """Get current state vector"""
        if self.current_game_idx >= len(self.games_data):
            # Return zero state if no more games
            return np.zeros(self.state_dim, dtype=np.float32)
        
        # Get game features
        game_row = self.games_data.row(self.current_game_idx, named=True)
        
        # Extract features
        features = []
        for col in self.feature_columns:
            if col in game_row:
                features.append(float(game_row[col]))
            else:
                features.append(0.0)  # Default value
        
        # Add bankroll (normalized)
        normalized_bankroll = (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll
        features.append(normalized_bankroll)
        
        return np.array(features, dtype=np.float32)
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get episode statistics"""
        if not self.episode_rewards:
            return {}
        
        total_reward = sum(self.episode_rewards)
        final_bankroll = self.episode_bankrolls[-1] if self.episode_bankrolls else self.initial_bankroll
        roi = (final_bankroll - self.initial_bankroll) / self.initial_bankroll
        
        # Calculate Sharpe ratio (simplified)
        if len(self.episode_rewards) > 1:
            sharpe = np.mean(self.episode_rewards) / (np.std(self.episode_rewards) + 1e-8)
        else:
            sharpe = 0.0
        
        # Calculate max drawdown
        peak = self.initial_bankroll
        max_drawdown = 0.0
        for bankroll in self.episode_bankrolls:
            if bankroll > peak:
                peak = bankroll
            drawdown = (peak - bankroll) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'total_reward': total_reward,
            'final_bankroll': final_bankroll,
            'roi': roi,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_games': len(self.episode_rewards),
            'avg_bet_fraction': np.mean(self.episode_bets) if self.episode_bets else 0.0,
            'bet_std': np.std(self.episode_bets) if self.episode_bets else 0.0
        }

class VecBaseballEnv:
    """Vectorized environment wrapper for parallel training"""
    
    def __init__(self, num_envs: int = 6, **env_kwargs):
        self.num_envs = num_envs
        self.envs = [BaseballBetEnv(**env_kwargs) for _ in range(num_envs)]
        
        # Use first env to get spaces
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        
    def reset(self) -> Tuple[np.ndarray, List[Dict]]:
        """Reset all environments"""
        states = []
        infos = []
        
        for env in self.envs:
            state, info = env.reset()
            states.append(state)
            infos.append(info)
        
        return np.array(states), infos
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments"""
        states = []
        rewards = []
        dones = []
        truncated = []
        infos = []
        
        for i, env in enumerate(self.envs):
            state, reward, done, trunc, info = env.step(actions[i])
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            truncated.append(trunc)
            infos.append(info)
        
        return (np.array(states), np.array(rewards), 
                np.array(dones), np.array(truncated), infos)
    
    def get_episode_stats(self) -> List[Dict[str, Any]]:
        """Get episode stats for all environments"""
        return [env.get_episode_stats() for env in self.envs]

def create_env(data_dir: str = "data", num_envs: int = 6) -> VecBaseballEnv:
    """Factory function to create vectorized environment"""
    data_pipeline = MLBDataPipeline(data_dir)
    
    # Ensure data is processed
    if not (Path(data_dir) / "processed_features.parquet").exists():
        logger.info("Processing data pipeline...")
        data_pipeline.download_retrosheet_data(2010, 2023)
        data_pipeline.download_odds_data(2010, 2023)
        data_pipeline.process_features()
    
    return VecBaseballEnv(
        num_envs=num_envs,
        data_pipeline=data_pipeline,
        initial_bankroll=10000.0,
        num_sims=1000,
        max_games=162
    )

if __name__ == "__main__":
    # Test the environment
    env = create_env(num_envs=2)
    
    print("Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test a few steps
    states, infos = env.reset()
    print(f"Initial states shape: {states.shape}")
    
    for step in range(5):
        actions = np.random.uniform(-0.05, 0.05, (2, 1))
        states, rewards, dones, truncated, infos = env.step(actions)
        print(f"Step {step}: rewards={rewards}, dones={dones}")
        
        if dones.any():
            break
    
    # Get episode stats
    stats = env.get_episode_stats()
    print(f"Episode stats: {stats}") 