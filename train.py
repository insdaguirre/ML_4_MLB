#!/usr/bin/env python3
"""
Main Training Script for MLB Betting System
CPU-optimized training with 6-core parallelization
"""

import os
import sys
import logging
from pathlib import Path
import time
import argparse

# Force CPU-only and single thread for optimal performance
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.data_pipeline import MLBDataPipeline
from src.rl.train import train_ppo_model, evaluate_model
from src.utils.backtest import BacktestingSystem
from src.live_betting import LiveBettingLauncher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup CPU-optimized environment"""
    logger.info("Setting up CPU-optimized environment...")
    
    # Force single-threaded operations
    import torch
    torch.set_num_threads(1)
    
    logger.info("Environment setup complete")

def run_data_pipeline():
    """Run Phase 1: Data pipeline"""
    logger.info("=== Phase 1: Data Pipeline ===")
    
    pipeline = MLBDataPipeline()
    
    # Download and process data
    pipeline.download_retrosheet_data(2010, 2023)
    pipeline.download_odds_data(2010, 2023)
    features_df = pipeline.process_features()
    
    logger.info(f"Data pipeline complete. Processed {len(features_df)} games.")
    return pipeline

def run_training():
    """Run Phase 3: RL training"""
    logger.info("=== Phase 3: RL Training ===")
    
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
    
    logger.info(f"Training config: {config}")
    
    # Train model
    model, train_env = train_ppo_model(**config)
    
    logger.info("Training complete!")
    return model

def run_backtesting():
    """Run Phase 5: Backtesting"""
    logger.info("=== Phase 5: Backtesting ===")
    
    backtest = BacktestingSystem()
    
    # Run walk-forward backtest
    results = backtest.walk_forward_backtest(2010, 2023)
    
    logger.info("Backtesting Results:")
    logger.info(f"Average ROI: {results['avg_roi']:.3f}")
    logger.info(f"Average Sharpe: {results['avg_sharpe']:.3f}")
    logger.info(f"Consistency: {results['consistency']:.3f}")
    logger.info(f"Cumulative ROI: {results['cumulative_roi']:.3f}")
    
    # Run Monte-Carlo validation if model exists
    model_path = "models/mlb_ppo_final.zip"
    if Path(model_path).exists():
        logger.info("Running Monte-Carlo validation...")
        mc_results = backtest.monte_carlo_validation(model_path, 2019)
        
        logger.info("Monte-Carlo Validation Results:")
        logger.info(f"Expected ROI: {mc_results['expected_roi']:.3f}")
        logger.info(f"ROI Volatility: {mc_results['roi_volatility']:.3f}")
        logger.info(f"Sharpe Ratio: {mc_results['sharpe_ratio']:.3f}")
        logger.info(f"P(ROI > 0): {mc_results['prob_positive_roi']:.3f}")
        
        # Plot results
        backtest.plot_results(mc_results, "backtest_results.png")
    
    return results

def run_live_test():
    """Run Phase 6: Live betting test"""
    logger.info("=== Phase 6: Live Betting Test ===")
    
    launcher = LiveBettingLauncher()
    
    # Run daily analysis
    launcher.run_daily_analysis()
    
    # Get performance stats
    stats = launcher.get_performance_stats()
    if stats:
        logger.info("Performance Stats (Last 7 days):")
        logger.info(f"Total suggestions: {stats['total_suggestions']}")
        logger.info(f"Total stake: ${stats['total_stake']:.2f}")
        logger.info(f"Average edge: {stats['avg_edge']*100:.2f}%")
        logger.info(f"Average confidence: {stats['avg_confidence']*100:.1f}%")

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description="MLB Betting RL Training")
    parser.add_argument("--phase", type=str, choices=["all", "data", "train", "backtest", "live"],
                       default="all", help="Which phase to run")
    parser.add_argument("--timesteps", type=int, default=2_000_000,
                       help="Number of training timesteps")
    parser.add_argument("--num-envs", type=int, default=6,
                       help="Number of parallel environments")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting MLB Betting RL System")
    logger.info(f"Phase: {args.phase}")
    logger.info(f"Timesteps: {args.timesteps}")
    logger.info(f"Parallel environments: {args.num_envs}")
    
    # Setup environment
    setup_environment()
    
    start_time = time.time()
    
    try:
        if args.phase in ["all", "data"]:
            run_data_pipeline()
        
        if args.phase in ["all", "train"]:
            run_training()
        
        if args.phase in ["all", "backtest"]:
            run_backtesting()
        
        if args.phase in ["all", "live"]:
            run_live_test()
        
        total_time = time.time() - start_time
        logger.info(f"✅ Pipeline completed in {total_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main() 