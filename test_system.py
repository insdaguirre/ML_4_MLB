#!/usr/bin/env python3
"""
Test script for MLB Betting RL System
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_pipeline():
    """Test data pipeline"""
    logger.info("Testing data pipeline...")
    
    try:
        from src.data.data_pipeline import MLBDataPipeline
        
        pipeline = MLBDataPipeline()
        pipeline.download_retrosheet_data(2010, 2011)  # Small test
        pipeline.download_odds_data(2010, 2011)
        features_df = pipeline.process_features()
        
        logger.info(f"✅ Data pipeline test passed. Processed {len(features_df)} games.")
        return True
    except Exception as e:
        logger.error(f"❌ Data pipeline test failed: {e}")
        return False

def test_simulation():
    """Test game simulation"""
    logger.info("Testing game simulation...")
    
    try:
        from src.simulation.game_simulator import simulate_game
        import pandas as pd
        
        # Create sample game data
        sample_game = pd.Series({
            'home_pitcher_era': 3.8,
            'away_pitcher_era': 4.2,
            'home_bullpen_fip': 3.9,
            'away_bullpen_fip': 4.1,
            'park_factor': 1.05,
            'home_moneyline': -120
        })
        
        # Simulate game
        game_results, payoffs = simulate_game(sample_game, num_sims=100)
        
        logger.info(f"✅ Simulation test passed. Simulated {len(payoffs)} outcomes.")
        return True
    except Exception as e:
        logger.error(f"❌ Simulation test failed: {e}")
        return False

def test_kelly_betting():
    """Test Kelly betting system"""
    logger.info("Testing Kelly betting system...")
    
    try:
        from src.betting.kelly_betting import KellyBettingSystem
        
        kelly = KellyBettingSystem()
        
        # Sample game features
        game_features = {
            'home_moneyline': -120,
            'home_implied_prob': 0.545,
            'edge': 0.045,
            'pitcher_quality_diff': 0.4,
            'bullpen_quality_diff': 0.2
        }
        
        # Make betting decision
        decision = kelly.make_betting_decision(game_features, 10000.0)
        
        logger.info(f"✅ Kelly betting test passed. Bet fraction: {decision.bet_fraction:.4f}")
        return True
    except Exception as e:
        logger.error(f"❌ Kelly betting test failed: {e}")
        return False

def test_environment():
    """Test RL environment"""
    logger.info("Testing RL environment...")
    
    try:
        from src.rl.baseball_env import create_env
        
        # Create environment
        env = create_env(num_envs=2)  # Small test
        
        # Test reset and step
        states, infos = env.reset()
        actions = [[0.02], [0.01]]  # Sample actions
        states, rewards, dones, truncated, infos = env.step(actions)
        
        logger.info(f"✅ Environment test passed. States shape: {states.shape}")
        return True
    except Exception as e:
        logger.error(f"❌ Environment test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🧪 Starting MLB Betting RL System Tests")
    
    tests = [
        ("Data Pipeline", test_data_pipeline),
        ("Simulation", test_simulation),
        ("Kelly Betting", test_kelly_betting),
        ("RL Environment", test_environment),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        if test_func():
            passed += 1
        else:
            logger.error(f"❌ {test_name} test failed")
    
    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready to use.")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 