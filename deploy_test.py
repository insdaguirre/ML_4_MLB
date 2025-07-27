#!/usr/bin/env python3
"""
Deployment Test Script for MLB Betting RL System
Tests all components and provides deployment status
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_environment():
    """Test Python environment and dependencies"""
    logger.info("🔧 Testing environment...")
    
    try:
        import torch
        import stable_baselines3
        import gymnasium
        import pandas
        import polars
        import numpy
        import requests
        logger.info("✅ All dependencies installed")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False

def test_cpu_optimization():
    """Test CPU optimization settings"""
    logger.info("⚡ Testing CPU optimization...")
    
    # Set environment variables
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    try:
        import torch
        torch.set_num_threads(1)
        logger.info(f"✅ CPU optimization: PyTorch threads = {torch.get_num_threads()}")
        return True
    except Exception as e:
        logger.error(f"❌ CPU optimization failed: {e}")
        return False

def test_data_sources():
    """Test real data sources"""
    logger.info("📊 Testing data sources...")
    
    try:
        from src.data.real_data_sources import RealDataSources
        data_sources = RealDataSources()
        
        # Test data processing
        features_df = data_sources.process_real_data()
        logger.info(f"✅ Data sources: {len(features_df)} games processed")
        return True
    except Exception as e:
        logger.error(f"❌ Data sources failed: {e}")
        return False

def test_simulation():
    """Test Monte Carlo simulation"""
    logger.info("🎲 Testing simulation...")
    
    try:
        from src.simulation.game_simulator import MonteCarloSimulator
        simulator = MonteCarloSimulator()
        
        # Test simulation
        home_stats = {"team_ops": 0.750, "team_wrc": 100}
        away_stats = {"team_ops": 0.740, "team_wrc": 98}
        result = simulator.simulate_game_vectorized(home_stats, away_stats, num_sims=100)
        logger.info(f"✅ Simulation: {result.shape} games simulated")
        logger.info(f"✅ Simulation: {result}")
        return True
    except Exception as e:
        logger.error(f"❌ Simulation failed: {e}")
        return False

def test_rl_environment():
    """Test RL environment"""
    logger.info("🤖 Testing RL environment...")
    
    try:
        from src.rl.baseball_env import BaseballBetEnv
        from src.data.data_pipeline import MLBDataPipeline
        
        # Create data pipeline for environment
        data_pipeline = MLBDataPipeline()
        env = BaseballBetEnv(data_pipeline)
        
        # Test environment
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        logger.info(f"✅ RL environment: observation shape = {obs.shape}")
        return True
    except Exception as e:
        logger.error(f"❌ RL environment failed: {e}")
        return False

def test_kelly_betting():
    """Test Kelly betting system"""
    logger.info("💰 Testing Kelly betting...")
    
    try:
        from src.betting.kelly_betting import KellyBettingSystem
        kelly = KellyBettingSystem()
        
        # Test Kelly calculation (convert moneyline to decimal odds)
        decimal_odds = kelly.moneyline_to_decimal(-110)
        bet_fraction = kelly.calculate_kelly_fraction(0.55, decimal_odds)
        logger.info(f"✅ Kelly betting: bet fraction = {bet_fraction:.4f}")
        return True
    except Exception as e:
        logger.error(f"❌ Kelly betting failed: {e}")
        return False

def test_live_betting():
    """Test live betting system"""
    logger.info("📱 Testing live betting...")
    
    try:
        from src.live_betting import LiveBettingLauncher
        launcher = LiveBettingLauncher()
        
        # Test database initialization
        launcher._init_database()
        logger.info("✅ Live betting: database initialized")
        return True
    except Exception as e:
        logger.error(f"❌ Live betting failed: {e}")
        return False

def test_api_keys():
    """Test API key configuration"""
    logger.info("🔑 Testing API keys...")
    
    keys_to_check = [
        ('ODDS_API_KEY', 'The Odds API'),
        ('TELEGRAM_BOT_TOKEN', 'Telegram Bot'),
        ('TELEGRAM_CHAT_ID', 'Telegram Chat ID'),
    ]
    
    missing_keys = []
    for key, name in keys_to_check:
        if not os.getenv(key):
            missing_keys.append(name)
    
    if missing_keys:
        logger.warning(f"⚠️ Missing API keys: {', '.join(missing_keys)}")
        logger.info("💡 Set these in your .env file or environment variables")
    else:
        logger.info("✅ All API keys configured")
    
    return len(missing_keys) == 0

def test_file_structure():
    """Test file structure"""
    logger.info("📁 Testing file structure...")
    
    required_dirs = ['data', 'models', 'logs', 'config']
    required_files = [
        'requirements.txt',
        'README.md',
        'DEPLOYMENT_GUIDE.md',
        'src/data/real_data_sources.py',
        'src/simulation/game_simulator.py',
        'src/rl/baseball_env.py',
        'src/betting/kelly_betting.py',
        'src/live_betting.py'
    ]
    
    missing_dirs = []
    missing_files = []
    
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing_dirs.append(dir_name)
    
    for file_name in required_files:
        if not Path(file_name).exists():
            missing_files.append(file_name)
    
    if missing_dirs or missing_files:
        logger.error(f"❌ Missing directories: {missing_dirs}")
        logger.error(f"❌ Missing files: {missing_files}")
        return False
    else:
        logger.info("✅ File structure complete")
        return True

def main():
    """Run all deployment tests"""
    logger.info("🚀 Starting deployment tests...")
    
    tests = [
        ("Environment", test_environment),
        ("CPU Optimization", test_cpu_optimization),
        ("File Structure", test_file_structure),
        ("Data Sources", test_data_sources),
        ("Simulation", test_simulation),
        ("RL Environment", test_rl_environment),
        ("Kelly Betting", test_kelly_betting),
        ("Live Betting", test_live_betting),
        ("API Keys", test_api_keys),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("📊 DEPLOYMENT TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    logger.info("="*50)
    logger.info(f"Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! System ready for deployment.")
        logger.info("📖 Next steps: Follow DEPLOYMENT_GUIDE.md")
    else:
        logger.error("⚠️ Some tests failed. Check errors above.")
        logger.info("🔧 Fix issues before deployment.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 