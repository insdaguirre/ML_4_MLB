#!/usr/bin/env python3
"""
Live Deployment Script for MLB Betting RL System
Sets up the complete production environment
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add project to path
sys.path.append('.')

from src.live_betting import LiveBettingLauncher
from src.data.real_data_sources import RealDataSources

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if environment is properly configured"""
    logger.info("🔍 Checking environment setup...")
    
    # Check required files
    required_files = [
        "models/simple_mlb_ppo_final.zip",
        "data/processed_features.parquet",
        "src/live_betting.py",
        "src/betting/kelly_betting.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"❌ Missing required files: {missing_files}")
        return False
    
    # Check API key
    api_key = os.getenv('ODDS_API_KEY')
    if not api_key or api_key == 'your_odds_api_key_here':
        logger.warning("⚠️  ODDS_API_KEY not set. Live odds will use sample data.")
    else:
        logger.info("✅ ODDS_API_KEY configured")
    
    logger.info("✅ Environment check completed")
    return True

def setup_database():
    """Initialize the betting database"""
    logger.info("🗄️  Setting up database...")
    
    try:
        from src.live_betting import LiveBettingLauncher
        launcher = LiveBettingLauncher(model_path="models/simple_mlb_ppo_final")
        # Database is automatically initialized in __init__
        logger.info("✅ Database initialized")
        return True
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return False

def test_live_system():
    """Test the live betting system"""
    logger.info("🧪 Testing live betting system...")
    
    try:
        from src.live_betting import LiveBettingLauncher
        launcher = LiveBettingLauncher(model_path="models/simple_mlb_ppo_final")
        
        # Test with sample data (it will automatically use sample data if no API key)
        suggestions = launcher.run_daily_analysis()
        
        if suggestions:
            logger.info(f"✅ Live system test successful - {len(suggestions)} suggestions generated")
            for suggestion in suggestions[:3]:  # Show first 3
                logger.info(f"  - {suggestion}")
        else:
            logger.warning("⚠️  No betting suggestions generated (this is normal for sample data)")
        
        return True
    except Exception as e:
        logger.error(f"❌ Live system test failed: {e}")
        return False

def create_cron_jobs():
    """Create automation scripts"""
    logger.info("⏰ Setting up automation...")
    
    # Create daily analysis script
    daily_script = """#!/bin/bash
cd /Users/diegoaguirre/MLB_ML
source venv/bin/activate  # If using virtual environment
python -c "
from src.live_betting import LiveBettingLauncher
launcher = LiveBettingLauncher(model_path='models/simple_mlb_ppo_final')
launcher.run_daily_analysis()
"
"""
    
    with open("run_daily_analysis.sh", "w") as f:
        f.write(daily_script)
    
    os.chmod("run_daily_analysis.sh", 0o755)
    
    # Create weekly retraining script
    weekly_script = """#!/bin/bash
cd /Users/diegoaguirre/MLB_ML
source venv/bin/activate  # If using virtual environment
python production_train.py
"""
    
    with open("run_weekly_training.sh", "w") as f:
        f.write(weekly_script)
    
    os.chmod("run_weekly_training.sh", 0o755)
    
    logger.info("✅ Automation scripts created:")
    logger.info("  - run_daily_analysis.sh (for daily betting analysis)")
    logger.info("  - run_weekly_training.sh (for weekly model retraining)")
    
    return True

def create_monitoring_dashboard():
    """Create a simple monitoring dashboard"""
    logger.info("📊 Creating monitoring dashboard...")
    
    dashboard_script = """#!/usr/bin/env python3
\"\"\"
Simple Monitoring Dashboard for MLB Betting System
\"\"\"

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def show_performance():
    \"\"\"Display recent betting performance\"\"\"
    try:
        conn = sqlite3.connect('data/betting_log.db')
        df = pd.read_sql_query('''
            SELECT * FROM betting_suggestions 
            WHERE timestamp > datetime('now', '-7 days')
            ORDER BY timestamp DESC
        ''', conn)
        
        if not df.empty:
            print("📈 Recent Betting Performance (Last 7 Days)")
            print("=" * 50)
            print(f"Total Suggestions: {len(df)}")
            print(f"Average Confidence: {df['confidence'].mean():.2f}")
            print(f"Total Suggested Bet: ${df['suggested_bet'].sum():.2f}")
            print("\\nRecent Suggestions:")
            for _, row in df.head(5).iterrows():
                print(f"  {row['timestamp']}: {row['game']} - ${row['suggested_bet']:.2f}")
        else:
            print("📊 No recent betting activity")
            
    except Exception as e:
        print(f"❌ Error loading performance data: {e}")

if __name__ == "__main__":
    show_performance()
"""
    
    with open("monitor_dashboard.py", "w") as f:
        f.write(dashboard_script)
    
    os.chmod("monitor_dashboard.py", 0o755)
    logger.info("✅ Monitoring dashboard created: monitor_dashboard.py")
    
    return True

def main():
    """Main deployment function"""
    logger.info("🚀 Starting live deployment...")
    
    # Step 1: Environment check
    if not check_environment():
        logger.error("❌ Environment check failed. Please fix issues before continuing.")
        return False
    
    # Step 2: Setup database
    if not setup_database():
        logger.error("❌ Database setup failed.")
        return False
    
    # Step 3: Test live system
    if not test_live_system():
        logger.error("❌ Live system test failed.")
        return False
    
    # Step 4: Create automation
    if not create_cron_jobs():
        logger.error("❌ Automation setup failed.")
        return False
    
    # Step 5: Create monitoring
    if not create_monitoring_dashboard():
        logger.error("❌ Monitoring setup failed.")
        return False
    
    logger.info("🎉 Live deployment completed successfully!")
    logger.info("")
    logger.info("📋 Next Steps:")
    logger.info("1. Get API key from https://the-odds-api.com/")
    logger.info("2. Set ODDS_API_KEY environment variable")
    logger.info("3. Run: python production_train.py (optional - for full training)")
    logger.info("4. Test: ./run_daily_analysis.sh")
    logger.info("5. Monitor: python monitor_dashboard.py")
    logger.info("")
    logger.info("🔧 Manual Commands:")
    logger.info("- Daily analysis: python -c \"from src.live_betting import LiveBettingLauncher; LiveBettingLauncher('models/simple_mlb_ppo_final').run_daily_analysis()\"")
    logger.info("- Performance check: python monitor_dashboard.py")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 