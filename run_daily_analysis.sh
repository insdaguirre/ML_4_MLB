#!/bin/bash
cd /Users/diegoaguirre/MLB_ML
source venv/bin/activate  # If using virtual environment
python -c "
from src.live_betting import LiveBettingLauncher
launcher = LiveBettingLauncher(model_path='models/simple_mlb_ppo_final')
launcher.run_daily_analysis()
"
