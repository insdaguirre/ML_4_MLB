#!/bin/bash

echo "🔑 MLB Betting System - API Key Setup"
echo "======================================"
echo ""

# Check if API key is already set
if [ ! -z "$ODDS_API_KEY" ] && [ "$ODDS_API_KEY" != "your_actual_api_key_here" ]; then
    echo "✅ API key is already set: $ODDS_API_KEY"
    echo ""
    echo "Testing with current API key..."
    python -c "
from src.live_betting import LiveBettingLauncher
launcher = LiveBettingLauncher('models/mlb_ppo_production_final.zip')
launcher.run_daily_analysis()
"
    exit 0
fi

echo "❌ No API key found or not properly configured."
echo ""
echo "To get your API key:"
echo "1. Go to: https://the-odds-api.com/"
echo "2. Sign up for FREE account"
echo "3. Get your API key"
echo ""
echo "Then choose one of these options:"
echo ""
echo "Option 1: Set for this session only"
echo "  export ODDS_API_KEY='your_actual_api_key_here'"
echo ""
echo "Option 2: Set permanently (recommended)"
echo "  echo 'export ODDS_API_KEY=\"your_actual_api_key_here\"' >> ~/.zshrc"
echo "  source ~/.zshrc"
echo ""
echo "Option 3: Create .env file"
echo "  echo 'ODDS_API_KEY=your_actual_api_key_here' > .env"
echo ""
echo "After setting the API key, run this script again to test." 