#!/bin/bash

echo "🚀 Launching MLB Betting Dashboard with API Key..."
echo "======================================================"

# Set API key
export ODDS_API_KEY="e4c6061c908cc7937f65feb02d9340c2"

# Verify API key is set
echo "✅ API Key: $ODDS_API_KEY"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
ODDS_API_KEY=e4c6061c908cc7937f65feb02d9340c2
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
SLACK_WEBHOOK_URL=your_slack_webhook_url_here
DATABASE_PATH=betting_log.db
MODEL_PATH=models/mlb_ppo_production_final.zip
TRAINING_TIMESTEPS=2000000
EOF
    echo "✅ .env file created"
fi

echo "🌐 Opening dashboard at http://localhost:8501"
echo "Press Ctrl+C to stop the dashboard"
echo ""

# Launch Streamlit with environment variables
streamlit run app.py --server.port 8501 