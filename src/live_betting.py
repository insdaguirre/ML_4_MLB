"""
Live Betting Launcher for MLB Betting System
Phase 6: Deploy CPU bot with automated betting suggestions
"""

import requests
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import os
import time
import schedule
import numpy as np

from src.data.data_pipeline import MLBDataPipeline
from src.rl.train import CustomPPO
from src.betting.kelly_betting import KellyBettingSystem, integrate_with_rl_model
from src.simulation.game_simulator import MonteCarloSimulator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveBettingLauncher:
    """Live betting system with automated suggestions"""
    
    def __init__(self, model_path: str = "models/mlb_ppo_final", 
                 data_dir: str = "data", db_path: str = "betting_log.db"):
        self.model_path = model_path
        self.data_dir = data_dir
        self.db_path = db_path
        
        # Initialize components
        self.model = None
        self.kelly_system = KellyBettingSystem()
        self.data_pipeline = MLBDataPipeline(data_dir)
        
        # Load model if available
        if Path(model_path).exists():
            self.model = CustomPPO.load(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}")
        
        # Initialize database
        self._init_database()
        
        # API keys (you'll need to set these)
        self.odds_api_key = os.getenv("ODDS_API_KEY")
        self.telegram_token = os.getenv("TELEGRAM_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
    def _init_database(self):
        """Initialize SQLite database for logging"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS betting_suggestions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                game_id TEXT,
                home_team TEXT,
                away_team TEXT,
                moneyline INTEGER,
                bet_fraction REAL,
                bet_amount REAL,
                edge REAL,
                confidence REAL,
                suggestion_sent BOOLEAN
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                total_bets INTEGER,
                winning_bets INTEGER,
                current_bankroll REAL,
                roi REAL,
                sharpe_ratio REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("Database initialized")
    
    def get_todays_games(self) -> List[Dict]:
        """Get today's MLB games with odds"""
        if not self.odds_api_key:
            logger.warning("No odds API key available, using sample data")
            return self._get_sample_games()
        
        try:
            # Get today's date
            today = datetime.now().strftime("%Y-%m-%d")
            
            # Fetch odds from API
            url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
            params = {
                'apiKey': self.odds_api_key,
                'regions': 'us',
                'markets': 'h2h',
                'dateFormat': 'iso'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            games = response.json()
            
            # Filter for today's games
            todays_games = []
            for game in games:
                game_date = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00'))
                if game_date.date() == datetime.now().date():
                    todays_games.append(game)
            
            return todays_games
            
        except Exception as e:
            logger.error(f"Error fetching odds: {e}")
            return self._get_sample_games()
    
    def _get_sample_games(self) -> List[Dict]:
        """Get sample games for testing"""
        return [
            {
                'id': 'sample_game_1',
                'home_team': 'NYY',
                'away_team': 'BOS',
                'commence_time': datetime.now().isoformat(),
                'bookmakers': [{
                    'key': 'sample_bookie',
                    'markets': [{
                        'key': 'h2h',
                        'outcomes': [
                            {'name': 'NYY', 'price': -120},
                            {'name': 'BOS', 'price': +110}
                        ]
                    }]
                }]
            },
            {
                'id': 'sample_game_2',
                'home_team': 'LAD',
                'away_team': 'SFG',
                'commence_time': datetime.now().isoformat(),
                'bookmakers': [{
                    'key': 'sample_bookie',
                    'markets': [{
                        'key': 'h2h',
                        'outcomes': [
                            {'name': 'LAD', 'price': -140},
                            {'name': 'SFG', 'price': +120}
                        ]
                    }]
                }]
            }
        ]
    
    def extract_game_features(self, game: Dict) -> Dict:
        """Extract features from game data"""
        # Get moneyline for home team
        home_team = game['home_team']
        away_team = game['away_team']
        
        moneyline = -110  # Default
        for bookmaker in game.get('bookmakers', []):
            for market in bookmaker.get('markets', []):
                if market['key'] == 'h2h':
                    for outcome in market['outcomes']:
                        if outcome['name'] == home_team:
                            moneyline = outcome['price']
                            break
        
        # Convert moneyline to implied probability
        if moneyline > 0:
            implied_prob = 100 / (moneyline + 100)
        else:
            implied_prob = abs(moneyline) / (abs(moneyline) + 100)
        
        # Create game features (simplified for live betting)
        features = {
            'home_team': home_team,
            'away_team': away_team,
            'home_moneyline': moneyline,
            'home_implied_prob': implied_prob,
            'away_implied_prob': 1 - implied_prob,
            'edge': implied_prob - 0.5,
            'home_pitcher_era': 4.0,  # Would need real data
            'away_pitcher_era': 4.0,
            'home_bullpen_fip': 4.0,
            'away_bullpen_fip': 4.0,
            'park_factor': 1.0,
            'pitcher_quality_diff': 0.0,
            'bullpen_quality_diff': 0.0,
            'park_adjusted_home_runs': 4.5,
            'park_adjusted_away_runs': 4.2,
            'total_runs': 8.7,
            'kelly_fraction': 0.0
        }
        
        return features
    
    def make_betting_decision(self, game_features: Dict, bankroll: float = 10000.0) -> Dict:
        """Make betting decision for a game"""
        if self.model is None:
            logger.warning("No model available, using Kelly only")
            decision = self.kelly_system.make_betting_decision(game_features, bankroll)
        else:
            # Get model prediction
            state = self._game_to_state(game_features, bankroll)
            action, _ = self.model.predict(state, deterministic=True)
            model_prediction = action[0]
            
            # Integrate with Kelly system
            decision = integrate_with_rl_model(model_prediction, game_features, bankroll)
        
        return {
            'game_id': f"{game_features['home_team']}_vs_{game_features['away_team']}",
            'home_team': game_features['home_team'],
            'away_team': game_features['away_team'],
            'moneyline': game_features['home_moneyline'],
            'bet_fraction': decision.bet_fraction,
            'bet_amount': decision.bet_amount,
            'edge': decision.edge,
            'confidence': decision.confidence,
            'should_bet': abs(decision.bet_fraction) > 0.001
        }
    
    def _game_to_state(self, game_features: Dict, bankroll: float) -> np.ndarray:
        """Convert game features to state vector"""
        import numpy as np
        
        feature_columns = [
            'home_pitcher_era', 'away_pitcher_era', 'home_bullpen_fip', 'away_bullpen_fip',
            'park_factor', 'pitcher_quality_diff', 'bullpen_quality_diff', 
            'park_adjusted_home_runs', 'park_adjusted_away_runs', 'edge',
            'total_runs', 'kelly_fraction', 'home_implied_prob', 'away_implied_prob'
        ]
        
        features = []
        for col in feature_columns:
            features.append(game_features.get(col, 0.0))
        
        # Add normalized bankroll
        normalized_bankroll = (bankroll - 10000.0) / 10000.0
        features.append(normalized_bankroll)
        
        return np.array(features, dtype=np.float32)
    
    def log_suggestion(self, suggestion: Dict):
        """Log betting suggestion to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO betting_suggestions 
            (timestamp, game_id, home_team, away_team, moneyline, bet_fraction, 
             bet_amount, edge, confidence, suggestion_sent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            suggestion['game_id'],
            suggestion['home_team'],
            suggestion['away_team'],
            suggestion['moneyline'],
            suggestion['bet_fraction'],
            suggestion['bet_amount'],
            suggestion['edge'],
            suggestion['confidence'],
            False
        ))
        
        conn.commit()
        conn.close()
    
    def send_telegram_message(self, message: str):
        """Send message via Telegram"""
        if not self.telegram_token or not self.telegram_chat_id:
            logger.warning("Telegram credentials not configured")
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, data=data)
            response.raise_for_status()
            
            logger.info("Telegram message sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
    
    def format_betting_message(self, suggestion: Dict) -> str:
        """Format betting suggestion for messaging"""
        if not suggestion['should_bet']:
            return f"❌ No bet recommended for {suggestion['home_team']} vs {suggestion['away_team']}"
        
        bet_direction = "ON" if suggestion['bet_fraction'] > 0 else "AGAINST"
        team = suggestion['home_team'] if suggestion['bet_fraction'] > 0 else suggestion['away_team']
        
        message = f"""
🎯 <b>MLB Betting Suggestion</b>

🏟️ <b>Game:</b> {suggestion['away_team']} @ {suggestion['home_team']}
💰 <b>Bet:</b> {bet_direction} {team}
📊 <b>Stake:</b> ${suggestion['bet_amount']:.2f} ({suggestion['bet_fraction']*100:.2f}% of bankroll)
📈 <b>Edge:</b> {suggestion['edge']*100:.2f}%
🎯 <b>Confidence:</b> {suggestion['confidence']*100:.1f}%
📉 <b>Moneyline:</b> {suggestion['moneyline']:+d}

⏰ <i>Generated at {datetime.now().strftime('%H:%M:%S')}</i>
        """.strip()
        
        return message
    
    def run_daily_analysis(self):
        """Run daily betting analysis"""
        logger.info("Starting daily betting analysis...")
        
        # Get today's games
        games = self.get_todays_games()
        
        if not games:
            logger.info("No games found for today")
            return
        
        logger.info(f"Found {len(games)} games for today")
        
        # Analyze each game
        suggestions = []
        for game in games:
            # Extract features
            game_features = self.extract_game_features(game)
            
            # Make betting decision
            suggestion = self.make_betting_decision(game_features)
            
            # Log suggestion
            self.log_suggestion(suggestion)
            
            # Send message if bet is recommended
            if suggestion['should_bet']:
                message = self.format_betting_message(suggestion)
                self.send_telegram_message(message)
                suggestions.append(suggestion)
        
        # Send summary
        if suggestions:
            summary = f"""
📊 <b>Daily Summary</b>

Found {len(suggestions)} betting opportunities out of {len(games)} games.
Total suggested stake: ${sum(s['bet_amount'] for s in suggestions):.2f}
Average edge: {np.mean([s['edge'] for s in suggestions])*100:.2f}%
            """.strip()
            
            self.send_telegram_message(summary)
        else:
            self.send_telegram_message("📊 <b>Daily Summary</b>\n\nNo betting opportunities found today.")
        
        logger.info(f"Daily analysis completed. {len(suggestions)} suggestions sent.")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent suggestions
        cursor.execute('''
            SELECT bet_fraction, bet_amount, edge, confidence
            FROM betting_suggestions
            WHERE timestamp >= datetime('now', '-7 days')
        ''')
        
        recent_suggestions = cursor.fetchall()
        
        if not recent_suggestions:
            return {}
        
        # Calculate stats
        total_suggestions = len(recent_suggestions)
        total_stake = sum(row[1] for row in recent_suggestions)
        avg_edge = np.mean([row[2] for row in recent_suggestions])
        avg_confidence = np.mean([row[3] for row in recent_suggestions])
        
        conn.close()
        
        return {
            'total_suggestions': total_suggestions,
            'total_stake': total_stake,
            'avg_edge': avg_edge,
            'avg_confidence': avg_confidence
        }

def main():
    """Main function for live betting launcher"""
    launcher = LiveBettingLauncher()
    
    # Run daily analysis
    launcher.run_daily_analysis()
    
    # Get performance stats
    stats = launcher.get_performance_stats()
    if stats:
        print("Performance Stats (Last 7 days):")
        print(f"Total suggestions: {stats['total_suggestions']}")
        print(f"Total stake: ${stats['total_stake']:.2f}")
        print(f"Average edge: {stats['avg_edge']*100:.2f}%")
        print(f"Average confidence: {stats['avg_confidence']*100:.1f}%")

if __name__ == "__main__":
    main() 