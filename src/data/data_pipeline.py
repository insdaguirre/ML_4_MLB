"""
Data Pipeline for MLB Betting System
Phase 1: Historical game & odds data processing
"""

import pandas as pd
import polars as pl
import numpy as np
import requests
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLBDataPipeline:
    """Handles data collection and processing for MLB betting system"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # API endpoints (you'll need to add your own API keys)
        self.odds_api_key = os.getenv("ODDS_API_KEY")
        self.stats_api_key = os.getenv("STATS_API_KEY")
        
    def download_retrosheet_data(self, start_year: int = 2010, end_year: int = 2023) -> None:
        """Download Retrosheet play-by-play data"""
        logger.info(f"Downloading Retrosheet data from {start_year} to {end_year}")
        
        # This would normally download from Retrosheet
        # For now, we'll create sample data structure
        sample_data = self._create_sample_retrosheet_data(start_year, end_year)
        
        # Save as Parquet for fast loading
        sample_data.write_parquet(self.data_dir / "retrosheet_data.parquet")
        logger.info("Retrosheet data saved")
        
    def download_odds_data(self, start_year: int = 2010, end_year: int = 2023) -> None:
        """Download money-line closing odds data"""
        logger.info(f"Downloading odds data from {start_year} to {end_year}")
        
        # This would normally fetch from Odds API or Kaggle dataset
        # For now, we'll create sample odds data
        sample_odds = self._create_sample_odds_data(start_year, end_year)
        
        # Save as Parquet
        sample_odds.write_parquet(self.data_dir / "odds_data.parquet")
        logger.info("Odds data saved")
        
    def process_features(self) -> pl.DataFrame:
        """Join data and compute derived features"""
        logger.info("Processing features...")
        
        # Load data
        retrosheet_df = pl.read_parquet(self.data_dir / "retrosheet_data.parquet")
        odds_df = pl.read_parquet(self.data_dir / "odds_data.parquet")
        
        # Join on game_id
        joined_df = retrosheet_df.join(odds_df, on="game_id", how="inner")
        
        # Compute derived features
        features_df = self._compute_derived_features(joined_df)
        
        # Save processed features
        features_df.write_parquet(self.data_dir / "processed_features.parquet")
        logger.info("Features processed and saved")
        
        return features_df
    
    def _create_sample_retrosheet_data(self, start_year: int, end_year: int) -> pl.DataFrame:
        """Create sample Retrosheet-style data for development"""
        np.random.seed(42)
        
        games = []
        for year in range(start_year, end_year + 1):
            # Generate ~2430 games per season (162 games * 15 teams / 2)
            num_games = 2430
            
            for game_id in range(num_games):
                game_date = datetime(year, 4, 1) + timedelta(days=game_id % 180)
                
                # Sample teams
                teams = ["NYY", "BOS", "LAD", "SFG", "CHC", "STL", "ATL", "PHI", 
                        "HOU", "TEX", "OAK", "LAA", "SEA", "MIN", "CLE", "DET",
                        "CWS", "KCR", "TOR", "TBR", "BAL", "NYY", "MIA", "WSN"]
                
                home_team = np.random.choice(teams)
                away_team = np.random.choice([t for t in teams if t != home_team])
                
                # Generate realistic scores
                home_runs = np.random.poisson(4.5)
                away_runs = np.random.poisson(4.2)
                
                # Starting pitcher stats (simplified)
                home_pitcher_era = np.random.normal(3.8, 0.8)
                away_pitcher_era = np.random.normal(3.9, 0.8)
                
                games.append({
                    "game_id": f"{year}_{game_id:06d}",
                    "date": game_date,
                    "year": year,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_runs": home_runs,
                    "away_runs": away_runs,
                    "home_pitcher_era": home_pitcher_era,
                    "away_pitcher_era": away_pitcher_era,
                    "park_factor": np.random.normal(1.0, 0.1),
                    "home_bullpen_fip": np.random.normal(4.0, 0.5),
                    "away_bullpen_fip": np.random.normal(4.1, 0.5)
                })
        
        return pl.DataFrame(games)
    
    def _create_sample_odds_data(self, start_year: int, end_year: int) -> pl.DataFrame:
        """Create sample odds data for development"""
        np.random.seed(42)
        
        odds = []
        for year in range(start_year, end_year + 1):
            num_games = 2430
            
            for game_id in range(num_games):
                # Generate realistic moneyline odds
                home_ml = np.random.choice([-150, -140, -130, -120, -110, -105, 
                                          -100, +100, +105, +110, +120, +130, +140, +150])
                
                # Convert to implied probability
                if home_ml > 0:
                    implied_prob = 100 / (home_ml + 100)
                else:
                    implied_prob = abs(home_ml) / (abs(home_ml) + 100)
                
                odds.append({
                    "game_id": f"{year}_{game_id:06d}",
                    "home_moneyline": home_ml,
                    "away_moneyline": -home_ml,
                    "home_implied_prob": implied_prob,
                    "away_implied_prob": 1 - implied_prob,
                    "total_line": np.random.choice([7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5]),
                    "bookmaker": "sample_bookie"
                })
        
        return pl.DataFrame(odds)
    
    def _compute_derived_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute derived features for the RL model"""
        
        # Convert to pandas for easier feature engineering
        pdf = df.to_pandas()
        
        # Game outcome
        pdf['home_win'] = (pdf['home_runs'] > pdf['away_runs']).astype(int)
        
        # Pitcher quality features
        pdf['pitcher_quality_diff'] = pdf['away_pitcher_era'] - pdf['home_pitcher_era']
        pdf['bullpen_quality_diff'] = pdf['away_bullpen_fip'] - pdf['home_bullpen_fip']
        
        # Park-adjusted features
        pdf['park_adjusted_home_runs'] = pdf['home_runs'] / pdf['park_factor']
        pdf['park_adjusted_away_runs'] = pdf['away_runs'] / pdf['park_factor']
        
        # Betting features
        pdf['edge'] = pdf['home_implied_prob'] - 0.5  # Positive = home favored
        pdf['total_runs'] = pdf['home_runs'] + pdf['away_runs']
        
        # Kelly criterion inputs
        pdf['kelly_fraction'] = (pdf['home_implied_prob'] * 2 - 1) * 0.05  # Capped at 5%
        pdf['kelly_fraction'] = np.clip(pdf['kelly_fraction'], -0.05, 0.05)
        
        # Convert back to polars
        return pl.from_pandas(pdf)
    
    def get_training_data(self, start_year: int = 2010, end_year: int = 2018) -> pl.DataFrame:
        """Get processed data for training period"""
        features_df = pl.read_parquet(self.data_dir / "processed_features.parquet")
        
        # Filter for training period
        training_data = features_df.filter(
            (pl.col("year") >= start_year) & (pl.col("year") <= end_year)
        )
        
        return training_data
    
    def get_test_data(self, start_year: int = 2019, end_year: int = 2023) -> pl.DataFrame:
        """Get processed data for testing period"""
        features_df = pl.read_parquet(self.data_dir / "processed_features.parquet")
        
        # Filter for testing period
        test_data = features_df.filter(
            (pl.col("year") >= start_year) & (pl.col("year") <= end_year)
        )
        
        return test_data

def main():
    """Run the data pipeline"""
    pipeline = MLBDataPipeline()
    
    # Download data
    pipeline.download_retrosheet_data(2010, 2023)
    pipeline.download_odds_data(2010, 2023)
    
    # Process features
    features_df = pipeline.process_features()
    
    print(f"Processed {len(features_df)} games with features")
    print("Feature columns:", features_df.columns)

if __name__ == "__main__":
    main() 