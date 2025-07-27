"""
Monte-Carlo Game Simulator for MLB Betting System
Phase 2: Baseline simulator with vectorized operations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GameState:
    """Represents the state of a baseball game"""
    home_runs: int = 0
    away_runs: int = 0
    inning: int = 1
    top_bottom: str = "top"  # "top" or "bottom"
    outs: int = 0
    bases: List[int] = None  # [first, second, third] - 0 = empty, 1 = occupied
    
    def __post_init__(self):
        if self.bases is None:
            self.bases = [0, 0, 0]
    
    def reset(self):
        """Reset game state for new simulation"""
        self.home_runs = 0
        self.away_runs = 0
        self.inning = 1
        self.top_bottom = "top"
        self.outs = 0
        self.bases = [0, 0, 0]

class MonteCarloSimulator:
    """Vectorized Monte-Carlo game simulator for fast CPU execution"""
    
    def __init__(self, num_sims: int = 1000):
        self.num_sims = num_sims
        
        # Historical MLB averages (simplified)
        self.avg_runs_per_game = 4.5
        self.avg_runs_per_inning = self.avg_runs_per_game / 9
        
        # Event probabilities (simplified model)
        self.event_probs = {
            'out': 0.67,      # 67% of plate appearances result in outs
            'single': 0.15,   # 15% singles
            'double': 0.05,   # 5% doubles  
            'triple': 0.01,   # 1% triples
            'home_run': 0.03, # 3% home runs
            'walk': 0.09      # 9% walks
        }
        
    def simulate_game_vectorized(self, home_team_stats: Dict, away_team_stats: Dict, 
                                num_sims: int = None) -> np.ndarray:
        """
        Vectorized game simulation for speed
        Returns: array of shape (num_sims, 2) with [home_runs, away_runs] for each sim
        """
        if num_sims is None:
            num_sims = self.num_sims
            
        # Adjust probabilities based on team stats
        home_offense_factor = self._calculate_offense_factor(home_team_stats)
        away_offense_factor = self._calculate_offense_factor(away_team_stats)
        home_pitching_factor = self._calculate_pitching_factor(home_team_stats)
        away_pitching_factor = self._calculate_pitching_factor(away_team_stats)
        
        # Simulate runs for each team
        home_runs = self._simulate_runs_vectorized(
            num_sims, home_offense_factor, away_pitching_factor
        )
        away_runs = self._simulate_runs_vectorized(
            num_sims, away_offense_factor, home_pitching_factor
        )
        
        return np.column_stack([home_runs, away_runs])
    
    def _calculate_offense_factor(self, team_stats: Dict) -> float:
        """Calculate offensive strength factor"""
        # Simplified - in real implementation would use more sophisticated stats
        base_factor = 1.0
        
        # Adjust based on team offensive stats
        if 'team_ops' in team_stats:
            base_factor *= (team_stats['team_ops'] / 0.750)  # Normalize to league average
        
        if 'team_wrc' in team_stats:
            base_factor *= (team_stats['team_wrc'] / 100)  # Normalize to league average
            
        return np.clip(base_factor, 0.7, 1.3)  # Cap at reasonable bounds
    
    def _calculate_pitching_factor(self, team_stats: Dict) -> float:
        """Calculate pitching strength factor (lower = better pitching)"""
        base_factor = 1.0
        
        # Adjust based on team pitching stats
        if 'team_era' in team_stats:
            base_factor *= (team_stats['team_era'] / 4.0)  # Normalize to league average
        
        if 'team_fip' in team_stats:
            base_factor *= (team_stats['team_fip'] / 4.0)  # Normalize to league average
            
        return np.clip(base_factor, 0.7, 1.3)
    
    def _simulate_runs_vectorized(self, num_sims: int, offense_factor: float, 
                                 pitching_factor: float) -> np.ndarray:
        """Vectorized run simulation"""
        # Use Poisson distribution for run generation
        # Adjust mean based on offense/pitching factors
        mean_runs = self.avg_runs_per_game * offense_factor / pitching_factor
        
        # Generate runs for all simulations at once
        runs = np.random.poisson(mean_runs, num_sims)
        
        return runs
    
    def simulate_season_vectorized(self, schedule: pd.DataFrame, team_stats: Dict) -> np.ndarray:
        """
        Simulate entire season vectorized
        Returns: array of shape (num_games, num_sims, 2) with results for each game
        """
        num_games = len(schedule)
        results = np.zeros((num_games, self.num_sims, 2))
        
        for i, game in schedule.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            
            home_stats = team_stats.get(home_team, {})
            away_stats = team_stats.get(away_team, {})
            
            game_results = self.simulate_game_vectorized(home_stats, away_stats)
            results[i] = game_results
            
        return results

class BettingSimulator:
    """Simulates betting outcomes given game results and betting strategy"""
    
    def __init__(self, initial_bankroll: float = 10000.0):
        self.initial_bankroll = initial_bankroll
        
    def simulate_bet(self, game_results: np.ndarray, bet_fraction: float, 
                     moneyline: int, bet_on_home: bool = True) -> np.ndarray:
        """
        Simulate betting outcomes
        Args:
            game_results: shape (num_sims, 2) with [home_runs, away_runs]
            bet_fraction: fraction of bankroll to bet (0.0 to 0.05 for Kelly cap)
            moneyline: moneyline odds (e.g., -150, +120)
            bet_on_home: True if betting on home team, False for away
        Returns:
            payoffs: shape (num_sims,) with payoff for each simulation
        """
        num_sims = len(game_results)
        payoffs = np.zeros(num_sims)
        
        # Determine which team we're betting on
        if bet_on_home:
            our_runs = game_results[:, 0]  # Home runs
            their_runs = game_results[:, 1]  # Away runs
        else:
            our_runs = game_results[:, 1]  # Away runs  
            their_runs = game_results[:, 0]  # Home runs
        
        # Determine wins
        wins = our_runs > their_runs
        
        # Calculate payoffs based on moneyline
        if moneyline > 0:  # Underdog
            # Bet $100 to win $moneyline
            odds_multiplier = moneyline / 100.0
            payoffs[wins] = bet_fraction * odds_multiplier
            payoffs[~wins] = -bet_fraction
        else:  # Favorite
            # Bet $abs(moneyline) to win $100
            odds_multiplier = 100.0 / abs(moneyline)
            payoffs[wins] = bet_fraction * odds_multiplier
            payoffs[~wins] = -bet_fraction
            
        return payoffs
    
    def simulate_season_betting(self, season_results: np.ndarray, bet_fractions: np.ndarray,
                               moneylines: np.ndarray, bet_on_home: np.ndarray) -> np.ndarray:
        """
        Simulate betting over entire season
        Args:
            season_results: shape (num_games, num_sims, 2)
            bet_fractions: shape (num_games,) - bet fraction for each game
            moneylines: shape (num_games,) - moneyline for each game
            bet_on_home: shape (num_games,) - True if betting on home team
        Returns:
            cumulative_payoffs: shape (num_games, num_sims) - cumulative payoffs
        """
        num_games, num_sims, _ = season_results.shape
        cumulative_payoffs = np.zeros((num_games, num_sims))
        
        for game_idx in range(num_games):
            game_results = season_results[game_idx]
            bet_fraction = bet_fractions[game_idx]
            moneyline = moneylines[game_idx]
            bet_on_home_team = bet_on_home[game_idx]
            
            game_payoffs = self.simulate_bet(
                game_results, bet_fraction, moneyline, bet_on_home_team
            )
            
            # Cumulative payoffs
            if game_idx == 0:
                cumulative_payoffs[game_idx] = game_payoffs
            else:
                cumulative_payoffs[game_idx] = cumulative_payoffs[game_idx - 1] + game_payoffs
                
        return cumulative_payoffs

def simulate_game(row: pd.Series, num_sims: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a single game and return payoffs
    This is the main function called by the RL environment
    """
    # Extract game features
    home_team_stats = {
        'team_era': row.get('home_pitcher_era', 4.0),
        'team_fip': row.get('home_bullpen_fip', 4.0),
        'park_factor': row.get('park_factor', 1.0)
    }
    
    away_team_stats = {
        'team_era': row.get('away_pitcher_era', 4.0), 
        'team_fip': row.get('away_bullpen_fip', 4.0)
    }
    
    # Simulate game
    simulator = MonteCarloSimulator(num_sims)
    game_results = simulator.simulate_game_vectorized(home_team_stats, away_team_stats, num_sims)
    
    # Simulate betting
    betting_sim = BettingSimulator()
    
    # For now, use a simple betting strategy (this will be replaced by RL agent)
    bet_fraction = 0.02  # 2% of bankroll
    moneyline = row.get('home_moneyline', -110)
    bet_on_home = True
    
    payoffs = betting_sim.simulate_bet(game_results, bet_fraction, moneyline, bet_on_home)
    
    return game_results, payoffs

if __name__ == "__main__":
    # Test the simulator
    import time
    
    # Create sample game data
    sample_game = pd.Series({
        'home_pitcher_era': 3.8,
        'away_pitcher_era': 4.2,
        'home_bullpen_fip': 3.9,
        'away_bullpen_fip': 4.1,
        'park_factor': 1.05,
        'home_moneyline': -120
    })
    
    # Time the simulation
    start_time = time.time()
    game_results, payoffs = simulate_game(sample_game, num_sims=10000)
    end_time = time.time()
    
    print(f"Simulated 10,000 games in {end_time - start_time:.3f} seconds")
    print(f"Average home runs: {game_results[:, 0].mean():.2f}")
    print(f"Average away runs: {game_results[:, 1].mean():.2f}")
    print(f"Home win rate: {(game_results[:, 0] > game_results[:, 1]).mean():.3f}")
    print(f"Average payoff: {payoffs.mean():.4f}")
    print(f"Payoff std: {payoffs.std():.4f}") 