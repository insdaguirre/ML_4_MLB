"""
Backtesting System for MLB Betting System
Phase 5: Walk-forward validation with Monte-Carlo bankroll paths
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
from datetime import datetime

from ..data.data_pipeline import MLBDataPipeline
from ..rl.baseball_env import create_env
from ..rl.train import CustomPPO, evaluate_model
from ..betting.kelly_betting import KellyBettingSystem, integrate_with_rl_model
from ..simulation.game_simulator import MonteCarloSimulator

logger = logging.getLogger(__name__)

class BacktestingSystem:
    """Walk-forward backtesting system with Monte-Carlo validation"""
    
    def __init__(self, data_dir: str = "data", model_dir: str = "models"):
        self.data_pipeline = MLBDataPipeline(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Monte-Carlo parameters
        self.num_mc_paths = 5000  # 5k paths as specified
        self.num_sims_per_game = 1000
        
        # Performance tracking
        self.results = {}
        
    def walk_forward_backtest(self, start_year: int = 2010, end_year: int = 2023,
                             train_window: int = 9, test_window: int = 1) -> Dict[str, Any]:
        """
        Walk-forward backtesting
        
        Args:
            start_year: Start year for backtesting
            end_year: End year for backtesting
            train_window: Number of years to train on
            test_window: Number of years to test on
            
        Returns:
            Dictionary with backtesting results
        """
        logger.info(f"Starting walk-forward backtest: {start_year}-{end_year}")
        
        all_results = []
        
        for test_year in range(start_year + train_window, end_year + 1, test_window):
            train_start = test_year - train_window
            train_end = test_year - 1
            
            logger.info(f"Testing {test_year} (trained on {train_start}-{train_end})")
            
            # Train model on historical data
            model = self._train_model_on_period(train_start, train_end)
            
            # Test on out-of-sample data
            test_results = self._test_model_on_period(model, test_year)
            
            # Store results
            test_results['train_period'] = f"{train_start}-{train_end}"
            test_results['test_year'] = test_year
            all_results.append(test_results)
            
            logger.info(f"Test year {test_year} results: ROI={test_results['roi']:.3f}, "
                       f"Sharpe={test_results['sharpe']:.3f}")
        
        # Aggregate results
        aggregated_results = self._aggregate_results(all_results)
        
        # Save results
        self._save_results(all_results, aggregated_results)
        
        return aggregated_results
    
    def _train_model_on_period(self, start_year: int, end_year: int) -> CustomPPO:
        """Train model on specific time period"""
        # Create training environment with period-specific data
        train_env = self._create_period_env(start_year, end_year, num_envs=6)
        
        # Train model
        model = CustomPPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0
        )
        
        # Train for fewer steps for backtesting (faster)
        model.learn(total_timesteps=500_000, progress_bar=False)
        
        return model
    
    def _create_period_env(self, start_year: int, end_year: int, num_envs: int = 6):
        """Create environment for specific time period"""
        from ..rl.baseball_env import VecBaseballEnv
        
        # Get data for period
        data_pipeline = MLBDataPipeline()
        period_data = data_pipeline.get_training_data(start_year, end_year)
        
        # Create environment with period data
        env = VecBaseballEnv(
            num_envs=num_envs,
            data_pipeline=data_pipeline,
            initial_bankroll=10000.0,
            num_sims=1000,
            max_games=162
        )
        
        return env
    
    def _test_model_on_period(self, model: CustomPPO, test_year: int) -> Dict[str, Any]:
        """Test model on specific year"""
        # Get test data
        test_data = self.data_pipeline.get_test_data(test_year, test_year)
        
        # Initialize Kelly system
        kelly_system = KellyBettingSystem()
        
        # Track performance
        bankroll = 10000.0
        bet_history = []
        bankroll_history = [bankroll]
        
        # Process each game
        for i in range(len(test_data)):
            game_row = test_data.row(i, named=True)
            
            # Convert to features dict
            game_features = {col: float(game_row[col]) for col in game_row.keys()}
            
            # Get model prediction
            state = self._game_to_state(game_features, bankroll)
            action, _ = model.predict(state, deterministic=True)
            model_prediction = action[0]
            
            # Make betting decision
            decision = integrate_with_rl_model(model_prediction, game_features, bankroll)
            
            # Simulate game outcome
            simulator = MonteCarloSimulator(self.num_sims_per_game)
            game_results, payoffs = simulator.simulate_game_vectorized(
                self._extract_team_stats(game_features),
                self._extract_team_stats(game_features, home=False)
            )
            
            # Determine outcome
            home_wins = (game_results[:, 0] > game_results[:, 1]).mean()
            outcome = home_wins > 0.5  # Simple threshold
            
            # Update bankroll
            old_bankroll = bankroll
            bankroll = kelly_system.update_bankroll(
                decision, outcome, game_features['home_moneyline'], bankroll
            )
            
            # Track history
            bet_history.append({
                'game_idx': i,
                'bet_fraction': decision.bet_fraction,
                'bet_amount': decision.bet_amount,
                'edge': decision.edge,
                'outcome': outcome,
                'bankroll_change': bankroll - old_bankroll
            })
            
            bankroll_history.append(bankroll)
        
        # Calculate metrics
        results = self._calculate_period_metrics(bet_history, bankroll_history)
        
        return results
    
    def _game_to_state(self, game_features: Dict, bankroll: float) -> np.ndarray:
        """Convert game features to state vector"""
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
    
    def _extract_team_stats(self, game_features: Dict, home: bool = True) -> Dict:
        """Extract team stats from game features"""
        prefix = "home" if home else "away"
        
        return {
            'team_era': game_features.get(f'{prefix}_pitcher_era', 4.0),
            'team_fip': game_features.get(f'{prefix}_bullpen_fip', 4.0),
            'park_factor': game_features.get('park_factor', 1.0) if home else 1.0
        }
    
    def _calculate_period_metrics(self, bet_history: List[Dict], 
                                bankroll_history: List[float]) -> Dict[str, float]:
        """Calculate performance metrics for a test period"""
        if not bet_history:
            return {}
        
        # Basic metrics
        total_bets = len(bet_history)
        winning_bets = sum(1 for bet in bet_history if bet['outcome'])
        win_rate = winning_bets / total_bets if total_bets > 0 else 0
        
        # ROI
        initial_bankroll = bankroll_history[0]
        final_bankroll = bankroll_history[-1]
        roi = (final_bankroll - initial_bankroll) / initial_bankroll
        
        # Sharpe ratio
        returns = []
        for i in range(1, len(bankroll_history)):
            ret = (bankroll_history[i] - bankroll_history[i-1]) / bankroll_history[i-1]
            returns.append(ret)
        
        if returns:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
        else:
            sharpe = 0.0
        
        # Max drawdown
        peak = initial_bankroll
        max_drawdown = 0.0
        for bankroll in bankroll_history:
            if bankroll > peak:
                peak = bankroll
            drawdown = (peak - bankroll) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Average edge and bet size
        avg_edge = np.mean([abs(bet['edge']) for bet in bet_history])
        avg_bet_fraction = np.mean([abs(bet['bet_fraction']) for bet in bet_history])
        
        return {
            'total_bets': total_bets,
            'win_rate': win_rate,
            'roi': roi,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'avg_edge': avg_edge,
            'avg_bet_fraction': avg_bet_fraction,
            'final_bankroll': final_bankroll
        }
    
    def monte_carlo_validation(self, model_path: str, test_year: int = 2019,
                              num_paths: int = 5000) -> Dict[str, Any]:
        """
        Run Monte-Carlo validation with 5k bankroll paths
        
        Args:
            model_path: Path to trained model
            test_year: Year to test on
            num_paths: Number of Monte-Carlo paths
            
        Returns:
            Monte-Carlo validation results
        """
        logger.info(f"Running Monte-Carlo validation for {test_year} with {num_paths} paths")
        
        # Load model
        model = CustomPPO.load(model_path)
        
        # Get test data
        test_data = self.data_pipeline.get_test_data(test_year, test_year)
        
        # Initialize Monte-Carlo paths
        initial_bankroll = 10000.0
        bankroll_paths = np.zeros((len(test_data), num_paths))
        bankroll_paths[0] = initial_bankroll
        
        # Kelly system
        kelly_system = KellyBettingSystem()
        
        # Process each game
        for game_idx in range(len(test_data)):
            game_row = test_data.row(game_idx, named=True)
            game_features = {col: float(game_row[col]) for col in game_row.keys()}
            
            # Get model prediction
            state = self._game_to_state(game_features, initial_bankroll)
            action, _ = model.predict(state, deterministic=True)
            model_prediction = action[0]
            
            # Make betting decision
            decision = integrate_with_rl_model(model_prediction, game_features, initial_bankroll)
            
            # Simulate game outcomes for all paths
            simulator = MonteCarloSimulator(num_paths)
            game_results, payoffs = simulator.simulate_game_vectorized(
                self._extract_team_stats(game_features),
                self._extract_team_stats(game_features, home=False)
            )
            
            # Calculate outcomes for each path
            home_wins = game_results[:, 0] > game_results[:, 1]
            
            # Update bankroll for each path
            if game_idx == 0:
                current_bankrolls = np.full(num_paths, initial_bankroll)
            else:
                current_bankrolls = bankroll_paths[game_idx - 1]
            
            # Calculate payoffs
            if decision.bet_fraction != 0:
                bet_amount = decision.bet_fraction * current_bankrolls
                decimal_odds = kelly_system.moneyline_to_decimal(game_features['home_moneyline'])
                
                # Calculate winnings/losses
                winnings = np.where(home_wins, bet_amount * (decimal_odds - 1), 0)
                losses = np.where(~home_wins, bet_amount, 0)
                
                bankroll_changes = winnings - losses
                new_bankrolls = current_bankrolls + bankroll_changes
            else:
                new_bankrolls = current_bankrolls
            
            bankroll_paths[game_idx] = new_bankrolls
        
        # Calculate distribution statistics
        final_bankrolls = bankroll_paths[-1]
        rois = (final_bankrolls - initial_bankroll) / initial_bankroll
        
        # Calculate percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        roi_percentiles = np.percentile(rois, percentiles)
        
        # Calculate probability of positive ROI
        prob_positive_roi = (rois > 0).mean()
        
        # Calculate expected ROI and volatility
        expected_roi = np.mean(rois)
        roi_volatility = np.std(rois)
        
        # Calculate Sharpe ratio
        sharpe_ratio = expected_roi / (roi_volatility + 1e-8)
        
        # Calculate Value at Risk (VaR)
        var_95 = np.percentile(rois, 5)  # 95% VaR
        var_99 = np.percentile(rois, 1)  # 99% VaR
        
        results = {
            'expected_roi': expected_roi,
            'roi_volatility': roi_volatility,
            'sharpe_ratio': sharpe_ratio,
            'prob_positive_roi': prob_positive_roi,
            'var_95': var_95,
            'var_99': var_99,
            'roi_percentiles': dict(zip(percentiles, roi_percentiles)),
            'final_bankrolls': final_bankrolls,
            'rois': rois,
            'bankroll_paths': bankroll_paths
        }
        
        return results
    
    def _aggregate_results(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results across all test periods"""
        if not all_results:
            return {}
        
        # Calculate averages
        avg_roi = np.mean([r['roi'] for r in all_results])
        avg_sharpe = np.mean([r['sharpe'] for r in all_results])
        avg_win_rate = np.mean([r['win_rate'] for r in all_results])
        avg_max_drawdown = np.mean([r['max_drawdown'] for r in all_results])
        
        # Calculate consistency
        positive_roi_years = sum(1 for r in all_results if r['roi'] > 0)
        total_years = len(all_results)
        consistency = positive_roi_years / total_years if total_years > 0 else 0
        
        # Calculate cumulative performance
        cumulative_roi = 1.0
        for result in all_results:
            cumulative_roi *= (1 + result['roi'])
        cumulative_roi -= 1
        
        return {
            'avg_roi': avg_roi,
            'avg_sharpe': avg_sharpe,
            'avg_win_rate': avg_win_rate,
            'avg_max_drawdown': avg_max_drawdown,
            'consistency': consistency,
            'cumulative_roi': cumulative_roi,
            'total_years': total_years,
            'positive_years': positive_roi_years,
            'yearly_results': all_results
        }
    
    def _save_results(self, yearly_results: List[Dict], aggregated_results: Dict):
        """Save backtesting results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'timestamp': timestamp,
            'yearly_results': yearly_results,
            'aggregated_results': aggregated_results
        }
        
        results_path = self.model_dir / f"backtest_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
    
    def plot_results(self, mc_results: Dict[str, Any], save_path: str = None):
        """Plot Monte-Carlo validation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # ROI distribution
        axes[0, 0].hist(mc_results['rois'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(mc_results['expected_roi'], color='red', linestyle='--', 
                           label=f"Expected ROI: {mc_results['expected_roi']:.3f}")
        axes[0, 0].set_xlabel('ROI')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('ROI Distribution')
        axes[0, 0].legend()
        
        # Bankroll paths (sample)
        sample_paths = mc_results['bankroll_paths'][:, ::100]  # Sample every 100th path
        for i in range(min(10, sample_paths.shape[1])):
            axes[0, 1].plot(sample_paths[:, i], alpha=0.3)
        axes[0, 1].set_xlabel('Game')
        axes[0, 1].set_ylabel('Bankroll')
        axes[0, 1].set_title('Sample Bankroll Paths')
        
        # ROI percentiles
        percentiles = list(mc_results['roi_percentiles'].keys())
        roi_values = list(mc_results['roi_percentiles'].values())
        axes[1, 0].bar(percentiles, roi_values)
        axes[1, 0].set_xlabel('Percentile')
        axes[1, 0].set_ylabel('ROI')
        axes[1, 0].set_title('ROI Percentiles')
        
        # Performance summary
        summary_text = f"""
        Expected ROI: {mc_results['expected_roi']:.3f}
        ROI Volatility: {mc_results['roi_volatility']:.3f}
        Sharpe Ratio: {mc_results['sharpe_ratio']:.3f}
        P(ROI > 0): {mc_results['prob_positive_roi']:.3f}
        VaR (95%): {mc_results['var_95']:.3f}
        VaR (99%): {mc_results['var_99']:.3f}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center')
        axes[1, 1].set_title('Performance Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def main():
    """Run backtesting"""
    backtest = BacktestingSystem()
    
    # Run walk-forward backtest
    results = backtest.walk_forward_backtest(2010, 2023)
    
    print("Backtesting Results:")
    print(f"Average ROI: {results['avg_roi']:.3f}")
    print(f"Average Sharpe: {results['avg_sharpe']:.3f}")
    print(f"Consistency: {results['consistency']:.3f}")
    print(f"Cumulative ROI: {results['cumulative_roi']:.3f}")
    
    # Run Monte-Carlo validation
    if Path("models/mlb_ppo_final.zip").exists():
        mc_results = backtest.monte_carlo_validation("models/mlb_ppo_final", 2019)
        
        print("\nMonte-Carlo Validation Results:")
        print(f"Expected ROI: {mc_results['expected_roi']:.3f}")
        print(f"ROI Volatility: {mc_results['roi_volatility']:.3f}")
        print(f"Sharpe Ratio: {mc_results['sharpe_ratio']:.3f}")
        print(f"P(ROI > 0): {mc_results['prob_positive_roi']:.3f}")
        
        # Plot results
        backtest.plot_results(mc_results, "backtest_results.png")

if __name__ == "__main__":
    main() 