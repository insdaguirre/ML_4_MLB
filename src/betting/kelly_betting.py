"""
Kelly Betting Layer for MLB Betting System
Phase 4: Dynamic Kelly with max-payout constraint and risk management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BettingDecision:
    """Represents a betting decision"""
    bet_fraction: float  # Fraction of bankroll to bet
    bet_amount: float    # Absolute bet amount
    edge: float          # Calculated edge
    kelly_fraction: float  # Raw Kelly fraction
    confidence: float    # Confidence in the bet
    risk_adjusted: bool  # Whether risk adjustment was applied

class KellyBettingSystem:
    """Kelly criterion betting system with risk management"""
    
    def __init__(self, max_bet_fraction: float = 0.05, max_payout: float = 0.20,
                 confidence_threshold: float = 0.6, risk_free_rate: float = 0.02):
        """
        Initialize Kelly betting system
        
        Args:
            max_bet_fraction: Maximum fraction of bankroll to bet (5% as specified)
            max_payout: Maximum payout as fraction of bankroll (20%)
            confidence_threshold: Minimum confidence to place bet
            risk_free_rate: Risk-free rate for opportunity cost
        """
        self.max_bet_fraction = max_bet_fraction
        self.max_payout = max_payout
        self.confidence_threshold = confidence_threshold
        self.risk_free_rate = risk_free_rate
        
        # Historical performance tracking
        self.bet_history = []
        self.bankroll_history = []
        
    def calculate_kelly_fraction(self, win_prob: float, odds: float) -> float:
        """
        Calculate Kelly fraction: f = (bp - q) / b
        
        Args:
            win_prob: Probability of winning
            odds: Decimal odds (e.g., 1.5 for -200 moneyline)
            
        Returns:
            Kelly fraction (optimal bet size as fraction of bankroll)
        """
        if odds <= 1.0:
            raise ValueError("Odds must be greater than 1.0")
        
        # Convert to Kelly formula variables
        b = odds - 1  # Net odds received on win
        p = win_prob   # Probability of winning
        q = 1 - p     # Probability of losing
        
        # Kelly formula: f = (bp - q) / b
        kelly_fraction = (b * p - q) / b
        
        return kelly_fraction
    
    def moneyline_to_decimal(self, moneyline: int) -> float:
        """Convert moneyline odds to decimal odds"""
        if moneyline > 0:
            # Underdog: +150 means bet $100 to win $150
            return (moneyline + 100) / 100
        else:
            # Favorite: -150 means bet $150 to win $100
            # For -110: bet $110 to win $100, so return is $210/$110 = 1.909
            return (abs(moneyline) + 100) / abs(moneyline)
    
    def calculate_edge(self, win_prob: float, moneyline: int) -> float:
        """
        Calculate betting edge
        
        Args:
            win_prob: Our estimated win probability
            moneyline: Moneyline odds
            
        Returns:
            Edge as decimal (positive = good bet)
        """
        decimal_odds = self.moneyline_to_decimal(moneyline)
        implied_prob = 1 / decimal_odds
        
        edge = win_prob - implied_prob
        return edge
    
    def estimate_win_probability(self, game_features: Dict, model_prediction: float = None) -> float:
        """
        Estimate win probability from game features and model prediction
        
        Args:
            game_features: Game features dictionary
            model_prediction: RL model prediction (optional)
            
        Returns:
            Estimated win probability
        """
        # Base probability from implied odds
        implied_prob = game_features.get('home_implied_prob', 0.5)
        
        # Adjust based on model prediction if available
        if model_prediction is not None:
            # Convert model output to probability adjustment
            # Model outputs bet fraction, convert to probability adjustment
            prob_adjustment = model_prediction * 0.1  # Scale factor
            adjusted_prob = implied_prob + prob_adjustment
        else:
            adjusted_prob = implied_prob
        
        # Apply confidence bounds
        confidence = self._calculate_confidence(game_features)
        adjusted_prob = np.clip(adjusted_prob, 0.1, 0.9)
        
        return adjusted_prob, confidence
    
    def _calculate_confidence(self, game_features: Dict) -> float:
        """Calculate confidence in our probability estimate"""
        # Factors that increase confidence:
        # - Large edge
        # - Consistent team performance
        # - Good data quality
        
        edge = abs(game_features.get('edge', 0))
        pitcher_diff = abs(game_features.get('pitcher_quality_diff', 0))
        bullpen_diff = abs(game_features.get('bullpen_quality_diff', 0))
        
        # Normalize factors
        edge_factor = min(edge * 10, 1.0)  # Edge of 0.1 = max confidence
        pitcher_factor = min(pitcher_diff / 2.0, 1.0)  # ERA diff of 2 = max confidence
        bullpen_factor = min(bullpen_diff / 1.0, 1.0)  # FIP diff of 1 = max confidence
        
        # Combine factors
        confidence = (edge_factor + pitcher_factor + bullpen_factor) / 3
        confidence = np.clip(confidence, 0.1, 0.95)
        
        return confidence
    
    def apply_risk_constraints(self, kelly_fraction: float, bankroll: float, 
                             win_prob: float, moneyline: int) -> float:
        """
        Apply risk management constraints to Kelly fraction
        
        Args:
            kelly_fraction: Raw Kelly fraction
            bankroll: Current bankroll
            win_prob: Win probability
            moneyline: Moneyline odds
            
        Returns:
            Risk-adjusted bet fraction
        """
        # 1. Kelly cap (5% as specified)
        kelly_fraction = np.clip(kelly_fraction, -self.max_bet_fraction, self.max_bet_fraction)
        
        # 2. Max payout constraint
        decimal_odds = self.moneyline_to_decimal(moneyline)
        max_payout_fraction = self.max_payout / decimal_odds
        kelly_fraction = np.clip(kelly_fraction, -max_payout_fraction, max_payout_fraction)
        
        # 3. Volatility adjustment (reduce bet size for high variance)
        variance = win_prob * (1 - win_prob)
        volatility_factor = 1.0 - (variance * 0.5)  # Reduce bet size for high variance
        kelly_fraction *= volatility_factor
        
        # 4. Bankroll protection (don't bet if bankroll too small)
        min_bankroll = 1000  # Minimum bankroll to continue betting
        if bankroll < min_bankroll:
            kelly_fraction *= 0.5  # Reduce bet size
        
        return kelly_fraction
    
    def make_betting_decision(self, game_features: Dict, bankroll: float,
                             model_prediction: float = None) -> BettingDecision:
        """
        Make betting decision using Kelly criterion with risk management
        
        Args:
            game_features: Game features dictionary
            bankroll: Current bankroll
            model_prediction: RL model prediction (optional)
            
        Returns:
            BettingDecision object
        """
        # Extract odds
        moneyline = game_features.get('home_moneyline', -110)
        
        # Estimate win probability
        win_prob, confidence = self.estimate_win_probability(game_features, model_prediction)
        
        # Calculate edge
        edge = self.calculate_edge(win_prob, moneyline)
        
        # Calculate raw Kelly fraction
        decimal_odds = self.moneyline_to_decimal(moneyline)
        raw_kelly = self.calculate_kelly_fraction(win_prob, decimal_odds)
        
        # Apply risk constraints
        risk_adjusted_kelly = self.apply_risk_constraints(
            raw_kelly, bankroll, win_prob, moneyline
        )
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            risk_adjusted_kelly *= 0.5  # Reduce bet size for low confidence
        
        # Calculate bet amount
        bet_amount = risk_adjusted_kelly * bankroll
        
        # Determine if we should bet
        should_bet = (abs(risk_adjusted_kelly) > 0.001 and  # Minimum bet size
                     confidence >= self.confidence_threshold and
                     abs(edge) > 0.02)  # Minimum edge
        
        if not should_bet:
            risk_adjusted_kelly = 0.0
            bet_amount = 0.0
        
        return BettingDecision(
            bet_fraction=risk_adjusted_kelly,
            bet_amount=bet_amount,
            edge=edge,
            kelly_fraction=raw_kelly,
            confidence=confidence,
            risk_adjusted=True
        )
    
    def update_bankroll(self, bet_decision: BettingDecision, outcome: bool, 
                       moneyline: int, bankroll: float) -> float:
        """
        Update bankroll after bet outcome
        
        Args:
            bet_decision: Original betting decision
            outcome: True if bet won, False if lost
            moneyline: Original moneyline odds
            bankroll: Current bankroll
            
        Returns:
            New bankroll
        """
        if bet_decision.bet_amount == 0:
            return bankroll
        
        decimal_odds = self.moneyline_to_decimal(moneyline)
        
        if outcome:
            # Won the bet
            winnings = bet_decision.bet_amount * (decimal_odds - 1)
            new_bankroll = bankroll + winnings
        else:
            # Lost the bet
            new_bankroll = bankroll - bet_decision.bet_amount
        
        # Track history
        self.bet_history.append({
            'bet_fraction': bet_decision.bet_fraction,
            'bet_amount': bet_decision.bet_amount,
            'edge': bet_decision.edge,
            'outcome': outcome,
            'moneyline': moneyline,
            'old_bankroll': bankroll,
            'new_bankroll': new_bankroll
        })
        
        self.bankroll_history.append(new_bankroll)
        
        return new_bankroll
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Calculate performance statistics"""
        if not self.bet_history:
            return {}
        
        # Basic stats
        total_bets = len(self.bet_history)
        winning_bets = sum(1 for bet in self.bet_history if bet['outcome'])
        win_rate = winning_bets / total_bets if total_bets > 0 else 0
        
        # ROI
        initial_bankroll = self.bankroll_history[0] if self.bankroll_history else 10000
        final_bankroll = self.bankroll_history[-1] if self.bankroll_history else initial_bankroll
        roi = (final_bankroll - initial_bankroll) / initial_bankroll
        
        # Average edge
        avg_edge = np.mean([bet['edge'] for bet in self.bet_history])
        
        # Average bet size
        avg_bet_fraction = np.mean([abs(bet['bet_fraction']) for bet in self.bet_history])
        
        # Sharpe ratio (simplified)
        returns = []
        for i in range(1, len(self.bankroll_history)):
            ret = (self.bankroll_history[i] - self.bankroll_history[i-1]) / self.bankroll_history[i-1]
            returns.append(ret)
        
        if returns:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
        else:
            sharpe = 0.0
        
        return {
            'total_bets': total_bets,
            'win_rate': win_rate,
            'roi': roi,
            'avg_edge': avg_edge,
            'avg_bet_fraction': avg_bet_fraction,
            'sharpe_ratio': sharpe,
            'final_bankroll': final_bankroll
        }

def integrate_with_rl_model(model_prediction: float, game_features: Dict, 
                          bankroll: float) -> BettingDecision:
    """
    Integrate RL model prediction with Kelly betting system
    
    Args:
        model_prediction: RL model output (bet fraction)
        game_features: Game features
        bankroll: Current bankroll
        
    Returns:
        BettingDecision
    """
    kelly_system = KellyBettingSystem()
    
    # Use model prediction to adjust win probability
    decision = kelly_system.make_betting_decision(
        game_features, bankroll, model_prediction
    )
    
    return decision

if __name__ == "__main__":
    # Test Kelly betting system
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
    
    print("Betting Decision:")
    print(f"Bet fraction: {decision.bet_fraction:.4f}")
    print(f"Bet amount: ${decision.bet_amount:.2f}")
    print(f"Edge: {decision.edge:.4f}")
    print(f"Confidence: {decision.confidence:.3f}")
    
    # Simulate outcome
    outcome = True  # Win
    new_bankroll = kelly.update_bankroll(decision, outcome, -120, 10000.0)
    print(f"New bankroll: ${new_bankroll:.2f}")
    
    # Get performance stats
    stats = kelly.get_performance_stats()
    print(f"Performance: {stats}") 