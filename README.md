# ⚾ MLB Betting RL System

**CPU-Optimized Reinforcement Learning System for MLB Analytics with Kelly Criterion Bankroll Management**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%20Only-red.svg)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-green.svg)](https://stable-baselines3.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange.svg)](https://streamlit.io/)

## 🎯 System Overview

This is a **production-ready MLB betting system** that combines reinforcement learning (RL) with Monte-Carlo simulation and Kelly Criterion bankroll management. The system is specifically optimized for **CPU-only operation**, making it accessible without expensive GPU hardware.

### 🏆 Key Features

- **🤖 RL Agent**: Proximal Policy Optimization (PPO) with tiny neural networks
- **🎲 Monte-Carlo Simulation**: Vectorized game outcome simulation
- **💰 Kelly Criterion**: Optimal bet sizing and bankroll management
- **⚡ CPU Optimized**: Pinned to 6 cores with single-threaded operations
- **📊 Live Dashboard**: Streamlit web interface for real-time monitoring
- **🔗 Real Data**: Integration with The Odds API for live MLB odds
- **📈 Performance Tracking**: SQLite database for betting history and ROI

## 🏗️ System Architecture

### System Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Processing     │    │   Live System   │
│                 │    │                 │    │                 │
│ • The Odds API  │───▶│ • Feature Eng.  │───▶│ • RL Agent      │
│ • Retrosheet    │    │ • Monte-Carlo   │    │ • Kelly System  │
│ • Statcast      │    │ • Simulation    │    │ • Dashboard     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Optimization  │
                       │                 │
                       │ • CPU Pinning   │
                       │ • Vectorization │
                       │ • Tiny Networks │
                       └─────────────────┘
```

### Data Flow Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Data Pipeline  │    │ Simulation      │
│                 │    │                 │    │ Engine          │
│ • The Odds API  │───▶│ • Feature Eng.  │───▶│ • Monte-Carlo   │
│ • Retrosheet    │    │ • Processing    │    │ • Game Outcomes │
│ • Statcast      │    │ • Aggregation   │    │ • Bankroll Paths│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  RL Environment │    │   Kelly Layer   │
                       │                 │    │                 │
                       │ • BaseballBetEnv│───▶│ • Kelly System  │
                       │ • PPO Agent     │    │ • Optimal Sizing│
                       │ • Action Space  │    │ • Risk Mgmt     │
                       └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────────────────────────────┐
                       │           Live System                   │
                       │                                         │
                       │ • LiveBettingLauncher                   │
                       │ • SQLite Database                       │
                       │ • Telegram/Slack Alerts                 │
                       │ • Streamlit Dashboard                   │
                       └─────────────────────────────────────────┘
```

### Component Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    Core Components                            │
├─────────────────┬─────────────────┬─────────────────┬─────────┤
│ Data Pipeline   │ Simulation      │ RL Environment  │ Kelly   │
│                 │ Engine          │                 │ System  │
│ • Feature Eng.  │ • Monte-Carlo   │ • BaseballBetEnv│ • Kelly │
│ • Processing    │ • Game Outcomes │ • PPO Agent     │ • Risk  │
│ • Aggregation   │ • Bankroll Paths│ • Action Space  │ • Sizing│
└─────────────────┴─────────────────┴─────────────────┴─────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                  Optimization Layer                           │
├─────────────────┬─────────────────┬─────────────────┬─────────┤
│ CPU Pinning     │ Single Threading│ Vectorized Ops  │ Tiny    │
│                 │                 │                 │ Networks│
│ • Core 0-5      │ • OMP_NUM_THREADS│ • NumPy Arrays │ • 128→64│
│ • taskset       │ • MKL_NUM_THREADS│ • Polars       │ • ReLU  │
│ • Performance   │ • torch.set_num  │ • Efficiency   │ • Tanh  │
└─────────────────┴─────────────────┴─────────────────┴─────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────┐
│                   Deployment Layer                            │
├─────────────────┬─────────────────┬─────────────────┬─────────┤
│ Streamlit       │ Automation      │ API Integration │ Database│
│ Dashboard       │ Scripts         │                 │ Logging │
│ • Real-time UI  │ • Cron Jobs     │ • The Odds API  │ • SQLite│
│ • Performance   │ • Daily Analysis│ • Webhooks      │ • Hist. │
│ • Monitoring    │ • Alerts        │ • Rate Limiting │ • ROI   │
└─────────────────┴─────────────────┴─────────────────┴─────────┘
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+ with pip
python --version

# Install dependencies
pip install -r requirements.txt
```

### Setup API Keys

```bash
# Set your Odds API key
export ODDS_API_KEY="your_api_key_here"

# Or create .env file
echo "ODDS_API_KEY=your_api_key_here" > .env
```

### Launch Dashboard

```bash
# Start the web dashboard
./launch_dashboard_fixed.sh

# Or manually
streamlit run app.py --server.port 8501
```

### Run Analysis

```bash
# Run daily betting analysis
python -c "
from src.live_betting import LiveBettingLauncher
launcher = LiveBettingLauncher('models/mlb_ppo_production_final.zip')
launcher.run_daily_analysis()
"
```

## 📊 How It Works

### 1. Data Collection & Processing

**Justification**: Real-time odds data is essential for live betting. The Odds API provides reliable, structured data with minimal latency.

```
Data Pipeline Flow:
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Raw Odds   │───▶│ Feature Eng.    │───▶│ Processed Data  │
└─────────────┘    └─────────────────┘    └─────────────────┘
       │                     │                     │
       ▼                     ▼                     ▼
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
│Historical   │───▶│Statistical      │───▶│Model Training   │
│Games        │    │Features         │    │Data             │
└─────────────┘    └─────────────────┘    └─────────────────┘
       │                     │                     │
       ▼                     ▼                     ▼
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
│Advanced     │───▶│Performance      │───▶│Edge Calculation │
│Metrics      │    │Indicators       │    │& Kelly Fraction │
└─────────────┘    └─────────────────┘    └─────────────────┘
```

**Key Features Extracted**:
- **Moneyline odds** → Implied probability
- **Pitcher ERA/FIP** → Quality differentials
- **Park factors** → Run environment adjustments
- **Bullpen metrics** → Late-game advantage
- **Historical matchups** → Team performance patterns

### 2. Monte-Carlo Simulation Engine

**Justification**: Monte-Carlo simulation provides robust probability estimates by running thousands of game scenarios, accounting for uncertainty in sports outcomes.

```
Simulation Process:
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
│Game Features│───▶│10,000 Sims      │───▶│Outcome Dist.    │
└─────────────┘    └─────────────────┘    └─────────────────┘
       │                     │                     │
       ▼                     ▼                     ▼
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
│Win Prob.    │───▶│Expected Value   │───▶│Kelly Fraction   │
└─────────────┘    └─────────────────┘    └─────────────────┘
       │                     │                     │
       ▼                     ▼                     ▼
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
│Bankroll Path│───▶│Risk Assessment  │───▶│Bet Decision     │
└─────────────┘    └─────────────────┘    └─────────────────┘
```

**Simulation Parameters**:
- **10,000 iterations** per game for statistical significance
- **Vectorized operations** for CPU efficiency
- **Realistic variance** based on historical data
- **Correlated outcomes** (runs, hits, errors)

### 3. Reinforcement Learning Environment

**Justification**: RL allows the system to learn optimal betting strategies through trial and error, adapting to changing market conditions and improving over time.

```
RL Environment Design:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│Observation      │───▶│Action Space     │───▶│Reward Function  │
│Space:           │    │Bet Fraction     │    │Bankroll Delta   │
│[Game Features + │    │0.0 to 0.1       │    │Direct P&L       │
│ Bankroll +      │    │(10% max bet)    │    │                 │
│ Market State]   │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
       │                       │                       │
       ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│Training:        │    │PPO Algorithm    │    │Stable, Sample-  │
│Stable, Sample-  │───▶│Sample-efficient │───▶│efficient        │
│efficient        │    │Learning         │    │Learning         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Environment Features**:
- **Continuous action space** for precise bet sizing
- **Realistic constraints** (Kelly limits, bankroll management)
- **Delayed rewards** (game outcomes take time)
- **Non-stationary environment** (odds change, teams improve)

### 4. Kelly Criterion Integration

**Justification**: Kelly Criterion maximizes long-term growth rate while managing risk. It's mathematically optimal for repeated betting scenarios.

```
Kelly Formula Implementation:
┌────────────────────────────────────────────────────────────┐
│                    Kelly Formula                           │
│                                                            │
│  Kelly Fraction = (bp - q) / b                             │
│                                                            │
│  where:                                                    │
│  • b = decimal odds - 1                                    │
│  • p = probability of winning                              │
│  • q = probability of losing (1 - p)                       │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│                  Risk Management                           │
│                                                            │
│  Max Bet = min(Kelly Fraction, 0.05)  # 5% cap             │
│  Min Bet = max(calculated_bet, 0.001)  # $1 minimum        │
└────────────────────────────────────────────────────────────┘
```

**Risk Constraints**:
- **5% maximum bet** to prevent catastrophic losses
- **Minimum bet size** for practical implementation
- **Bankroll protection** during losing streaks
- **Dynamic adjustment** based on performance

### 5. Live Betting System

**Justification**: Automated execution ensures consistent application of the strategy, removes emotional bias, and captures opportunities 24/7.

```
Live System Flow:
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
│API Polling  │───▶│Game Detection   │───▶│FeatureExtraction│
└─────────────┘    └─────────────────┘    └─────────────────┘
       │                     │                       │
       ▼                     ▼                       ▼
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
│Model        │───▶│Kelly Calculation│───▶│Decision Logic   │
│Prediction   │    │                 │    │                 │
└─────────────┘    └─────────────────┘    └─────────────────┘
       │                     │                       │
       ▼                     ▼                       ▼
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
│Database     │───▶│Notification     │───▶│Performance      │
│Logging      │    │System           │    │Tracking         │
└─────────────┘    └─────────────────┘    └─────────────────┘
```

**System Components**:
- **Real-time odds polling** (every 5 minutes)
- **Automated decision making** (no human intervention)
- **Comprehensive logging** (all decisions and outcomes)
- **Performance monitoring** (ROI, Sharpe ratio, win rate)

## 🎯 Performance Targets

### Expected Results (Based on Backtesting)

| Metric | Target | Justification |
|--------|--------|---------------|
| **ROI** | 8-12% annually | Conservative edge capture with risk management |
| **Sharpe Ratio** | >1.5 | Risk-adjusted returns above market |
| **Win Rate** | 52-55% | Slight edge over 50/50 with good odds |
| **Max Drawdown** | <15% | Kelly Criterion limits exposure |
| **Bet Frequency** | 15-25% of games | Selective betting on positive EV |

### System Performance Metrics

```
┌────────────────────────────────────────────────────────────┐
│                    Performance Targets                     │
├────────────────────────────────────────────────────────────┤
│  Training Speed:    10k rollouts/hour on 6 cores           │
│  Simulation Speed:  1 season ≈ 0.1 seconds                 │
│  Model Size:        128→64→1 neurons (tiny network)        │
│  Memory Usage:      <2GB RAM for full pipeline             │
│  API Latency:       <100ms for odds retrieval              │
│  Database Queries:  <10ms for betting history              │
└────────────────────────────────────────────────────────────┘
```

### Risk Management

```python
# Risk Parameters
Max Bankroll Exposure: 5% per bet
Maximum Daily Bets: 10 games
Stop Loss: 20% bankroll decline
Position Sizing: Kelly Fraction × 0.5 (conservative)
```

## 🔧 Technical Implementation

### CPU Optimization Strategy

**Justification**: CPU-only operation reduces costs and complexity while maintaining performance through intelligent optimization.

```bash
# Core Pinning (Linux)
taskset -c 0-5 python train.py

# Thread Limits
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
torch.set_num_threads(6)
```

**Optimization Techniques**:
- **Core pinning** to specific CPU cores
- **Single-threaded operations** to prevent context switching
- **Vectorized NumPy operations** for batch processing
- **Tiny neural networks** (128→64→1) for fast inference
- **Memory-efficient data structures** (Polars vs Pandas)

### Model Architecture

```python
# PPO Network Design
Policy Network: [128 → ReLU → 64 → Tanh]
Value Network: [128 → ReLU → 64 → Linear]
Action Space: Continuous [0.0, 0.1]
Observation Space: 15 features + bankroll state
```

**Design Rationale**:
- **Small networks** for CPU efficiency
- **Continuous actions** for precise bet sizing
- **Separate policy/value** for stable learning
- **Bounded outputs** for risk management

### Data Pipeline Efficiency

```python
# Processing Pipeline
Raw Data → Polars DataFrame → Vectorized Operations → Features
     ↓
Batch Processing → Memory Mapping → Compressed Storage
     ↓
Real-time Updates → Incremental Processing → Live Features
```

**Performance Optimizations**:
- **Polars** for fast DataFrame operations
- **Parquet format** for compressed storage
- **Vectorized feature computation**
- **Incremental updates** for live data

## 📈 Deployment Guide

### Production Setup

1. **Environment Configuration**
```bash
# Set production environment
export ODDS_API_KEY="your_production_key"
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

2. **Model Training**
```bash
# Train production model (2M timesteps)
python production_train.py
```

3. **Automation Setup**
```bash
# Add to crontab for daily runs
0 9 * * * cd /path/to/MLB_ML && ./run_daily_analysis.sh
```

### Monitoring & Alerts

```python
# Performance Monitoring
Daily ROI Tracking
Weekly Sharpe Ratio Calculation
Monthly Drawdown Analysis
Quarterly Strategy Review
```

## 🧪 Testing & Validation

### Backtesting Methodology

**Justification**: Walk-forward validation prevents overfitting and provides realistic performance estimates.

```python
# Walk-Forward Validation
Training Window: 2 years
Testing Window: 6 months
Rolling Window: Monthly updates
Validation Metric: Out-of-sample ROI
```

### Stress Testing

```python
# Stress Test Scenarios
Market Crash: -30% bankroll simulation
Losing Streak: 10 consecutive losses
Odds Movement: Rapid line changes
Data Quality: Missing/incomplete data
```

## 📊 Dashboard Features

### Real-Time Monitoring

- **Live odds integration** with The Odds API
- **Real-time betting suggestions** with edge calculations
- **Performance tracking** with ROI and Sharpe ratio
- **Risk monitoring** with drawdown alerts
- **Historical analysis** with detailed game logs

### Dashboard Components

```python
# Dashboard Sections
1. System Status: API connectivity, model health
2. Live Analysis: Current betting opportunities
3. Performance: Historical ROI and statistics
4. Settings: Configuration and parameters
5. Automation: Scheduled tasks and alerts
```

## 🔒 Security & Compliance

### Data Protection

- **API key encryption** in environment variables
- **Database encryption** for sensitive betting data
- **Access logging** for audit trails
- **Backup systems** for data recovery

### Legal Compliance

- **Paper trading mode** for testing
- **Jurisdiction compliance** for live betting
- **Tax reporting** integration
- **Responsible gambling** limits

## 🚀 Future Enhancements

### Planned Improvements

1. **Multi-sport expansion** (NBA, NFL, NHL)
2. **Advanced ML models** (XGBoost, Neural Networks)
3. **Real-time streaming** (WebSocket integration)
4. **Mobile app** (React Native dashboard)
5. **Social features** (betting communities)

### Research Areas

- **Alternative RL algorithms** (SAC, TD3)
- **Ensemble methods** (multiple model combination)
- **Market microstructure** (odds movement analysis)
- **Sentiment analysis** (news impact on odds)

## 📚 References & Resources

### Academic Papers

- [Kelly Criterion for Optimal Betting](https://en.wikipedia.org/wiki/Kelly_criterion)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)
- [Monte Carlo Methods](https://en.wikipedia.org/wiki/Monte_Carlo_method)

### Data Sources

- [The Odds API](https://the-odds-api.com/) - Live sports odds
- [Retrosheet](https://www.retrosheet.org/) - Historical game data
- [Baseball Savant](https://baseballsavant.mlb.com/) - Advanced metrics

### Tools & Libraries

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [Streamlit](https://streamlit.io/) - Web dashboard
- [Polars](https://polars.apache.org/) - Fast DataFrame operations

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/MLB_ML.git
cd MLB_ML

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black src/ tests/
```

### Code Standards

- **Type hints** for all functions
- **Docstrings** for all classes and methods
- **Unit tests** for critical components
- **Integration tests** for end-to-end workflows

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**This system is for educational and research purposes only. Sports betting involves risk and may result in financial loss. Always gamble responsibly and within your means. The authors are not responsible for any financial losses incurred through the use of this system.**

---

**Built with ❤️ for the intersection of sports, data science, and reinforcement learning.** 
