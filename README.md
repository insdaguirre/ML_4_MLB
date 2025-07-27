# MLB Reinforcement Learning Betting System

A CPU-only reinforcement learning system for MLB betting using Monte-Carlo simulation and Kelly-style bankroll management.

## Project Overview

This system implements a reinforcement learning agent that learns optimal betting strategies for MLB games using:
- **Model #6**: RL layer driven by Monte-Carlo simulation
- **Kelly-style bankroll management**: Dynamic bet sizing with risk constraints
- **CPU-optimized**: Designed to run on exactly 6 cores of Intel i9

## Setup Instructions

### Phase 0: Environment Setup

1. **Create virtual environment:**
```bash
python -m venv mlb_betting_env
source mlb_betting_env/bin/activate  # On Windows: mlb_betting_env\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set CPU thread limits:**
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

4. **Run training with core pinning:**
```bash
taskset -c 0-5 python train.py
```

## Project Structure

```
MLB_ML/
├── data/                   # Data pipeline outputs
├── models/                 # Trained RL models
├── src/
│   ├── data/              # Data processing modules
│   ├── simulation/         # Monte-Carlo game simulator
│   ├── rl/                # RL agent and environment
│   ├── betting/           # Kelly betting logic
│   └── utils/             # Utility functions
├── notebooks/              # Analysis notebooks
├── config/                 # Configuration files
└── logs/                  # Training logs
```

## Roadmap Progress

- [x] Phase 0: Environment Setup
- [ ] Phase 1: Data Pipeline (Historical game & odds data)
- [ ] Phase 2: Baseline Simulator (Monte-Carlo engine)
- [ ] Phase 3: RL Agent (PPO training)
- [ ] Phase 4: Kelly & Bankroll Layer
- [ ] Phase 5: Back-test & Stress-test
- [ ] Phase 6: Live Launcher
- [ ] Phase 7: Maintenance Loop

## Key Features

- **CPU-optimized**: Runs on 6 cores with parallel environments
- **Monte-Carlo simulation**: Fast game outcome simulation
- **Kelly criterion**: Dynamic bet sizing with risk management
- **Walk-forward validation**: Robust backtesting methodology
- **Live deployment**: Automated betting suggestions

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv mlb_betting_env
source mlb_betting_env/bin/activate  # On Windows: mlb_betting_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set CPU thread limits
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

### 2. Test the System
```bash
# Run system tests
python test_system.py
```

### 3. Run Full Pipeline
```bash
# Run complete pipeline (data → train → backtest → live)
python train.py --phase all

# Or run individual phases
python train.py --phase data      # Phase 1: Data pipeline
python train.py --phase train     # Phase 3: RL training
python train.py --phase backtest  # Phase 5: Backtesting
python train.py --phase live      # Phase 6: Live betting
```

### 4. CPU-Optimized Training
```bash
# Run training with core pinning (6 cores)
taskset -c 0-5 python train.py --phase train
```

## Usage Examples

### Training the Agent
```bash
# Full training (2M steps, 6-8 hours overnight)
python train.py --phase train --timesteps 2000000 --num-envs 6
```

### Running Backtests
```bash
# Walk-forward backtesting
python -c "from src.utils.backtest import BacktestingSystem; BacktestingSystem().walk_forward_backtest(2010, 2023)"
```

### Live Betting
```bash
# Run daily analysis
python -c "from src.live_betting import LiveBettingLauncher; LiveBettingLauncher().run_daily_analysis()"
```

### Monte-Carlo Validation
```bash
# Run 5k Monte-Carlo paths
python -c "from src.utils.backtest import BacktestingSystem; BacktestingSystem().monte_carlo_validation('models/mlb_ppo_final.zip', 2019)"
```

## How It Works

### 🎯 System Overview

The MLB Betting RL System implements **"Model #6"** - a reinforcement learning layer driven by Monte-Carlo simulation and Kelly-style bankroll management, optimized for exactly 6 cores on your Intel i9.

### 📊 Data Pipeline (Phase 1)

The system starts by processing historical MLB data:

```python
# Historical game data + odds processing
pipeline = MLBDataPipeline()
pipeline.download_retrosheet_data(2010, 2023)  # Game results, stats
pipeline.download_odds_data(2010, 2023)         # Moneyline odds
features_df = pipeline.process_features()         # Join & compute features
```

**What it does:**
- Downloads historical MLB game data (Retrosheet play-by-play)
- Fetches moneyline closing odds from betting APIs
- Joins data on game_id and computes derived features:
  - Starting pitcher ERA, park factors, bullpen FIP
  - Implied probabilities from odds
  - Kelly fractions, edges, confidence scores
- Stores as Parquet for fast reload

### 🎲 Monte-Carlo Game Simulator (Phase 2)

The system uses vectorized game simulation for speed:

```python
# Vectorized game simulation
simulator = MonteCarloSimulator(num_sims=1000)
game_results, payoffs = simulate_game(game_row, num_sims=1000)
```

**How it works:**
- **Vectorized simulation**: Uses NumPy for speed (1 season ≈ 0.1s)
- **Game modeling**: Simulates runs using Poisson distributions
- **Team adjustments**: Factors in pitcher quality, park factors, bullpen strength
- **Betting outcomes**: Calculates payoffs based on moneyline odds
- **Kelly integration**: Converts to bet fractions with 5% cap

**Key insight**: Instead of simulating every pitch, it models run distributions directly for speed.

### 🧠 Reinforcement Learning Environment (Phase 3)

The system uses a Gym-style environment with continuous actions:

```python
# Gym-style environment with continuous actions
env = BaseballBetEnv(
    data_pipeline=pipeline,
    initial_bankroll=10000.0,
    num_sims=1000,
    max_games=162
)
```

**State Space**: Game features + current bankroll
- Pitcher ERAs, park factors, implied probabilities
- Normalized bankroll (current/initial)

**Action Space**: Bet fraction ∈ [-0.05, +0.05]
- Positive = bet on home team
- Negative = bet on away team
- 5% Kelly cap enforced

**Reward**: Bankroll delta per game
- Win = positive payoff
- Loss = negative payoff
- Episode = full season (162 games)

### 🧠 PPO Training (Phase 3)

The system uses a tiny network architecture for CPU efficiency:

```python
# Tiny network architecture for CPU efficiency
model = CustomPPO(
    "MlpPolicy",
    train_env,
    policy_kwargs={
        'net_arch': [dict(pi=[128, 64], vf=[128, 64])],
        'activation_fn': nn.ReLU,
        'final_activation_fn': nn.Tanh
    }
)
```

**Network Architecture**: [Linear(128) → ReLU → Linear(64) → Tanh]
- **Why tiny?** Keeps forward/backward FLOPs minimal
- **CPU-friendly**: No massive networks to bottleneck training
- **Parallel environments**: 6 environments, one per core

**Training Process**:
1. Agent observes game state (features + bankroll)
2. Outputs bet fraction (continuous action)
3. Environment simulates game outcome
4. Updates bankroll based on result
5. Agent learns optimal betting strategy

### 💰 Kelly Betting System (Phase 4)

The system implements dynamic Kelly criterion with risk management:

```python
# Dynamic Kelly with risk management
kelly = KellyBettingSystem(max_bet_fraction=0.05)
decision = kelly.make_betting_decision(game_features, bankroll, model_prediction)
```

**Kelly Formula**: `f = (bp - q) / b`
- `b` = net odds received on win
- `p` = our estimated win probability
- `q` = probability of losing (1-p)

**Risk Management**:
- **5% cap**: Maximum bet fraction
- **Max payout constraint**: Limits exposure
- **Volatility adjustment**: Reduces bet size for high variance
- **Confidence threshold**: Only bet when confident

**Integration with RL**:
- RL model adjusts win probability estimates
- Kelly system calculates optimal bet size
- Risk constraints applied for safety

### 📈 Walk-Forward Backtesting (Phase 5)

The system uses robust validation methodology:

```python
# Walk-forward validation
backtest = BacktestingSystem()
results = backtest.walk_forward_backtest(2010, 2023)
```

**Process**:
1. **Train on 2010-2018**, test on 2019
2. **Slide window**: Train on 2011-2019, test on 2020
3. **Continue**: Each year becomes test data after training
4. **Aggregate**: Average performance across all test years

**Monte-Carlo Validation**:
```python
# 5k bankroll paths per season
mc_results = backtest.monte_carlo_validation(model_path, 2019, num_paths=5000)
```

- Runs 5,000 different possible season outcomes
- Plots distribution of ending wealth
- Calculates Value at Risk (VaR)
- Estimates probability of positive ROI

### 🤖 Live Betting System (Phase 6)

The system provides automated betting suggestions:

```python
# Automated betting suggestions
launcher = LiveBettingLauncher()
launcher.run_daily_analysis()
```

**Daily Process**:
1. **9 AM ET**: Pull today's MLB odds
2. **Feature extraction**: Convert odds to game features
3. **Model prediction**: Get RL agent's bet recommendation
4. **Kelly calculation**: Apply Kelly criterion with risk management
5. **Decision**: Bet or pass based on edge and confidence
6. **Notification**: Send Telegram message with suggestion
7. **Logging**: Store in SQLite for analysis

### 🔄 Maintenance Loop (Phase 7)

The system includes automated maintenance:

```python
# Weekly retraining
# 500k PPO steps runs in ~90 min on 6 cores
model.learn(total_timesteps=500_000)
```

**Weekly Process**:
- **Sunday night**: Retrain with last 7 days of data
- **Performance monitoring**: Track edge decay
- **Adaptive tuning**: Adjust if ROI < 0 for 4 weeks
- **Model updates**: Save new checkpoints

### 🚀 Why It Works on 6 Cores

#### **Parallel Environments**
```python
# Each core handles one environment
env = VecBaseballEnv(num_envs=6)  # 6 parallel environments
```

- PPO spends most time generating roll-outs
- Roll-outs scale linearly with CPU cores
- Each environment runs independently

#### **Vectorized Simulation**
```python
# NumPy vectorization for speed
runs = np.random.poisson(mean_runs, num_sims)  # All sims at once
```

- Monte-Carlo simulations are embarrassingly parallel
- NumPy operations are CPU-optimized
- No GPU needed for this workload

#### **Tiny Network**
```python
# Minimal forward/backward passes
net_arch = [dict(pi=[128, 64], vf=[128, 64])]  # Small network
```

- Forward pass: 128 → 64 neurons
- Backward pass: Minimal gradients
- CPU can handle this easily

### 📊 Performance Flow

```
1. Data Pipeline     → Process historical games & odds
2. Monte-Carlo      → Simulate 10k games in seconds  
3. RL Environment   → Agent learns betting strategy
4. Kelly System     → Calculate optimal bet sizes
5. Backtesting      → Validate on out-of-sample data
6. Live Deployment  → Automated betting suggestions
7. Maintenance      → Weekly retraining & monitoring
```

### 🎯 Key Innovations

1. **CPU-Optimized**: No GPU needed, scales with cores
2. **Kelly Integration**: Mathematical optimal bet sizing
3. **Risk Management**: Multiple layers of protection
4. **Walk-Forward**: Robust validation methodology
5. **Live Deployment**: Automated production system

The system essentially learns to be a smart sports bettor by playing thousands of simulated seasons, optimizing for long-term bankroll growth while managing risk through Kelly criterion and multiple safety constraints.

## Performance Targets

- **Simulation Speed**: 1 season ≈ 0.1s
- **Training**: 10k rollouts/hour on 6 cores
- **Overnight Training**: 2-3M steps in 6-8 hours

## Disclaimer

This system is for educational and research purposes. Please bet responsibly and be aware of the risks involved in sports betting. 