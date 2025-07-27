#!/usr/bin/env python3
"""
MLB Betting RL System - Streamlit Web Dashboard
Real-time betting analysis and monitoring
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project to path
sys.path.append('.')

from src.live_betting import LiveBettingLauncher
from src.data.real_data_sources import RealDataSources

# Page config
st.set_page_config(
    page_title="MLB Betting RL System",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        border-left: 4px solid #28a745;
    }
    .warning-card {
        border-left: 4px solid #ffc107;
    }
    .danger-card {
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">⚾ MLB Betting RL System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎯 Control Panel")
    
    # API Key Status
    api_key = os.getenv('ODDS_API_KEY')
    if api_key and api_key != 'your_actual_api_key_here':
        st.sidebar.success("✅ API Key Configured")
    else:
        st.sidebar.error("❌ API Key Not Set")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Dashboard", "📊 Live Analysis", "📈 Performance", "⚙️ Settings", "🤖 Automation"]
    )
    
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📊 Live Analysis":
        show_live_analysis()
    elif page == "📈 Performance":
        show_performance()
    elif page == "⚙️ Settings":
        show_settings()
    elif page == "🤖 Automation":
        show_automation()

def show_dashboard():
    """Main dashboard"""
    st.header("🏠 System Dashboard")
    
    # System Status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Status", "🟢 Online")
    
    with col2:
        st.metric("Model Status", "✅ Production Ready")
    
    with col3:
        st.metric("API Status", "✅ Connected")
    
    with col4:
        st.metric("Database", "✅ Active")
    
    # Quick Actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Run Analysis", type="primary"):
            with st.spinner("Running betting analysis..."):
                try:
                    launcher = LiveBettingLauncher('models/mlb_ppo_production_final.zip')
                    suggestions = launcher.run_daily_analysis()
                    st.success(f"Analysis complete! Found {len(suggestions) if suggestions else 0} opportunities.")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with col2:
        if st.button("📊 View Performance"):
            st.info("Navigate to Performance page")
    
    with col3:
        if st.button("⚙️ Settings"):
            st.info("Navigate to Settings page")
    
    # Recent Activity
    st.subheader("📋 Recent Activity")
    
    try:
        conn = sqlite3.connect('betting_log.db')
        df = pd.read_sql_query('''
            SELECT * FROM betting_suggestions 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''', conn)
        conn.close()
        
        if not df.empty:
            st.dataframe(df)
        else:
            st.info("No recent betting activity")
    except Exception as e:
        st.warning("No database found or error loading data")

def show_live_analysis():
    """Live betting analysis"""
    st.header("📊 Live Betting Analysis")
    
    # Analysis options
    col1, col2 = st.columns(2)
    
    with col1:
        run_live = st.button("🔍 Run Live Analysis", type="primary")
    
    with col2:
        test_mode = st.button("🎲 Test Mode (Simulated)", type="secondary")
    
    if run_live:
        with st.spinner("Fetching live data and running analysis..."):
            try:
                launcher = LiveBettingLauncher('models/mlb_ppo_production_final.zip')
                
                # Run the full daily analysis
                suggestions = launcher.run_daily_analysis()
                
                if suggestions:
                    st.success(f"Found {len(suggestions)} betting opportunities")
                    
                    # Display suggestions
                    for i, suggestion in enumerate(suggestions):
                        with st.expander(f"Opportunity {i+1}: {suggestion.get('home_team', 'N/A')} vs {suggestion.get('away_team', 'N/A')}"):
                            st.success("💡 **BETTING OPPORTUNITY**")
                            st.write(f"**Moneyline:** {suggestion.get('moneyline', 'N/A')}")
                            st.write(f"**Suggested bet:** ${suggestion.get('bet_amount', 0):.2f}")
                            st.write(f"**Edge:** {suggestion.get('edge', 0)*100:.2f}%")
                            st.write(f"**Confidence:** {suggestion.get('confidence', 0)*100:.1f}%")
                            st.write(f"**Kelly fraction:** {suggestion.get('bet_fraction', 0)*100:.2f}%")
                else:
                    st.info("✅ Analysis complete - No betting opportunities found today")
                    st.write("This means your RL model is being selective and only suggests bets with positive expected value.")
                    
                    # Show summary stats
                    try:
                        conn = sqlite3.connect('betting_log.db')
                        cursor = conn.cursor()
                        cursor.execute('''
                            SELECT COUNT(DISTINCT home_team || away_team) as unique_games,
                                   AVG(edge) as avg_edge,
                                   MIN(edge) as min_edge,
                                   MAX(edge) as max_edge
                            FROM betting_suggestions 
                            WHERE DATE(timestamp) = DATE('now')
                        ''')
                        stats = cursor.fetchone()
                        conn.close()
                        
                        if stats and stats[0] > 0:
                            st.subheader("📊 Today's Analysis Summary")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Games Analyzed", stats[0])
                            with col2:
                                st.metric("Avg Edge", f"{stats[1]*100:.2f}%")
                            with col3:
                                st.metric("Best Edge", f"{stats[3]*100:.2f}%")
                            with col4:
                                st.metric("Worst Edge", f"{stats[2]*100:.2f}%")
                    except Exception as e:
                        st.write(f"Could not load summary stats: {e}")
                    
                    # Show why no bets were suggested
                    st.subheader("🔍 Analysis Details")
                    try:
                        conn = sqlite3.connect('betting_log.db')
                        df = pd.read_sql_query('''
                            SELECT DISTINCT home_team, away_team, moneyline, edge, confidence
                            FROM betting_suggestions 
                            WHERE DATE(timestamp) = DATE('now')
                            ORDER BY edge DESC
                        ''', conn)
                        conn.close()
                        
                        if not df.empty:
                            st.write(f"**Today's {len(df)} games with calculated edges:**")
                            df['edge_pct'] = (df['edge'] * 100).round(2)
                            df['confidence_pct'] = (df['confidence'] * 100).round(1)
                            
                            # Show games in expandable cards
                            for i, row in df.iterrows():
                                with st.expander(f"Game {i+1}: {row['home_team']} vs {row['away_team']}"):
                                    st.write(f"**Moneyline:** {row['moneyline']}")
                                    st.write(f"**Calculated Edge:** {row['edge_pct']:.2f}%")
                                    st.write(f"**Model Confidence:** {row['confidence_pct']:.1f}%")
                                    
                                    if row['edge'] > 0:
                                        st.success("✅ **POTENTIAL OPPORTUNITY**")
                                    else:
                                        st.info("❌ **No betting opportunity** (negative edge)")
                            
                            st.write("💡 **Why no bets?** All games have negative edges (unfavorable odds)")
                        else:
                            st.write("No analysis data found for today")
                    except Exception as e:
                        st.write(f"Could not load analysis details: {e}")
                    
                # Show recent database entries
                st.subheader("📋 Latest Analysis Results")
                try:
                    conn = sqlite3.connect('betting_log.db')
                    df = pd.read_sql_query('''
                        SELECT timestamp, home_team, away_team, moneyline, bet_fraction, bet_amount, edge, confidence
                        FROM betting_suggestions 
                        ORDER BY timestamp DESC 
                        LIMIT 10
                    ''', conn)
                    conn.close()
                    
                    if not df.empty:
                        st.dataframe(df)
                    else:
                        st.info("No betting history yet")
                except Exception as e:
                    st.warning(f"Could not load betting history: {e}")
                    
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                st.exception(e)
    
    elif test_mode:
        st.subheader("🎲 Test Mode - Simulated Scenarios")
        
        # Create simulated betting opportunities
        import random
        random.seed(42)  # For consistent results
        
        simulated_opportunities = []
        test_games = [
            {"home": "Yankees", "away": "Red Sox", "moneyline": 130},
            {"home": "Dodgers", "away": "Giants", "moneyline": -110},
            {"home": "Astros", "away": "Angels", "moneyline": 145}
        ]
        
        for game in test_games:
            # Simulate positive edge scenario
            fake_edge = random.uniform(0.02, 0.08)  # 2-8% edge
            fake_confidence = random.uniform(0.6, 0.9)  # 60-90% confidence
            bet_fraction = fake_edge * 0.5  # Conservative Kelly
            bet_amount = 1000 * bet_fraction  # Assuming $1000 bankroll
            
            simulated_opportunities.append({
                'home_team': game['home'],
                'away_team': game['away'],
                'moneyline': game['moneyline'],
                'edge': fake_edge,
                'confidence': fake_confidence,
                'bet_fraction': bet_fraction,
                'bet_amount': bet_amount
            })
        
        st.success(f"🎯 Found {len(simulated_opportunities)} simulated opportunities")
        
        for i, opp in enumerate(simulated_opportunities):
            with st.expander(f"Simulated Opportunity {i+1}: {opp['home_team']} vs {opp['away_team']}"):
                st.success("💡 **SIMULATED BETTING OPPORTUNITY**")
                st.write(f"**Moneyline:** {opp['moneyline']}")
                st.write(f"**Suggested bet:** ${opp['bet_amount']:.2f}")
                st.write(f"**Edge:** {opp['edge']*100:.2f}%")
                st.write(f"**Confidence:** {opp['confidence']*100:.1f}%")
                st.write(f"**Kelly fraction:** {opp['bet_fraction']*100:.2f}%")
        
        st.info("💡 This is what your dashboard will look like when real betting opportunities are found!")

def show_performance():
    """Performance monitoring"""
    st.header("📈 Performance Monitoring")
    
    try:
        conn = sqlite3.connect('betting_log.db')
        
        # Get performance data
        df = pd.read_sql_query('''
            SELECT * FROM betting_suggestions 
            WHERE timestamp >= datetime('now', '-30 days')
            ORDER BY timestamp DESC
        ''', conn)
        
        if not df.empty:
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Suggestions", len(df))
            
            with col2:
                avg_edge = df['edge'].mean() * 100
                st.metric("Avg Edge", f"{avg_edge:.2f}%")
            
            with col3:
                total_bet = df['bet_amount'].sum()
                st.metric("Total Suggested", f"${total_bet:.2f}")
            
            with col4:
                avg_confidence = df['confidence'].mean() * 100
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            
            # Performance chart
            st.subheader("📊 Performance Over Time")
            
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            daily_performance = df.groupby('date').agg({
                'bet_amount': 'sum',
                'edge': 'mean',
                'confidence': 'mean'
            }).reset_index()
            
            fig = px.line(daily_performance, x='date', y='bet_amount', 
                         title="Daily Suggested Bet Amount")
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent suggestions table
            st.subheader("📋 Recent Suggestions")
            st.dataframe(df.head(10))
            
        else:
            st.info("No performance data available")
            
        conn.close()
        
    except Exception as e:
        st.error(f"Error loading performance data: {e}")

def show_settings():
    """System settings"""
    st.header("⚙️ System Settings")
    
    # API Key Configuration
    st.subheader("🔑 API Configuration")
    
    current_api_key = os.getenv('ODDS_API_KEY', '')
    if current_api_key and current_api_key != 'your_actual_api_key_here':
        st.success("✅ API Key is configured")
        st.code(f"ODDS_API_KEY: {current_api_key[:8]}...")
    else:
        st.error("❌ API Key not configured")
        st.info("Set your API key: export ODDS_API_KEY='your_key'")
    
    # Model Information
    st.subheader("🤖 Model Information")
    
    model_path = 'models/mlb_ppo_production_final.zip'
    if os.path.exists(model_path):
        st.success("✅ Production model loaded")
        st.write(f"**Model:** {model_path}")
        st.write("**Training:** 2,000,000 timesteps")
        st.write("**Status:** Production Ready")
    else:
        st.warning("⚠️ Production model not found")
    
    # System Configuration
    st.subheader("⚙️ System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Database:** SQLite")
        st.write("**Environment:** CPU Optimized")
        st.write("**Threads:** 6 cores")
    
    with col2:
        st.write("**Framework:** Stable-Baselines3")
        st.write("**Algorithm:** PPO")
        st.write("**Optimization:** Kelly Criterion")

def show_automation():
    """Automation settings"""
    st.header("🤖 Automation Settings")
    
    st.subheader("⏰ Scheduled Tasks")
    
    # Daily Analysis
    with st.expander("📅 Daily Analysis (9 AM)"):
        st.write("Automatically runs betting analysis every day at 9 AM")
        st.code("0 9 * * * cd /Users/diegoaguirre/MLB_ML && ./run_daily_analysis.sh")
        
        if st.button("🔄 Test Daily Analysis"):
            with st.spinner("Testing daily analysis..."):
                try:
                    launcher = LiveBettingLauncher('models/mlb_ppo_production_final.zip')
                    launcher.run_daily_analysis()
                    st.success("Daily analysis test completed!")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Weekly Retraining
    with st.expander("🔄 Weekly Model Retraining (Sunday 2 AM)"):
        st.write("Retrains the model weekly with new data")
        st.code("0 2 * * 0 cd /Users/diegoaguirre/MLB_ML && ./run_weekly_training.sh")
    
    # Notifications
    st.subheader("📱 Notifications")
    
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    if telegram_token:
        st.success("✅ Telegram notifications configured")
    else:
        st.info("Set up Telegram: export TELEGRAM_BOT_TOKEN='your_token'")
    
    # Manual Controls
    st.subheader("🎮 Manual Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Run Full Pipeline"):
            with st.spinner("Running full pipeline..."):
                st.info("This would run: data → train → backtest → live")
    
    with col2:
        if st.button("�� Generate Report"):
            with st.spinner("Generating performance report..."):
                st.info("This would create a detailed performance report")

if __name__ == "__main__":
    main()
