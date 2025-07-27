#!/usr/bin/env python3
"""
Simple Monitoring Dashboard for MLB Betting System
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def show_performance():
    """Display recent betting performance"""
    try:
        conn = sqlite3.connect('data/betting_log.db')
        df = pd.read_sql_query('''
            SELECT * FROM betting_suggestions 
            WHERE timestamp > datetime('now', '-7 days')
            ORDER BY timestamp DESC
        ''', conn)
        
        if not df.empty:
            print("📈 Recent Betting Performance (Last 7 Days)")
            print("=" * 50)
            print(f"Total Suggestions: {len(df)}")
            print(f"Average Confidence: {df['confidence'].mean():.2f}")
            print(f"Total Suggested Bet: ${df['suggested_bet'].sum():.2f}")
            print("\nRecent Suggestions:")
            for _, row in df.head(5).iterrows():
                print(f"  {row['timestamp']}: {row['game']} - ${row['suggested_bet']:.2f}")
        else:
            print("📊 No recent betting activity")
            
    except Exception as e:
        print(f"❌ Error loading performance data: {e}")

if __name__ == "__main__":
    show_performance()
